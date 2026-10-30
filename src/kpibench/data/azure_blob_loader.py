"""Carrega e concatena interações em JSON a partir de blobs Azure.

Uso rápido:
    python -m kpibench.data.azure_blob_loader \
      --start-date 2026-09-01 --end-date 2026-10-28 --num-calls 500

Se `--start-date`/`--end-date` não forem informados (modo de produção), a
janela é calculada automaticamente como a ÚLTIMA SEMANA REGISTRADA NA BASE —
os 7 dias mais recentes que de fato têm blobs no container, e não os últimos
7 dias corridos a partir de hoje (a ingestão de produção pode atrasar e 'hoje'
ainda não ter dados). As datas efetivamente usadas são sempre impressas.

`--num-calls` limita quantas chamadas (1 blob = 1 chamada) entram na análise;
se informado, é sempre elevado para o mínimo de `MIN_CALLS` (10).

A rotina busca os JSONs sob `ACCOUNT_URL`/`CONTAINER_NAME`/`PATH_PREFIX` (ou do
intervalo informado) e normaliza nomes de métricas que variam entre bases.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential


DEFAULT_METRIC_ALIASES: dict[str, str] = {
    "csat": "csat",
    "customer_satisfaction": "csat",
    "satisfacao": "csat",
    "satisfacao_cliente": "csat",
    "sentimento": "sentimento",
    "sentiment": "sentimento",
    "qualidade_resumo": "qualidade_resumo",
    "summary_quality": "qualidade_resumo",
    "resumo_qualidade": "qualidade_resumo",
    "resolucao": "resolucao",
    "resolved": "resolucao",
    "solicitacao_resolvida": "solicitacao_resolvida",
    "first_call_resolution": "solicitacao_resolvida",
    "first_call_resolution_rate": "solicitacao_resolvida",
    "razao_ligacao": "razao_ligacao",
    "call_reason": "razao_ligacao",
    "motivo_ligacao": "razao_ligacao",
    "tempo_total": "tempo_total",
    "total_duration": "tempo_total",
    "duracao_total": "tempo_total",
    "empatia": "empatia",
    "empathetic": "empatia",
    "clareza_informacoes": "clareza_informacoes",
    "clarity": "clareza_informacoes",
    "odc": "odc",
    "on_demand_conversion": "odc",
    "valor_negocio": "valor_negocio",
}

ACCOUNT_URL = "https://saatenadev.blob.core.windows.net/"
CONTAINER_NAME = "cs-brasilcap"
PATH_PREFIX = "reports/ven-fin-cap/"
CREDENTIAL = DefaultAzureCredential()

# Janela padrão (em dias) usada quando start_date/end_date não são informados:
# "última semana" = 7 dias, terminando no dia mais recente que de fato tem
# blobs registrados na base (não necessariamente 'hoje').
DEFAULT_WINDOW_DAYS = 1

# Quantidade mínima de chamadas (blobs) usada em uma análise, mesmo que o
# chamador peça um número menor via `num_calls` — abaixo disso as métricas
# estatísticas (bootstrap, concordância por KPI) ficam pouco confiáveis.
MIN_CALLS = 10



def normalize_metric_name(name: str | None, aliases: dict[str, str] | None = None) -> str:
    """Padroniza nomes de colunas/metricas para um mesmo formato."""
    if name is None:
        return ""
    norm = str(name).strip().lower()
    norm = unicodedata_nfkd(norm)
    norm = re.sub(r"[^a-z0-9]+", "_", norm)
    norm = re.sub(r"_+", "_", norm).strip("_")
    mapping = DEFAULT_METRIC_ALIASES.copy()
    if aliases:
        mapping.update({k.lower(): v.lower() for k, v in aliases.items()})
    if norm in mapping:
        return mapping[norm]
    # se o nome já estiver em um alias inverso, resolve também em formato canonizado
    for key, value in mapping.items():
        if key == norm:
            return value
    return norm


def unicodedata_nfkd(text: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


# Chaves de nível superior candidatas a conter os registros de KPI, em ordem de
# prioridade. "metrics_items" vem primeiro pois é onde ficam os KPIs de produção
# nos blobs de reports/ven-fin-cap/ (os demais blocos do mesmo JSON — sentiment_items,
# summary_items, transcription_items, audio_info — não são KPIs e são ignorados
# aqui para não poluir o DataFrame com listas enormes).
_RECORD_KEYS = ("metrics_items", "interactions", "records", "items", "data", "results")


def _iter_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        selected_key = next((k for k in _RECORD_KEYS if k in payload), None)
        selected = payload[selected_key] if selected_key is not None else payload
        items = selected if isinstance(selected, list) else [selected]
    else:
        items = []

    records: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            # "cost" é telemetria de tokens/deployment por campo, não um KPI —
            # removida antes de achatar para não colidir com nomes de coluna reais.
            records.append({k: v for k, v in item.items() if k != "cost"})
    return records


def _flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_record(value, new_key))
        elif isinstance(value, list):
            flat[new_key] = value
        else:
            flat[new_key] = value
    return flat


# Padrões de particionamento observados nos blobs reais, ex.:
#   summary/ven-fin-cap/date=20260918/<uuid>.summary.json
# O padrão com prefixo "date=" vem primeiro para evitar casar dígitos do uuid.
_DATE_PATTERNS = (
    re.compile(r"date=(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"date=(\d{4})(\d{2})(\d{2})"),
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"(\d{4})(\d{2})(\d{2})"),
    re.compile(r"(\d{4})/(\d{2})/(\d{2})"),
)


def _extract_blob_date(blob_name: str) -> date | None:
    """Extrai a data de particionamento do caminho do blob (pasta 'date=YYYYMMDD').

    Monta a data a partir dos grupos capturados (ano/mês/dia), em vez de
    reformatar a string casada e reanalisá-la com strptime — isso evitava que
    casamentos sem separador (ex.: '20260918') nunca virassem uma data válida.
    """
    for pattern in _DATE_PATTERNS:
        match = pattern.search(blob_name)
        if match:
            year, month, day = (int(g) for g in match.groups())
            try:
                return date(year, month, day)
            except ValueError:
                continue
    return None


def _extract_call_id(blob_name: str) -> str:
    """Extrai o identificador da chamada a partir do nome do arquivo (uuid antes do primeiro '.')."""
    stem = blob_name.rsplit("/", 1)[-1]
    return stem.split(".")[0]


# =============================================================================
# Seleção dos KPIs do projeto (config/kpis.yaml) a partir dos campos brutos
# encontrados em metrics_items dos blobs de produção (reports/ven-fin-cap/).
# -----------------------------------------------------------------------------
# Os 10 KPIs oficiais do benchmark:
ALL_KPI_NAMES: tuple[str, ...] = (
    "qualidade_resumo",
    "csat",
    "razao_csat",
    "solicitacao_resolvida",
    "solver_reasoning",
    "razao_ligacao",
    "first_call_resolution",
    "empatia",
    "clareza_informacoes",
    "odc",
)

# kpi -> campo bruto observado em metrics_items (já após normalize_metric_name,
# que renomeia "call_reason" para "razao_ligacao" via DEFAULT_METRIC_ALIASES —
# por isso o valor abaixo é "razao_ligacao", não "call_reason"). "qualidade_resumo"
# fica de fora de propósito: não há nenhum campo de qualidade do resumo nos
# relatórios de produção (ver check_kpi_coverage).
KPI_FIELD_MAP: dict[str, str] = {
    "csat": "csat",
    "razao_csat": "csat_reasoning",
    "solicitacao_resolvida": "solved",
    "solver_reasoning": "solved_reasoning",
    "razao_ligacao": "razao_ligacao",
    "first_call_resolution": "dash_advins_fcr",
    "empatia": "dash_advins_empathy",
    "clareza_informacoes": "dash_advins_clarity",
    "odc": "dash_advins_odc",
}

# KPIs cujo campo de produção existe, mas tem domínio DIFERENTE do esperado por
# config/kpis.yaml — o valor é carregado como prod_<kpi> mesmo assim (é o dado
# real disponível), mas não deve ser usado como baseline sem reclassificação.
KPI_DOMAIN_MISMATCH: dict[str, str] = {
    "empatia": "produção retorna Sim/Não; kpis.yaml espera nota ordinal 1-5",
    "clareza_informacoes": "produção retorna Sim/Não; kpis.yaml espera nota ordinal 1-5",
    "odc": "produção retorna Sim/Não; kpis.yaml espera taxonomia categórica de 7 rótulos",
    "razao_ligacao": "produção retorna texto livre (call_reason); kpis.yaml espera 1 dos 8 rótulos fixos",
    "solicitacao_resolvida": "produção só distingue Sim/Não; kpis.yaml também espera 'parcial'",
    "first_call_resolution": "produção só distingue Sim/Não; kpis.yaml também espera 'indeterminado'",
}

_SCORE_PREFIX_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)")


def _parse_scored_text(value: Any) -> float | None:
    """'Nota: 6' -> 6.0; 'Probabilidade: 55%' -> 55.0."""
    if value is None:
        return None
    match = _SCORE_PREFIX_RE.search(str(value))
    return float(match.group(1).replace(",", ".")) if match else None


def _parse_sim_nao(value: Any) -> str | None:
    """Normaliza 'Sim'/'Não'/variações de capitalização e pontuação para 'sim'/'nao'."""
    if value is None:
        return None
    norm = unicodedata_nfkd(str(value).strip().lower()).rstrip(".")
    if norm.startswith("sim"):
        return "sim"
    if norm.startswith("nao"):
        return "nao"
    return None


_SIM_NAO_KPIS = frozenset({"first_call_resolution", "solicitacao_resolvida", "empatia", "clareza_informacoes", "odc"})


def _kpi_parser(kpi: str):
    if kpi == "csat":
        return _parse_scored_text
    if kpi in _SIM_NAO_KPIS:
        return _parse_sim_nao
    return lambda v: v  # texto livre: razao_csat, solver_reasoning, razao_ligacao


def select_kpi_columns(df: pd.DataFrame, field_map: dict[str, str] = KPI_FIELD_MAP) -> pd.DataFrame:
    """Deriva as colunas `prod_<kpi>` (baseline de produção) a partir dos campos
    brutos de `metrics_items`, aplicando o parser adequado a cada KPI.

    Mantém as colunas originais (campos brutos, call_id, data_referencia,
    blob_name) e só ADICIONA uma coluna `prod_<kpi>` para cada KPI cujo campo
    bruto esteja presente no DataFrame.
    """
    out = df.copy()
    for kpi, raw_field in field_map.items():
        if raw_field not in out.columns:
            continue
        out[f"prod_{kpi}"] = out[raw_field].map(_kpi_parser(kpi))
    return out


def check_kpi_coverage(df: pd.DataFrame, kpi_names: tuple[str, ...] = ALL_KPI_NAMES) -> dict[str, list[str]]:
    """Confere, para cada KPI de config/kpis.yaml, se a base carregada tem a
    coluna prod_<kpi> correspondente; imprime um resumo em português.

    Retorna {"ausentes": [...], "com_divergencia_de_dominio": [...]}.
    """
    ausentes = [k for k in kpi_names if f"prod_{k}" not in df.columns]
    divergentes = [k for k in kpi_names if k in KPI_DOMAIN_MISMATCH and f"prod_{k}" in df.columns]

    if ausentes:
        print("KPIs sem correspondência na base de produção (nenhuma coluna prod_<kpi> gerada):")
        for k in ausentes:
            print(f"  - {k}")
    else:
        print("Todos os KPIs de config/kpis.yaml têm uma coluna correspondente na base.")

    if divergentes:
        print("\nKPIs com dado de produção disponível, mas com domínio DIFERENTE do esperado:")
        for k in divergentes:
            print(f"  - {k}: {KPI_DOMAIN_MISMATCH[k]}")

    return {"ausentes": ausentes, "com_divergencia_de_dominio": divergentes}


def _url_components(blob_url: str) -> tuple[str, str, str]:
    parsed = urlparse(blob_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"URL do blob inválida: {blob_url}")
    raw_path = parsed.path.lstrip("/")
    parts = raw_path.split("/", 2)
    if len(parts) < 2:
        raise ValueError(
            "A URL precisa apontar para um container e manter o prefixo do diretório, ex.: "
            "https://contoso.blob.core.windows.net/<container>/<prefix>?<sas>"
        )
    container_name = parts[0]
    prefix = parts[1] if len(parts) == 2 else parts[1]
    if len(parts) == 3:
        prefix = "/".join(parts[1:])
    return parsed.scheme + "://" + parsed.netloc, container_name, prefix


def load_interaction_jsons_from_azure(
    blob_url: str,
    *,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    metric_aliases: dict[str, str] | None = None,
    num_calls: int | None = None,
) -> pd.DataFrame:
    """Baixa todos os blobs JSON de um container/prefixo e concatena em um DataFrame.

    Se `start_date`/`end_date` não forem informados (modo de produção), a janela
    é derivada automaticamente a partir do que de fato está registrado na base:
    os `DEFAULT_WINDOW_DAYS` (7) dias mais recentes com blobs presentes, e não
    os últimos N dias corridos a partir de `date.today()` — a ingestão de
    produção pode atrasar, então "hoje" pode ainda não ter blobs. As datas
    usadas (janela calculada + dias efetivamente encontrados) são impressas.

    `num_calls` limita a quantidade de chamadas (1 blob = 1 chamada) usadas na
    análise, pegando as mais recentes dentro da janela. Se informado, é sempre
    elevado para `MIN_CALLS` (10) quando menor — abaixo disso a análise
    estatística por KPI não é confiável. Se omitido, usa todas as chamadas da
    janela.
    """
    if BlobServiceClient is None:
        raise ImportError("Instale azure-storage-blob para usar este carregador.")

    if num_calls is not None and num_calls < MIN_CALLS:
        print(f"[azure_blob_loader] num_calls={num_calls} abaixo do mínimo; usando {MIN_CALLS}.")
        num_calls = MIN_CALLS

    service = BlobServiceClient(account_url=ACCOUNT_URL, credential=CREDENTIAL)
    container = service.get_container_client(CONTAINER_NAME)

    # list_blobs só traz metadados (nome, tamanho etc.), sem baixar conteúdo —
    # materializar uma vez é barato e serve tanto para descobrir a última data
    # disponível na base quanto para filtrar o período, sem listar duas vezes.
    all_blobs = [b for b in container.list_blobs(name_starts_with=PATH_PREFIX) if b.name.lower().endswith(".json")]
    blob_dates_by_name = {b.name: _extract_blob_date(b.name) for b in all_blobs}
    available_dates = sorted({d for d in blob_dates_by_name.values() if d is not None})

    auto_window = start_date is None and end_date is None
    if auto_window:
        if available_dates:
            latest = available_dates[-1]
            end_date = latest
            start_date = latest - timedelta(days=DEFAULT_WINDOW_DAYS - 1)
        else:
            # container vazio ou nenhum blob com data reconhecível no nome:
            # cai de volta para a janela corrida a partir de hoje.
            today = date.today()
            end_date = today
            start_date = today - timedelta(days=DEFAULT_WINDOW_DAYS - 1)

    start_d = date.fromisoformat(str(start_date)) if isinstance(start_date, str) else start_date
    end_d = date.fromisoformat(str(end_date)) if isinstance(end_date, str) else end_date

    if auto_window:
        used_dates = [d for d in available_dates if start_d <= d <= end_d]
        print(
            f"[azure_blob_loader] Janela automática (última semana registrada na base): "
            f"{start_d.isoformat()} a {end_d.isoformat()}"
        )
        if used_dates:
            print(f"[azure_blob_loader] Dias com dados usados: {', '.join(d.isoformat() for d in used_dates)}")
        else:
            print("[azure_blob_loader] Nenhum dia com dados encontrado nessa janela.")

    rows: list[dict[str, Any]] = []
    calls_processed = 0
    for blob in all_blobs:
        blob_date = blob_dates_by_name[blob.name]
        if start_d is not None and blob_date is not None and blob_date < start_d:
            continue
        if end_d is not None and blob_date is not None and blob_date > end_d:
            continue
        if num_calls is not None and calls_processed >= num_calls:
            break

        # container.list_blobs() devolve BlobProperties (só metadados, sem
        # download_blob) — é preciso pegar o client do blob específico para
        # baixar o conteúdo.
        content = container.get_blob_client(blob.name).download_blob().readall()
        payload = json.loads(content.decode("utf-8"))
        call_id = _extract_call_id(blob.name)
        for record in _iter_json_records(payload):
            flat = _flatten_record(record)
            normalized: dict[str, Any] = {}
            for raw_key, value in flat.items():
                clean_key = normalize_metric_name(raw_key.split(".")[-1], metric_aliases)
                normalized[clean_key] = value
            # chave da chamada + data de partição: sem isso não dá para agrupar por dia
            # nem cruzar com outras bases (o payload em si não traz call_id nem a data).
            normalized["call_id"] = call_id
            normalized["data_referencia"] = blob_date.isoformat() if blob_date else None
            normalized["blob_name"] = blob.name
            rows.append(normalized)
        calls_processed += 1

    if num_calls is not None:
        print(f"[azure_blob_loader] Chamadas usadas na análise: {calls_processed}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["data_referencia"] = pd.to_datetime(df["data_referencia"], errors="coerce")
    df = select_kpi_columns(df)
    return df


def summarize_by_day(df: pd.DataFrame, date_column: str = "data_referencia") -> pd.DataFrame:
    """Agrega o DataFrame carregado por dia: total de chamadas + média das colunas numéricas.

    Útil para validar visualmente se o agrupamento por dia está correto antes
    de alimentar a análise/report da aplicação.
    """
    if date_column not in df.columns:
        raise ValueError(
            f"coluna '{date_column}' ausente — carregue os dados com load_interaction_jsons_from_azure"
        )
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    grouped = df.groupby(df[date_column].dt.date)
    daily = grouped.size().rename("total_chamadas").to_frame()
    if numeric_cols:
        daily = daily.join(grouped[numeric_cols].mean())
    return daily.reset_index().rename(columns={date_column: "dia"}).sort_values("dia")


def load_last_month_interactions(blob_url: str, *, metric_aliases: dict[str, str] | None = None, num_calls: int | None = None) -> pd.DataFrame:
    today = date.today()
    start = today - timedelta(days=30)
    return load_interaction_jsons_from_azure(blob_url, start_date=start, end_date=today, metric_aliases=metric_aliases, num_calls=num_calls)


def load_last_week_interactions(blob_url: str | None = None, *, metric_aliases: dict[str, str] | None = None, num_calls: int | None = None) -> pd.DataFrame:
    """Modo de produção: carrega apenas a última semana REGISTRADA NA BASE.

    Equivalente a chamar `load_interaction_jsons_from_azure` sem `start_date`/
    `end_date` — a janela (e os dias efetivamente usados) são impressos.
    `num_calls` limita (com mínimo de `MIN_CALLS`) a quantidade de chamadas
    usadas na análise.
    """
    return load_interaction_jsons_from_azure(blob_url, metric_aliases=metric_aliases, num_calls=num_calls)


def main() -> int:
    parser = argparse.ArgumentParser(description="Concatena JSONs de interações vindos do Azure Blob Storage.")
    parser.add_argument(
        "--url",
        required=False,
        default=None,
        help=(
            "URL opcional do container/prefixo (SAS), ex.: https://contoso.blob.core.windows.net/container/prefix?sig=... "
            "Se omitida, usa ACCOUNT_URL/CONTAINER_NAME/PATH_PREFIX fixos deste módulo."
        ),
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help=(
            "Data inicial no formato YYYY-MM-DD. Se omitida junto com --end-date, "
            "usa automaticamente a última semana REGISTRADA NA BASE (produção)."
        ),
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Data final no formato YYYY-MM-DD (ver --start-date para o padrão automático).",
    )
    parser.add_argument(
        "--num-calls", type=int, default=None, dest="num_calls",
        help=f"Quantidade de chamadas (blobs) para a análise; mínimo {MIN_CALLS} (valores menores são elevados). Se omitido, usa todas as chamadas da janela.",
    )
    parser.add_argument("--out", type=str, default=None, help="Caminho opcional para salvar o CSV consolidado")
    parser.add_argument("--daily-out", type=str, default=None, help="Caminho opcional para salvar a agregação diária em CSV")
    args = parser.parse_args()

    df = load_interaction_jsons_from_azure(args.url, start_date=args.start_date, end_date=args.end_date, num_calls=args.num_calls)
    if df.empty:
        print("Nenhum registro encontrado no período informado.")
        return 0

    print(df.head().to_string(index=False))
    print(f"\nTotal de registros: {len(df)}")

    print("\nCobertura dos KPIs do benchmark (config/kpis.yaml):")
    check_kpi_coverage(df)

    daily = summarize_by_day(df)
    print("\nRegistros por dia:")
    print(daily.to_string(index=False))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nArquivo salvo em: {out}")
    if args.daily_out:
        daily_out = Path(args.daily_out)
        daily_out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(daily_out, index=False)
        print(f"Agregação diária salva em: {daily_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
