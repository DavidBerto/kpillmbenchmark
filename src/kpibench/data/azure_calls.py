"""Ligações reais para o teste de modelos: KPIs de produção + texto do resumo/transcrição.

Reaproveita `azure_blob_loader` (mesma conta, container, prefixo e a conexão padrão
`DefaultAzureCredential`) para trazer os KPIs de produção (`prod_<kpi>`) e
completa cada ligação com o TEXTO do resumo e, quando existir, da transcrição — que
é o que os prompts de KPI recebem.

Onde o texto é procurado, nesta ordem:
  1. campos da própria linha de métricas que já contenham texto;
  2. campos com URL de blob (ex.: `summary`, que aponta para `<uuid>.summary.json`),
     baixados com a mesma credencial;
  3. blocos `summary_items` / `transcription_items` do próprio blob de report.

A estrutura exata dos JSONs de resumo/transcrição não está documentada no
repositório. As heurísticas abaixo cobrem os formatos comuns; se o texto vier
vazio ou errado, use `inspect_blob_structure()` em um blob real e ajuste
`SUMMARY_KEYS` / `TRANSCRIPT_KEYS` (ou passe `summary_keys=` / `transcript_keys=`).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import pandas as pd

# chaves candidatas, em ordem de prioridade (comparação sem maiúsculas)
SUMMARY_KEYS: tuple[str, ...] = ("summary_items", "summary", "resumo", "summary_text", "text", "content", "value")
TRANSCRIPT_KEYS: tuple[str, ...] = ("transcription_items", "transcription", "transcript", "transcricao",
                                    "segments", "utterances", "phrases", "turns", "items")
SPEAKER_KEYS: tuple[str, ...] = ("speaker", "role", "channel", "participant", "speaker_label", "locutor")
TEXT_KEYS: tuple[str, ...] = ("text", "content", "transcript", "display", "utterance", "sentence", "value", "summary")

MIN_SUMMARY_CHARS = 20


# =============================================================================== extração de texto
def _is_url(v: Any) -> bool:
    return isinstance(v, str) and v.strip().lower().startswith(("http://", "https://"))


def _ci_get(d: dict, keys: Iterable[str]):
    lower = {str(k).lower(): k for k in d}
    for k in keys:
        if k.lower() in lower:
            return d[lower[k.lower()]]
    return None


def _longest_string(obj: Any, depth: int = 0) -> str:
    """Maior string (que não seja URL) encontrada numa busca em profundidade."""
    if depth > 8:
        return ""
    if isinstance(obj, str):
        return "" if _is_url(obj) else obj
    best = ""
    items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, list) else []
    for v in items:
        s = _longest_string(v, depth + 1)
        if len(s) > len(best):
            best = s
    return best


def extract_summary_text(payload: Any, keys: Iterable[str] = SUMMARY_KEYS) -> str:
    """Texto do resumo a partir de um JSON de formato não fixo."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        s = payload.strip()
        if s.startswith(("{", "[")):
            try:
                return extract_summary_text(json.loads(s), keys)
            except json.JSONDecodeError:
                pass
        return "" if _is_url(s) else s
    if isinstance(payload, list):
        parts = [extract_summary_text(x, keys) for x in payload]
        parts = [p for p in parts if p]
        return "\n".join(parts)
    if isinstance(payload, dict):
        v = _ci_get(payload, keys)
        if v is not None and not _is_url(v):
            txt = extract_summary_text(v, keys)
            if len(txt) >= MIN_SUMMARY_CHARS:
                return txt
        return _longest_string(payload).strip()
    return str(payload)


def extract_transcript_text(payload: Any, keys: Iterable[str] = TRANSCRIPT_KEYS) -> str:
    """Transcrição como 'LOCUTOR: fala' por linha, a partir de um JSON de formato não fixo."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        s = payload.strip()
        if s.startswith(("{", "[")):
            try:
                return extract_transcript_text(json.loads(s), keys)
            except json.JSONDecodeError:
                pass
        return "" if _is_url(s) else s
    if isinstance(payload, dict):
        v = _ci_get(payload, keys)
        if v is not None and not _is_url(v):
            return extract_transcript_text(v, keys)
        txt = _ci_get(payload, TEXT_KEYS)
        if isinstance(txt, str):
            spk = _ci_get(payload, SPEAKER_KEYS)
            return f"{str(spk).upper()}: {txt.strip()}" if spk not in (None, "") else txt.strip()
        return _longest_string(payload).strip()
    if isinstance(payload, list):
        lines = []
        for turn in payload:
            if isinstance(turn, dict):
                txt = _ci_get(turn, TEXT_KEYS)
                if isinstance(txt, str) and txt.strip():
                    spk = _ci_get(turn, SPEAKER_KEYS)
                    lines.append(f"{str(spk).upper()}: {txt.strip()}" if spk not in (None, "") else txt.strip())
                else:
                    sub = extract_transcript_text(turn, keys)
                    if sub:
                        lines.append(sub)
            elif isinstance(turn, str) and turn.strip():
                lines.append(turn.strip())
        return "\n".join(lines)
    return str(payload)


# =============================================================================== Azure
def _blob_service():
    """Cliente do container com a conexão padrão do projeto (DefaultAzureCredential)."""
    from kpibench.data import azure_blob_loader as abl

    from azure.storage.blob import BlobServiceClient

    svc = BlobServiceClient(account_url=abl.ACCOUNT_URL, credential=abl.CREDENTIAL)
    return svc.get_container_client(abl.CONTAINER_NAME), abl


def _download_json(container, abl, name_or_url: str) -> Any:
    """Baixa um blob por nome (no container padrão) ou por URL completa (mesma credencial)."""
    if _is_url(name_or_url):
        from azure.storage.blob import BlobClient

        client = BlobClient.from_blob_url(name_or_url.split("?")[0], credential=abl.CREDENTIAL)
    else:
        client = container.get_blob_client(name_or_url)
    raw = client.download_blob().readall()
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def inspect_blob_structure(name_or_url: str, max_depth: int = 4, max_items: int = 3) -> str:
    """Esqueleto do JSON de um blob (chaves, tipos e amostras curtas) para ajustar as chaves."""
    container, abl = _blob_service()
    payload = _download_json(container, abl, name_or_url)
    lines: list[str] = []

    def walk(obj, indent=0, key="raiz"):
        pad = "  " * indent
        if indent > max_depth:
            lines.append(f"{pad}{key}: …")
            return
        if isinstance(obj, dict):
            lines.append(f"{pad}{key}: dict({len(obj)})")
            for k, v in list(obj.items())[:40]:
                walk(v, indent + 1, k)
        elif isinstance(obj, list):
            lines.append(f"{pad}{key}: list({len(obj)})")
            for i, v in enumerate(obj[:max_items]):
                walk(v, indent + 1, f"[{i}]")
        else:
            sample = repr(obj)
            lines.append(f"{pad}{key}: {type(obj).__name__} = {sample[:90]}{'…' if len(sample) > 90 else ''}")

    walk(payload)
    out = "\n".join(lines)
    print(out)
    return out


def _resolve_row(row: dict, container, abl, summary_keys, transcript_keys) -> dict:
    resumo, transcricao, origem, erro = "", "", "", None
    try:
        # 1) texto já presente na linha / 2) URL em campo de resumo
        for col in ("summary", "resumo", "summary_url", "summary_text"):
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            if _is_url(v):
                resumo = extract_summary_text(_download_json(container, abl, v), summary_keys)
                origem = f"url:{col}"
            else:
                resumo = extract_summary_text(v, summary_keys)
                origem = f"campo:{col}"
            if len(resumo) >= MIN_SUMMARY_CHARS:
                break
        for col in [c for c in row if "transcri" in str(c).lower()]:
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            transcricao = extract_transcript_text(_download_json(container, abl, v) if _is_url(v) else v,
                                                  transcript_keys)
            if transcricao:
                break
        # 3) blocos do próprio blob de report
        if (len(resumo) < MIN_SUMMARY_CHARS or not transcricao) and row.get("blob_name"):
            rep = _download_json(container, abl, row["blob_name"])
            if isinstance(rep, dict):
                if len(resumo) < MIN_SUMMARY_CHARS and _ci_get(rep, ("summary_items",)) is not None:
                    resumo = extract_summary_text(_ci_get(rep, ("summary_items",)), summary_keys)
                    origem = "report:summary_items"
                if not transcricao and _ci_get(rep, ("transcription_items",)) is not None:
                    transcricao = extract_transcript_text(_ci_get(rep, ("transcription_items",)), transcript_keys)
    except Exception as e:  # noqa: BLE001 - ligação sem texto é reportada, não derruba a carga
        erro = f"{type(e).__name__}: {e}"[:200]
    return {"call_id": row["call_id"], "resumo": resumo.strip(), "transcricao": transcricao.strip(),
            "origem_resumo": origem, "erro_texto": erro}


def load_calls(start_date: str | None, end_date: str | None, num_calls: int | None, *,
               call_ids: list[str] | None = None, max_workers: int = 8,
               summary_keys: Iterable[str] = SUMMARY_KEYS,
               transcript_keys: Iterable[str] = TRANSCRIPT_KEYS) -> pd.DataFrame:
    """Ligações do período com KPIs de produção + resumo/transcrição em texto.

    `num_calls` segue a regra do loader (mínimo de 10). `call_ids`, se informado,
    filtra as ligações carregadas na janela (útil para reproduzir casos específicos).
    """
    from kpibench.data.azure_blob_loader import load_interaction_jsons_from_azure

    df = load_interaction_jsons_from_azure(None, start_date=start_date, end_date=end_date, num_calls=num_calls)
    if df.empty:
        return df
    df = df.drop_duplicates("call_id").reset_index(drop=True)
    if call_ids:
        df = df[df["call_id"].astype(str).isin([str(c) for c in call_ids])].reset_index(drop=True)
        if df.empty:
            print("[azure_calls] nenhum dos call_ids informados está na janela carregada")
            return df

    container, abl = _blob_service()
    rows = df.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        texts = list(pool.map(lambda r: _resolve_row(r, container, abl, tuple(summary_keys),
                                                     tuple(transcript_keys)), rows))
    txt = pd.DataFrame(texts)
    out = df.drop(columns=[c for c in ("resumo", "transcricao") if c in df.columns]).merge(txt, on="call_id")
    sem_resumo = int((out["resumo"].str.len() < MIN_SUMMARY_CHARS).sum())
    com_transc = int((out["transcricao"].str.len() > 0).sum())
    print(f"[azure_calls] {len(out)} ligações · com resumo: {len(out) - sem_resumo} · "
          f"com transcrição: {com_transc}")
    if sem_resumo:
        print(f"[azure_calls] {sem_resumo} ligações sem resumo legível — rode inspect_blob_structure() "
              f"num blob de resumo e ajuste SUMMARY_KEYS")
    return out
