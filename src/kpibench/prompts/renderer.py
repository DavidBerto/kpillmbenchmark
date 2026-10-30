"""Renderização dos prompts.

Um arquivo por KPI (prompts/kpis/<kpi>.md), com seções '### SYSTEM' e '### USER'.
O MESMO template é usado para todos os modelos: o hash do arquivo (incluindo os
includes) é registrado em cada predição para auditoria e reprodutibilidade.

A parte estática (instruções, rubrica, formato) fica no SYSTEM e a parte variável
(resumo/transcrição) no USER, o que maximiza o reaproveitamento de prompt caching.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from kpibench.llm.tokens import estimate_tokens


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    template: str
    template_hash: str

    @property
    def static_prefix_tokens(self) -> int:
        return estimate_tokens(self.system)


class PromptRenderer:
    def __init__(self, prompts_dir: Path):
        self.dir = Path(prompts_dir)
        self.env = Environment(loader=FileSystemLoader(str(self.dir)), undefined=StrictUndefined,
                               keep_trailing_newline=False, trim_blocks=False)

    @lru_cache(maxsize=None)
    def template_hash(self, rel_path: str) -> str:
        """sha256 do template + includes (12 primeiros hex)."""
        h = hashlib.sha256()
        h.update((self.dir / rel_path).read_bytes())
        h.update((self.dir / "_base_system.md").read_bytes())
        return h.hexdigest()[:12]

    def render(self, rel_path: str, **variables) -> RenderedPrompt:
        text = self.env.get_template(rel_path).render(**variables)
        if "### USER" not in text or "### SYSTEM" not in text:
            raise ValueError(f"{rel_path}: o template precisa das seções '### SYSTEM' e '### USER'")
        system, user = text.split("### USER", 1)
        system = system.replace("### SYSTEM", "", 1).strip()
        return RenderedPrompt(system=system, user=user.strip(), template=rel_path,
                              template_hash=self.template_hash(rel_path))
