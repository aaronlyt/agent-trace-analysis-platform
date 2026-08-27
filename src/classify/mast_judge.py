"""MAST judge labeling —— LLM-as-a-judge classification by 14 failure modes (arXiv:2503.13657).

Mechanism (paper §3.3 judge pipeline): prompt = all MAST definitions +
few-shot example + trajectory (folded view) → the judge outputs the failure
mode codes it hits + reasons + evidence steps. Validation: codes must exist
in the MAST vocabulary; unknown codes are dropped and recorded (never
silently trusted).

Differences from the paper (engineering adaptations):
* the few-shot is one self-constructed example —— the paper uses the
  human-annotated data examples from Appendix N; few-shot is precisely the
  decisive factor behind the κ 0.58→0.77 jump in the paper's Table 2, so
  matching that number requires real examples;
* ``max_labels`` (default 3) truncates the label count —— codes are fully
  validated first, then truncated; overflowing valid labels are recorded in
  ``truncated_codes`` (never silently dropped); the paper puts no cap on
  multi-labels;
* by default only failed trajectories are labeled (MAST annotates failure
  modes; ``include_success=True`` overrides). Note that MAST Appendix J.1
  does not give the judge the success/failure outcome (verbatim: "we do not
  provide the success or failure result to the LLM Annotator"), whereas
  this rendered view includes an outcome line —— in the include_success
  scenario this deviates from that setup; the default path is unaffected.

Phase-four extension (AgentDebugX 2607.18754 §3.4 residual vocabulary
entry):
* ``allow_novel=True`` —— the prompt gains "if the symptom belongs to no
  allowed code, output code=\"novel\" + a symptom phrase in the symptom
  field"; a novel label must carry a symptom (missing symptom is treated as
  invalid). When the judge meets a **recurring** failure outside the seeds
  it records a novel-mode candidate, for clustering nomination by
  classify/inducer [adaptation: the paper's judge vocabulary is closed; the
  novel channel is the inducer's residual entry];
* ``extra_modes_file`` —— loads the human-accepted extended modes (the JSON
  produced after an inducer proposal passes ``atap taxonomy accept``) and
  merges them into the allowed codes and the definitions block ——
  "proposals never take effect automatically"; acceptance must go through
  human adjudication and land in a file. Modes whose definition is empty or
  whitespace-only are skipped (a bare code+name line in the judge prompt
  would invite free-form guessing) and recorded in the artifact's
  ``skipped_extra_modes`` —— never silently dropped.

Artifact: ``{"labels": [...], "fusion": [...], "invalid_codes": [...]}``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from atap.classify.base import Classifier
from atap.classify.taxonomy import MAST_MODES, FusionLabel, mast_definitions_block
from atap.core.registry import register
from atap.core.render import judge_view


class MastLabel(BaseModel):
    code: str = Field(description="MAST failure mode code, e.g. FM-1.3")
    reason: str
    step: int | None = Field(default=None, description="Evidence step (R0 index)")
    symptom: str | None = Field(
        default=None,
        description="Symptom phrase when code=novel (summarizing the most typical observable symptom)",
    )


class MastLabels(BaseModel):
    labels: list[MastLabel] = Field(default_factory=list)


_SYSTEM = (
    "You are a MAST failure-mode annotation judge for multi-agent systems. "
    "Below are the definitions of MAST's 3 categories and 14 failure modes, "
    "plus one execution trajectory. Select the failure modes that actually "
    "occurred in the trajectory (multiple allowed); for each, give the code, "
    "the reason, and the evidence step. Do not select when uncertain. Choose "
    "only from the given codes.\n\n"
    "MAST definitions:\n{definitions}"
)
# Anti-leak constraint: the example must not be an answer key for any sandbox
# fault — no (GT agent, GT code) pair, no GT onset step, no GT step-run like
# "3/5/7". Agents/steps here are fictional and outside the sandbox roster.
_FEW_SHOT = (
    "Example: the editor receives an ambiguous change request at step 6, "
    "never asks for clarification, and at step 10 rewrites the wrong section "
    "—— the decisive error is acting on ambiguous input without "
    "clarification —— output "
    "{\"labels\": [{\"code\": \"FM-2.2\", \"reason\": \"proceeded on ambiguous input without asking for clarification\", \"step\": 10}]}."
)
_NOVEL_INSTRUCTION = (
    "\n\nIf the failure symptom in the trajectory does not belong to any of "
    "the failure modes above, output code=\"novel\" and summarize the most "
    "typical observable symptom as a short phrase in the symptom field "
    "(quote keywords from the trajectory itself; do not invent symptoms that "
    "do not exist in the trajectory)."
)
_EXTRA_TAG = " [extended] "


def _load_extra_modes(path: str) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Load the human-accepted extended modes file (product of atap taxonomy accept).

    Returns ``(modes, skipped)``: a mode whose definition is empty or
    whitespace-only is excluded from ``modes`` (an empty definition would
    enter the judge prompt as a bare code+name line) and reported in
    ``skipped`` as ``{"code", "reason"}`` —— visible in the artifact, never
    silently dropped.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    modes = data.get("modes") if isinstance(data, dict) else data
    if not isinstance(modes, list):
        raise ValueError(f"Malformed extended modes file (expected {{modes: [...]}} or a list): {path}")
    out: dict[str, dict[str, str]] = {}
    skipped: list[dict[str, str]] = []
    for m in modes:
        code, name = str(m.get("code", "")), str(m.get("name") or m.get("label", ""))
        if not code or not name:
            raise ValueError(f"Extended mode missing code/name fields: {m}")
        definition = str(m.get("definition", "")).strip()
        if not definition:
            skipped.append({"code": code, "reason": "empty definition"})
            continue
        out[code] = {
            "category": "EXT",
            "name": name,
            "definition": definition,
        }
    return out, skipped


@register
class MastJudgeClassifier(Classifier):
    stage = "classify"
    name = "mast_judge"

    def __init__(self, **params) -> None:
        super().__init__(**params)
        path = self.param("extra_modes_file")
        self.extra_modes, self.skipped_extra_modes = (
            _load_extra_modes(str(path)) if path else ({}, [])
        )
        # conflict check covers skipped codes too: a file entry shadowing a
        # MAST code is wrong regardless of whether its definition was usable
        overlap = (
            (set(self.extra_modes) | {s["code"] for s in self.skipped_extra_modes})
            & set(MAST_MODES)
        )
        if overlap:
            raise ValueError(f"Extended mode codes conflict with MAST: {sorted(overlap)}")

    def _allowed_modes(self) -> dict[str, dict[str, str]]:
        return {**MAST_MODES, **self.extra_modes}

    def _definitions_block(self) -> str:
        lines = [mast_definitions_block()]
        for code, m in sorted(self.extra_modes.items()):
            lines.append(f"{code}{_EXTRA_TAG}{m['name']} —— {m['definition']}")
        return "\n".join(lines)

    def run_one(self, bundle, ctx) -> None:
        if not bundle.trajectory.events:
            raise ValueError(
                f"{bundle.trace_id} has no R0 event stream: configure canonical_events first"
            )
        if bundle.succeeded and not self.param("include_success", False):
            bundle.put("classify", self.name, {"labels": [], "fusion": [], "invalid_codes": []})
            return
        if ctx.llm is None:
            raise RuntimeError("mast_judge requires an LLM client (RunContext.llm)")

        allow_novel = bool(self.param("allow_novel", False))
        system = _SYSTEM.format(definitions=self._definitions_block())
        if self.param("few_shot", True):
            system += "\n\n" + _FEW_SHOT
        if allow_novel:
            system += _NOVEL_INSTRUCTION
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: Label the failure modes in the following trajectory:\n{judge_view(bundle)}"},
        ]
        result = ctx.llm.complete(messages, schema=MastLabels, tag=self.name)
        parsed = result.parsed
        assert isinstance(parsed, MastLabels)

        allowed = self._allowed_modes()
        max_labels = int(self.param("max_labels", 3))
        valid, invalid = [], []
        for lab in parsed.labels:          # Validate all codes first, then truncate
            if lab.code in allowed:        # (so invalid codes among the first 3
                valid.append(lab)          # cannot crowd out later valid labels)
            elif allow_novel and lab.code == "novel" and (lab.symptom or "").strip():
                valid.append(lab)          # novel labels must carry a symptom phrase
            else:
                invalid.append(lab.code)
        truncated = [lab.code for lab in valid[max_labels:]]
        valid = valid[:max_labels]
        fusion = [
            FusionLabel(mast=lab.code, evidence_step=lab.step, reason=lab.reason)
            for lab in valid
        ]
        bundle.put(
            "classify",
            self.name,
            {
                "labels": [lab.model_dump() for lab in valid],
                "fusion": [f.to_dict() for f in fusion],
                "invalid_codes": invalid,
                "truncated_codes": truncated,  # valid labels beyond max_labels, kept on record rather than silently dropped
                "novel_channel": allow_novel,
                "extra_modes": sorted(self.extra_modes),
                "skipped_extra_modes": self.skipped_extra_modes,  # modes rejected for an empty definition, kept on record
            },
        )
