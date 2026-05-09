"""Module 4: concise MRI report-generation workflow.

This restored version keeps the project explanation clear without carrying the
large Colab/MedGemma/LangGraph implementation. It shows the same conceptual
pipeline used in the graduation project:

1. Generate a draft report from structured clinical findings.
2. Critique the draft for safety and evidence grounding.
3. Refine the draft into a final radiology-style report.

The final website currently uses the deterministic report path in
``TriNeuro_XAI_Web/api.py`` for speed. This module is kept for explaining and
testing the report-generation idea independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypedDict


class ReportState(TypedDict, total=False):
    clinical_dict: Dict[str, Any]
    prompts: Dict[str, str]
    draft_report: str
    critique: str
    final_report: str


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _fmt(value: Any, decimals: int = 3) -> str:
    value = _plain(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.{decimals}f}"
        return text.rstrip("0").rstrip(".")
    return str(value)


def _bbox_text(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, (list, tuple)):
        return " x ".join(_fmt(v, 1) for v in value) + " mm"
    return "N/A"


def _subregion_text(subregions: Dict[str, Any]) -> str:
    if not subregions:
        return "No subregion volume breakdown available."

    parts = []
    for name, values in subregions.items():
        if not isinstance(values, dict):
            continue
        volume = _fmt(values.get("volume_cm3", 0.0))
        percentage = _fmt(values.get("percentage", 0.0), 1)
        parts.append(f"{name}: {volume} cm3 ({percentage}%)")
    return "; ".join(parts) if parts else "No subregion volume breakdown available."


def _report_label(clinical_dict: Dict[str, Any]) -> str:
    return str(clinical_dict.get("classification_label") or "brain tumor")


def _facts_block(clinical_dict: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"patient_id: {clinical_dict.get('patient_id', 'UNKNOWN')}",
            f"classification_label: {_report_label(clinical_dict)}",
            f"tumor_present: {clinical_dict.get('tumor_present', False)}",
            f"segmented_lesion_volume_cm3: {_fmt(clinical_dict.get('segmented_lesion_volume_cm3', 0.0))}",
            f"tumor_core_volume_cm3: {_fmt(clinical_dict.get('tumor_volume_cm3', 0.0))}",
            f"max_diameter_mm: {_fmt(clinical_dict.get('max_diameter_mm', 0.0), 1)}",
            f"axial_span_mm: {_fmt(clinical_dict.get('axial_span_mm', 0.0), 1)}",
            f"axial_span_slices: {_fmt(clinical_dict.get('axial_span_slices', 0), 0)}",
            f"hemisphere: {clinical_dict.get('hemisphere', 'N/A')}",
            f"bounding_box_size_mm: {_bbox_text(clinical_dict.get('bounding_box_size_mm'))}",
            f"sphericity: {_fmt(clinical_dict.get('sphericity', 0.0), 3)}",
            f"subregions: {_subregion_text(clinical_dict.get('subregion_volumes', {}) or {})}",
        ]
    )


def build_seed_report(clinical_dict: Dict[str, Any]) -> str:
    patient_id = clinical_dict.get("patient_id", "UNKNOWN")
    label = _report_label(clinical_dict)
    tumor_present = bool(clinical_dict.get("tumor_present", False))
    lesion_volume = _fmt(clinical_dict.get("segmented_lesion_volume_cm3", 0.0))

    if not tumor_present:
        return f"""Patient: {patient_id}
Volume: 0 cm3

TECHNIQUE:
Automated brain MRI review was performed using the uploaded modality set.

COMPARISON:
No prior comparison study was provided.

FINDINGS:
- No measurable intracranial tumor burden is reported by the automated workflow.

IMPRESSION:
- No tumor detected by the automated TriNeuro AI workflow.

RECOMMENDATIONS:
Radiologist review remains required before clinical use."""

    return f"""Patient: {patient_id}
Volume: {lesion_volume} cm3

TECHNIQUE:
Automated brain MRI review was performed using the uploaded modality set. The workflow included classification, segmentation-derived measurements, Grad-CAM++ visualization, and report generation.

COMPARISON:
No prior comparison study was provided.

FINDINGS:
- Automated classification is most consistent with {label}.
- Segmented lesion volume is {lesion_volume} cm3, including estimated tumor core volume of {_fmt(clinical_dict.get('tumor_volume_cm3', 0.0))} cm3.
- Maximum lesion diameter is {_fmt(clinical_dict.get('max_diameter_mm', 0.0), 1)} mm, with axial span of {_fmt(clinical_dict.get('axial_span_mm', 0.0), 1)} mm across {_fmt(clinical_dict.get('axial_span_slices', 0), 0)} slices.
- Laterality estimate is {clinical_dict.get('hemisphere', 'N/A')}. Bounding box dimensions are {_bbox_text(clinical_dict.get('bounding_box_size_mm'))}.
- Subregion measurements: {_subregion_text(clinical_dict.get('subregion_volumes', {}) or {})}

IMPRESSION:
- Automated MRI pipeline detects measurable tumor burden most consistent with {label}.
- Segmentation-derived measurements support a structured estimate of lesion size, extent, and morphology.

RECOMMENDATIONS:
Review the original MRI sequences, segmentation overlay, and Grad-CAM++ map. Correlate with clinical findings and histopathology when available."""


def build_generate_prompt(clinical_dict: Dict[str, Any]) -> str:
    return (
        "You are a neuroradiology report writer. Rewrite the seed report using "
        "only the structured facts. Keep wording cautious and evidence-based.\n\n"
        f"Structured facts:\n{_facts_block(clinical_dict)}\n\n"
        f"Seed report:\n{build_seed_report(clinical_dict)}"
    )


def build_critique_prompt(draft_report: str, clinical_dict: Dict[str, Any]) -> str:
    return (
        "Review this MRI report for evidence grounding, safety, completeness, "
        "and professional radiology wording. Return concise JSON.\n\n"
        f"Structured facts:\n{_facts_block(clinical_dict)}\n\n"
        f"Draft report:\n{draft_report}"
    )


def build_refine_prompt(draft_report: str, critique: str, clinical_dict: Dict[str, Any]) -> str:
    return (
        "Revise the draft report using the critique. Do not invent unsupported "
        "findings. Return only the final report.\n\n"
        f"Structured facts:\n{_facts_block(clinical_dict)}\n\n"
        f"Critique:\n{critique}\n\n"
        f"Draft report:\n{draft_report}"
    )


def build_default_prompts(clinical_dict: Dict[str, Any]) -> Dict[str, str]:
    return {
        "generate": build_generate_prompt(clinical_dict),
        "critique": build_critique_prompt("<<DRAFT_REPORT>>", clinical_dict),
        "refine": build_refine_prompt("<<DRAFT_REPORT>>", "<<CRITIQUE>>", clinical_dict),
    }


def deterministic_critique(draft_report: str, clinical_dict: Dict[str, Any]) -> str:
    issues = []
    if "radiologist review" not in draft_report.lower() and "decision-support" not in draft_report.lower():
        issues.append("Report should clearly state that the output is decision support and requires radiologist review.")
    if clinical_dict.get("tumor_present") and "Volume:" not in draft_report:
        issues.append("Report should include the segmented lesion volume.")
    if not issues:
        issues.append("Draft is broadly consistent with the structured facts.")

    return json.dumps(
        {
            "final_verdict": "PASS" if len(issues) == 1 and issues[0].startswith("Draft") else "REVISE",
            "critical_issues": issues,
            "improvement_suggestions": [
                "Keep all claims grounded in classification and segmentation measurements.",
                "Use cautious wording and avoid definitive diagnosis beyond the model label.",
            ],
        },
        indent=2,
    )


def generate_report_node(state: ReportState, llm: Optional[Any] = None) -> ReportState:
    clinical = state["clinical_dict"]
    prompt = state.get("prompts", {}).get("generate") or build_generate_prompt(clinical)
    if llm is not None:
        draft = llm.invoke(prompt, system_text="Write a cautious neuroradiology report.")
    else:
        draft = build_seed_report(clinical)
    return {"draft_report": draft.strip()}


def critique_report_node(state: ReportState, llm: Optional[Any] = None) -> ReportState:
    clinical = state["clinical_dict"]
    draft = state["draft_report"]
    prompt = build_critique_prompt(draft, clinical)
    if llm is not None:
        critique = llm.invoke(prompt, system_text="Return concise JSON critique only.")
    else:
        critique = deterministic_critique(draft, clinical)
    return {"critique": critique.strip()}


def refine_report_node(state: ReportState, llm: Optional[Any] = None) -> ReportState:
    clinical = state["clinical_dict"]
    draft = state["draft_report"]
    critique = state["critique"]
    prompt = build_refine_prompt(draft, critique, clinical)
    if llm is not None:
        final_report = llm.invoke(prompt, system_text="Return only the final report.")
    else:
        final_report = draft
    return {"final_report": final_report.strip()}


@dataclass
class SimpleReportGraph:
    report_llm: Optional[Any] = None
    critique_llm: Optional[Any] = None
    refine_llm: Optional[Any] = None

    def invoke(self, state: ReportState) -> ReportState:
        current: ReportState = dict(state)
        current.update(generate_report_node(current, self.report_llm))
        current.update(critique_report_node(current, self.critique_llm))
        current.update(refine_report_node(current, self.refine_llm))
        return current


def build_report_graph(report_llm: Optional[Any] = None, critique_llm: Optional[Any] = None, refine_llm: Optional[Any] = None) -> SimpleReportGraph:
    return SimpleReportGraph(report_llm=report_llm, critique_llm=critique_llm, refine_llm=refine_llm)


GPT_41_MODEL = "gpt-4.1"


class OpenAILLM:
    """Tiny OpenAI wrapper fixed to gpt-4.1 for this project."""

    def __init__(self, api_key: Optional[str] = None, max_output_tokens: int = 900) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = GPT_41_MODEL
        self.max_output_tokens = max_output_tokens

    def invoke(self, prompt: str, system_text: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=system_text,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
        )
        return (getattr(response, "output_text", "") or "").strip()


def generate_mri_report_with_graph(
    clinical_dict: Dict[str, Any],
    prompts: Optional[Dict[str, str]] = None,
    report_llm: Optional[Any] = None,
    critique_llm: Optional[Any] = None,
    refine_llm: Optional[Any] = None,
    **_unused: Any,
) -> Dict[str, Any]:
    graph = build_report_graph(report_llm, critique_llm, refine_llm)
    result = graph.invoke(
        {
            "clinical_dict": clinical_dict,
            "prompts": prompts or build_default_prompts(clinical_dict),
        }
    )
    return dict(result)


def generate_medgemma_report(clinical_dict: Dict[str, Any], *_args: Any, **_kwargs: Any) -> str:
    """Compatibility helper for older notebook cells."""
    return generate_mri_report_with_graph(clinical_dict)["final_report"]
