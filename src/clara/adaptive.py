"""Adaptive learning insights: update student profile and recommend next cases.

This prototype writes simple gap entries to the student profile JSON and returns a suggested next case.
"""
from typing import Dict
from .analytics import save_profile_update


def generate_adaptive_insights(student_id: str, diagnostic_feedback: Dict, communication_feedback: Dict) -> Dict:
    gaps = diagnostic_feedback.get("missing", [])
    insights = {"gaps": gaps, "comm_open_ratio": communication_feedback.get("open_ratio", 0.0)}

    # Save to profile
    save_profile_update(student_id, insights)

    # Recommend next case: if cardiac gaps, recommend a cardiac-focused case
    next_case = "general_clinical_case"
    cardiac_markers = set(["chest_pain", "radiation", "shortness_of_breath", "diaphoresis"])
    if cardiac_markers.intersection(set(gaps)):
        next_case = "cardiac_chest_pain_case"

    return {"insights": insights, "recommended_next_case": next_case}
