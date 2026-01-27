"""Adaptive learning insights: update student profile and recommend next cases.

This prototype writes simple gap entries to the student profile JSON and returns a suggested next case.
Optional LLM integration for intelligent recommendations.
"""
from typing import Dict, Optional, Any
from .analytics import save_profile_update

try:
    from .llm import LLMClient
    _LLM_AVAILABLE = True
except (ImportError, Exception):
    _LLM_AVAILABLE = False
    LLMClient = Any  # type: ignore


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


def generate_adaptive_insights_with_llm(
    student_id: str,
    diagnostic_feedback: Dict,
    communication_feedback: Dict,
    client: Optional[LLMClient] = None,
    use_llm_fallback: bool = False
) -> Dict:
    """
    Generate adaptive insights using LLM for intelligent recommendations.
    
    Args:
        student_id: Student identifier
        diagnostic_feedback: Diagnostic feedback dict
        communication_feedback: Communication feedback dict
        client: LLMClient instance. If None, creates one.
        use_llm_fallback: If True, falls back to rule-based insights if LLM fails.
    
    Returns:
        Dict with insights, recommended_next_case, and llm_recommendation
    """
    if not _LLM_AVAILABLE:
        if use_llm_fallback:
            return generate_adaptive_insights(student_id, diagnostic_feedback, communication_feedback)
        raise ImportError("LLM integration not available. Install python-dotenv and requests packages or set use_llm_fallback=True.")
    
    if client is None:
        client = LLMClient()
    
    try:
        # Get base insights
        base_insights = generate_adaptive_insights(student_id, diagnostic_feedback, communication_feedback)
        
        # Get LLM recommendation
        gaps = diagnostic_feedback.get("missing", [])
        performance = {
            "open_question_ratio": communication_feedback.get("open_ratio", 0.0),
            "empathy_count": communication_feedback.get("empathy_count", 0),
            "diagnostic_coverage": diagnostic_feedback.get("coverage", 0.0),
        }
        
        llm_recommendation = client.generate_case_recommendation(
            student_gaps=gaps,
            student_id=student_id,
            recent_performance=performance
        )
        
        return {
            "insights": base_insights["insights"],
            "recommended_next_case": base_insights["recommended_next_case"],
            "llm_recommendation": llm_recommendation
        }
    except Exception as e:
        if use_llm_fallback:
            return generate_adaptive_insights(student_id, diagnostic_feedback, communication_feedback)
        raise

