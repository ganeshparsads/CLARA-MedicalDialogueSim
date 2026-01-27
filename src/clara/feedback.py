"""Feedback generator: communication and diagnostic feedback based on NLP + semantic alignment outputs.

Optional LLM integration is available for enhanced feedback generation.
"""
from typing import Dict, List, Optional, Any

try:
    from .llm import LLMClient
    _LLM_AVAILABLE = True
except (ImportError, Exception):
    _LLM_AVAILABLE = False
    LLMClient = Any  # type: ignore


def communication_feedback(utterance_analyses: List[Dict]) -> Dict:
    total = len(utterance_analyses)
    if total == 0:
        return {"notes": "No interaction recorded.", "open_ratio": 0.0, "empathy_count": 0}

    open_count = sum(1 for u in utterance_analyses if u.get("type") == "open")
    empathy_count = sum(1 for u in utterance_analyses if u.get("empathy"))

    open_ratio = open_count / total
    notes = []
    if open_ratio >= 0.5:
        notes.append("Good use of open questions overall.")
    else:
        notes.append("Consider using more open-ended questions to elicit narrative responses.")

    if empathy_count > 0:
        notes.append(f"Empathy was expressed {empathy_count} time(s). Good practice.")
    else:
        notes.append("Try to include empathetic prompts to build rapport.")

    return {"notes": " ".join(notes), "open_ratio": open_ratio, "empathy_count": empathy_count}


def diagnostic_feedback(semantic_result: Dict) -> Dict:
    required = semantic_result.get("required_count", 0)
    matched = len(semantic_result.get("matched", []))
    missing = semantic_result.get("missing", [])
    critical = semantic_result.get("critical_missing", [])

    coverage = matched / required if required else 0.0

    suggestions = []
    if critical:
        suggestions.append("Critical red-flag items missing: " + ", ".join(critical))
    if missing:
        suggestions.append("Additional items to ask: " + ", ".join(missing))
    if not missing:
        suggestions.append("History appears comprehensive for the scenario.")

    return {"coverage": coverage, "matched": semantic_result.get("matched", []), "missing": missing, "suggestions": suggestions}


def enhanced_feedback_with_llm(
    utterance_analyses: List[Dict],
    semantic_result: Dict,
    client: Optional[LLMClient] = None,
    use_llm_fallback: bool = False
) -> Dict:
    """
    Generate enhanced feedback using LLM.
    
    Combines communication and diagnostic feedback with LLM-generated insights.
    
    Args:
        utterance_analyses: List of analyzed utterances
        semantic_result: Semantic alignment results
        client: LLMClient instance. If None, creates one using OPENROUTER_API_KEY env var.
        use_llm_fallback: If True, falls back to rule-based feedback if LLM fails.
    
    Returns:
        Dict with communication_feedback, diagnostic_feedback, and llm_feedback sections
    """
    if not _LLM_AVAILABLE:
        if use_llm_fallback:
            return {
                "communication_feedback": communication_feedback(utterance_analyses),
                "diagnostic_feedback": diagnostic_feedback(semantic_result),
                "llm_feedback": None
            }
        raise ImportError("LLM integration not available. Install python-dotenv and requests packages or set use_llm_fallback=True.")
    
    if client is None:
        client = LLMClient()
    
    try:
        # Get base feedback
        comm_fb = communication_feedback(utterance_analyses)
        diag_fb = diagnostic_feedback(semantic_result)
        
        # Extract student utterances for LLM
        student_concepts = semantic_result.get("matched", [])
        missing_concepts = semantic_result.get("missing", [])
        
        # Generate LLM feedback
        llm_fb = client.generate_feedback(
            student_concepts=student_concepts,
            missing_concepts=missing_concepts,
            communication_notes=comm_fb.get("notes", ""),
            scenario="chest_pain_diagnosis"
        )
        
        return {
            "communication_feedback": comm_fb,
            "diagnostic_feedback": diag_fb,
            "llm_feedback": llm_fb
        }
    except Exception as e:
        if use_llm_fallback:
            return {
                "communication_feedback": communication_feedback(utterance_analyses),
                "diagnostic_feedback": diagnostic_feedback(semantic_result),
                "llm_feedback": None,
                "error": str(e)
            }
        raise
