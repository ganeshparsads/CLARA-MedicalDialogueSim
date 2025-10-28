"""Feedback generator: communication and diagnostic feedback based on NLP + semantic alignment outputs."""
from typing import Dict, List


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
