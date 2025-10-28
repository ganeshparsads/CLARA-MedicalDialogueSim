"""Simple performance analytics and peer-comparison simulation."""
from typing import Dict
import json
from pathlib import Path

_PROFILE_PATH = Path(__file__).resolve().parents[1] / "data" / "student_profiles.json"


def compute_score(diagnostic_feedback: Dict, communication_feedback: Dict) -> Dict:
    # Simple weighted score: 70% diagnostic coverage, 30% open question ratio
    diag_score = diagnostic_feedback.get("coverage", 0.0)
    comm_score = communication_feedback.get("open_ratio", 0.0)
    total = 0.7 * diag_score + 0.3 * comm_score

    # Simulated peer stats (mock)
    peer_avg = 0.72
    peer_sd = 0.08

    return {"score": round(total, 3), "peer_avg": peer_avg, "peer_sd": peer_sd}


def save_profile_update(student_id: str, updates: Dict) -> None:
    # Append updates to profile store (simple JSON list). Create file if missing.
    try:
        data = {"profiles": {}}
        if _PROFILE_PATH.exists():
            with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        profiles = data.get("profiles", {})
        profile = profiles.get(student_id, {"id": student_id, "history": []})
        profile["history"].append(updates)
        profiles[student_id] = profile
        with open(_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump({"profiles": profiles}, f, indent=2)
    except Exception:
        # Swallow errors for this demo
        pass
