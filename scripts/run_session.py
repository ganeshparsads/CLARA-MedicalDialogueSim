"""Demo runner for Scenario 1 - Initial History Taking and Diagnostic Reasoning

This script simulates a student asking questions (from data/sample_transcript.txt),
processes them through the mock ASR -> NLP -> Semantic -> Feedback pipeline,
and prints real-time feedback & a post-encounter report.
"""
import sys
from pathlib import Path

# ensure src is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from clara.asr import transcribe
from clara.nlp import analyze_utterance
from clara.semantic import align_concepts_to_kb
from clara.feedback import communication_feedback, diagnostic_feedback
from clara.analytics import compute_score
from clara.adaptive import generate_adaptive_insights


def load_transcript(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def run_simulation(student_id: str = "student_001"):
    transcript_path = REPO_ROOT / "data" / "sample_transcript.txt"
    utterances = load_transcript(transcript_path)

    analyses = []
    asked_concepts_sequence = []

    # Real-time rule: if after 4 utterances critical chest pain details are missing, prompt real-time feedback
    for idx, utt in enumerate(utterances, start=1):
        # ASR (mock)
        asr_out = transcribe(utt)
        text = asr_out["text"]

        # Medical NLP
        analysis = analyze_utterance(text)
        analyses.append(analysis)

        # track sequence of concepts as they are asked
        for c in analysis.get("concepts", []):
            if c not in asked_concepts_sequence:
                asked_concepts_sequence.append(c)

        # Real-time check after a few initial questions
        if idx == 4:
            semantic_partial = align_concepts_to_kb(asked_concepts_sequence)
            if semantic_partial.get("critical_missing"):
                print("[REAL-TIME FEEDBACK] Critical red-flag items not yet asked:", ", ".join(semantic_partial.get("critical_missing")))
                print("Consider asking about onset, quality, radiation, and shortness of breath.")

    # Post-encounter processing
    semantic_result = align_concepts_to_kb(asked_concepts_sequence)
    comm_fb = communication_feedback(analyses)
    diag_fb = diagnostic_feedback(semantic_result)
    score = compute_score(diag_fb, comm_fb)
    adaptive = generate_adaptive_insights(student_id, diag_fb, comm_fb)

    # Print report
    print('\n=== Post-Encounter Report ===')
    print('Asked concepts sequence:', asked_concepts_sequence)
    print('\n-- Communication Feedback --')
    print(comm_fb.get('notes'))
    print('\n-- Diagnostic Feedback --')
    print('Coverage: {:.1%}'.format(diag_fb.get('coverage', 0)))
    for s in diag_fb.get('suggestions', []):
        print('-', s)
    print('\n-- Score --')
    print('Score:', score.get('score'), '(peer avg: {})'.format(score.get('peer_avg')))
    print('\n-- Adaptive Insights --')
    print('Recommended next case:', adaptive.get('recommended_next_case'))


if __name__ == '__main__':
    run_simulation()
