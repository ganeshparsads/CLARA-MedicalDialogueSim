"""Interactive session CLI for CLARA prototype.

Supports two modes:
- text: type utterances (blank line to finish)
- audio: provide a WAV file path to transcribe with local VOSK model (optional)

The script processes utterances through the pipeline and prints real-time and post-encounter feedback.
"""
import sys
from pathlib import Path

# ensure src on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from clara.asr import transcribe, transcribe_audio
from clara.nlp import analyze_utterance
from clara.semantic import align_concepts_to_kb
from clara.feedback import communication_feedback, diagnostic_feedback
from clara.analytics import compute_score
from clara.adaptive import generate_adaptive_insights


def interactive_text_mode(student_id: str = "student_001"):
    print("Interactive text mode. Type each student utterance and press Enter. Blank line to finish.")
    analyses = []
    asked_concepts_sequence = []
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        asr_out = transcribe(line)
        analysis = analyze_utterance(asr_out["text"])
        analyses.append(analysis)
        for c in analysis.get("concepts", []):
            if c not in asked_concepts_sequence:
                asked_concepts_sequence.append(c)

    # post processing
    semantic_result = align_concepts_to_kb(asked_concepts_sequence)
    comm_fb = communication_feedback(analyses)
    diag_fb = diagnostic_feedback(semantic_result)
    score = compute_score(diag_fb, comm_fb)
    adaptive = generate_adaptive_insights(student_id, diag_fb, comm_fb)

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


def audio_file_mode(student_id: str = "student_001"):
    wav = input("Enter path to WAV file (PCM 16k mono): ").strip()
    if not wav:
        print("No file provided. Exiting.")
        return
    try:
        asr_out = transcribe_audio(wav)
    except Exception as e:
        print("ASR error:", e)
        return

    print("Transcribed text:", asr_out.get("text"))
    analysis = analyze_utterance(asr_out.get("text", ""))
    semantic_result = align_concepts_to_kb(analysis.get("concepts", []))
    comm_fb = communication_feedback([analysis])
    diag_fb = diagnostic_feedback(semantic_result)
    score = compute_score(diag_fb, comm_fb)
    adaptive = generate_adaptive_insights(student_id, diag_fb, comm_fb)

    print('\n=== Post-Encounter Report ===')
    print('Asked concepts sequence:', analysis.get('concepts', []))
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


def main():
    print("CLARA interactive session")
    mode = input("Choose mode ('text' or 'audio'): ").strip().lower()
    if mode == 'audio':
        audio_file_mode()
    else:
        interactive_text_mode()


if __name__ == '__main__':
    main()
