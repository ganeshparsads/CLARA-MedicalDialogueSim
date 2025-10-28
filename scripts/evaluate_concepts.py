"""Evaluate current concept extraction on sample_transcript.txt

This script computes precision, recall, and F1 for the aggregated set of
concepts found in the sample transcript compared to a small ground-truth set
inferred from the transcript.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from clara.nlp import extract_concepts


def load_transcript(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def compute_metrics(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main():
    transcript_path = REPO_ROOT / "data" / "sample_transcript.txt"
    utterances = load_transcript(transcript_path)

    # Ground-truth concepts inferred from the sample transcript (manually curated)
    gold_concepts = set([
        "chest_pain",
        "onset_timing",
        "quality",
        "radiation",
        "shortness_of_breath",
        "past_medical_history",
        "smoking",
        "medications",
    ])

    predicted = set()
    per_utt = []
    for u in utterances:
        cs = set(extract_concepts(u))
        per_utt.append((u, cs))
        predicted |= cs

    metrics = compute_metrics(predicted, gold_concepts)

    print("Gold concepts:", sorted(gold_concepts))
    print("Predicted concepts:", sorted(predicted))
    print()
    print(f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")

    print('\nPer-utterance extraction:')
    for u, cs in per_utt:
        print('-', u)
        print('  ->', cs)


if __name__ == '__main__':
    main()
