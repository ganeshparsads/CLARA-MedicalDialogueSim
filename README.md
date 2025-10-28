# CLARA Medical Dialogue Simulation (Minimal Prototype)

This repository contains a minimal, self-contained prototype that implements Scenario 1: Initial History Taking and Diagnostic Reasoning. It uses simple, rule-based components to simulate the pipeline:

- ASR (mock)
- Medical NLP + concept extraction (rule-based)
- Semantic alignment engine (compare asked concepts against a small clinical knowledge base)
- Feedback generator (communication + diagnostic)
- Real-time feedback simulation
- Performance analytics + adaptive learning update (simple JSON profile)

How to run (quick)

Text-only demo (no external models required):

```bash
python3 scripts/run_session.py
```

Interactive text session (type utterances, blank line to finish):

```bash
python3 scripts/interactive_session.py
```

Optional: enable local ASR with VOSK

1. Install Python dependencies (recommended in a venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download a VOSK model and place it under `data/vosk-model` or set the environment variable `VOSK_MODEL_PATH` to the model directory. Example models: https://alphacephei.com/vosk/models

3. Use the interactive script and choose "audio" mode, then provide a WAV file path (PCM 16k mono) when prompted.

Notes
- The prototype uses rule-based NLP by default but will use spaCy if installed to improve parsing.
- The ASR module falls back to a mock/text mode when VOSK is not available. Real-time microphone capture is not included in this minimal prototype to keep dependencies light.

spaCy (optional enhancement)

To enable improved concept extraction via spaCy's PhraseMatcher:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

After installing, the prototype will automatically use spaCy for concept extraction.


Files created
- `src/clara/` - package with pipeline modules
- `scripts/run_session.py` - demo runner
- `data/clinical_knowledge.json` - small knowledge base
- `data/sample_transcript.txt` - example student utterances

Note: This is a lightweight prototype with no external dependencies.
