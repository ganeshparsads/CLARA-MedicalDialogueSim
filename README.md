# CLARA Medical Dialogue Simulation (Minimal Prototype)

This repository contains a minimal, self-contained prototype that implements Scenario 1: Initial History Taking and Diagnostic Reasoning. The pipeline can run in two modes:

- **Rule-based mode**: Fast, deterministic analysis using pattern matching
- **LLM-enhanced mode**: Uses OpenRouter models (Deepseek R1 + Qwen3) for intelligent analysis and feedback

## Pipeline Components

- ASR (mock transcription)
- Medical NLP + concept extraction (rule-based or LLM-enhanced)
- Semantic alignment engine (compare asked concepts against clinical knowledge base)
- Feedback generator (communication + diagnostic feedback)
- Real-time feedback simulation
- Performance analytics + adaptive learning

## Quick Start

### Run Sample Transcript
```bash
python scripts/run_session.py
```
You'll be prompted to choose:
- **Analysis mode**: Rule-based (default) or LLM-enhanced
- Then processes sample_transcript.txt through the pipeline

### Interactive Session
```bash
python scripts/interactive_session.py
```
You'll be prompted to choose:
- **Analysis mode**: Rule-based or LLM-enhanced
- **Input mode**: Text (type utterances) or Audio (WAV file)
- Then interactively analyze medical questions

## LLM Integration (Optional)

To enable LLM-powered analysis using Deepseek R1 and Qwen3:

### 1. Setup API Key
Create a `.env` file in the repository root:
```
OPENROUTER_API_KEY=your_api_key_here
```

### 2. Get OpenRouter API Key
Visit [openrouter.ai](https://openrouter.ai) and create a free account.
- Supports free models including **Deepseek R1** (reasoning) and **Qwen3** (structured outputs)
- No credit card required for free models

### 3. Run Pipeline in LLM Mode
When you run either script, select option **2** for LLM-enhanced analysis:
- **Deepseek R1**: Used for open-ended reasoning and detailed explanations
- **Qwen3**: Used for structured JSON outputs (medical analysis, feedback, case recommendations)

## Optional Features

### Local ASR with VOSK
To enable audio transcription without external APIs:

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Download a VOSK model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and extract to `data/vosk-model/`

3. Use interactive script and select audio mode

### spaCy for Enhanced NLP
For improved medical concept extraction:
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

The pipeline will automatically use spaCy if available.

## Architecture

```
User Input
    ↓
ASR (mock or VOSK)
    ↓
Analysis (Rule-based or LLM)
    ├─ Rule mode: Pattern matching
    └─ LLM mode: Qwen3 for structured analysis
    ↓
Semantic Alignment
    ↓
Feedback Generation
    ├─ Rule mode: Rule-based feedback
    └─ LLM mode: Deepseek R1 + Qwen3 for rich feedback
    ↓
Post-Encounter Report
    └─ Analytics, Score, Adaptive Insights
```

## Files

- `src/clara/` - Pipeline modules
  - `llm.py` - OpenRouter API client
  - `nlp.py` - NLP analysis (rule-based & LLM)
  - `semantic.py` - Knowledge base alignment
  - `feedback.py` - Feedback generation
  - `adaptive.py` - Case recommendations
  - `analytics.py` - Scoring & analytics
  - `asr.py` - ASR integration
- `scripts/run_session.py` - Process sample transcript
- `scripts/interactive_session.py` - Interactive mode
- `data/clinical_knowledge.json` - Knowledge base
- `data/sample_transcript.txt` - Example utterances
- `.env` - Configuration (API keys)

## Notes

- Rule-based mode requires no external dependencies (fast startup)
- LLM mode is optional; gracefully falls back to rule-based if API unavailable
- All LLM calls include automatic rate-limit retry handling
- Reasoning models (like Deepseek R1) take time but provide excellent analysis quality
