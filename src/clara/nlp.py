"""Medical NLP and concept extraction (rule-based, minimal)

This module exposes functions to:
- classify question type (open vs closed)
- extract clinical concepts from a sentence using keyword lists
- detect empathy / communication features

Optional LLM integration is available via analyze_utterance_with_llm().
"""
from typing import Dict, List, Tuple, Optional, Any
import re
from pathlib import Path

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

try:
    from .llm import LLMClient
    _LLM_AVAILABLE = True
except (ImportError, Exception):
    _LLM_AVAILABLE = False
    LLMClient = Any  # type: ignore

# Small keyword-to-concept mapping for chest pain scenario
_CONCEPT_KEYWORDS = {
    "chest_pain": ["chest pain", "pain in the chest", "central chest pain"],
    "radiation": ["left arm", "jaw", "back", "radiat"],
    "shortness_of_breath": ["shortness of breath", "short of breath", "sob", "breathless", "dyspn"],
    "diaphoresis": ["sweat", "diaphoresis", "clammy"],
    "nausea_vomiting": ["nausea", "vomit"],
    "onset_timing": ["when did", "onset", "how long"],
    "quality": ["sharp", "dull", "pressure", "tightness", "crushing"],
    "severity": ["severe", "mild", "scale of 1 to 10", "how bad"],
    "past_medical_history": ["history of", "past medical", "pmh", "hypertension", "diabetes", "high blood"],
    "family_history": ["family history", "father had", "mother had", "siblings"],
    "smoking": ["smoke", "smoking", "tobacco"],
    "medications": ["medication", "taking", "aspirin", "beta blocker"],
}

_OPEN_Q_PATTERNS = re.compile(r"^(what|how|tell|describe|could you tell|please describe|can you tell)", re.I)
_CLOSED_Q_PATTERNS = re.compile(r"^(do|did|have|has|is|are|was|were|can|could|would|should)\\b", re.I)


def _build_spacy_matcher(nlp):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns_by_concept = {}
    for concept, kws in _CONCEPT_KEYWORDS.items():
        patterns = [nlp.make_doc(kw) for kw in kws]
        matcher.add(concept, patterns)
        patterns_by_concept[concept] = patterns
    return matcher


def classify_question(text: str) -> str:
    t = text.strip()
    if not t:
        return "unknown"
    if _OPEN_Q_PATTERNS.search(t):
        return "open"
    if _CLOSED_Q_PATTERNS.search(t):
        return "closed"
    # fallback: treat WH- words as open
    if any(t.lower().startswith(w) for w in ["who", "what", "when", "where", "why", "how"]):
        return "open"
    return "closed"


def extract_concepts(text: str) -> List[str]:
    text_low = text.lower()
    found = set()

    # Try spaCy PhraseMatcher if available
    if _SPACY_AVAILABLE:
        try:
            # Load small English model if not already loaded; this is lazy to avoid import overhead
            nlp = getattr(extract_concepts, "_nlp", None)
            matcher = getattr(extract_concepts, "_matcher", None)
            if nlp is None:
                nlp = spacy.load("en_core_web_sm")
                matcher = _build_spacy_matcher(nlp)
                extract_concepts._nlp = nlp
                extract_concepts._matcher = matcher

            doc = nlp(text)
            matches = matcher(doc)
            for match_id, start, end in matches:
                concept = nlp.vocab.strings[match_id]
                found.add(concept)
            if found:
                return sorted(found)
        except Exception:
            # Fall back to rule-based extraction if anything goes wrong with spaCy
            pass

    # Rule-based fallback
    for concept, keywords in _CONCEPT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_low:
                found.add(concept)
                break
    return sorted(found)


def detect_empathy(text: str) -> bool:
    # Very simple heuristic: presence of phrases expressing concern
    empathy_phrases = ["how are you feeling", "i'm sorry", "that must be", "that sounds", "i understand"]
    t = text.lower()
    return any(p in t for p in empathy_phrases)


def analyze_utterance(text: str) -> Dict:
    qtype = classify_question(text)
    concepts = extract_concepts(text)
    empathy = detect_empathy(text)
    return {"text": text, "type": qtype, "concepts": concepts, "empathy": empathy}


def analyze_utterance_with_llm(
    text: str,
    client: Optional[LLMClient] = None,
    use_llm_fallback: bool = False
) -> Dict:
    """
    Analyze a medical utterance using LLM (with rule-based fallback).
    
    Args:
        text: The utterance to analyze
        client: LLMClient instance. If None, creates one using OPENROUTER_API_KEY env var.
        use_llm_fallback: If True, falls back to rule-based analysis if LLM fails.
                         If False, raises exception on LLM failure.
    
    Returns:
        Dict with analysis including text, type, concepts, empathy, and llm_analysis
    """
    if not _LLM_AVAILABLE:
        if use_llm_fallback:
            return analyze_utterance(text)
        raise ImportError("LLM integration not available. Install python-dotenv and requests packages or set use_llm_fallback=True.")
    
    if client is None:
        client = LLMClient()
    
    try:
        # Get LLM analysis
        llm_result = client.analyze_medical_utterance(text)
        
        # Get rule-based analysis for comparison
        rule_analysis = analyze_utterance(text)
        
        # Merge results: use LLM concepts if available, fall back to rule-based
        concepts = llm_result.get("concepts", [])
        if isinstance(concepts, str):
            concepts = [c.strip() for c in concepts.split(",")]
        if not concepts:
            concepts = rule_analysis.get("concepts", [])
        
        qtype = llm_result.get("question_type", rule_analysis.get("type", "unknown"))
        if qtype not in ["open", "closed", "statement", "unclear"]:
            qtype = rule_analysis.get("type", qtype)
        
        return {
            "text": text,
            "type": qtype,
            "concepts": concepts,
            "empathy": rule_analysis.get("empathy", False),
            "llm_analysis": llm_result,
            "clinical_relevance": llm_result.get("clinical_relevance", "medium"),
        }
    except Exception as e:
        if use_llm_fallback:
            # Fall back to rule-based analysis
            return analyze_utterance(text)
        raise
