"""
alignment.py — Dynamic Lexicon Alignment for NL-FOL translation.

This module statistically aligns predicates in FOL premises with n-grams in NL premises
using co-occurrence (Jaccard similarity). For singleton predicates (appearing in only 1 premise),
it falls back to structural token matching and flags them as low-confidence.
"""

import re
from collections import defaultdict
from typing import Any

from src.fol.fol_parser import parse, collect_predicates, ParseError

STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'does', 'do', 'did',
    'have', 'has', 'had', 'be', 'been', 'can', 'could', 'will', 'would',
    'should', 'shall', 'may', 'might', 'must', 'to', 'of', 'in', 'for',
    'with', 'on', 'at', 'by', 'from', 'about', 'and', 'or', 'but', 'not',
    'no', 'all', 'any', 'some', 'every', 'each', 'that', 'this', 'those',
    'these', 'if', 'then', 'only', 'when', 'where', 'who', 'which',
    'there', 'exists', 'at', 'least', 'one', 'person', 'student', 'project',
    'code', 'it', 'they', 'he', 'she', 'we', 'you', 'I'
}

def _extract_ngrams(text: str, max_n: int = 3) -> set[str]:
    """Extract 1, 2, 3-grams from text, excluding purely stop-word n-grams."""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    words = text.split()
    
    ngrams = set()
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            gram_words = words[i:i+n]
            # Ignore n-grams that are entirely stop words
            if all(w in STOP_WORDS for w in gram_words):
                continue
            ngrams.add(' '.join(gram_words))
    return ngrams

def _tokenize_predicate(pred_name: str) -> set[str]:
    """Split CamelCase or snake_case predicate name into lowercase words."""
    # Split by underscore
    name = pred_name.replace('_', ' ')
    # Split CamelCase
    name = re.sub(r'([A-Z])', r' \1', name)
    words = set(name.lower().split())
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}

def build_dynamic_lexicon(premises_nl: list[str], premises_fol: list[str]) -> dict[str, dict[str, Any]]:
    """
    Build a dynamic mapping from FOL predicates to NL phrases.
    Returns:
      {
         "PredicateName": {
             "phrase": "nl phrase",
             "confidence": "high" | "low",
             "jaccard": 1.0
         }
      }
    """
    # 1. Map predicates to their premise indices
    pred_to_indices = defaultdict(set)
    for i, fol_str in enumerate(premises_fol):
        try:
            ast = parse(fol_str)
            preds = {}
            collect_predicates(ast, preds)
            for p in preds:
                pred_to_indices[p].add(i)
        except ParseError:
            continue

    # 2. Extract n-grams from NL premises and map to indices
    nl_ngrams_per_premise = []
    ngram_to_indices = defaultdict(set)
    for i, nl_str in enumerate(premises_nl):
        ngrams = _extract_ngrams(nl_str, max_n=3)
        nl_ngrams_per_premise.append(ngrams)
        for g in ngrams:
            ngram_to_indices[g].add(i)

    lexicon = {}

    # 3. Align each predicate
    for pred, p_indices in pred_to_indices.items():
        if len(p_indices) > 1:
            # High confidence: use Jaccard co-occurrence
            best_phrase = pred.lower()
            best_jaccard = -1.0
            best_phrases = []
            
            for ngram, n_indices in ngram_to_indices.items():
                intersection = len(p_indices & n_indices)
                union = len(p_indices | n_indices)
                jaccard = intersection / union if union > 0 else 0
                
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_phrases = [ngram]
                elif jaccard == best_jaccard and jaccard > 0:
                    best_phrases.append(ngram)
            
            lexicon[pred] = {
                "phrases": best_phrases,
                "confidence": "high",
                "jaccard": best_jaccard
            }
        else:
            # Singleton predicate (low confidence)
            # Use structural token matching on the specific premise it appears in
            idx = list(p_indices)[0]
            if idx < len(nl_ngrams_per_premise):
                premise_ngrams = nl_ngrams_per_premise[idx]
            else:
                # If idx is out of bounds, search across all n-grams
                premise_ngrams = set().union(*nl_ngrams_per_premise) if nl_ngrams_per_premise else set()
                
            pred_tokens = _tokenize_predicate(pred)
            
            best_score = 0
            best_phrases = []
            
            if pred_tokens:
                for ngram in premise_ngrams:
                    ngram_tokens = set(ngram.split())
                    overlap = len(pred_tokens & ngram_tokens)
                    
                    if overlap > best_score:
                        best_score = overlap
                        best_phrases = [ngram]
                    elif overlap == best_score and overlap > 0:
                        best_phrases.append(ngram)
            
            # If no token overlap at all
            if best_score <= 0 and len(pred) <= 4 and pred.isupper():
                # check acronym match
                for ngram in premise_ngrams:
                    words = [w for w in ngram.split() if w not in STOP_WORDS]
                    if len(words) == len(pred):
                        acronym = ''.join(w[0].upper() for w in words)
                        if acronym == pred:
                            best_phrases.append(ngram)
                            best_score = 1.0
            
            if not best_phrases:
                best_phrases = [pred.lower()]

            lexicon[pred] = {
                "phrases": best_phrases,
                "confidence": "low",
                "jaccard": 1.0  # By definition since it's a singleton
            }

    return lexicon

if __name__ == '__main__':
    # Quick Test
    nl = [
        "If a Python code is well-tested, then the project is optimized.",
        "If a Python code does not follow PEP 8 standards, then it is not well-tested.",
        "All Python projects are easy to maintain.",
        "All Python code is well-tested."
    ]
    fol = [
        "∀x (WT(x) → O(x))",
        "∀x (¬PEP8(x) → ¬WT(x))",
        "∀x EM(x)",
        "∀x WT(x)"
    ]
    
    lex = build_dynamic_lexicon(nl, fol)
    import json
    print(json.dumps(lex, indent=2))
