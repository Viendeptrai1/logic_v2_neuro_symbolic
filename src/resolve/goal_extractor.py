"""
goal_extractor.py — Extracts FOL goals using Dynamic Lexicon.

Replaces the regex-only approach with Lexicon Alignment from alignment.py.
"""

from __future__ import annotations
import re
from typing import Any

from src.fol.fol_parser import parse, ParseError, collect_constants
from src.resolve.alignment import build_dynamic_lexicon, _extract_ngrams

# Same constants used to ignore question words
_QUESTION_WORDS = frozenset({
    'does', 'do', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
    'can', 'could', 'will', 'would', 'should', 'shall', 'may', 'might',
    'if', 'then', 'else', 'that', 'this', 'which', 'what', 'when', 'where',
    'who', 'whose', 'how', 'why', 'there', 'here', 'based', 'according',
    'following', 'given', 'above', 'premise', 'premises', 'statement',
    'logical', 'logic', 'conclusion', 'true', 'false', 'correct', 'valid',
    'not', 'no', 'and', 'or', 'but', 'so', 'for', 'the', 'a', 'an',
    'all', 'every', 'some', 'any', 'each', 'both', 'either', 'neither',
})

def _get_constants(premises_fol: list[str]) -> set[str]:
    all_consts = set()
    for fol in premises_fol:
        try:
            collect_constants(parse(fol), all_consts)
        except ParseError:
            pass
    return all_consts

def detect_entity_in_question(question_text: str, constants: set[str]) -> str | None:
    for const_name in sorted(constants, key=len, reverse=True):
        if len(const_name) < 2: continue
        if const_name.lower() in _QUESTION_WORDS: continue
        if re.search(r'\b' + re.escape(const_name) + r'\b', question_text, re.IGNORECASE):
            return const_name
    return None

def match_predicate_lexicon(phrase: str, lexicon: dict) -> tuple[str | None, bool]:
    """Returns the best predicate name and a boolean is_low_confidence."""
    # Normalize phrase
    phrase_ngrams = _extract_ngrams(phrase, max_n=3)
    
    best_pred = None
    best_score = -1
    best_is_low = False
    
    # Try exact match against phrases first
    phrase_clean = re.sub(r'[^a-z0-9\s]', ' ', phrase.lower())
    for pred, data in lexicon.items():
        for lex_phrase in data["phrases"]:
            if lex_phrase in phrase_clean:
                score = len(lex_phrase.split())
                if score > best_score:
                    best_score = score
                    best_pred = pred
                    best_is_low = (data["confidence"] == "low")
                    
    if best_pred:
        return best_pred, best_is_low
        
    return None, False

def extract_goal_with_lexicon(question_text: str, premises_nl: list[str], premises_fol: list[str]) -> tuple[str | None, bool]:
    """
    Returns (fol_str, has_low_confidence_pred).
    """
    lexicon = build_dynamic_lexicon(premises_nl, premises_fol)
    constants = _get_constants(premises_fol)
    
    stmt = _extract_embedded_statement(question_text) or question_text.strip()
    
    has_low = False
    
    # Check for direct entity
    entity = detect_entity_in_question(stmt, constants)
    if entity:
        negated = bool(re.search(r'\b(?:not|cannot|does not|is not|are not|no)\b', stmt, re.IGNORECASE))
        # Remove entity and question words
        phrase = re.sub(re.escape(entity), '', stmt, flags=re.IGNORECASE)
        phrase = re.sub(r'^(?:Does|Do|Can|Is|Are|Has|Have|Will)\s+', '', phrase, flags=re.IGNORECASE)
        pred, is_low = match_predicate_lexicon(phrase, lexicon)
        if pred:
            goal = f"¬{pred}({entity})" if negated else f"{pred}({entity})"
            return goal, is_low
            
    # Check for If...Then
    m = re.match(r'^[Ii]f (.+?),?\s+then (.+)$', stmt)
    if m:
        ant_nl = m.group(1).strip()
        con_nl = m.group(2).strip()
        
        ant_neg = ' not ' in ant_nl.lower() or ant_nl.lower().startswith('not ')
        con_neg = ' not ' in con_nl.lower() or con_nl.lower().startswith('not ') or ' does not ' in con_nl.lower()
        
        ant_pred, low1 = match_predicate_lexicon(ant_nl, lexicon)
        con_pred, low2 = match_predicate_lexicon(con_nl, lexicon)
        
        if ant_pred and con_pred:
            ant_fol = f"¬{ant_pred}(x)" if ant_neg else f"{ant_pred}(x)"
            con_fol = f"¬{con_pred}(x)" if con_neg else f"{con_pred}(x)"
            return f"∀x ({ant_fol} → {con_fol})", (low1 or low2)

    # Check for All...
    m = re.match(r'^[Aa]ll (.+?) (?:are|is|have|has) (.+)$', stmt)
    if m:
        con_pred, low = match_predicate_lexicon(m.group(2), lexicon)
        if con_pred:
            return f"∀x ({con_pred}(x))", low
            
    # Check for Exists...
    m = re.match(r'^[Tt]here (?:exists?|is) (?:at least one )?(.+)', stmt)
    if m:
        pred, low = match_predicate_lexicon(m.group(1), lexicon)
        if pred:
            return f"∃x ({pred}(x))", low
            
    return None, False

def _extract_embedded_statement(question_text: str) -> str | None:
    q = question_text.strip()
    PATTERNS = [
        r'[Dd]oes it follow that (.+?)(?:,?\s+according\s+to\s+the\s+premises\??)?$',
        r'[Ii]s it true that (.+?)\??$',
        r'[Ss]tatement:\s*(.+?)$',
        r'[Dd]oes the logical (?:chain|progression) (?:demonstrate|show|prove) that (.+?)\??$',
        r'[Ii]s it correct that (.+?)\??$',
        r'([Ii]f .+?) follow (?:from|according to) the premises\??$',
        r'[Dd]oes it hold that (.+?)\??$',
        r'[Ss]tatement:\s*\n?(.+?)$',
        r'[Ww]hether (.+?)\??$',
    ]
    for pattern in PATTERNS:
        m = re.search(pattern, q, re.MULTILINE | re.DOTALL)
        if m:
            stmt = m.group(1).strip().rstrip('.,;?')
            if len(stmt) > 5:
                return stmt
    return None
