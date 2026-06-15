"""
oracle_resolver.py — Map reasoner results to answer tokens.

Given gold FOL premises + a parsed question, this module:
  1. Extracts the question type (truth / MCQ)
  2. Extracts the goal formula(s)
  3. Calls the reasoner
  4. Selects the answer token

Two key limitations acknowledged here:
  - MCQ options are NL only → we use heuristic NL→FOL conversion
  - Truth questions: the "statement" is NL → we try to parse it as FOL

These limitations are intentional for the Oracle Harness:
  we measure how much accuracy we get with the FOL premise oracle
  PLUS a best-effort goal extraction.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any

from src.fol.reasoner import check_entailment, check_entailment_z3, ReasonerResult
from src.fol.fol_parser import parse, ParseError
from src.fol.fol_to_z3 import Z3Context, TranslateError
from src.resolve.goal_extractor import extract_goal, build_item_vocab, match_predicate


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ResolverResult:
    answer: str                         # exact answer token
    premise_ids: list[int]              # 0-based (from reasoner)
    status: str                         # 'yes'|'no'|'uncertain'|'error'|'unparseable'
    goal_str: str = ''                  # the extracted goal FOL string (for debug)
    error: str = ''
    all_option_results: dict = None     # for MCQ: {option: ReasonerResult}


# ---------------------------------------------------------------------------
# Question type detection
# ---------------------------------------------------------------------------

_TRUTH_ANSWER_TOKENS = {'yes', 'no', 'uncertain', 'unknown', 'true', 'false'}

def is_truth_question(options: list[str]) -> bool:
    """Return True if this is a Yes/No/Uncertain question."""
    if not options:
        return False
    return all(o.lower() in _TRUTH_ANSWER_TOKENS for o in options)

def is_mcq_question(options: list[str]) -> bool:
    """Return True if options are MCQ letter choices (A/B/C/D)."""
    return all(re.match(r'^[A-D]$', o) for o in options)


# ---------------------------------------------------------------------------
# Goal extractor for truth questions
# ---------------------------------------------------------------------------

# Patterns to extract the logical statement from truth question text
_TRUTH_PATTERNS = [
    # "Does it follow that [STATEMENT]?"
    r'[Dd]oes it follow that (.+?)\??$',
    # "Is it true that [STATEMENT]?"
    r'[Ii]s it true that (.+?)\??$',
    # "According to the premises, is the following statement true?\nStatement: [STMT]"
    r'[Ss]tatement:\s*(.+?)$',
    # "Does the logical chain demonstrate that [STATEMENT]?"
    r'[Dd]oes the logical (?:chain|progression) demonstrate that (.+?)\??$',
    # "Is it correct that [STATEMENT]?"
    r'[Ii]s it correct that (.+?)\??$',
    # "Does [STATEMENT] follow from the premises?"
    r'[Dd]oes (.+?) follow (?:from|according to) the premises\??$',
    # "According to the premises, [STATEMENT]?"
    r'[Aa]ccording to the premises,?\s+(.+?)\??$',
    # Generic: extract after "that " or after "?"
    r'[Ww]hether (.+?)\??$',
]


def extract_truth_goal(question_text: str) -> str | None:
    """
    Try to extract the statement being asked about from a truth question.
    Returns the statement text (NL), or None if extraction fails.
    """
    # Normalise whitespace
    text = ' '.join(question_text.split())
    # Remove trailing punctuation for cleaner matching
    for pattern in _TRUTH_PATTERNS:
        m = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if m:
            stmt = m.group(1).strip().rstrip('.,;')
            if len(stmt) > 5:
                return stmt
    return None


# ---------------------------------------------------------------------------
# NL→FOL heuristic for truth statement goals
# ---------------------------------------------------------------------------

# Maps NL patterns to FOL templates using predicate names inferred from premises
_NL_TO_FOL_TEMPLATES = [
    # "If X, then Y" / "If X then Y"
    (r'^[Ii]f (.+?),?\s+then (.+)$', 'if_then'),
    # "All X are Y" / "All X have Y"
    (r'^[Aa]ll (.+?) (?:are|is|have|has) (.+)$', 'all_are'),
    # "X is Y" (subject predication)
    (r'^(\w+) is (.+)$', 'subject_pred'),
    # "There exists X that Y"
    (r'^[Tt]here exists? (?:at least one )?(.+?) that (.+)$', 'exists_that'),
    # "No X is Y"
    (r'^[Nn]o (.+?) (?:is|are) (.+)$', 'no_is'),
    # "X follows PEP 8 / X is well-tested" etc. (direct fact about named entity)
    (r'^(\w+) (?:follows|is|has|can|meets) (.+)$', 'entity_fact'),
]


def nl_to_fol_heuristic(nl_text: str, predicates: dict[str, int]) -> str | None:
    """
    Very rough NL→FOL conversion using regex templates.
    Returns a FOL string or None.
    
    This is NOT a proper NL→FOL translator — just enough to handle 
    common patterns in the dataset's truth questions.
    
    `predicates` is a dict of {pred_name: arity} from the gold premises.
    """
    text = nl_text.strip().rstrip('.')

    # Pattern: "If ... then ..."
    m = re.match(r'^[Ii]f (.+?),?\s+then (.+)$', text)
    if m:
        ant_nl, con_nl = m.group(1).strip(), m.group(2).strip()
        ant_fol = _map_nl_phrase_to_pred(ant_nl, predicates)
        con_fol = _map_nl_phrase_to_pred(con_nl, predicates)
        if ant_fol and con_fol:
            return f"∀x ({ant_fol}(x) → {con_fol}(x))"

    # Pattern: "All X are Y"
    m = re.match(r'^[Aa]ll (.+?) (?:are|is) (.+)$', text)
    if m:
        con_fol = _map_nl_phrase_to_pred(m.group(2).strip(), predicates)
        if con_fol:
            return f"∀x ({con_fol}(x))"

    # Pattern: "There exists X"
    m = re.match(r'^[Tt]here exists? (?:at least one )?(.+)', text)
    if m:
        pred_fol = _map_nl_phrase_to_pred(m.group(1).strip(), predicates)
        if pred_fol:
            return f"∃x ({pred_fol}(x))"

    return None


def _map_nl_phrase_to_pred(nl: str, predicates: dict[str, int]) -> str | None:
    """
    Try to find the best-matching predicate name for an NL phrase.
    Uses simple substring/word overlap scoring.
    """
    nl_lower = nl.lower()
    nl_words = set(re.findall(r'[a-z]+', nl_lower))
    
    best_pred = None
    best_score = 0
    
    for pred_name in predicates:
        # Convert predicate name to words (CamelCase → words, snake_case → words)
        pred_words = set(re.findall(r'[a-z]+', re.sub(r'([A-Z])', r' \1', pred_name).lower()))
        overlap = len(nl_words & pred_words)
        if overlap > best_score:
            best_score = overlap
            best_pred = pred_name
    
    if best_score >= 1:
        return best_pred
    return None


# ---------------------------------------------------------------------------
# MCQ option extractor
# ---------------------------------------------------------------------------

def extract_mcq_options(question_text: str) -> dict[str, str]:
    """
    Extract MCQ options from question text.
    Returns {letter: option_text} e.g. {'A': 'If X then Y', ...}
    """
    options = {}
    # Pattern: "A. text\nB. text" or "A) text\nB) text"
    for m in re.finditer(r'\n([A-D])[\.)\s]\s*(.+?)(?=\n[A-D][\.)\s]|\Z)', question_text, re.DOTALL):
        letter = m.group(1)
        text   = m.group(2).strip()
        options[letter] = text
    return options


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve_question(
    premises_fol: list[str],
    question_text: str,
    options: list[str],
    timeout_ms: int = 5000,
) -> ResolverResult:
    """
    Resolve a single question using gold FOL premises.
    
    Parameters
    ----------
    premises_fol : gold premises as FOL strings (from dataset)
    question_text : the full question text (NL)
    options : answer tokens e.g. ['Yes', 'No', 'Uncertain'] or ['A', 'B', 'C', 'D']
    timeout_ms : per-reasoner-call timeout in milliseconds
    
    Returns
    -------
    ResolverResult
    """
    if is_truth_question(options):
        return _resolve_truth(premises_fol, question_text, options, timeout_ms)
    elif is_mcq_question(options):
        return _resolve_mcq(premises_fol, question_text, options, timeout_ms)
    else:
        # Unknown option format: try truth first
        return _resolve_truth(premises_fol, question_text, options, timeout_ms)


# ---------------------------------------------------------------------------
# Truth question resolver
# ---------------------------------------------------------------------------

def _resolve_truth(
    premises_fol: list[str],
    question_text: str,
    options: list[str],
    timeout_ms: int,
) -> ResolverResult:
    """Resolve a Yes/No/Uncertain question."""
    
    # Use improved goal extractor
    goal_fol = extract_goal(question_text, premises_fol)
    
    if not goal_fol:
        return ResolverResult(
            answer=_map_uncertain(options),
            premise_ids=[],
            status='unparseable',
            error='Could not extract goal from question text',
        )

    # Verify goal is parseable
    try:
        parse(goal_fol)
    except ParseError:
        # goal_extractor returned NL text, not FOL — try to convert
        from src.fol.fol_parser import collect_predicates
        all_preds: dict[str, int] = {}
        for prem in premises_fol:
            try:
                ast = parse(prem)
                collect_predicates(ast, all_preds)
            except Exception:
                pass
        conv = nl_to_fol_heuristic(goal_fol, all_preds)
        if not conv:
            return ResolverResult(
                answer=_map_uncertain(options),
                premise_ids=[],
                status='unparseable',
                goal_str=goal_fol,
                error=f'Goal not parseable as FOL: {goal_fol[:60]!r}',
            )
        goal_fol = conv

    # Call reasoner
    result = check_entailment(premises_fol, goal_fol, timeout_ms=timeout_ms)

    # Map status → answer token
    answer = _map_truth_status(result.status, options)
    return ResolverResult(
        answer=answer,
        premise_ids=result.premise_ids,
        status=result.status,
        goal_str=goal_fol,
        error=result.error,
    )


def _map_truth_status(status: str, options: list[str]) -> str:
    """Map 'yes'/'no'/'uncertain'/'error' to exact option token."""
    opts_lower = {o.lower(): o for o in options}
    if status == 'yes':
        return opts_lower.get('yes', opts_lower.get('true', options[0]))
    if status == 'no':
        return opts_lower.get('no', opts_lower.get('false', options[1] if len(options) > 1 else options[0]))
    # uncertain / error / unparseable → map to the uncertainty token
    return _map_uncertain(options)


def _map_uncertain(options: list[str]) -> str:
    """Return the uncertainty token from options list."""
    opts_lower = {o.lower(): o for o in options}
    for key in ('uncertain', 'unknown', 'not known', 'maybe'):
        if key in opts_lower:
            return opts_lower[key]
    # No explicit uncertainty token: default to last option
    return options[-1] if options else 'Uncertain'


# ---------------------------------------------------------------------------
# MCQ resolver
# ---------------------------------------------------------------------------

def _resolve_mcq(
    premises_fol: list[str],
    question_text: str,
    options: list[str],
    timeout_ms: int,
) -> ResolverResult:
    """
    Resolve an MCQ question.
    
    Strategy:
    1. Extract option texts from question
    2. For each option, try goal extraction + NL→FOL and run reasoner
    3. Collect entailed options
    4. If exactly 1 entailed → pick it
    5. If multiple entailed → fewest-premises (tie → A < B < C < D)
    6. If none entailed → return 'uncertain' (strict — no guess)
    """
    option_texts = extract_mcq_options(question_text)
    predicates, constants = build_item_vocab(premises_fol)

    # Run reasoner for each option
    all_results: dict[str, ReasonerResult] = {}
    option_fols: dict[str, str] = {}

    for letter in options:  # options is ['A', 'B', 'C', 'D'] or subset
        option_nl = option_texts.get(letter, '')
        if not option_nl:
            continue

        # Try to extract goal from the option text using improved extractor
        goal_fol = extract_goal(option_nl, premises_fol)
        
        # Fallback: try structural NL patterns
        if not goal_fol:
            goal_fol = nl_to_fol_heuristic(option_nl, predicates)
        
        # Fallback: try direct FOL parse
        if not goal_fol:
            try:
                parse(option_nl)
                goal_fol = option_nl
            except ParseError:
                pass

        if goal_fol:
            # Verify parseable
            try:
                parse(goal_fol)
                option_fols[letter] = goal_fol
                res = check_entailment(premises_fol, goal_fol, timeout_ms=timeout_ms)
                all_results[letter] = res
            except ParseError:
                all_results[letter] = ReasonerResult(
                    status='error',
                    premise_ids=[],
                    error=f'Goal parse error for option {letter}: {goal_fol[:60]!r}',
                )
        else:
            all_results[letter] = ReasonerResult(
                status='error',
                premise_ids=[],
                error=f'Cannot convert option {letter} to FOL: {option_nl[:60]!r}',
            )

    # Collect entailed options
    entailed = [l for l, r in all_results.items() if r.status == 'yes']

    if not entailed:
        # Strict mode: no option entailed
        return ResolverResult(
            answer='',  # empty = no answer (strict)
            premise_ids=[],
            status='uncertain',
            all_option_results=all_results,
            error='No option entailed by KB',
        )

    if len(entailed) == 1:
        best = entailed[0]
    else:
        # Fewest premises tiebreak, then alphabetical
        best = min(entailed, key=lambda l: (len(all_results[l].premise_ids), l))

    best_result = all_results[best]
    return ResolverResult(
        answer=best,
        premise_ids=best_result.premise_ids,
        status='yes',
        goal_str=option_fols.get(best, ''),
        all_option_results=all_results,
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== Oracle Resolver Self-Tests ===\n")

    # Test 1: Truth question — direct entailment
    prems = [
        "∀x (WT(x) → O(x))",
        "∀x (¬PEP8(x) → ¬WT(x))",
        "∀x (WT(x))",
        "∀x (WT(x) → PEP8(x))",
        "∀x (WS(x) → O(x))",
        "∀x (EM(x) → WT(x))",
        "∀x (O(x) → CR(x))",
        "∀x (WS(x))",
        "∀x (CR(x))",
        "∃x (BP(x))",
        "∃x (O(x))",
    ]
    q_truth = "Does it follow that if all Python projects are well-structured, then all Python projects are optimized, according to the premises?"
    opts_truth = ["Yes", "No", "Uncertain"]

    res = resolve_question(prems, q_truth, opts_truth)
    print(f"Test 1 (truth - Yes expected):")
    print(f"  Answer: {res.answer!r}  Status: {res.status}")
    print(f"  Goal extracted: {res.goal_str!r}")
    print(f"  Error: {res.error!r}" if res.error else "")
    print()

    # Test 2: truth question — No
    prems2 = ["∀x (A(x) → B(x))", "∀x (A(x))"]
    q2 = "Is it true that if A then not B?"
    res2 = resolve_question(prems2, q2, ["Yes", "No", "Uncertain"])
    print(f"Test 2 (truth - No expected):")
    print(f"  Answer: {res2.answer!r}  Status: {res2.status}")
    print()

    # Test 3: MCQ
    q_mcq = ("Which conclusion follows with the fewest premises?\n"
              "A. If a Python project is not optimized, then it is not well-tested\n"
              "B. If all Python projects are optimized, then all Python projects are well-structured\n"
              "C. If a Python project is well-tested, then it must be clean and readable\n"
              "D. If a Python project is not optimized, then it does not follow PEP 8 standards")
    opts_mcq = ["A", "B", "C", "D"]
    res3 = resolve_question(prems, q_mcq, opts_mcq)
    print(f"Test 3 (MCQ - A expected for fewest premises):")
    print(f"  Answer: {res3.answer!r}  Status: {res3.status}")
    print(f"  Premise_ids: {res3.premise_ids}")
    print(f"  Error: {res3.error!r}" if res3.error else "")
    if res3.all_option_results:
        for letter, r in res3.all_option_results.items():
            print(f"    Option {letter}: status={r.status}  premises_used={r.premise_ids}  err={r.error[:50] if r.error else ''}")
