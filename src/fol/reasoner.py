"""
reasoner.py — Z3-based FOL reasoner with 3-state semantics and unsat-core tracking.

Protocol (đúng 2-chiều):
  1. Negate goal: check  KB ∧ ¬goal  →  if UNSAT  →  'yes'  (KB entails goal)
  2. Prove negation: check  KB ∧ goal  →  if UNSAT  →  'no'  (KB entails ¬goal)
  3. Both SAT  →  'uncertain'

Premise tracking:
  - Each premise is asserted with a unique Bool tracking variable.
  - After UNSAT, z3.Solver.unsat_core() returns the used tracking vars.
  - We map those back to premise indices (0-based).
  - This gives us `premises_used` for free.

Usage
-----
>>> from src.fol.fol_parser import parse
>>> from src.fol.fol_to_z3 import Z3Context
>>> from src.fol.reasoner import check_entailment
>>> premises_str = ["∀x (WT(x) → O(x))", "∀x (WT(x))"]
>>> goal_str = "∀x (O(x))"
>>> result = check_entailment(premises_str, goal_str)
>>> result['status']
'yes'
>>> result['premise_ids']
[0, 1]
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

import z3

from src.fol.fol_parser import parse, ParseError
from src.fol.fol_to_z3 import Z3Context, TranslateError


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReasonerResult:
    status: str                         # 'yes' | 'no' | 'uncertain' | 'error'
    premise_ids: list[int]              # 0-based indices of premises used in proof
    elapsed_ms: float = 0.0
    error: str = ''
    proof_direction: str = ''           # 'forward' (yes) or 'backward' (no) or ''
    solver_unknown: bool = False        # True when Z3 gave z3.unknown (≠ logical uncertain)
    core_minimized: bool = False        # True when unsat core was minimized



# ---------------------------------------------------------------------------
# Core entailment check
# ---------------------------------------------------------------------------

def check_entailment(
    premises_str: list[str],
    goal_str: str,
    timeout_ms: int = 5000,
    parse_cache: dict | None = None,
) -> ReasonerResult:
    """
    Check whether the given premises (as FOL strings) entail the goal (FOL string).

    Returns ReasonerResult with status in {'yes', 'no', 'uncertain', 'error'}.
    """
    t0 = time.perf_counter()

    # --- Parse all formulas -------------------------------------------------
    try:
        premises_ast = []
        for i, s in enumerate(premises_str):
            if parse_cache and s in parse_cache:
                ast = parse_cache[s]
            else:
                ast = parse(s)
                if parse_cache is not None:
                    parse_cache[s] = ast
            premises_ast.append(ast)
        goal_ast = parse(goal_str)
    except ParseError as e:
        return ReasonerResult(
            status='error',
            premise_ids=[],
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=f'ParseError: {e}',
        )

    # --- Translate to Z3 ----------------------------------------------------
    try:
        ctx = Z3Context()
        premises_z3 = ctx.translate_all(premises_ast)
        goal_z3     = ctx.translate(goal_ast)
    except (TranslateError, Exception) as e:
        return ReasonerResult(
            status='error',
            premise_ids=[],
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=f'TranslateError: {e}',
        )

    result = _check_z3(premises_z3, goal_z3, timeout_ms)
    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


def check_entailment_z3(
    premises_z3: list[z3.ExprRef],
    goal_z3: z3.ExprRef,
    timeout_ms: int = 5000,
) -> ReasonerResult:
    """
    Same as check_entailment but takes already-translated Z3 expressions.
    Useful when multiple goals share the same premise set.
    """
    t0 = time.perf_counter()
    result = _check_z3(premises_z3, goal_z3, timeout_ms)
    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# Internal Z3 check logic
# ---------------------------------------------------------------------------

def _check_z3(
    premises_z3: list[z3.ExprRef],
    goal_z3: z3.ExprRef,
    timeout_ms: int,
) -> ReasonerResult:
    """Run the 2-direction check and return a ReasonerResult (without timing)."""

    # Set up solver with tracking
    solver = z3.Solver()
    solver.set('timeout', timeout_ms)

    # Create tracking Bool for each premise
    track_vars = [z3.Bool(f'__p{i}') for i in range(len(premises_z3))]
    for tv, prem in zip(track_vars, premises_z3):
        solver.assert_and_track(prem, tv)

    got_unknown = False  # tracks if any z3.unknown result seen

    # ---- Direction 1: KB ∧ ¬goal  →  UNSAT  →  'yes' ---------------------
    solver.push()
    solver.add(z3.Not(goal_z3))
    r1 = solver.check()

    if r1 == z3.unsat:
        core = solver.unsat_core()
        prem_ids = _core_to_ids(core, track_vars)
        # Minimize unsat core via deletion loop
        prem_ids, minimized = _minimize_core(
            premises_z3, goal_z3, prem_ids, track_vars, direction='yes', timeout_ms=timeout_ms
        )
        solver.pop()
        return ReasonerResult(
            status='yes',
            premise_ids=prem_ids,
            proof_direction='forward',
            core_minimized=minimized,
        )
    if r1 == z3.unknown:
        got_unknown = True
    solver.pop()

    # ---- Direction 2: KB ∧ goal  →  UNSAT  →  'no' -----------------------
    solver.push()
    solver.add(goal_z3)
    r2 = solver.check()

    if r2 == z3.unsat:
        core = solver.unsat_core()
        prem_ids = _core_to_ids(core, track_vars)
        prem_ids, minimized = _minimize_core(
            premises_z3, goal_z3, prem_ids, track_vars, direction='no', timeout_ms=timeout_ms
        )
        solver.pop()
        return ReasonerResult(
            status='no',
            premise_ids=prem_ids,
            proof_direction='backward',
            core_minimized=minimized,
        )
    if r2 == z3.unknown:
        got_unknown = True
    solver.pop()

    # ---- Z3 gave unknown on at least one direction ------------------------
    if got_unknown:
        # Z3 could not decide — distinct from "genuinely uncertain" (both sat)
        return ReasonerResult(
            status='uncertain',
            premise_ids=[],
            error='z3_solver_unknown',
            solver_unknown=True,
        )

    # Both SAT → genuinely logically uncertain / independent
    return ReasonerResult(status='uncertain', premise_ids=[])


def _minimize_core(
    premises_z3: list[z3.ExprRef],
    goal_z3: z3.ExprRef,
    core_ids: list[int],
    track_vars: list[z3.ExprRef],
    direction: str,
    timeout_ms: int,
) -> tuple[list[int], bool]:
    """
    Deletion-based unsat core minimization.
    Tries removing each premise from the core; keeps the removal if still unsat.
    Returns (minimized_ids, was_minimized).
    
    Note: Z3's unsat_core() is not guaranteed minimal. This finds ONE minimal subset
    (there may be others — F1 vs gold idx is a soft metric by design).
    Timeout is halved to stay within budget.
    """
    if len(core_ids) <= 1:
        return core_ids, False

    mini_timeout = max(500, timeout_ms // 4)
    current = list(core_ids)
    minimized = False

    for cand in list(current):  # iterate over a copy
        trial = [i for i in current if i != cand]
        if not trial:
            continue
        # Test if trial subset alone is still sufficient
        s = z3.Solver()
        s.set('timeout', mini_timeout)
        for idx in trial:
            s.add(premises_z3[idx])
        if direction == 'yes':
            s.add(z3.Not(goal_z3))
        else:
            s.add(goal_z3)
        result = s.check()
        if result == z3.unsat:
            current = trial
            minimized = True

    return sorted(current), minimized



def _core_to_ids(core: z3.AstVector, track_vars: list[z3.ExprRef]) -> list[int]:
    """Map an unsat core back to premise indices."""
    core_set = {str(c) for c in core}
    ids = [i for i, tv in enumerate(track_vars) if str(tv) in core_set]
    return sorted(ids)


# ---------------------------------------------------------------------------
# Acceptance test suite
# ---------------------------------------------------------------------------

def run_acceptance_tests():
    """Run the 9 acceptance test cases from the implementation plan."""
    tests = [
        # (description, premises_fol_list, goal_fol, expected_status)
        (
            "1. Multi-hop Horn: ∀x WT(x), ∀x WT(x)→O(x) ⊢ ∀x O(x)",
            ["∀x (WT(x))", "∀x (WT(x) → O(x))"],
            "∀x (O(x))",
            "yes",
        ),
        (
            "2. Universal tautology: ∀x A(x)→B(x) ⊢ ∀x ¬A(x)∨B(x)",
            ["∀x (A(x) → B(x))"],
            "∀x (¬A(x) ∨ B(x))",
            "yes",
        ),
        (
            "3. Contraposition: ∀x A(x)→B(x) ⊢ ∀x ¬B(x)→¬A(x)",
            ["∀x (A(x) → B(x))"],
            "∀x (¬B(x) → ¬A(x))",
            "yes",
        ),
        (
            "4. Biconditional: FailTest ↔ ¬PassTest, PassTest(c) ⊢ ¬FailTest(c) [propositional]",
            ["FailTest ↔ ¬PassTest", "PassTest"],
            "¬FailTest",
            "yes",
        ),
        (
            "5. Existential: ∃x BP(x) ⊢ ∃x BP(x)",
            ["∃x (BP(x))"],
            "∃x (BP(x))",
            "yes",
        ),
        (
            "6. Disjunction: ∀x A(x)→(B(x)∨C(x)), ∀x A(x), ∀x ¬B(x) ⊢ ∀x C(x)",
            ["∀x (A(x) → (B(x) ∨ C(x)))", "∀x (A(x))", "∀x (¬B(x))"],
            "∀x (C(x))",
            "yes",
        ),
        (
            "7. Uncertain: unrelated premises → goal_unrelated",
            ["∀x (P(x) → Q(x))"],
            "∀x (R(x))",
            "uncertain",
        ),
        (
            "8. Negation of goal: ∀x (A(x)), ∀x (A(x) → ¬B(x)) ⊢ B(c) → no",
            ["∀x (A(x))", "∀x (A(x) → ¬B(x))"],
            "∀x (B(x))",
            "no",
        ),
        (
            "9. Numeric threshold: ∀x score(x) >= 8.5 → pass(x), score(alex) = 9.0 ⊢ pass(alex)",
            ["ForAll(x, score(x) >= 8.5 → pass(x))", "score(alex) = 9.0"],
            "pass(alex)",
            "yes",
        ),
    ]

    ok = fail = 0
    for desc, prems, goal, expected in tests:
        result = check_entailment(prems, goal)
        status_ok = result.status == expected
        prem_info = f"premises_used={result.premise_ids}" if result.premise_ids else ""
        icon = "✓" if status_ok else "✗"
        print(f"  {icon}  {desc}")
        if not status_ok:
            print(f"       Expected: {expected!r}  Got: {result.status!r}")
            if result.error:
                print(f"       Error: {result.error}")
        else:
            print(f"       status={result.status!r}  {prem_info}  ({result.elapsed_ms:.1f}ms)")
        if status_ok:
            ok += 1
        else:
            fail += 1

    print(f"\n{ok}/{ok+fail} acceptance tests passed")
    return ok, fail


if __name__ == '__main__':
    run_acceptance_tests()
