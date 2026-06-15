"""
oracle_harness.py — Main evaluation harness for the Oracle Experiment (Step 1).

Chạy gold FOL premises qua Z3 engine trên toàn bộ 808 câu.
Không dùng LLM — đây là thí nghiệm "trần năng lực symbolic".

Usage
-----
    PYTHONPATH=. python3 eval/oracle_harness.py \\
        --data data/Logic_Based_Educational_Queries.json \\
        --output eval/oracle_results.json

Output
------
  - Console: per-construct breakdown + summary table
  - JSON: full results per question (for later analysis)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fol.fol_parser import parse, ParseError, collect_predicates
from src.fol.reasoner import check_entailment, ReasonerResult
from src.resolve.oracle_resolver import (
    resolve_question,
    is_truth_question,
    is_mcq_question,
    extract_truth_goal,
    extract_mcq_options,
    nl_to_fol_heuristic,
    _map_truth_status,
    _map_uncertain,
    ResolverResult,
)


# ---------------------------------------------------------------------------
# Construct detector
# ---------------------------------------------------------------------------

def detect_constructs(premises_fol: list[str], question_text: str) -> set[str]:
    """
    Detect which logical constructs are present in an item.
    Returns a set of construct labels.
    """
    constructs = set()
    full_fol = ' '.join(premises_fol)

    # Quantifiers
    if '∀' in full_fol or 'ForAll' in full_fol:
        constructs.add('universal')
    if '∃' in full_fol or 'Exists(' in full_fol:
        constructs.add('existential')

    # Connectives
    if '¬' in full_fol or '!' in full_fol:
        constructs.add('negation')
    if '∧' in full_fol:
        constructs.add('conjunction')
    if '∨' in full_fol:
        constructs.add('disjunction')
    if '↔' in full_fol or '<->' in full_fol:
        constructs.add('biconditional')

    # Numeric
    if any(op in full_fol for op in ['>=', '<=', '>', '<', '= ']):
        # filter out => in ForAll/Exists context
        if re.search(r'[0-9]', full_fol):
            constructs.add('numeric')

    # Ground facts (constants)
    if re.search(r'\b[A-Z][a-z]+\s*\)', full_fol):
        constructs.add('ground_facts')

    # Nested quantifiers
    if 'ForAll(x, ForAll' in full_fol or 'ForAll(s, ForAll' in full_fol:
        constructs.add('nested_quantifier')

    # Multi-hop (long implication chains — heuristic: >=3 premises with →)
    n_arrows = full_fol.count('→') + full_fol.count('->')
    if n_arrows >= 3:
        constructs.add('multi_hop')

    # Fewest-premises question
    if re.search(r'fewest\s+premises?', question_text, re.IGNORECASE):
        constructs.add('fewest_premises')

    # Contraposition in question text
    if re.search(r'contrapos', question_text, re.IGNORECASE):
        constructs.add('contraposition')

    return constructs


# ---------------------------------------------------------------------------
# F1 computation for premises_used vs gold idx
# ---------------------------------------------------------------------------

def premises_f1(pred_ids_0based: list[int], gold_ids_1based: list[int]) -> tuple[float, float, float]:
    """
    Compute P, R, F1 between predicted (0-based) and gold (1-based → convert to 0-based).
    """
    gold_0based = set(i - 1 for i in gold_ids_1based)
    pred_set    = set(pred_ids_0based)

    if not pred_set and not gold_0based:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_0based:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gold_0based)
    p  = tp / len(pred_set)
    r  = tp / len(gold_0based)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Single-question runner
# ---------------------------------------------------------------------------

def run_question(
    item_idx: int,
    q_idx: int,
    premises_fol: list[str],
    question_text: str,
    gold_answer: str,
    gold_premise_ids_1based: list[int],
    timeout_ms: int = 5000,
) -> dict:
    """Run one question and return a result dict."""

    # Determine option list from question text and gold answer
    if gold_answer in ('Yes', 'No', 'Uncertain', 'Unknown', 'True', 'False'):
        options = _infer_truth_options(question_text, gold_answer)
    elif re.match(r'^[A-D]$', gold_answer):
        options = _infer_mcq_options(question_text)
    else:
        options = [gold_answer]

    result = resolve_question(premises_fol, question_text, options, timeout_ms=timeout_ms)

    # Compute correctness
    correct = (result.answer.strip().lower() == gold_answer.strip().lower())

    # Compute F1
    p, r, f1 = premises_f1(result.premise_ids, gold_premise_ids_1based)

    constructs = detect_constructs(premises_fol, question_text)

    return {
        'item_idx':    item_idx,
        'q_idx':       q_idx,
        'question_type': 'truth' if is_truth_question(options) else 'mcq',
        'gold_answer': gold_answer,
        'pred_answer': result.answer,
        'correct':     correct,
        'status':      result.status,
        'goal_str':    result.goal_str,
        'error':       result.error,
        'pred_ids_0':  result.premise_ids,
        'gold_ids_1':  gold_premise_ids_1based,
        'prem_f1':     f1,
        'prem_p':      p,
        'prem_r':      r,
        'constructs':  list(constructs),
    }


def _infer_truth_options(question_text: str, gold_answer: str) -> list[str]:
    """Infer truth options from question and gold answer."""
    # Check for Uncertain/Unknown in question text
    if 'uncertain' in question_text.lower():
        return ['Yes', 'No', 'Uncertain']
    if 'unknown' in question_text.lower():
        return ['Yes', 'No', 'Unknown']
    # Default: Yes/No/Uncertain
    return ['Yes', 'No', 'Uncertain']


def _infer_mcq_options(question_text: str) -> list[str]:
    """Infer MCQ options from question text."""
    letters = sorted(set(re.findall(r'\n([A-D])[\.)\s]', question_text)))
    return letters if letters else ['A', 'B', 'C', 'D']


# ---------------------------------------------------------------------------
# Main harness loop
# ---------------------------------------------------------------------------

def run_harness(
    data_path: str,
    output_path: str | None,
    timeout_ms: int = 5000,
    max_items: int | None = None,
    verbose: bool = False,
) -> dict:
    """Run the full oracle harness and return aggregated results."""

    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    if max_items:
        data = data[:max_items]

    all_q_results = []
    t_start = time.perf_counter()

    for item_i, item in enumerate(data):
        premises_fol   = item['premises-FOL']
        questions      = item['questions']
        answers        = item['answers']
        gold_idx       = item['idx']           # list of lists, 1-based

        for q_i, (q_text, gold_ans, gold_ids_1) in enumerate(
            zip(questions, answers, gold_idx)
        ):
            if verbose:
                print(f"\rProcessing item {item_i+1}/{len(data)}, Q{q_i+1}...", end='', flush=True)

            q_result = run_question(
                item_idx=item_i,
                q_idx=q_i,
                premises_fol=premises_fol,
                question_text=q_text,
                gold_answer=gold_ans,
                gold_premise_ids_1based=gold_ids_1,
                timeout_ms=timeout_ms,
            )
            all_q_results.append(q_result)

    if verbose:
        print()  # newline after \r

    elapsed = time.perf_counter() - t_start

    # Aggregate results
    summary = _aggregate(all_q_results, elapsed)

    # Save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary':  summary,
                'results':  all_q_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    return summary


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def _aggregate(results: list[dict], elapsed: float) -> dict:
    total       = len(results)
    correct     = sum(r['correct'] for r in results)
    parseable   = sum(r['status'] != 'unparseable' and r['status'] != 'error' for r in results)
    unparseable = sum(r['status'] == 'unparseable' for r in results)
    errors      = sum(r['status'] == 'error' for r in results)
    uncertain   = sum(r['status'] == 'uncertain' for r in results)
    no_answer   = sum(r['pred_answer'] == '' for r in results)

    # Answer accuracy overall
    acc_total   = correct / total if total else 0.0
    # Strict: exclude questions where we had no answer (empty string)
    answered    = [r for r in results if r['pred_answer'] != '']
    acc_strict  = (sum(r['correct'] for r in answered) / len(answered)) if answered else 0.0

    # Premises F1
    f1_vals = [r['prem_f1'] for r in results if r['pred_ids_0']]
    avg_f1  = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0

    # Per question type
    truth_r = [r for r in results if r['question_type'] == 'truth']
    mcq_r   = [r for r in results if r['question_type'] == 'mcq']
    truth_acc = sum(r['correct'] for r in truth_r) / len(truth_r) if truth_r else 0.0
    mcq_acc   = sum(r['correct'] for r in mcq_r)   / len(mcq_r)   if mcq_r   else 0.0

    # Per construct
    construct_stats: dict[str, dict] = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        for c in r['constructs']:
            construct_stats[c]['total'] += 1
            construct_stats[c]['correct'] += int(r['correct'])

    construct_acc = {
        c: {
            'accuracy': v['correct'] / v['total'] if v['total'] else 0.0,
            'total': v['total'],
            'correct': v['correct'],
        }
        for c, v in sorted(construct_stats.items())
    }

    summary = {
        'total_questions':      total,
        'correct':              correct,
        'accuracy_total':       round(acc_total, 4),
        'accuracy_strict':      round(acc_strict, 4),
        'answered_count':       len(answered),
        'unparseable_count':    unparseable,
        'error_count':          errors,
        'uncertain_count':      uncertain,
        'no_answer_count':      no_answer,
        'premises_f1_macro':    round(avg_f1, 4),
        'truth_questions':      len(truth_r),
        'truth_accuracy':       round(truth_acc, 4),
        'mcq_questions':        len(mcq_r),
        'mcq_accuracy':         round(mcq_acc, 4),
        'construct_accuracy':   construct_acc,
        'elapsed_sec':          round(elapsed, 2),
    }
    return summary


def print_report(summary: dict):
    """Print a human-readable report."""
    print()
    print("=" * 65)
    print("  ORACLE HARNESS RESULTS  (Gold FOL → Z3 Engine, No LLM)")
    print("=" * 65)
    print(f"  Total questions:           {summary['total_questions']}")
    print(f"  Correct (total):           {summary['correct']} / {summary['total_questions']}"
          f"  ({summary['accuracy_total']*100:.1f}%)")
    print(f"  Accuracy (strict, w/ans):  {summary['accuracy_strict']*100:.1f}%"
          f"  ({summary['answered_count']} answered)")
    print(f"  Unparseable (no goal):     {summary['unparseable_count']}")
    print(f"  Error:                     {summary['error_count']}")
    print(f"  No answer (strict empty):  {summary['no_answer_count']}")
    print(f"  Premises F1 (macro):       {summary['premises_f1_macro']*100:.1f}%")
    print()
    print(f"  By question type:")
    print(f"    Truth (Yes/No/Unc):  {summary['truth_accuracy']*100:.1f}%"
          f"  ({summary['truth_questions']} questions)")
    print(f"    MCQ  (A/B/C/D):     {summary['mcq_accuracy']*100:.1f}%"
          f"  ({summary['mcq_questions']} questions)")
    print()
    print(f"  Per-construct accuracy:")
    print(f"  {'Construct':<25}  {'Acc':>6}  {'Correct':>7}  {'Total':>5}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*7}  {'-'*5}")
    for c, stats in summary['construct_accuracy'].items():
        acc_pct = stats['accuracy'] * 100
        print(f"  {c:<25}  {acc_pct:>5.1f}%  {stats['correct']:>7}  {stats['total']:>5}")
    print()
    print(f"  Elapsed:  {summary['elapsed_sec']:.1f}s")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Oracle Harness: Gold FOL → Z3 evaluation')
    parser.add_argument('--data',    default='data/Logic_Based_Educational_Queries.json',
                        help='Path to dataset JSON')
    parser.add_argument('--output',  default='eval/oracle_results.json',
                        help='Path to save results JSON')
    parser.add_argument('--timeout', type=int, default=5000,
                        help='Z3 timeout per question in milliseconds (default: 5000)')
    parser.add_argument('--max',     type=int, default=None,
                        help='Max items to process (for quick smoke test)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress')
    args = parser.parse_args()

    print(f"Oracle Harness")
    print(f"  Data:    {args.data}")
    print(f"  Output:  {args.output}")
    print(f"  Timeout: {args.timeout}ms")
    if args.max:
        print(f"  Max:     {args.max} items")
    print()

    summary = run_harness(
        data_path=args.data,
        output_path=args.output,
        timeout_ms=args.timeout,
        max_items=args.max,
        verbose=args.verbose,
    )
    print_report(summary)


if __name__ == '__main__':
    main()
