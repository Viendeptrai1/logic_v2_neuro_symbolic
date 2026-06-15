"""
generate_silver_data.py — Generates Silver-Standard FOL Goals for LLM Training

Uses goal_extractor.py to auto-translate NL questions to FOL.
Applies rigorous Triple Filter verification using Z3 to guarantee mathematical correctness.
Outputs a clean SFT dataset JSON.
"""

from __future__ import annotations
import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.resolve.goal_extractor import extract_goal_with_lexicon
from src.fol.reasoner import check_entailment

def _parse_mcq_options(question_text: str) -> dict[str, str]:
    """Extracts A/B/C/D option texts from an MCQ question."""
    options = {}
    
    # Try finding A., B., C., D.
    m = re.split(r'\n([A-D])\.\s*', question_text)
    if len(m) >= 5: # text before A, 'A', text A, 'B', text B...
        for i in range(1, len(m), 2):
            letter = m[i].strip()
            text = m[i+1].strip()
            options[letter] = text
            
    return options

def evaluate_item(item_idx: int, q_idx: int, question_text: str, gold_ans: str, gold_idx_1based: list[int], premises_nl: list[str], premises_fol: list[str]) -> tuple[bool, bool, dict | None]:
    """
    Returns (strict_pass, raw_pass, verified_item).
    """
    gold_idx = {i - 1 for i in gold_idx_1based}
    
    # Check if MCQ or Truth
    is_mcq = gold_ans in ["A", "B", "C", "D"]
    is_unknown = gold_ans in ["Unknown", "Uncertain"]
    is_truth = gold_ans in ["Yes", "No", "True", "False"]
    
    # Extract
    if is_mcq:
        mcq_texts = _parse_mcq_options(question_text)
        if gold_ans not in mcq_texts:
            return False, False, None
        
        extracted_options = {}
        for letter, opt_text in mcq_texts.items():
            fol, has_low = extract_goal_with_lexicon(opt_text, premises_nl, premises_fol)
            if not fol:
                return False, False, None # Must extract all
            extracted_options[letter] = (fol, has_low)
            
        gold_fol, gold_low = extracted_options[gold_ans]
        
        # Profile Test
        res = check_entailment(premises_fol, gold_fol)
        if res.status != 'yes':
            return False, False, None
            
        raw_pass = True
        
        gold_core = set(res.premise_ids)
        
        # Check Distractors
        for letter, (fol, low) in extracted_options.items():
            if letter == gold_ans: continue
            dist_res = check_entailment(premises_fol, fol)
            if dist_res.status == 'yes':
                # Distractor is entailed! We drop this to be safe.
                return False, True, None
        
        # Intersection Test
        if not gold_core: return False, True, None
        overlap = len(gold_core & gold_idx) / len(gold_core)
        if overlap < 0.5:
            return False, True, None
            
        return True, True, {
            "type": "mcq",
            "question": question_text,
            "gold_answer": gold_ans,
            "gold_fol": gold_fol,
            "options_fol": {k: v[0] for k, v in extracted_options.items()}
        }
        
    elif is_truth or is_unknown:
        fol, has_low = extract_goal_with_lexicon(question_text, premises_nl, premises_fol)
        if not fol: return False, False, None
        
        res = check_entailment(premises_fol, fol)
        gold_core = set(res.premise_ids)
        
        if is_truth:
            # Track A for Truth
            expected_status = 'yes' if gold_ans in ['Yes', 'True'] else 'no'
            if res.status != expected_status:
                return False, False, None
                
            raw_pass = True
            
            # Flip Test
            flip_fol = f"¬({fol})"
            flip_res = check_entailment(premises_fol, flip_fol)
            
            if expected_status == 'yes' and flip_res.status != 'no' and flip_res.status != 'uncertain':
                return False, True, None
            if expected_status == 'no' and flip_res.status != 'yes':
                return False, True, None
                
            # Intersection Test
            used_core = gold_core if expected_status == 'yes' else set(flip_res.premise_ids)
            if not used_core: return False, True, None
            overlap = len(used_core & gold_idx) / len(used_core)
            if overlap < 0.5:
                return False, True, None
                
            return True, True, {
                "type": "truth",
                "question": question_text,
                "gold_answer": gold_ans,
                "gold_fol": fol
            }
            
        elif is_unknown:
            # Track B for Unknown (Faithfulness)
            if res.status != 'uncertain':
                return False, False, None
                
            raw_pass = True
            
            if has_low: 
                return False, True, None # Strict structural requirement
                
            flip_fol = f"¬({fol})"
            flip_res = check_entailment(premises_fol, flip_fol)
            if flip_res.status != 'uncertain':
                return False, True, None
                
            return True, True, {
                "type": "truth",
                "question": question_text,
                "gold_answer": gold_ans,
                "gold_fol": fol
            }
            
    return False, False, None

def main():
    data_path = 'data/Logic_Based_Educational_Queries.json'
    out_path = 'data/silver_sft_targets.json'
    
    with open(data_path) as f:
        data = json.load(f)
        
    total_qs = 0
    raw_pass_count = 0
    strict_pass_count = 0
    verified = []
    
    print("Generating Silver SFT Data with Triple Filter...")
    
    for item_idx, item in enumerate(data):
        premises_nl = item['premises-NL']
        premises_fol = item['premises-FOL']
        idx_list = item['idx']
        
        for q_idx, q_text in enumerate(item['questions']):
            total_qs += 1
            gold_ans = item['answers'][q_idx]
            gold_idx_1based = idx_list[q_idx]
            
            strict_pass, raw_pass, result = evaluate_item(
                item_idx, q_idx, q_text, gold_ans, gold_idx_1based,
                premises_nl, premises_fol
            )
            
            if raw_pass:
                raw_pass_count += 1
            if strict_pass and result:
                strict_pass_count += 1
                result['item_idx'] = item_idx
                result['q_idx'] = q_idx
                verified.append(result)
                
        print(f"\rProcessed {item_idx+1}/{len(data)} items. Raw: {raw_pass_count}, Strict: {strict_pass_count}", end="")
        
    print(f"\n\n--- RESULTS ---")
    print(f"Total Questions: {total_qs}")
    print(f"Raw Yield (Matched Answer Only): {raw_pass_count} ({raw_pass_count/total_qs*100:.1f}%)")
    print(f"Strict Yield (Passed Triple Filter): {strict_pass_count} ({strict_pass_count/total_qs*100:.1f}%)")
    print(f"Garbage caught by filters: {raw_pass_count - strict_pass_count} items")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)
        
    print(f"Saved strict verified items to {out_path}")

if __name__ == "__main__":
    main()
