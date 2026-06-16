import json
import logging
from openai import OpenAI
from eval.sample_encoded import ENCODED_CASES

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

SYSTEM_PROMPT = """You are an expert in formal logic. You are given a set of premises with their First-Order Logic (FOL). Translate the QUESTION/GOAL into a single FOL formula. Reuse ONLY predicate names that appear in the premises. Use the symbols ∀, ∃, →, ¬, ∧, ∨. Output ONLY the FOL string."""

from src.fol.reasoner import check_entailment

def evaluate_baseline(model_name="qwen-logic-v2"):
    with open('data/Logic_Based_Educational_Queries.json') as f:
        data = json.load(f)
    
    total = 0
    correct_string = 0
    correct_logic = 0
    
    print(f"Testing Baseline Model: {model_name} on {len(ENCODED_CASES)} cases...")
    
    results = []
    
    for case in ENCODED_CASES:
        item = data[case.item_idx]
        q_text = item['questions'][case.q_idx]
        prems_fol = item['premises-FOL']
        
        # Build prompt matching the new Goal SFT schema
        prompt = "Premises (FOL):\n"
        for i, fol in enumerate(prems_fol, 1):
            prompt += f"{i}. {fol}\n"
            
        prompt += f"\nQuestion: {q_text}\n\nTranslate the goal of this question into one FOL formula."
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            gen_fol = response.choices[0].message.content.strip()
            
            # Simple equivalence check (remove spaces)
            gen_clean = gen_fol.replace(" ", "")
            gold_clean = case.goal_fol.replace(" ", "")
            
            is_correct = False
            match_type = "none"
            
            if gen_clean == gold_clean:
                correct_string += 1
                correct_logic += 1
                is_correct = True
                match_type = "string"
            else:
                # Use Z3 Solver to check logical equivalence: Gold <-> Gen
                # Gold -> Gen
                res1 = check_entailment([case.goal_fol], gen_fol)
                # Gen -> Gold
                res2 = check_entailment([gen_fol], case.goal_fol)
                
                if res1.status == 'yes' and res2.status == 'yes':
                    correct_logic += 1
                    is_correct = True
                    match_type = "logic"
                    print(f"\n[Logic Match] Target: {case.goal_fol}")
                    print(f"              Output: {gen_fol}\n")
                else:
                    print(f"\n[Mismatch] Target: {case.goal_fol}")
                    print(f"           Output: {gen_fol}")
                    print(f"           Z3 (Gold->Gen): {res1.status}, Z3 (Gen->Gold): {res2.status}\n")
            
            total += 1
            print(f"\rProgress: {total}/{len(ENCODED_CASES)} | String Match: {correct_string} | Logic Match: {correct_logic}", end="")
            
            results.append({
                "item_idx": case.item_idx,
                "q_idx": case.q_idx,
                "gold_fol": case.goal_fol,
                "gen_fol": gen_fol,
                "is_correct": is_correct,
                "match_type": match_type
            })
            
        except Exception as e:
            print(f"\nError on case {total}: {e}")
            total += 1
            
    print(f"\n\nBaseline Results for {model_name}:")
    print(f"Strict String Accuracy: {correct_string}/{total} ({correct_string/total*100:.1f}%)")
    print(f"Logical Equivalence Accuracy: {correct_logic}/{total} ({correct_logic/total*100:.1f}%)")
    
    with open('eval/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluate_baseline(model_name="qwen-logic-v2")

