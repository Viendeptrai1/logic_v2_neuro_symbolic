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

# System Prompt for NL -> FOL Translation
SYSTEM_PROMPT = """You are an expert in formal logic. Your task is to translate a given natural language statement into First-Order Logic (FOL).
You will be provided with:
1. A list of background premises in Natural Language (NL).
2. Their corresponding First-Order Logic (FOL) translations. This acts as your lexicon/glossary for predicate names.
3. A Target Statement in Natural Language that you must translate.

RULES:
- You MUST use the exact abbreviated predicate names found in the provided FOL premises.
- Output ONLY the final FOL string. Do NOT output any markdown blocks, explanations, or quotes.
- Use standard mathematical notation: ∀, ∃, →, ¬, ∧, ∨, ↔
"""

def evaluate_baseline(model_name="qwen2.5-coder:7b"):
    with open('data/Logic_Based_Educational_Queries.json') as f:
        data = json.load(f)
    
    total = 0
    correct = 0
    
    print(f"Testing Baseline Model: {model_name} on {len(ENCODED_CASES)} cases...")
    
    for case in ENCODED_CASES:
        item = data[case.item_idx]
        q_text = item['questions'][case.q_idx]
        prems_nl = item['premises-NL']
        prems_fol = item['premises-FOL']
        
        # Build prompt
        prompt = "Background Premises:\n"
        for nl, fol in zip(prems_nl, prems_fol):
            prompt += f"- NL: {nl}\n  FOL: {fol}\n"
            
        prompt += f"\nTarget Statement to Translate:\n{q_text}\n\nFOL Output:"
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            gen_fol = response.choices[0].message.content.strip()
            
            # Simple equivalence check (remove spaces)
            gen_clean = gen_fol.replace(" ", "")
            gold_clean = case.goal_fol.replace(" ", "")
            
            if gen_clean == gold_clean:
                correct += 1
            
            total += 1
            print(f"\rProgress: {total}/{len(ENCODED_CASES)} | Correct: {correct}", end="")
            
        except Exception as e:
            print(f"\nError on case {total}: {e}")
            total += 1
            
    print(f"\n\nBaseline Results for {model_name}:")
    print(f"Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    evaluate_baseline(model_name="qwen-logic:latest")

