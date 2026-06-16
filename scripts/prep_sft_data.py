import json
import re

def _parse_mcq_options(question_text: str) -> dict[str, str]:
    options = {}
    m = re.split(r'\n([A-D])\.\s*', question_text)
    if len(m) >= 5:
        for i in range(1, len(m), 2):
            letter = m[i].strip()
            text = m[i+1].strip()
            options[letter] = text
    return options

def build_premise_prompt(nl_text):
    return f"Translate the following natural language statement into First-Order Logic (FOL).\n\nStatement: {nl_text}"

def build_goal_prompt(item, q_text):
    prompt = "Background Premises:\n"
    for nl, fol in zip(item['premises-NL'], item['premises-FOL']):
        prompt += f"- NL: {nl}\n  FOL: {fol}\n"
    prompt += f"\nTarget Statement to Translate:\n{q_text}"
    return prompt

def main():
    with open('data/Logic_Based_Educational_Queries.json') as f:
        data = json.load(f)
        
    with open('data/silver_sft_targets.json') as f:
        silver = json.load(f)
        
    sft_data = []
    
    SYSTEM_PREMISE = "You are an expert in formal logic. Translate the natural language statement into First-Order Logic (FOL). Use standard notation (∀, ∃, →, ¬, ∧, ∨) and abbreviated predicate names. Output ONLY the FOL string."
    
    SYSTEM_GOAL = """You are an expert in formal logic. Your task is to translate a given natural language statement into First-Order Logic (FOL).
You will be provided with:
1. Background premises in NL and their FOL translations (serving as your lexicon).
2. A Target Statement in NL to translate.
RULES: Use EXACT abbreviated predicate names. Output ONLY the FOL string. Use: ∀, ∃, →, ¬, ∧, ∨"""
    
    # 1. Add premise pairs
    seen_premises = set()
    for item in data:
        for nl, fol in zip(item['premises-NL'], item['premises-FOL']):
            if nl not in seen_premises:
                seen_premises.add(nl)
                sft_data.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PREMISE},
                        {"role": "user", "content": build_premise_prompt(nl)},
                        {"role": "assistant", "content": fol}
                    ]
                })
                
    # 2. Add Silver Goals
    for s in silver:
        item = data[s['item_idx']]
        
        if s['type'] == 'truth':
            q_text = s['question']
            goal_fol = s['gold_fol']
            sft_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_GOAL},
                    {"role": "user", "content": build_goal_prompt(item, q_text)},
                    {"role": "assistant", "content": goal_fol}
                ]
            })
        elif s['type'] == 'mcq':
            # For MCQ, add all 4 options as independent translation tasks
            mcq_texts = _parse_mcq_options(s['question'])
            for letter, fol in s['options_fol'].items():
                if letter in mcq_texts:
                    opt_text = mcq_texts[letter]
                    sft_data.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_GOAL},
                            {"role": "user", "content": build_goal_prompt(item, opt_text)},
                            {"role": "assistant", "content": fol}
                        ]
                    })
                    
    # 3. Add Augmented Goals
    with open('data/silver_goals_all_meta.json') as f:
        aug_goals = json.load(f)
        
    for ag in aug_goals:
        mock_item = {
            'premises-NL': ag['premises_nl'],
            'premises-FOL': ag['premises_fol']
        }
        q_text = ag['question']
        goal_fol = ag['gold_fol']
        
        sft_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_GOAL},
                {"role": "user", "content": build_goal_prompt(mock_item, q_text)},
                {"role": "assistant", "content": goal_fol}
            ]
        })
                    
    print(f"Generated {len(sft_data)} SFT samples.")
    out_path = "data/qwen_logic_sft.jsonl"
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in sft_data:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
