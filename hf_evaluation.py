# Install dependencies if needed
# !pip install torch transformers peft z3-solver
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from eval.sample_encoded import ENCODED_CASES
from src.fol.reasoner import check_entailment

# Load Model and Tokenizer
base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
lora_path = "./models/qwen_logic_v2_lora" # or the path on your server

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()
print("Model loaded successfully!")

# Setup Evaluation
with open('data/Logic_Based_Educational_Queries.json') as f:
    data = json.load(f)

SYSTEM_PROMPT = "You are an expert in formal logic. You are given a set of premises with their First-Order Logic (FOL). Translate the QUESTION/GOAL into a single FOL formula. Reuse ONLY predicate names that appear in the premises. Use the symbols ∀, ∃, →, ¬, ∧, ∨. Output ONLY the FOL string."

total = 0
correct_string = 0
correct_logic = 0
results = []

print(f"Testing Model on {len(ENCODED_CASES)} cases...")

for case in ENCODED_CASES:
    item = data[case.item_idx]
    q_text = item['questions'][case.q_idx]
    prems_fol = item['premises-FOL']
    
    # Build prompt
    prompt = "Premises (FOL):\n"
    for i, fol in enumerate(prems_fol, 1):
        prompt += f"{i}. {fol}\n"
    prompt += f"\nQuestion: {q_text}\n\nTranslate the goal of this question into one FOL formula."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    # Format and tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    gen_fol = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Check String Match
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
        # Check Logic Equivalence using Z3
        res1 = check_entailment([case.goal_fol], gen_fol)
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

print(f"\n\nResults:")
print(f"Strict String Accuracy: {correct_string}/{total} ({correct_string/total*100:.1f}%)")
print(f"Logical Equivalence Accuracy: {correct_logic}/{total} ({correct_logic/total*100:.1f}%)")

with open('eval/hf_eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
