import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm.auto import tqdm
import sys
import os

# Add src to path to import check_entailment
sys.path.append(os.getcwd())
from src.fol.reasoner import check_entailment

# Load Model and Tokenizer
base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
lora_path = "kotorii1/qwen_logic_v2_lora" # or the path on your server

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

# Load BTC Test Data
with open('btc_queries.json') as f:
    btc_data = json.load(f)

type1_data = [q for q in btc_data if q['type'] == 'type1']
print(f"Loaded {len(type1_data)} Type 1 BTC queries.")

TRANSLATE_SYS = "You are an expert in formal logic. Translate the natural language statement into First-Order Logic (FOL)."

def translate_to_fol(nl_text):
    messages = [
        {"role": "system", "content": TRANSLATE_SYS},
        {"role": "user", "content": nl_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

predictions = []

for case in tqdm(type1_data):
    print(f"\\nProcessing {case['query_id']}...")
    
    # 1. Translate all premises
    premises_fol = []
    for p in case['premises']:
        premises_fol.append(translate_to_fol(p))
        
    final_answer = "ERROR"
    final_premises_used = []
    
    # Check if MCQ or Truth
    options_letters = case.get('options', [])
    is_mcq = len(options_letters) > 0 and options_letters[0] in ['A', 'B', 'C', 'D']
    
    if is_mcq:
        # Extract option text from query
        # Example query: "...which conclusion is correct?\\nA. text\\nB. text..."
        # We findall letters A-D followed by a dot
        opt_texts = {}
        for opt in options_letters:
            pattern = rf'{opt}\.\s*(.*?)(?=\n[A-D]\.|$)'
            match = re.search(pattern, case['query'], flags=re.DOTALL)
            if match:
                opt_texts[opt] = match.group(1).strip()
            else:
                opt_texts[opt] = "" # Fallback
                
        # Evaluate each option using Z3
        found = False
        for opt, text in opt_texts.items():
            if not text: continue
            opt_fol = translate_to_fol(text)
            res = check_entailment(premises_fol, opt_fol)
            if res.status == 'yes':
                final_answer = opt
                final_premises_used = res.premise_ids
                found = True
                break
        
        # If none works, we fallback or leave as ERROR
        if not found:
            final_answer = "Uncertain (No options proved)"
            
    else:
        # Truth question (Yes/No/Uncertain)
        # Query: "Does the Amber Amulet qualify for public display, according to the premises?"
        goal_nl = case['query']
        goal_fol = translate_to_fol(goal_nl)
        res = check_entailment(premises_fol, goal_fol)
        
        if res.status == 'yes':
            final_answer = "Yes"
            final_premises_used = res.premise_ids
        elif res.status == 'no':
            final_answer = "No"
            final_premises_used = res.premise_ids
        else:
            final_answer = "Uncertain"
            final_premises_used = [] # No proof core for uncertain usually
            
    predictions.append({
        "query_id": case["query_id"],
        "answer": final_answer,
        "unit": "",
        "explanation": f"Neuro-Symbolic resolution. Translated and proved via Z3 Reasoner.",
        "premises_used": final_premises_used,
        "reasoning": {
            "type": "neuro_symbolic",
            "steps": ["LLM translated premises to FOL.", "LLM translated candidate goals to FOL.", f"Z3 Reasoner proved goal entailment with status: {final_answer}."]
        }
    })

# Save results to file
out_file = "btc_neuro_symbolic_predictions.json"
with open(out_file, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\\n--- Evaluation Complete ---")
print(f"Predictions saved to {out_file}")
