import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm.auto import tqdm

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

# Load SFT Validation Data
with open('qwen_sft_full_val.jsonl') as f:
    val_data = [json.loads(line) for line in f]

print(f"Loaded {len(val_data)} validation samples.")

# We will classify tasks by their system prompt
TRANSLATE_SYS = "You are an expert in formal logic. Translate the natural language statement into First-Order Logic (FOL)."
GOAL_SYS = "You are an expert in formal logic. You are given a set of premises with their First-Order Logic (FOL). Translate the QUESTION/GOAL into a single FOL formula."
SOLVE_SYS = "You are an expert in formal logic reasoning. Given the premises and the question, decide the answer and respond with a single JSON object"

results = {"translate": {"correct": 0, "total": 0}, "goal": {"correct": 0, "total": 0}, "solve": {"correct": 0, "total": 0}}

for item in tqdm(val_data):
    sys_msg = item['messages'][0]['content']
    user_msg = item['messages'][1]['content']
    gold_answer = item['messages'][2]['content']
    
    task_type = "unknown"
    if TRANSLATE_SYS in sys_msg: task_type = "translate"
    elif GOAL_SYS in sys_msg: task_type = "goal"
    elif SOLVE_SYS in sys_msg: task_type = "solve"
    
    if task_type == "unknown": continue
    
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ]
    
    # Format and tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=300, 
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Strict matching (remove spaces for FOL formulas, keep spaces for JSON)
    if task_type in ["translate", "goal"]:
        is_correct = gen_text.replace(" ", "") == gold_answer.replace(" ", "")
    else:
        # For JSON solve task, we just check if the 'answer' key matches
        try:
            gen_json = json.loads(gen_text.replace("```json", "").replace("```", "").strip())
            gold_json = json.loads(gold_answer)
            is_correct = gen_json.get("answer", "").lower() == gold_json.get("answer", "").lower()
        except:
            is_correct = False
            
    results[task_type]["total"] += 1
    if is_correct:
        results[task_type]["correct"] += 1
        
print("\n--- Validation Set Results ---")
for task, stats in results.items():
    if stats['total'] > 0:
        acc = stats['correct'] / stats['total'] * 100
        print(f"{task.capitalize()} Task: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
