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

# Load BTC Test Data
with open('btc_queries.json') as f:
    btc_data = json.load(f)

print(f"Loaded {len(btc_data)} BTC queries.")

SOLVE_SYS = "You are an expert in formal logic reasoning. Given the premises and the question, decide the answer and respond with a single JSON object with keys: \"answer\", \"unit\", \"explanation\", \"premises_used\", \"reasoning\". \"premises_used\" is a 0-based list of the premises actually needed. \"reasoning\" has keys \"type\" and \"steps\". Output ONLY valid JSON."

predictions = []

for case in tqdm(btc_data):
    # Construct the user prompt
    user_msg = "Premises:\n"
    for i, p in enumerate(case["premises"], 1):
        user_msg += f"{i}. {p}\n"
    user_msg += f"\nQuestion: {case['query']}"
    
    messages = [
        {"role": "system", "content": SOLVE_SYS},
        {"role": "user", "content": user_msg}
    ]
    
    # Format and tokenize
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=500, 
            temperature=0.0, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Try to parse the predicted answer
    pred_ans = "ERROR"
    parsed_json = {}
    try:
        clean_json = gen_text.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(clean_json)
        pred_ans = parsed_json.get("answer", "ERROR")
    except Exception as e:
        pass
        
    predictions.append({
        "query_id": case["query_id"],
        "predicted_answer": pred_ans,
        "full_output": parsed_json if pred_ans != "ERROR" else gen_text
    })

# Save results to file
with open("btc_predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

print("\n--- BTC Evaluation Complete ---")
print(f"Predictions saved to btc_predictions.json")

# Count errors vs valid formats
errors = sum(1 for p in predictions if p["predicted_answer"] == "ERROR")
valid = len(predictions) - errors
print(f"Valid JSON outputs: {valid}/{len(predictions)}")
print(f"Failed JSON formats: {errors}/{len(predictions)}")
