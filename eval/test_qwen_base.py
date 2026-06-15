import json
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Test the base model
def test_model(model_name, prompt):
    print(f"Testing {model_name}...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        print(f"Response:\n{response.choices[0].message.content}\n")
    except Exception as e:
        print(f"Error: {e}\n")

if __name__ == "__main__":
    prompt = "Translate this premise to First Order Logic (FOL) using standard notation (∀, ∃, →, ¬, ∧, ∨). Use logical variables (x, y, z) and abbreviated predicate names.\n\nPremise: If a Python code is well-tested, then the project is optimized.\n\nOutput only the FOL string."
    test_model("qwen2.5-coder:7b", prompt)

