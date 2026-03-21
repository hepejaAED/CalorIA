# Este código es para que si el usuario introduce un texto se procese de la mejor forma para el prompt global
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PROMPT = """
You are a strict information extractor.

TASK:
Rewrite and structure the user's food description into a clean JSON format.

CRITICAL CONSTRAINT:
- DO NOT add any new information.
- DO NOT infer missing ingredients.
- DO NOT assume anything that is not explicitly written.
- Only reorganize and clarify what the user said.

OUTPUT FORMAT (STRICT JSON ONLY):

{
  "ingredients": [
    {
      "name": "ingredient name",
      "details": "extra info from user (if any)",
      "uncertainty": "low | medium | high"
    }
  ],
  "notes": "any additional comments from the user rewritten clearly"
}

RULES:

1. INGREDIENTS:
- Extract only ingredients explicitly mentioned.
- If the user is unsure (e.g. "maybe", "I think"), mark uncertainty as "high".
- If clearly stated → "low".
- If somewhat unclear → "medium".

2. DETAILS:
- Preserve descriptors like:
  - "a lot of oil"
  - "a bit spicy"
  - "homemade"
- Do NOT reinterpret them.

3. NORMALIZATION:
- Convert to simple English ingredient names when possible.
- Keep original meaning exactly.

4. NOTES:
- Include any relevant extra info that is not an ingredient.

5. FORBIDDEN:
- No guessing
- No completing recipes
- No adding typical ingredients
- No nutritional estimation

BEHAVIOR:
- Be literal and conservative.
- If something is unclear, reflect uncertainty instead of fixing it.

OUTPUT RULES:
- Return ONLY JSON.
- No explanations.
"""

PROMPT_TEMPLATE = """
User input:
"{user_text}"

{instruction}
"""

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )

    return model, tokenizer


def process_text(model, tokenizer, user_text, instruction):

    prompt = PROMPT_TEMPLATE.format(
        user_text=user_text,
        instruction=instruction
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1  #  muy bajo para evita invención
        )

    response = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return response


if __name__ == "__main__":
    model, tokenizer = load_model()

    user_text = input("Describe your meal: ")

    result = process_text(model, tokenizer, user_text, PROMPT)

    print("\nProcessed output:\n")
    print(result)