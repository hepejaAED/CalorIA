# Uso del archivo food_detector.py: python food_detector.py --image images/foto2.jpg 


import torch
from PIL import Image
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# El prompt en inglés es bastante mejor para Qwen2
PROMPT = """
You are an expert nutritionist.

Look carefully at the image and identify all foods present.

Return ONLY a object with this format:

{
"foods": ["food1", "food2", "food3"]
}

Rules:
- Only include actual foods.
- Do not include plates, tables, or utensils.
- Use simple food names.
"""

def load_model():

    print("Loading model...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )

    print("Model loaded\n")

    return model, processor

def image_to_foods(model, processor, image_path):

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.2
        )

    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]

    response = processor.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)

    args = parser.parse_args()

    model, processor = load_model()

    result = image_to_foods(model, processor, args.image)

    print("\nModel output:\n")
    print(result)

    # Intentar parsear JSON
    try:
        foods = json.loads(result)
        print("\nParsed foods:\n")
        print(foods["foods"])
    except:
        print("\nCould not parse JSON")