# Uso del archivo food_detector.py: python food_detector.py --image images/foto2.jpg 
# Permite obtener una descripción de los productos y de un peso estimado a partir de una imagen.
# El peso es puramente orientativo y se usa únicamente si no se especifica por el usuario.

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# El prompt en inglés es bastante mejor para Qwen2
# Indicarle que sea un experto nutricionista permite ser más visto identificando comidas
PROMPT = """
You are an expert nutritionist.

Look carefully at the image and identify all foods present.

Also estimate the approximate quantity of each food in grams.


Return ONLY a object with this format:

{
 "foods": [
   {
     "name": "food_name",
     "estimated_amount_g": number
   }
 ]
}

Rules:
- Only include actual foods.
- Do not include plates, tables, or utensils.
- Use simple food names.

If uncertain, give your best approximation.
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
            temperature=0.2 # Pequeña para evitar mucha variabilidad 
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

