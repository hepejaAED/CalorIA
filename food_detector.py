# # Uso del archivo food_detector.py: python food_detector.py --image images/foto2.jpg 
# # Permite obtener una descripción de los productos y de un peso estimado a partir de una imagen.
# # El peso es puramente orientativo y se usa únicamente si no se especifica por el usuario.

# import torch
# from PIL import Image
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct" # Qwen/Qwen2-VL-2B-Instruct no es tan bueno estimando cantidades
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # El prompt en inglés es bastante mejor para Qwen2
# # Indicarle que sea un experto nutricionista permite ser más visto identificando comidas y un chef experto permite hacer mejor el contexto del plato.
# PROMPT = """
# You are an expert chef AND nutritionist.

# TASK:
# Analyze the image and reconstruct the dish as it would be prepared in real life.
# Then estimate realistic ingredient quantities.

# IMPORTANT:
# You MUST first understand the dish context before estimating quantities.

# OUTPUT FORMAT (STRICT JSON ONLY):

# {
#   "dish_name": "name of the dish",
#   "dish_type": "e.g. stew, soup, pasta",
#   "ingredients": [
#     {
#       "name": "ingredient",
#       "grams": number,
#       "source": "visible | inferred",
#       "importance": "high | medium | low"
#     }
#   ],
#   "confidence": "low | medium | high"
# }

# CRITICAL RULES:

# 1. DISH UNDERSTANDING (VERY IMPORTANT):
# - Identify the dish as a whole BEFORE listing ingredients.
# - If the dish is a known recipe, use that knowledge.
# - Example: a stew or "cocido" MUST include broth, not water.

# 2. INGREDIENT NAMING:
# - Use culinary terms, not generic ones.
# - DO NOT say "water" if it is part of a cooked dish.
# - Instead use: "broth", "meat broth", "chicken broth", "sauce", etc.

# 3. INGREDIENTS:
# - Include both visible and inferred ingredients.
# - Inferred ingredients must reflect real recipes.

# 4. GRAMS:
# - Estimate realistic quantities.
# - Total dish weight: 250g–700g.

# 5. IMPORTANCE:
# HIGH:
# - Oil, meat, rice, pasta, bread, cheese

# MEDIUM:
# - Vegetables, legumes, sauces

# LOW:
# - Spices, herbs

# 6. BEHAVIOR:
# - First think like a chef (context).
# - Then think like a nutritionist (quantities).
# - Prefer realistic recipes over literal visual interpretation.

# OUTPUT RULES:
# - JSON only.
# - No explanations.
# """

# def load_model():

#     print("Loading model...")

#     processor = AutoProcessor.from_pretrained(MODEL_ID)

#     model = Qwen2VLForConditionalGeneration.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
#         device_map="auto"
#     )

#     print("Model loaded\n")

#     return model, processor

# def image_to_foods(model, processor, image_path):

#     image = Image.open(image_path).convert("RGB")
#     image = image.resize((512, 512))
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": PROMPT},
#             ],
#         }
#     ]

#     text = processor.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )

#     inputs = processor(
#         text=[text],
#         images=[image],
#         return_tensors="pt"
#     ).to(DEVICE)

#     with torch.no_grad():

#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=500,
#             temperature=0.4 # Pequeña para evitar mucha variabilidad, con 0.4 ayuda a detectar posibles ingredientes ocultos
#         )

#     generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]

#     response = processor.decode(
#         generated_tokens,
#         skip_special_tokens=True
#     )

#     return response

# if __name__ == "__main__":

#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image", required=True)

#     args = parser.parse_args()

#     model, processor = load_model()

#     result = image_to_foods(model, processor, args.image)

#     print("\nModel output:\n")
#     print(result)

import torch
import json
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class FoodDetector:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.prompt = """
        You are an expert chef AND nutritionist.

        TASK:
        Analyze the image and reconstruct the dish as it would be prepared in real life.
        Then estimate realistic ingredient quantities.

        IMPORTANT:
        You MUST first understand the dish context before estimating quantities.

        OUTPUT FORMAT (STRICT JSON ONLY):

        {
        "dish_name": "name of the dish",
        "dish_type": "e.g. stew, soup, pasta",
        "ingredients": [
            {
            "name": "ingredient",
            "grams": number,
            "source": "visible | inferred",
            "importance": "high | medium | low"
            }
        ],
        "confidence": "low | medium | high"
        }

        CRITICAL RULES:

        1. VISUAL PRIORITY (MOST IMPORTANT):
        - Only mark an ingredient as "visible" if it is clearly seen in the image.
        - If an ingredient is not clearly visible, it MUST NOT be labeled as visible.

        2. SOLID INGREDIENT RESTRICTION:
        - DO NOT infer solid ingredients that are not visible.
        - Example: if potatoes are not visible, DO NOT include them at all.

        3. INFERRED INGREDIENTS (LIMITED):
        - Only infer elements that are unavoidable in the dish:
        - liquids (broth, sauce)
        - cooking fats (oil, butter)
        - Do NOT infer additional solid ingredients.

        4. DISH UNDERSTANDING:
        - Identify the dish, but do NOT force a full recipe.
        - The image is the source of truth.

        5. INGREDIENT NAMING:
        - Use precise culinary terms.
        - Use "broth" instead of "water" when appropriate.

        6. GRAMS:
        - Estimate realistic quantities.
        - Total dish weight: 250g–700g.

        7. IMPORTANCE:
        HIGH:
        - Meat, fish, oil, rice, pasta, bread

        MEDIUM:
        - Vegetables, legumes, sauces

        LOW:
        - Spices, herbs

        8. BEHAVIOR:
        - First think like a chef (context).
        - Then think like a nutritionist (quantities).
        - Prioritize visual evidence over prior knowledge.
        - Be conservative: when in doubt, omit the ingredient.
        - Do NOT complete recipes.

        OUTPUT RULES:
        - JSON only.
        - No explanations.
        """
        
        print(f"Loading model {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        print("Model loaded successfully.")

    def analyze_image(self, image_path, temperature=0.1):
        # la temperatura pequeña para evitar mucha variabilidad, con 0.2 ayuda a detectar posibles ingredientes ocultos
        # Cargar y preparar imagen
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )

        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        response = self.processor.decode(generated_tokens, skip_special_tokens=True)
        
        # Intentar parsear a JSON automáticamente para que sea útil en el notebook
        try:
            return json.loads(response)
        except:
            return response

# --- INSTRUCCIONES DE USO EN EL NOTEBOOK ---
# 1. Instanciar (Solo una vez)
# detector = FoodDetector()

# 2. Llamar (Tantas veces como quieras)
# resultado = detector.analyze_image("mi_comida.jpg")
# print(resultado["dish_name"])