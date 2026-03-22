
# Permite obtener una descripción de los productos y de un peso estimado a partir de una imagen.
# El peso es puramente orientativo y se usa únicamente si no se especifica por el usuario.

import torch
import json
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class FoodDetector:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"): # Qwen/Qwen2-VL-2B-Instruct no es tan bueno estimando cantidades
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
        # la temperatura pequeña para evitar mucha variabilidad, con 0.1 ayuda a detectar posibles ingredientes ocultos
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