import torch
import json
import gc
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig

# --- CONFIGURACIÓN ---
TEXT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
VISION_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FoodNutritionSystem:
    def __init__(self, unload_text_model=True):
        self.unload_text_model = unload_text_model

        # Prompts base (fijos)
        self._init_prompts()

        # Inicializar componentes de Visión (los más pesados, los mantenemos)
        print(f"Loading Vision Model: {VISION_MODEL_ID} on {DEVICE}...")
        self.v_processor = AutoProcessor.from_pretrained(
            VISION_MODEL_ID,min_pixels=256 * 28 * 28,
            max_pixels=512 * 512,   # limitar tokens visuales, crítico con 8GB
        )

        if DEVICE == "cuda":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            self.v_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VISION_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto"
                )
        else:
            self.v_model = Qwen2VLForConditionalGeneration.from_pretrained(
                VISION_MODEL_ID,
                torch_dtype=torch.float32,
                device_map="auto"
                )

        # Componentes de texto se cargan bajo demanda para ahorrar VRAM
        self.t_tokenizer = None
        self.t_model = None
        print("System initialized.")

    def _init_prompts(self):
        self.vision_expert_prompt_base = """
        You are an expert chef AND nutritionist.

        TASK:
        Analyze the image and reconstruct the dish as it would be prepared in real life.
        Then estimate realistic ingredient quantities.

        IMPORTANT:
        You MUST first understand the dish context before estimating quantities.

        --------------------------------
        USER ADDITIONAL CONTEXT (SECONDARY SOURCE):
        {user_context_instruction}
        --------------------------------

        CONTEXT USAGE RULES (VERY IMPORTANT):

        - The image is ALWAYS the primary source of truth.
        - The user context is a secondary hint and MUST NOT override the image.

        - If user context contradicts the image:
        → IGNORE the user input.

        - If user context adds information NOT visible:
        → You may ONLY use it if it refers to:
            - hidden ingredients (e.g. "there is sauce underneath")
            - cooking method (e.g. "fried", "grilled")
            - portion size

        - NEVER use user context to:
        - add new visible ingredients
        - override visual evidence
        - force a recipe

        --------------------------------

        OUTPUT FORMAT (STRICT JSON ONLY):

        {
        "dish_name": "name of the dish",
        "dish_type": "e.g. stew, soup, pasta",
        "ingredients": [
            {
            "name": "ingredient",
            "grams": number,
            "source": "visible | inferred"
            }
        ],
        "confidence": "low | medium | high"
        }

        --------------------------------

        CRITICAL RULES:

        1. VISUAL PRIORITY (MOST IMPORTANT):
        - Only mark an ingredient as "visible" if it is clearly seen in the image.
        - If an ingredient is not clearly visible, it MUST NOT be labeled as visible.

        2. SOLID INGREDIENT RESTRICTION:
        - DO NOT infer solid ingredients that are not visible.
        - Example: if potatoes are not visible, DO NOT include them.

        3. INFERRED INGREDIENTS (LIMITED):
        - Only infer unavoidable elements:
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
        - Total dish weight: 250g-700g.


        7. BEHAVIOR:
        - First think like a chef (context).
        - Then think like a nutritionist (quantities).
        - Prioritize visual evidence over prior knowledge.
        - Be conservative: when in doubt, omit the ingredient.
        - Do NOT complete recipes.

        --------------------------------

        STRICT OUTPUT CONTROL:

        - Output ONLY one JSON object.
        - Do NOT add explanations.
        - Do NOT repeat the JSON.
        - Stop immediately after the JSON.
        """
        # Prompt para el LLM de texto: traduce y optimiza
        self.text_refiner_prompt = """
        You are a translation and prompt engineering expert for AI vision models.
        TASK:
        1. Translate the user's input text (which is about food/dish context) into English.
        2. Rewrite the translated text to be highly concise, clear, and optimized for an AI food detection model.
        3. Focus only on factual information about ingredients, quantities, hidden elements, or dish type.
        4. DO NOT add any explanations or introductory text. Just the optimized English instruction.

        Examples:
        Input: "Hay patatas debajo de la salsa" -> Output: "Hidden potatoes are present beneath the sauce."
        Input: "Es una ración pequeña de unos 200g" -> Output: "The total dish weight is approximately 200g (small portion)."
        Input: "Creo que lleva pollo pero no se ve" -> Output: "Inferred ingredient: Chicken (not clearly visible)."
        """

    def _get_text_model(self):
        """Carga el modelo de texto solo si es necesario."""
        if self.t_model is None:
            print(f"Loading Text Model: {TEXT_MODEL_ID} on {DEVICE}...")
            self.t_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
            self.t_model = AutoModelForCausalLM.from_pretrained(
                TEXT_MODEL_ID,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto"
            )
        return self.t_model, self.t_tokenizer

    def _unload_text_model(self):
        """Descarga el modelo de texto para liberar VRAM."""
        if self.t_model is not None:
            print("Unloading Text Model from VRAM...")
            del self.t_model
            del self.t_tokenizer
            self.t_model = None
            self.t_tokenizer = None
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    def refine_user_input(self, user_text):
        """Traduce y optimiza el texto del usuario usando el LLM."""
        if not user_text or user_text.strip() == "":
            return "No additional context provided."

        model, tokenizer = self._get_text_model()
        
        messages = [
            {"role": "system", "content": self.text_refiner_prompt},
            {"role": "user", "content": f"Input: \"{user_text}\""}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        
        print(f"Refining user input: '{user_text}'...")
        with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs, 
                        max_new_tokens=100, # Suficiente para un prompt optimizado
                        temperature=0.1,
                        do_sample=False,    # Greedy search es más rápido y estable para traducciones
                        pad_token_id=tokenizer.eos_token_id
                    )
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        refined_text = response.split("Output:")[-1].strip() # Extraer solo la salida
        print(f"Refined context: '{refined_text}'")
        
        if self.unload_text_model:
            self._unload_text_model()
            
        return refined_text

    def analyze_food(self, image_path, user_context_text=None, temperature=0.1):
        """Flujo completo: refina texto (si hay), analiza imagen y texto, devuelve JSON."""
        
        # 1. Procesar el texto del usuario (si se proporciona)
        processed_context = "No additional context provided."
        if user_context_text:
            processed_context = self.refine_user_input(user_context_text)
            
        # 2. Preparar el prompt final de visión
        final_vision_prompt = self.vision_expert_prompt_base.replace(
            "{user_context_instruction}",
            processed_context
        )
        
        # 3. Procesar la imagen con Qwen2-VL
        print("Analyzing image with Qwen2-VL...")
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": final_vision_prompt},
                ],
            }
        ]

        text = self.v_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.v_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = self.v_model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=temperature,
                do_sample=True if temperature > 0 else False
            )

        generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        response = self.v_processor.decode(generated_tokens, skip_special_tokens=True)
        
        # 4. Parsear el JSON (con limpieza por si acaso)
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response.replace("```json", "", 1).replace("```", "", 1).strip()
        elif clean_response.startswith("```"):
            clean_response = clean_response.replace("```", "", 2).strip()

        try:
            return json.loads(clean_response)
        except Exception as e:
            print(f"Error parseando JSON. Respuesta original: {response}")
            return {"error": "Invalid JSON format", "raw": response, "context_used": processed_context}