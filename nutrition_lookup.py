# nutrition_lookup.py
# Dado el output de food_description.py (string JSON crudo o lista de {name, grams}),
# busca los valores nutricionales en USDA FoodData Central.
# Si USDA no encuentra bien un alimento, usa Qwen2 (ya cargado) como fallback LLM.

#
# USO CON LISTA DE ALIMENTOS (sin detección visual):
#   python nutrition_lookup.py --foods '[{"name":"chickpeas","grams":150}]'
#
# USO INTEGRADO (modelo ya cargado, desde otro script):
#   from nutrition_lookup import analyze_from_raw, analyze_meal
#   raw    = image_to_foods(model, processor, "foto.jpg")   # string de Qwen2
#   result = analyze_from_raw(raw, model=model, processor=processor)
#
# APIs necesarias:
#   - USDA FoodData Central: https://fdc.nal.usda.gov/api-guide.html (gratuita)

import os
import re
import json
import difflib
import unicodedata
import torch
import requests
from dotenv import load_dotenv
import time
load_dotenv()
 
# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
os.environ["USDA_API_KEY"] = "J49CUStV1jni8rGgxNgwPK3rfJg1Nd1jxIoffpEG"
USDA_API_KEY    = os.getenv("USDA_API_KEY", "DEMO_KEY")
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
NUTRIENT_IDS = {
    "kcal":      1008,
    "protein_g": 1003,
    "fat_g":     1004,
    "carbs_g":   1005,
    "fiber_g":   1079,
    "sugar_g":   2000,
    "sodium_mg": 1093,
}
 
USDA_MIN_SIMILARITY = 0.5
PROCESSED_WORDS     = {"dried", "powder", "dehydrated", "freeze-dried"}
 
LLM_FALLBACK_PROMPT = """You are a professional nutritionist database.
 
For the food item below, provide the average nutritional values per 100g.
Be as accurate as possible based on standard nutritional references (USDA, FAO).
 
Food: "{food_name}"
 
Return ONLY a valid JSON object, no explanation, no extra text:
 
{{
  "kcal": number,
  "protein_g": number,
  "fat_g": number,
  "carbs_g": number,
  "fiber_g": number,
  "sugar_g": number,
  "sodium_mg": number
}}
 
Rules:
- All values must be realistic numbers for this food per 100g.
- Use average values for typical preparation (cooked if applicable).
- sodium_mg in milligrams, all others in grams except kcal.
- Output JSON only.
"""
 
# ─── HELPERS ──────────────────────────────────────────────────────────────────
 
def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))
 
 
def _name_similarity(query: str, description: str) -> float:
    return difflib.SequenceMatcher(
        None,
        _normalize_text(query.lower().strip()),
        _normalize_text(description.lower().strip()),
    ).ratio()
 
 
def _has_processed_form(description: str) -> bool:
    return any(w in description.lower() for w in PROCESSED_WORDS)
 
 
def _word_containment(query: str, description: str) -> float:
    words     = re.sub(r"[^\w\s]", "", _normalize_text(query.lower())).split()
    desc_norm = re.sub(r"[^\w\s]", "", _normalize_text(description.lower()))
    if not words:
        return 0.0
    return sum(1 for w in words if w in desc_norm) / len(words)
 
 
def _simplify_query(food_name: str) -> str | None:
    PRESENTATION_WORDS = {
        "slices", "slice", "chopped", "diced", "minced", "grated",
        "cooked", "raw", "fresh", "whole", "pieces", "piece",
        "fillet", "fillets", "strips", "strip", "leaves", "leaf",
        "wedges", "wedge", "halves", "half", "chunks", "chunk",
        "grilled", "fried", "baked", "boiled", "steamed", "roasted",
    }
    words = food_name.lower().split()
    if len(words) <= 1:
        return None
    core = [w for w in words if w not in PRESENTATION_WORDS]
    if core and len(core) < len(words):
        return " ".join(core)
    return None
 
 
def _get_text_model_and_tokenizer(nutrition_system):
    """
    Obtiene model y tokenizer del sistema, cargándolos si están descargados.
    Funciona tanto si unload_text_model=True como False.
    """
    if nutrition_system is None:
        return None, None
    # _get_text_model() ya implementa la lógica lazy-load en FoodNutritionSystem
    model, tokenizer = nutrition_system._get_text_model()
    return model, tokenizer
 
# ─── USDA ─────────────────────────────────────────────────────────────────────
 
def usda_search(food_name: str, _allow_simplified: bool = True) -> dict | None:
    params = {
        "query":    food_name,
        "api_key":  USDA_API_KEY,
        "pageSize": 5,
        "dataType": "Foundation,SR Legacy",
    }
    try:
        resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [USDA] Error en búsqueda de '{food_name}': {e}")
        return None
 
    foods = data.get("foods", [])
    if not foods:
        print(f"  [USDA] Sin resultados para '{food_name}'")
        simplified = _simplify_query(food_name) if _allow_simplified else None
        if simplified:
            print(f"  [USDA] Reintentando con '{simplified}'")
            return usda_search(simplified, _allow_simplified=False)
        return None
 
    query_wants_processed = any(w in food_name.lower() for w in PROCESSED_WORDS)
    best, best_sim = None, -1.0
 
    for candidate in foods:
        desc = candidate.get("description", "")
        sim  = _name_similarity(food_name, desc)
        if not query_wants_processed and _has_processed_form(desc):
            sim *= 0.5
        if sim > best_sim:
            best_sim = sim
            best     = candidate
 
    query_words = re.sub(r"[^\w\s]", "", food_name.lower()).split()
    if len(query_words) >= 2:
        containment = _word_containment(food_name, best.get("description", ""))
        if containment < 0.7:
            simplified = _simplify_query(food_name) if _allow_simplified else None
            if simplified:
                print(f"  [USDA] Contención baja ({containment:.2f}), reintentando con '{simplified}'")
                result = usda_search(simplified, _allow_simplified=False)
                if result:
                    return result
            print(f"  [USDA] Sin match confiable para '{food_name}' → fallback LLM")
            return None
 
    if best_sim < USDA_MIN_SIMILARITY:
        print(f"  [USDA] Similitud insuficiente ({best_sim:.2f}) para '{food_name}'")
        return None
 
    description = best.get("description", food_name)
    result      = _extract_nutrients_usda(best.get("foodNutrients", []))
    result["source"]      = "USDA"
    result["description"] = description
    result["fdc_id"]      = best["fdcId"]
    print(f"  [USDA] '{food_name}' → '{description}' (similitud {best_sim:.2f})")
    return result
 
 
def _extract_nutrients_usda(nutrient_list: list) -> dict:
    lookup = {n["nutrientId"]: n.get("value", 0) for n in nutrient_list}
    return {key: round(lookup.get(nid, 0), 2) for key, nid in NUTRIENT_IDS.items()}
 
# ─── LLM FALLBACK (Qwen2.5-3B-Instruct) ──────────────────────────────────────
 
def llm_fallback(food_name: str, nutrition_system) -> dict | None:
    """
    Fallback con Qwen2.5-3B-Instruct.
    Recarga el modelo de texto si fue descargado (unload_text_model=True).
    Lo descarga de nuevo al terminar si así estaba configurado.
    """
    model, tokenizer = _get_text_model_and_tokenizer(nutrition_system)
 
    if model is None or tokenizer is None:
        print(f"  [LLM] Modelo de texto no disponible para '{food_name}'")
        return None
 
    print(f"  [LLM] Consultando Qwen2.5-3B para '{food_name}'...")
 
    messages = [
        {"role": "system", "content": "You are a professional nutritionist database. Output JSON only."},
        {"role": "user",   "content": LLM_FALLBACK_PROMPT.format(food_name=food_name)},
    ]
 
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    inputs = tokenizer([text], return_tensors="pt").to(device)
 
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
 
    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
 
    # # Si el sistema estaba configurado para descargar, volvemos a descargar
    # if nutrition_system is not None and nutrition_system.unload_text_model:
    #     nutrition_system._unload_text_model()
 
    return _parse_llm_response(response, food_name)
 
 
def _parse_llm_response(response: str, food_name: str) -> dict | None:
    clean = re.sub(r"```(?:json)?", "", response).strip().removesuffix("```")
    try:
        return _validate_llm_nutrients(json.loads(clean), food_name)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]+\}', clean, re.DOTALL)
    if match:
        try:
            return _validate_llm_nutrients(json.loads(match.group()), food_name)
        except json.JSONDecodeError:
            pass
    print(f"  [LLM] No se pudo parsear la respuesta para '{food_name}':\n    {response[:200]}")
    return None
 
 
def _validate_llm_nutrients(data: dict, food_name: str) -> dict | None:
    limits = {"kcal": 900, "protein_g": 100, "fat_g": 100, "carbs_g": 100,
              "fiber_g": 50, "sugar_g": 100, "sodium_mg": 5000}
    result = {}
    for key in NUTRIENT_IDS:
        val = data.get(key, 0)
        if not isinstance(val, (int, float)) or val < 0 or val > limits[key]:
            print(f"  [LLM] Valor sospechoso para '{key}': {val} — usando 0")
            val = 0
        result[key] = round(float(val), 2)
    result["source"]      = "LLM (estimado)"
    result["description"] = food_name
    print(f"  [LLM] '{food_name}': {result['kcal']} kcal/100g")
    return result
 
# ─── LOOKUP PRINCIPAL ─────────────────────────────────────────────────────────
 
def get_nutrients_per_100g(food_name: str, nutrition_system=None) -> dict | None:
    """
    1. USDA (con reintento simplificado automático)
    2. LLM fallback con Qwen2.5-3B (recarga el modelo si es necesario)
    3. None
    """
    result = usda_search(food_name)
    if result is None:
        result = llm_fallback(food_name, nutrition_system)

    return result
 
 
def scale_nutrients(nutrients_per_100g: dict, grams: float) -> dict:
    factor = grams / 100.0
    return {
        key: val if key in ("source", "description", "fdc_id") else round(val * factor, 1)
        for key, val in nutrients_per_100g.items()
    }
 
# ─── FUNCIONES PRINCIPALES ────────────────────────────────────────────────────
 
def lookup_dish(dish_name: str, total_grams: float, nutrition_system=None) -> dict | None:
    """Busca el plato completo en USDA. Devuelve None si no lo encuentra."""
    print(f"\n🍽️  Buscando plato completo: '{dish_name}' ({total_grams}g)")
    nutrients_100g = usda_search(dish_name)
    if nutrients_100g is None:
        print(f"  → '{dish_name}' no encontrado. Entrando en lookup por ingredientes.")
        return None
    scaled = scale_nutrients(nutrients_100g, total_grams)
    print(f"  ✅ Plato encontrado vía {nutrients_100g['source']}.")
    return {
        "mode": "dish", "source": nutrients_100g["source"],
        "name": dish_name, "grams": total_grams,
        **{k: scaled[k] for k in NUTRIENT_IDS},
    }
 
 
def analyze_meal(foods: list[dict], nutrition_system=None) -> dict:
    meal_items = []
    totals     = {key: 0.0 for key in NUTRIENT_IDS}

    # ── Cargar el modelo de texto UNA sola vez antes del bucle ───────────────
    if nutrition_system is not None:
        nutrition_system._get_text_model()
        print("  [LLM] Modelo de texto cargado para el lookup.")

    for food in foods:
        name, grams = food["name"], food["grams"]
        print(f"\nBuscando: '{name}' ({grams}g)")
        nutrients_100g = get_nutrients_per_100g(name, nutrition_system=nutrition_system)

        if nutrients_100g is None:
            print(f"  ✗ Sin datos para '{name}'. Omitido del total.")
            meal_items.append({"name": name, "grams": grams,
                                "error": "No encontrado", "source": None})
            continue

        scaled = scale_nutrients(nutrients_100g, grams)
        meal_items.append({
            "name": name, "grams": grams,
            "source": scaled.get("source"),
            "description": scaled.get("description"),
            **{k: scaled[k] for k in NUTRIENT_IDS},
        })
        for key in NUTRIENT_IDS:
            totals[key] = round(totals[key] + scaled.get(key, 0), 1)

    # ── Descargar al terminar todos los ingredientes ──────────────────────────
    if nutrition_system is not None and nutrition_system.unload_text_model:
        nutrition_system._unload_text_model()
        print("  [LLM] Modelo de texto descargado.")

    return {"items": meal_items, "totals": totals}
 
 
def analyze_from_raw(raw: str, nutrition_system=None) -> dict:
    foods = _parse_detector_output(raw)
    if not foods:
        return {"items": [], "totals": {k: 0.0 for k in NUTRIENT_IDS}}
    print(f"\n  ✅ {len(foods)} alimento(s) detectado(s):")
    for f in foods:
        print(f"     - {f['name']}: {f['grams']}g")
    return analyze_meal(foods, nutrition_system=nutrition_system)
 
 
def _parse_detector_output(raw: str) -> list[dict]:
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(clean).get("foods", [])
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()).get("foods", [])
        except json.JSONDecodeError:
            pass
    print("  ⚠ No se pudo parsear el output de food_detector.")
    return []
 
# ─── PRETTY PRINT ─────────────────────────────────────────────────────────────
 
def print_meal_summary(result: dict):
    SOURCE_ICONS = {"USDA": "✅", "LLM (estimado)": "⚠️ ", None: "❌"}
    print("\n" + "═" * 58)
    print("  RESUMEN NUTRICIONAL DE LA COMIDA")
    print("═" * 58)
    for item in result["items"]:
        if "error" in item:
            print(f"\n  ❌ {item['name']} ({item['grams']}g) — {item['error']}")
            continue
        icon = SOURCE_ICONS.get(item.get("source"), "❓")
        print(f"\n  {icon} {item['name'].capitalize()} ({item['grams']}g)  [{item['source']}]")
        print(f"     Kcal:     {item['kcal']} kcal")
        print(f"     Proteína: {item['protein_g']} g")
        print(f"     Carbos:   {item['carbs_g']} g")
        print(f"     Grasa:    {item['fat_g']} g")
        print(f"     Fibra:    {item['fiber_g']} g")
    t = result["totals"]
    print("\n" + "─" * 58)
    print("  TOTALES")
    print("─" * 58)
    print(f"  🔥 Calorías totales: {t['kcal']} kcal")
    print(f"  🥩 Proteína:         {t['protein_g']} g")
    print(f"  🍞 Carbohidratos:    {t['carbs_g']} g")
    print(f"  🫒 Grasa:            {t['fat_g']} g")
    print(f"  🌾 Fibra:            {t['fiber_g']} g")
    print(f"  🧂 Sodio:            {t['sodium_mg']} mg")
    print()
    print("  Fuentes: ✅ USDA  ⚠️  LLM (estimado)  ❌ No encontrado")
    print("═" * 58 + "\n")