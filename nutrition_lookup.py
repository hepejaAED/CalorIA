# nutrition_lookup.py
# Dado el output de food_detector.py (string JSON crudo o lista de {name, grams}),
# busca los valores nutricionales en USDA FoodData Central.
# Si USDA no encuentra bien un alimento, usa Qwen2-VL (ya cargado) como fallback LLM.
#
# USO DIRECTO CON IMAGEN (pipeline completo en un solo comando):
#   python nutrition_lookup.py --image images/foto.jpg
#   python nutrition_lookup.py --image images/foto.jpg --output results/comida.json
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

load_dotenv() #cargar el entorno con la API_KEY

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

USDA_API_KEY    = os.getenv("USDA_API_KEY", "DEMO_KEY") 
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Nutrientes que nos interesan y sus IDs en USDA
NUTRIENT_IDS = {
    "kcal":      1008,   # Energy (kcal)
    "protein_g": 1003,   # Protein
    "fat_g":     1004,   # Total lipid (fat)
    "carbs_g":   1005,   # Carbohydrate, by difference
    "fiber_g":   1079,   # Fiber, total dietary
    "sugar_g":   2000,   # Sugars, total
    "sodium_mg": 1093,   # Sodium
}

# Similitud mínima entre nombre buscado y descripción USDA (difflib.SequenceMatcher)
USDA_MIN_SIMILARITY = 0.3
# Palabras que indican formas procesadas no deseadas salvo que el query las incluya
PROCESSED_WORDS = {"dried", "powder", "dehydrated", "freeze-dried"}

# OpenFoodFacts — segunda fuente, gratuita, sin API key, productos europeos/españoles
OFF_SEARCH_URL      = "https://world.openfoodfacts.org/cgi/search.pl"
OFF_USER_AGENT      = "CalorIA/1.0 (miniproject)"
OFF_MIN_CONTAINMENT = 0.5   # más permisivo que USDA (0.6) — nombres pueden variar de idioma

# Prompt para el fallback LLM — texto puro, sin imagen
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

# ─── USDA ─────────────────────────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    """Elimina acentos y diacríticos para comparaciones robustas (ñ→n, é→e, etc.)."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _name_similarity(query: str, description: str) -> float:
    """Similitud de secuencia entre nombre buscado y descripción (case-insensitive, sin acentos)."""
    return difflib.SequenceMatcher(
        None,
        _normalize_text(query.lower().strip()),
        _normalize_text(description.lower().strip()),
    ).ratio()


def _has_processed_form(description: str) -> bool:
    """True si la descripción contiene formas procesadas no deseadas."""
    desc_lower = description.lower()
    return any(w in desc_lower for w in PROCESSED_WORDS)


def _word_containment(query: str, description: str) -> float:
    """
    Fracción de palabras del query que aparecen como substrings en la descripción.
    Normaliza acentos: "espanola" encaja con "española", "n" encaja con "ñ".
    Captura singulares/plurales sin stemming ("banana" ⊆ "bananas").
    Solo se aplica a queries con 2 o más palabras.
    """
    words     = re.sub(r"[^\w\s]", "", _normalize_text(query.lower())).split()
    desc_norm = re.sub(r"[^\w\s]", "", _normalize_text(description.lower()))
    if not words:
        return 0.0
    return sum(1 for w in words if w in desc_norm) / len(words)


def usda_search(food_name: str) -> dict | None:
    """
    Busca un alimento en USDA y devuelve sus nutrientes por 100g.
    Selecciona el candidato con mayor similitud de nombre, penalizando formas
    procesadas (dried, powder, dehydrated…) salvo que el query las incluya.
    Devuelve None si la similitud del mejor candidato es < USDA_MIN_SIMILARITY.
    """
    params = {
        "query":    food_name,
        "api_key":  USDA_API_KEY,
        "pageSize": 5,
        "dataType": "Foundation,SR Legacy",  # alimentos genéricos, más fiables
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
        return None

    # ── Problema 4: preferir formas no procesadas salvo que el query las pida ──
    query_wants_processed = any(w in food_name.lower() for w in PROCESSED_WORDS)

    # ── Problemas 1+2: seleccionar el candidato con mejor similitud de nombre ──
    best      = None
    best_sim  = -1.0

    for candidate in foods:
        desc = candidate.get("description", "")
        sim  = _name_similarity(food_name, desc)
        if not query_wants_processed and _has_processed_form(desc):
            sim *= 0.5   # penalizar formas procesadas no pedidas
        if sim > best_sim:
            best_sim  = sim
            best      = candidate

    # Para queries multi-palabra: exigir además contención léxica ≥ 0.6
    query_words = re.sub(r"[^\w\s]", "", food_name.lower()).split()
    if len(query_words) >= 2:
        containment = _word_containment(food_name, best.get("description", ""))
        if containment < 0.6:
            print(f"  [USDA] Contención insuficiente ({containment:.2f}) para '{food_name}' → fallback LLM")
            return None

    if best_sim < USDA_MIN_SIMILARITY:
        print(f"  [USDA] Similitud insuficiente ({best_sim:.2f}) para '{food_name}' → fallback LLM")
        return None

    description = best.get("description", food_name)
    nutrients   = best.get("foodNutrients", [])

    result = _extract_nutrients_usda(nutrients)
    result["source"]      = "USDA"
    result["description"] = description
    result["fdc_id"]      = best["fdcId"]

    print(f"  [USDA] '{food_name}' → '{description}' (similitud {best_sim:.2f})")
    return result


def _extract_nutrients_usda(nutrient_list: list) -> dict:
    """Extrae nutrientes de interés de la respuesta cruda de USDA (valores por 100g)."""
    lookup = {n["nutrientId"]: n.get("value", 0) for n in nutrient_list}
    return {
        key: round(lookup.get(nid, 0), 2)
        for key, nid in NUTRIENT_IDS.items()
    }

# ─── OPENFOODFACTS ────────────────────────────────────────────────────────────

def _extract_nutrients_off(nutriments: dict, product_name: str) -> dict | None:
    """Extrae nutrientes de un producto OpenFoodFacts (valores por 100g).
    Devuelve None si faltan más de 3 de los 7 campos clave."""
    off_map = {
        "kcal":      "energy-kcal_100g",
        "protein_g": "proteins_100g",
        "fat_g":     "fat_100g",
        "carbs_g":   "carbohydrates_100g",
        "fiber_g":   "fiber_100g",
        "sugar_g":   "sugars_100g",
        "sodium_mg": "sodium_100g",  # OFF almacena sodio en gramos → convertir a mg
    }
    result  = {}
    missing = 0
    for key, off_key in off_map.items():
        val = nutriments.get(off_key)
        if val is None:
            missing += 1
            result[key] = 0.0
        else:
            result[key] = round(float(val) * 1000, 2) if key == "sodium_mg" else round(float(val), 2)
    if missing > 3:
        return None
    result["source"]      = "OpenFoodFacts"
    result["description"] = product_name
    return result


def off_search(food_name: str) -> dict | None:
    """
    Busca en OpenFoodFacts como segunda fuente tras USDA.
    Selecciona el producto con mejor contención de palabras del query.
    Devuelve nutrientes por 100g o None si no hay match confiable.
    """
    params = {
        "search_terms": food_name,
        "json":         1,
        "action":       "process",
        "page_size":    5,
        "fields":       "product_name,nutriments",
        "sort_by":      "unique_scans_n",   # más escaneados = datos más fiables
    }
    headers = {"User-Agent": OFF_USER_AGENT}

    try:
        resp = requests.get(OFF_SEARCH_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [OFF] Error para '{food_name}': {e}")
        return None

    products = [p for p in data.get("products", []) if p.get("product_name")]
    if not products:
        print(f"  [OFF] Sin resultados para '{food_name}'")
        return None

    # Seleccionar el producto con mejor contención de palabras del query
    best       = None
    best_score = -1.0
    for product in products:
        score = _word_containment(food_name, product.get("product_name", ""))
        if score > best_score:
            best_score = score
            best       = product

    query_words = re.sub(r"[^\w\s]", "", _normalize_text(food_name).lower()).split()
    threshold   = OFF_MIN_CONTAINMENT if len(query_words) >= 2 else 0.3

    if best_score < threshold:
        print(f"  [OFF] Sin match confiable ({best_score:.2f}) para '{food_name}'")
        return None

    product_name = best.get("product_name", food_name)
    result       = _extract_nutrients_off(best.get("nutriments", {}), product_name)

    if result is None:
        print(f"  [OFF] Datos nutricionales incompletos para '{food_name}'")
        return None

    print(f"  [OFF] '{food_name}' → '{product_name}' (contención {best_score:.2f})")
    return result

# ─── LLM FALLBACK (Qwen2-VL en modo texto puro) ───────────────────────────────

def llm_fallback(food_name: str, model, processor) -> dict | None:
    """
    Fallback: pregunta a Qwen2-VL (sin imagen, solo texto) los nutrientes por 100g.
    El modelo y processor deben estar ya cargados — no se recargan aquí.
    Devuelve el dict de nutrientes o None si el parsing falla.
    """
    if model is None or processor is None:
        print(f"  [LLM] Modelo no disponible para fallback de '{food_name}'")
        return None

    print(f"  [LLM] Consultando Qwen2-VL para '{food_name}'...")

    prompt = LLM_FALLBACK_PROMPT.format(food_name=food_name)

    # Qwen2-VL acepta mensajes sin imagen — modo texto puro
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    device = next(model.parameters()).device

    inputs = processor(
        text=[text_input],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,    # greedy — más determinista para datos estructurados
        )

    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_tokens, skip_special_tokens=True).strip()

    return _parse_llm_response(response, food_name)


def _parse_llm_response(response: str, food_name: str) -> dict | None:
    """
    Parsea la respuesta JSON del LLM.
    Intenta extraer el JSON aunque venga con texto alrededor o en bloque ```json.
    """
    # Limpiar posibles bloques markdown
    clean = re.sub(r"```(?:json)?", "", response).strip()

    # Intentar parsear directamente
    try:
        data = json.loads(clean)
        return _validate_llm_nutrients(data, food_name)
    except json.JSONDecodeError:
        pass

    # Si falla, buscar el primer bloque JSON en la respuesta
    match = re.search(r'\{[^{}]+\}', clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_llm_nutrients(data, food_name)
        except json.JSONDecodeError:
            pass

    print(f"  [LLM] No se pudo parsear la respuesta para '{food_name}':\n    {response[:200]}")
    return None


def _validate_llm_nutrients(data: dict, food_name: str) -> dict | None:
    """
    Valida que el JSON del LLM tenga los campos esperados y valores razonables.
    Rellena con 0 los campos que falten.
    """
    # Límites máximos razonables por 100g para detectar alucinaciones
    limits = {
        "kcal":      900,
        "protein_g": 100,
        "fat_g":     100,
        "carbs_g":   100,
        "fiber_g":   50,
        "sugar_g":   100,
        "sodium_mg": 5000,
    }

    result = {}
    for key in NUTRIENT_IDS:
        val = data.get(key, 0)
        if not isinstance(val, (int, float)) or val < 0 or val > limits[key]:
            print(f"  [LLM] Valor sospechoso para '{key}': {val} — usando 0")
            val = 0
        result[key] = round(float(val), 2)

    result["source"]      = "LLM (estimado)"
    result["description"] = food_name

    print(f"  [LLM] Estimación obtenida para '{food_name}': {result['kcal']} kcal/100g")
    return result

# ─── LOOKUP PRINCIPAL ─────────────────────────────────────────────────────────

def get_nutrients_per_100g(food_name: str, model=None, processor=None) -> dict | None:
    """
    Estrategia de búsqueda:
      1. USDA FoodData Central → datos verificados        (fuente: "USDA")
      2. OpenFoodFacts         → datos comunitarios       (fuente: "OpenFoodFacts")
      3. Qwen2-VL texto puro   → estimación LLM           (fuente: "LLM (estimado)")
      4. None                  → alimento omitido del total
    """
    result = usda_search(food_name)
    if result is None:
        result = off_search(food_name)
    if result is None:
        result = llm_fallback(food_name, model, processor)
    return result


def scale_nutrients(nutrients_per_100g: dict, grams: float) -> dict:
    """Escala los valores nutricionales de 100g a los gramos reales del plato."""
    factor = grams / 100.0
    scaled = {}
    for key, val in nutrients_per_100g.items():
        if key in ("source", "description", "fdc_id"):
            scaled[key] = val
        else:
            scaled[key] = round(val * factor, 1)
    return scaled

# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────

def analyze_meal(foods: list[dict], model=None, processor=None) -> dict:
    """
    Recibe la lista de food_detector.py: [{"name": str, "grams": int}, ...]
    Opcionalmente recibe model y processor de Qwen2-VL ya cargados para el fallback LLM.
    Devuelve el desglose por alimento y los totales de la comida.

    Ejemplo de entrada:
        [{"name": "chickpeas", "grams": 150}, {"name": "paella", "grams": 300}]

    Fuentes posibles en el output:
        "USDA"           → datos verificados de base de datos oficial ✅
        "LLM (estimado)" → estimación de Qwen2-VL, orientativa       ⚠️
        None             → no encontrado, omitido del total           ❌
    """
    meal_items = []
    totals     = {key: 0.0 for key in NUTRIENT_IDS}

    for food in foods:
        name  = food["name"]
        grams = food["grams"]

        print(f"\nBuscando: '{name}' ({grams}g)")

        nutrients_100g = get_nutrients_per_100g(name, model=model, processor=processor)

        if nutrients_100g is None:
            if model is None:
                print(f"  ✗ '{name}' no encontrado en USDA. Fallback LLM no disponible.")
                print(f"     → Usa --use-llm para activar el fallback con Qwen2-VL.")
            else:
                print(f"  ✗ Sin datos para '{name}'. Omitido del total.")
            meal_items.append({
                "name":   name,
                "grams":  grams,
                "error":  "No encontrado en USDA ni estimable por LLM",
                "source": None,
            })
            continue

        scaled = scale_nutrients(nutrients_100g, grams)

        item_result = {
            "name":        name,
            "grams":       grams,
            "source":      scaled.get("source"),
            "description": scaled.get("description"),
            **{k: scaled[k] for k in NUTRIENT_IDS},
        }
        meal_items.append(item_result)

        for key in NUTRIENT_IDS:
            totals[key] = round(totals[key] + scaled.get(key, 0), 1)

    return {
        "items":  meal_items,
        "totals": totals,
    }

# ─── PARSEO DEL STRING RAW DE FOOD_DETECTOR ───────────────────────────────────

def parse_detector_output(raw: str) -> list[dict]:
    """
    Convierte el string JSON crudo de image_to_foods() en una lista de {name, grams}.
    Maneja el caso en que Qwen2 devuelva texto alrededor o bloques ```json.
    """
    clean = re.sub(r"```(?:json)?", "", raw).strip()

    # Intentar parsear directamente
    try:
        data = json.loads(clean)
        return data.get("foods", [])
    except json.JSONDecodeError:
        pass

    # Buscar el primer objeto JSON en la respuesta
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("foods", [])
        except json.JSONDecodeError:
            pass

    print("  ⚠ No se pudo parsear el output de food_detector.")
    print(f"  Raw: {raw[:300]}")
    return []


def analyze_from_raw(raw: str, model=None, processor=None) -> dict:
    """
    Entrada directa desde image_to_foods() — acepta el string crudo de Qwen2.
    Parsea el JSON, luego ejecuta analyze_meal() con los alimentos detectados.

    Uso típico:
        raw    = image_to_foods(model, processor, "foto.jpg")
        result = analyze_from_raw(raw, model=model, processor=processor)
    """
    foods = parse_detector_output(raw)

    if not foods:
        print("  ❌ No se detectaron alimentos en el output. Abortando.")
        return {"items": [], "totals": {k: 0.0 for k in NUTRIENT_IDS}}

    print(f"\n  ✅ {len(foods)} alimento(s) detectado(s):")
    for f in foods:
        print(f"     - {f['name']}: {f['grams']}g")

    return analyze_meal(foods, model=model, processor=processor)

# ─── PRETTY PRINT ─────────────────────────────────────────────────────────────

def print_meal_summary(result: dict):
    SOURCE_ICONS = {
        "USDA":           "✅",
        "OpenFoodFacts":  "🌍",
        "LLM (estimado)": "⚠️ ",
        None:             "❌",
    }

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
    print("  Fuentes: ✅ USDA  🌍 OpenFoodFacts  ⚠️  LLM (estimado)  ❌ No encontrado")
    print("═" * 58 + "\n")

# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="CalorIA — USDA lookup con fallback a Qwen2-VL LLM",
        epilog="""
Ejemplos:
  # Pipeline completo desde imagen (carga Qwen2-VL):
  python nutrition_lookup.py --image images/foto.jpg

  # Pipeline completo + guardar JSON:
  python nutrition_lookup.py --image images/foto.jpg --output results/comida.json

  # Solo lookup nutricional sin detección visual:
  python nutrition_lookup.py --foods '[{"name":"chickpeas","grams":150}]'
        """
    )

    # Modos de entrada — imagen O lista de alimentos (no los dos a la vez)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        help="Ruta a la imagen del plato — ejecuta el pipeline completo (food_detector + lookup)"
    )
    input_group.add_argument(
        "--foods",
        help='JSON string con lista de alimentos. Ejemplo: \'[{"name":"chickpeas","grams":150}]\''
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Ruta del JSON de salida (solo con --image). Por defecto: results/<imagen>_<timestamp>.json"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprimir resultado completo en JSON por stdout"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="(Solo con --foods) Carga Qwen2-VL para el fallback LLM cuando USDA falla"
    )
    args = parser.parse_args()

    # ── Modo imagen: pipeline completo ────────────────────────────────────────
    if args.image:
        import os
        from food_detector import image_to_foods
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        if not os.path.exists(args.image):
            print(f"❌ Imagen no encontrada: {args.image}")
            exit(1)

        MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
        DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 58)
        print("  Cargando Qwen2-VL-7B...")
        print("=" * 58)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model     = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto"
        )
        print(f"  Modelo cargado en {DEVICE.upper()}.\n")

        print(f"  📷 Analizando imagen: {args.image}\n")
        raw    = image_to_foods(model, processor, args.image)
        result = analyze_from_raw(raw, model=model, processor=processor)

        # Guardar JSON
        output_path = args.output
        if output_path is None:
            stem        = Path(args.image).stem
            timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/{stem}_{timestamp}.json"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        full_output = {
            "timestamp": datetime.now().isoformat(),
            "image":     args.image,
            "nutrition": result,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_output, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 Resultado guardado en: {output_path}")

    # ── Modo foods: solo lookup nutricional ───────────────────────────────────
    else:
        model = processor = None
        if args.use_llm:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
            DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
            print("=" * 58)
            print("  Cargando Qwen2-VL-7B para fallback LLM...")
            print("=" * 58)
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model     = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto"
            )
            print(f"  Modelo cargado en {DEVICE.upper()}.\n")
        foods_input = json.loads(args.foods)
        result      = analyze_meal(foods_input, model=model, processor=processor)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_meal_summary(result)
