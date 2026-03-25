# meal_suggester.py
#
# Sugiere un plato cuando se activa alguno de los dos triggers:
#   - Quedan ≤ 25 % de las kcal diarias.
#   - Quedan ≤ 3.5 horas para medianoche.
#
# Se apoya en los ficheros que ya genera main.ipynb:
#   - meals.json  → historial de comidas
#   - user.json   → perfil y objetivos diarios
#
# El LLM puede ser Qwen2-VL (vision) o Qwen2.5-3B (texto).
# Se detecta automáticamente — no hace falta recargar nada.
#
# USO EN EL NOTEBOOK (tras cada save_meal):
#
#   from meal_suggester import check_and_suggest
#
#   check_and_suggest(
#       model            = nutrition_system.v_model,
#       processor_or_tok = nutrition_system.v_processor,
#   )

import os
import json
import re
import torch
from datetime import datetime
from nutrition_lookup import get_nutrients_per_100g, scale_nutrients, NUTRIENT_IDS

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

KCAL_TRIGGER_PCT   = 0.25   # sugerir si quedan ≤ 25 % de kcal
TIME_TRIGGER_HOURS = 3.5    # sugerir si quedan ≤ 3.5 h para medianoche

SUGGESTER_PROMPT = """You are a professional nutritionist assistant.

A user needs to cover their remaining nutritional needs for today.

REMAINING MACROS TO COVER:
- Calories: {kcal_remaining} kcal
- Protein:  {protein_remaining} g
- Carbs:    {carbs_remaining} g
- Fat:      {fat_remaining} g

USER RESTRICTIONS: {restrictions}

MEAL TYPE RULES (follow strictly based on remaining calories):
- More than 600 kcal remaining  → suggest a FULL MEAL (main dish with sides)
- Between 250 and 600 kcal      → suggest a LIGHT MEAL (salad, wrap, bowl, soup)
- Less than 250 kcal            → suggest a SNACK or SMALL COMPLEMENT (fruit, yogurt, nuts, toast)

TASK:
Suggest ONE option that best fits the remaining macros and the meal type above.
It must be appetizing and varied — avoid generic "chicken and rice" style suggestions.

INGREDIENT NAMING RULES (critical for database lookup):
- Use simple, generic English names: "salmon" not "grilled salmon fillet"
- No preparation methods in the name: "potato" not "boiled potato"
- No compound names: "olive oil" and "tomato" separately, not "tomato with olive oil"
- Use the most common form: "chicken breast" not "boneless skinless chicken breast"

Return ONLY a valid JSON object, no explanation, no extra text:

{{
  "dish_name": "name of the overall dish or snack",
  "total_grams": number,
  "ingredients": [
    {{"name": "simple ingredient name", "grams": number}},
    ...
  ],
  "reason": "one sentence explaining why this fits the remaining macros and meal type"
}}

Rules:
- total_grams must be the sum of all ingredient grams.
- Output JSON only.
"""

# ─── PUNTO DE ENTRADA PRINCIPAL ───────────────────────────────────────────────

def check_and_suggest(
    model,
    processor_or_tok,
    meals_file: str = "meals.json",
    user_file:  str = "user.json",
) -> dict | None:
    """
    Función principal. Llámala tras cada save_meal() en el notebook.

    1. Lee el estado del día desde meals.json + user.json.
    2. Evalúa los triggers.
    3. Si alguno está activo, genera y verifica una sugerencia de plato.
    4. Imprime el resultado y lo devuelve (o None si no hay trigger activo).
    """
    # 1. Leer estado actual del día
    summary = _load_daily_summary(meals_file, user_file)
    if summary is None:
        return None

    remaining = summary["remaining"]
    targets   = summary["targets"]

    # Si ya no quedan kcal el día está completado
    if remaining.get("kcal", 0) <= 0:
        print("✅ Objetivos del día alcanzados. No se genera sugerencia.")
        return None

    # 2. Evaluar triggers
    triggered, reason = _check_triggers(remaining, targets)
    if not triggered:
        return None

    print(f"\n⚡ Trigger activo: {reason}")
    print("🍽️  Generando sugerencia de plato...")

    # 3. Leer restricciones desde user.json
    restrictions = _load_restrictions(user_file)

    # 4. LLM sugiere el plato
    llm_output = _ask_llm(remaining, restrictions, model, processor_or_tok)
    if llm_output is None:
        print("  ❌ El LLM no pudo generar una sugerencia válida.")
        return None

    # 5. Verificar macros ingrediente a ingrediente en USDA → OFF → LLM fallback
    suggestion = _verify_with_db(llm_output, model, processor_or_tok)
    suggestion["trigger_reason"]   = reason
    suggestion["remaining_before"] = remaining

    # 6. Imprimir y devolver
    print_suggestion(suggestion)
    return suggestion

# ─── TRIGGERS ─────────────────────────────────────────────────────────────────

def _check_triggers(remaining: dict, targets: dict) -> tuple[bool, str]:
    """
    Evalúa los dos triggers en orden. Devuelve (True, motivo) si alguno se activa.
    """
    target_kcal    = targets.get("kcal", 0)
    remaining_kcal = remaining.get("kcal", 0)

    # Trigger 1: ≤ 25 % de kcal restantes
    if target_kcal > 0:
        pct = remaining_kcal / target_kcal
        if pct <= KCAL_TRIGGER_PCT:
            return True, f"≤25 % kcal restantes ({pct*100:.1f} % = {remaining_kcal:.0f} kcal)"

    # Trigger 2: ≤ 3.5 h para medianoche
    now       = datetime.now()
    secs_left = (24 * 3600) - (now.hour * 3600 + now.minute * 60 + now.second)
    if secs_left <= TIME_TRIGGER_HOURS * 3600:
        h, m = divmod(secs_left // 60, 60)
        return True, f"≤3.5 h para medianoche ({h}h {m:02d}min restantes)"

    return False, ""

# ─── CARGA DE DATOS ───────────────────────────────────────────────────────────

def _load_daily_summary(
    meals_file: str = "meals.json",
    user_file:  str = "user.json",
) -> dict | None:
    """
    Replica la lógica de get_daily_summary() del notebook.
    Lee meals.json y user.json y devuelve {total, targets, remaining}.
    """
    try:
        with open(user_file, "r") as f:
            user = json.load(f)
    except FileNotFoundError:
        print(f"  ❌ No se encontró {user_file}. Ejecuta primero la celda de perfil.")
        return None

    targets  = user["daily_targets"]   # {kcal, protein_g, carbs_g, fat_g}
    date_str = datetime.now().strftime("%Y-%m-%d")
    total    = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}

    if os.path.exists(meals_file):
        with open(meals_file, "r") as f:
            meals_db = json.load(f)
        for meal in meals_db.get(date_str, []):
            nutr = meal.get("nutrition", {}).get("totals", {})
            for key in total:
                total[key] = round(total[key] + nutr.get(key, 0), 1)

    remaining = {
        key: max(0.0, round(targets[key] - total[key], 1))
        for key in targets
    }

    return {"total": total, "targets": targets, "remaining": remaining}


def _load_restrictions(user_file: str = "user.json") -> str:
    """Lee las restricciones del perfil y las formatea para el prompt."""
    try:
        with open(user_file, "r") as f:
            user = json.load(f)
    except FileNotFoundError:
        return "None"

    r     = user.get("restrictions", {})
    parts = []
    if r.get("allergies"):
        parts.append(f"Allergies: {', '.join(r['allergies'])}")
    if r.get("intolerances"):
        parts.append(f"Intolerances: {', '.join(r['intolerances'])}")
    if r.get("dislikes"):
        parts.append(f"Dislikes: {', '.join(r['dislikes'])}")
    return "; ".join(parts) if parts else "None"

# ─── GENERACIÓN LLM ───────────────────────────────────────────────────────────

def _ask_llm(
    remaining: dict,
    restrictions: str,
    model,
    proc_or_tok,
) -> dict | None:
    """
    Llama al LLM en modo texto puro.
    Detecta automáticamente si es Qwen2-VL (processor) o Qwen2.5-3B (tokenizer).
    """
    prompt = SUGGESTER_PROMPT.format(
        kcal_remaining    = round(remaining.get("kcal",      0)),
        protein_remaining = round(remaining.get("protein_g", 0)),
        carbs_remaining   = round(remaining.get("carbs_g",   0)),
        fat_remaining     = round(remaining.get("fat_g",     0)),
        restrictions      = restrictions,
    )

    messages   = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    device     = next(model.parameters()).device
    is_vision  = hasattr(proc_or_tok, "image_processor")

    text_input = proc_or_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if is_vision:
        inputs = proc_or_tok(text=[text_input], return_tensors="pt").to(device)
    else:
        inputs = proc_or_tok([text_input], return_tensors="pt", padding=True).to(device)

    print("  [LLM] Generando sugerencia...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=400, do_sample=False)

    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response         = proc_or_tok.decode(generated_tokens, skip_special_tokens=True).strip()

    return _parse_llm_suggestion(response)


def _parse_llm_suggestion(response: str) -> dict | None:
    clean = re.sub(r"```(?:json)?", "", response).strip().rstrip("`").strip()

    try:
        return _validate_suggestion(json.loads(clean))
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            return _validate_suggestion(json.loads(match.group()))
        except json.JSONDecodeError:
            pass

    print(f"  [LLM] No se pudo parsear la sugerencia:\n    {response[:300]}")
    return None


def _validate_suggestion(data: dict) -> dict | None:
    for field in ("dish_name", "total_grams", "ingredients"):
        if field not in data:
            print(f"  [LLM] Campo obligatorio ausente: '{field}'")
            return None
    clean_ings = [
        {"name": i["name"], "grams": float(i["grams"])}
        for i in data["ingredients"]
        if "name" in i and "grams" in i
    ]
    if not clean_ings:
        print("  [LLM] Lista de ingredientes vacía.")
        return None
    data["ingredients"] = clean_ings
    return data

# ─── VERIFICACIÓN EN BBDD ─────────────────────────────────────────────────────

def _verify_with_db(llm_output: dict, model, proc_or_tok) -> dict:
    """
    Verifica los macros de cada ingrediente sugerido.
    Cadena: USDA → OpenFoodFacts → LLM fallback (solo si hay modelo de visión).
    Reutiliza get_nutrients_per_100g() + scale_nutrients() de nutrition_lookup.py.
    """
    is_vision = hasattr(proc_or_tok, "image_processor")
    lm  = model       if is_vision else None
    prc = proc_or_tok if is_vision else None

    print(f"\n  Verificando {len(llm_output['ingredients'])} ingrediente(s) en USDA/OFF...")

    ingredients_verified = []
    totals = {k: 0.0 for k in NUTRIENT_IDS}

    for ing in llm_output["ingredients"]:
        name, grams    = ing["name"], ing["grams"]
        nutrients_100g = get_nutrients_per_100g(name, model=lm, processor=prc)

        if nutrients_100g is None:
            print(f"  ✗ '{name}' no encontrado. Omitido.")
            ingredients_verified.append({
                "name": name, "grams": grams, "source": None, "error": "No encontrado"
            })
            continue

        scaled = scale_nutrients(nutrients_100g, grams)
        ingredients_verified.append({
            "name":   name,
            "grams":  grams,
            "source": scaled.get("source"),
            **{k: scaled[k] for k in NUTRIENT_IDS},
        })
        for key in NUTRIENT_IDS:
            totals[key] = round(totals[key] + scaled.get(key, 0), 1)

    found_grams = sum(i["grams"] for i in ingredients_verified if "error" not in i)

    return {
        "dish_name":            llm_output["dish_name"],
        "total_grams":          found_grams,
        "reason":               llm_output.get("reason", ""),
        "ingredients_verified": ingredients_verified,
        "totals_verified":      {k: round(v, 1) for k, v in totals.items()},
    }

# ─── PRETTY PRINT ─────────────────────────────────────────────────────────────

def print_suggestion(suggestion: dict) -> None:
    SOURCE_ICONS = {"USDA": "✅", "LLM (estimado)": "⚠️"}

    print("\n" + "═" * 58)
    print("  SUGERENCIA DE PLATO")
    print("═" * 58)
    print(f"  ⚡ Motivo:  {suggestion.get('trigger_reason', '')}")
    print(f"\n  🍽️  {suggestion['dish_name']}  ({suggestion['total_grams']:.0f}g)")
    print(f"  💬 {suggestion.get('reason', '')}")

    print("\n  Ingredientes:")
    for ing in suggestion.get("ingredients_verified", []):
        if "error" in ing:
            print(f"    ❌ {ing['name']} ({ing['grams']:.0f}g) — no encontrado")
            continue
        icon = SOURCE_ICONS.get(ing.get("source"), "❓")
        print(f"    {icon} {ing['name']} ({ing['grams']:.0f}g) — "
              f"{ing['kcal']} kcal | P {ing['protein_g']}g | "
              f"C {ing['carbs_g']}g | G {ing['fat_g']}g  [{ing.get('source')}]")

    t   = suggestion.get("totals_verified", {})
    rem = suggestion.get("remaining_before", {})

    print("\n" + "─" * 58)
    print("  TOTALES  vs  NECESARIO")
    print("─" * 58)
    print(f"  {'Macro':<12} {'Necesario':>10} {'Sugerido':>10} {'Diferencia':>12}")
    print(f"  {'─' * 46}")
    for label, key, unit in [
        ("Kcal",     "kcal",      "kcal"),
        ("Proteína", "protein_g", "g"),
        ("Carbos",   "carbs_g",   "g"),
        ("Grasa",    "fat_g",     "g"),
    ]:
        needed = rem.get(key, 0)
        got    = t.get(key, 0)
        diff   = round(got - needed, 1)
        sign   = "+" if diff >= 0 else ""
        print(f"  {label:<12} {needed:>8.1f}{unit}  {got:>8.1f}{unit}  {sign}{diff:>8.1f}{unit}")

    print("\n  Fuentes: ✅ USDA  ⚠️  LLM (estimado)")
    print("═" * 58 + "\n")
