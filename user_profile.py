
from dataclasses import dataclass, field
from typing import Literal

ActivityLevel = Literal[0, 1, 2, 3, 4, 5, 6]  # entrenos/semana
Goal = Literal["loss", "maintenance", "gain"]

# Multiplicadores de actividad (escala de Katch-McArdle adaptada)
ACTIVITY_MULTIPLIERS = {
    0: 1.2,    # Sedentario
    1: 1.375,  # Ligero (1-2x/semana)
    2: 1.375,  # Ligero
    3: 1.55,   # Moderado (3-4x/semana)
    4: 1.55,   # Moderado
    5: 1.725,  # Activo (5-6x/semana)
    6: 1.9,    # Muy activo (diario + físico)
}

# Ajuste calórico y split de macros por objetivo
GOAL_CONFIG = {
    "loss": {
        "kcal_offset": -400,          # déficit
        "protein_g_per_kg": 2.2,      # alto para preservar músculo
        "fat_pct": 0.25,
    },
    "maintenance": {
        "kcal_offset": 0,
        "protein_g_per_kg": 1.8,
        "fat_pct": 0.28,
    },
    "gain": {
        "kcal_offset": +300,          # superávit moderado
        "protein_g_per_kg": 2.0,
        "fat_pct": 0.25,
    },
}

@dataclass
class UserProfile:
    name: str
    sex: Literal["male", "female"]
    age: int
    weight_kg: float
    height_cm: float
    activity_level: ActivityLevel   # 0-6 entrenos/semana
    goal: Goal                      # "loss" | "maintenance" | "gain"

    # Se calculan automáticamente al crear el perfil
    daily_targets: dict = field(init=False)

    def __post_init__(self):
        self.daily_targets = self._calculate_targets()

    def _bmr(self) -> float:
        """Harris-Benedict revisada (Mifflin-St Jeor, más precisa)."""
        if self.sex == "male":
            return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161

    def _tdee(self) -> float:
        return self._bmr() * ACTIVITY_MULTIPLIERS[self.activity_level]

    def _calculate_targets(self) -> dict:
        cfg = GOAL_CONFIG[self.goal]
        tdee = self._tdee()
        target_kcal = round(tdee + cfg["kcal_offset"])

        # Proteína fija por kg de peso corporal
        protein_g = round(self.weight_kg * cfg["protein_g_per_kg"])
        protein_kcal = protein_g * 4

        # Grasa como porcentaje de las kcal objetivo
        fat_g = round((target_kcal * cfg["fat_pct"]) / 9)
        fat_kcal = fat_g * 9

        # Carbos cubren el resto
        carb_kcal = target_kcal - protein_kcal - fat_kcal
        carb_g = round(carb_kcal / 4)

        return {
            "kcal":    target_kcal,
            "protein": protein_g,   # gramos
            "fat":     fat_g,       # gramos
            "carbs":   carb_g,      # gramos
            "bmr":     round(self._bmr()),
            "tdee":    round(tdee),
        }

    def summary(self) -> str:
        t = self.daily_targets
        goal_label = {"loss": "Pérdida de peso", "maintenance": "Mantenimiento", "gain": "Volumen"}[self.goal]
        return (
            f"=== {self.name} | {goal_label} ===\n"
            f"BMR: {t['bmr']} kcal  |  TDEE: {t['tdee']} kcal\n"
            f"Objetivo diario: {t['kcal']} kcal\n"
            f"  Proteína: {t['protein']}g\n"
            f"  Carbos:   {t['carbs']}g\n"
            f"  Grasa:    {t['fat']}g"
        )