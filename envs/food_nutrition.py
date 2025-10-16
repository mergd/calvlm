from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import re

from PIL import Image

from vlmrl.envs.base import BaseEnv, BaseState
from vlmrl.utils.vlm import decode_tokens


def _resolve_eos_id(tokenizer) -> int | None:
    if hasattr(tokenizer, "eos_token_id"):
        eos_id = getattr(tokenizer, "eos_token_id")
        if eos_id is not None:
            return eos_id
    if hasattr(tokenizer, "get_eos_token_id"):
        return tokenizer.get_eos_token_id()
    return None


@dataclass(frozen=True)
class FoodNutritionPrompt:
    """Observation for food nutrition prediction."""
    question: str
    image: object  # PIL Image
    image_path: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    is_healthy: bool


@dataclass(frozen=True)
class FoodNutritionState(BaseState):
    dataset_idx: int
    image: object
    food_name: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    is_healthy: bool

    def render(self) -> str:
        return "Analyze this food image and provide nutrition information."


class FoodNutritionEnv(BaseEnv):
    """Food nutrition prediction environment.
    
    The agent must predict nutrition information (calories, macros) from food images.
    Rewards based on prediction accuracy using normalized error.
    
    For now, uses a simple demo dataset. Can be extended to use:
    - Nutrition5k dataset
    - Custom food database
    - Your own labeled food images
    """

    def __init__(
        self,
        tokenizer,
        dataset_path: Optional[str] = None,
        calorie_tolerance: float = 100.0,  # calories tolerance for "correct" prediction
        macro_tolerance_g: float = 10.0,   # grams tolerance for macros
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_per_action = 256
        self.calorie_tolerance = calorie_tolerance
        self.macro_tolerance_g = macro_tolerance_g
        self.dataset = self._build_dataset(dataset_path)
        self.num_tasks = len(self.dataset)
        self.eos_token_id = _resolve_eos_id(tokenizer)

    def _build_dataset(self, dataset_path: Optional[str]) -> list[dict]:
        """Build food nutrition dataset.
        
        For demo purposes, creates a simple dataset with the bundled images.
        In production, you'd load from a real nutrition database.
        """
        if dataset_path:
            # TODO: Load from custom dataset path
            return self._load_custom_dataset(dataset_path)
        
        # Demo dataset with example foods
        asset_root = Path(__file__).resolve().parents[1] / "imgs"
        
        return [
            {
                "food_name": "Coffee and Laptop Setup",
                "image_path": str(asset_root / "coffee_laptop.png"),
                "calories": 5.0,  # black coffee
                "protein_g": 0.3,
                "carbs_g": 0.0,
                "fat_g": 0.0,
                "is_healthy": True,
            },
            # Add more foods here as you build your dataset
        ]

    def _load_custom_dataset(self, dataset_path: str) -> list[dict]:
        """Load custom nutrition dataset from CSV or JSON.
        
        Expected format:
        - food_name: str
        - image_path: str
        - calories: float
        - protein_g: float
        - carbs_g: float
        - fat_g: float
        - is_healthy: bool (optional)
        """
        import pandas as pd
        
        df = pd.read_csv(dataset_path)
        dataset = []
        
        for _, row in df.iterrows():
            # Calculate if healthy based on simple heuristic
            # (you can customize this logic)
            is_healthy = self._calculate_healthiness(
                calories=row.get("calories", 0),
                protein_g=row.get("protein_g", 0),
                carbs_g=row.get("carbs_g", 0),
                fat_g=row.get("fat_g", 0),
            )
            
            dataset.append({
                "food_name": str(row["food_name"]),
                "image_path": str(row["image_path"]),
                "calories": float(row["calories"]),
                "protein_g": float(row["protein_g"]),
                "carbs_g": float(row["carbs_g"]),
                "fat_g": float(row["fat_g"]),
                "is_healthy": bool(row.get("is_healthy", is_healthy)),
            })
        
        return dataset

    def _calculate_healthiness(
        self,
        calories: float,
        protein_g: float,
        carbs_g: float,
        fat_g: float,
    ) -> bool:
        """Simple heuristic for food healthiness.
        
        You can customize this with more sophisticated rules.
        """
        # High protein-to-calorie ratio is generally good
        if calories > 0:
            protein_ratio = (protein_g * 4) / calories  # protein has 4 cal/g
            if protein_ratio > 0.3:  # >30% protein
                return True
        
        # Low fat and reasonable calories
        if fat_g < 10 and calories < 300:
            return True
        
        # Default to moderate
        return calories < 500 and fat_g < 20

    def reset(self, idx):
        item = self.dataset[idx % len(self.dataset)]
        
        # Load image
        image = Image.open(item["image_path"]).convert("RGB")
        
        # Prompt that asks for structured nutrition info
        question = """Analyze this food image and provide a nutrition label in this format:

Food Name: [name]
Calories: [number]
Protein: [grams]g
Carbohydrates: [grams]g
Fat: [grams]g
Is Healthy: [Yes/No]"""

        state = FoodNutritionState(
            dataset_idx=idx % len(self.dataset),
            image=image,
            food_name=item["food_name"],
            calories=item["calories"],
            protein_g=item["protein_g"],
            carbs_g=item["carbs_g"],
            fat_g=item["fat_g"],
            is_healthy=item["is_healthy"],
        )

        obs = FoodNutritionPrompt(
            question=question,
            image=image,
            image_path=item["image_path"],
            calories=item["calories"],
            protein_g=item["protein_g"],
            carbs_g=item["carbs_g"],
            fat_g=item["fat_g"],
            is_healthy=item["is_healthy"],
        )

        return state, obs

    def step(self, state, action_tokens):
        if self.eos_token_id is not None:
            action_tokens = self.clean_action(action_tokens, self.eos_token_id)
        
        response_text = decode_tokens(self.tokenizer, action_tokens).strip()
        
        # Parse the response
        parsed = self._parse_nutrition_response(response_text)
        
        # Calculate reward based on prediction accuracy
        reward = 0.0
        info = {
            "response": response_text,
            "food_name": state.food_name,
            "ground_truth": {
                "calories": state.calories,
                "protein_g": state.protein_g,
                "carbs_g": state.carbs_g,
                "fat_g": state.fat_g,
                "is_healthy": state.is_healthy,
            },
            "predicted": parsed,
        }
        
        # Reward components (each worth up to 0.2, totaling 1.0)
        reward_weights = {
            "calories": 0.3,
            "protein": 0.2,
            "carbs": 0.2,
            "fat": 0.2,
            "healthy": 0.1,
        }
        
        # Calories accuracy (normalized error)
        if parsed.get("calories") is not None:
            calorie_error = abs(parsed["calories"] - state.calories)
            # Exponential decay reward: 1.0 at 0 error, ~0.0 at 2x tolerance
            calorie_reward = max(0.0, 1.0 - (calorie_error / (self.calorie_tolerance * 2)))
            reward += calorie_reward * reward_weights["calories"]
            info["calorie_error"] = calorie_error
            info["calorie_reward"] = calorie_reward
            info["calories_correct"] = int(calorie_error <= self.calorie_tolerance)
        
        # Protein accuracy
        if parsed.get("protein_g") is not None:
            protein_error = abs(parsed["protein_g"] - state.protein_g)
            protein_reward = max(0.0, 1.0 - (protein_error / (self.macro_tolerance_g * 2)))
            reward += protein_reward * reward_weights["protein"]
            info["protein_error"] = protein_error
            info["protein_reward"] = protein_reward
            info["protein_correct"] = int(protein_error <= self.macro_tolerance_g)
        
        # Carbs accuracy
        if parsed.get("carbs_g") is not None:
            carbs_error = abs(parsed["carbs_g"] - state.carbs_g)
            carbs_reward = max(0.0, 1.0 - (carbs_error / (self.macro_tolerance_g * 2)))
            reward += carbs_reward * reward_weights["carbs"]
            info["carbs_error"] = carbs_error
            info["carbs_reward"] = carbs_reward
            info["carbs_correct"] = int(carbs_error <= self.macro_tolerance_g)
        
        # Fat accuracy
        if parsed.get("fat_g") is not None:
            fat_error = abs(parsed["fat_g"] - state.fat_g)
            fat_reward = max(0.0, 1.0 - (fat_error / (self.macro_tolerance_g * 2)))
            reward += fat_reward * reward_weights["fat"]
            info["fat_error"] = fat_error
            info["fat_reward"] = fat_reward
            info["fat_correct"] = int(fat_error <= self.macro_tolerance_g)
        
        # Healthy classification
        if parsed.get("is_healthy") is not None:
            healthy_correct = int(parsed["is_healthy"] == state.is_healthy)
            reward += healthy_correct * reward_weights["healthy"]
            info["healthy_correct"] = healthy_correct
        
        # Format bonus for structured output
        if len(parsed) >= 3:
            format_bonus = 0.05
            reward += format_bonus
            info["format_bonus"] = format_bonus
        
        # Clip reward to [0, 1]
        reward = float(max(0.0, min(1.0, reward)))
        info["final_reward"] = reward
        
        # Overall correctness (for logging)
        all_correct = (
            info.get("calories_correct", 0) and
            info.get("protein_correct", 0) and
            info.get("carbs_correct", 0) and
            info.get("fat_correct", 0)
        )
        info["all_correct"] = int(all_correct)
        
        return state, [], reward, True, info

    def _parse_nutrition_response(self, text: str) -> Dict[str, Any]:
        """Parse nutrition information from model response.
        
        Looks for structured key-value pairs like:
        - Calories: 500
        - Protein: 25g
        - Is Healthy: Yes
        """
        parsed = {}
        
        if not text:
            return parsed
        
        # Extract calories
        cal_match = re.search(r'calories?\s*:?\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if cal_match:
            try:
                parsed["calories"] = float(cal_match.group(1))
            except ValueError:
                pass
        
        # Extract protein
        protein_match = re.search(r'protein\s*:?\s*(\d+(?:\.\d+)?)\s*g?', text, re.IGNORECASE)
        if protein_match:
            try:
                parsed["protein_g"] = float(protein_match.group(1))
            except ValueError:
                pass
        
        # Extract carbs
        carbs_match = re.search(r'carb(?:ohydrate)?s?\s*:?\s*(\d+(?:\.\d+)?)\s*g?', text, re.IGNORECASE)
        if carbs_match:
            try:
                parsed["carbs_g"] = float(carbs_match.group(1))
            except ValueError:
                pass
        
        # Extract fat
        fat_match = re.search(r'fat\s*:?\s*(\d+(?:\.\d+)?)\s*g?', text, re.IGNORECASE)
        if fat_match:
            try:
                parsed["fat_g"] = float(fat_match.group(1))
            except ValueError:
                pass
        
        # Extract healthy classification
        healthy_match = re.search(r'(?:is\s+)?healthy\s*:?\s*(yes|no|true|false)', text, re.IGNORECASE)
        if healthy_match:
            healthy_str = healthy_match.group(1).lower()
            parsed["is_healthy"] = healthy_str in ["yes", "true"]
        
        # Extract food name (optional)
        name_match = re.search(r'food\s+name\s*:?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if name_match:
            parsed["food_name"] = name_match.group(1).strip()
        
        return parsed

