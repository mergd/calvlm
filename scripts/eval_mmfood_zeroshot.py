"""Zero-shot nutrition evaluation on MM-Food-100K with a Hugging Face VLM."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import requests
from PIL import Image

try:
    from datasets import IterableDataset, load_dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "This script requires the `datasets` package. Install via `pip install datasets`."
    ) from exc

from core.simple_hf_backend import _load_hf_model, generate_response


TARGET_FIELDS = ("calories_kcal", "protein_g", "fat_g", "carbohydrate_g")


@dataclass
class SampleResult:
    """Container for a single prediction and associated metadata."""

    index: int
    image_url: str
    food_type: str
    ground_truth: Dict[str, float]
    prediction: Dict[str, Optional[float]]
    confidence: Optional[float]
    raw_text: str


class MetricsTracker:
    """Track MAE and MAPE across multiple nutrition fields."""

    def __init__(self) -> None:
        self._abs_error = defaultdict(float)
        self._abs_pct_error = defaultdict(float)
        self._mae_count = defaultdict(int)
        self._mape_count = defaultdict(int)

    def update(self, truth: Dict[str, float], pred: Dict[str, Optional[float]]) -> None:
        for field in TARGET_FIELDS:
            if field not in truth:
                continue
            target = truth[field]
            guess = pred.get(field)
            if guess is None or not math.isfinite(guess):
                continue
            self._abs_error[field] += abs(guess - target)
            self._mae_count[field] += 1
            if target != 0:
                self._abs_pct_error[field] += abs(guess - target) / abs(target)
                self._mape_count[field] += 1

    def as_dict(self) -> Dict[str, Dict[str, Optional[float]]]:
        report: Dict[str, Dict[str, Optional[float]]] = {}
        for field in TARGET_FIELDS:
            mae = (
                self._abs_error[field] / self._mae_count[field]
                if self._mae_count[field]
                else None
            )
            mape = (
                self._abs_pct_error[field] / self._mape_count[field]
                if self._mape_count[field]
                else None
            )
            report[field] = {"mae": mae, "mape": mape}
        return report


def _ensure_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _safe_json_loads(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _to_list_str(raw: Any) -> str:
    parsed = _safe_json_loads(raw)
    if isinstance(parsed, (list, tuple)):
        return ", ".join(map(str, parsed))
    return str(raw)


def _fetch_image(url: str, timeout: float = 20.0) -> Image.Image:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as img:
        return img.convert("RGB")


def _build_question(example: Dict[str, Any]) -> str:
    dish = example.get("dish_name") or example.get("food_name") or "Unknown dish"
    food_type = example.get("food_type") or "Unknown type"
    ingredients = _to_list_str(example.get("ingredients", ""))
    portion = _to_list_str(example.get("portion_size", ""))
    cooking = example.get("cooking_method") or example.get("cook_method") or ""

    details = [
        f"Dish name: {dish}",
        f"Food type: {food_type}",
    ]
    if ingredients.strip():
        details.append(f"Ingredients: {ingredients}")
    if portion.strip():
        details.append(f"Portion size: {portion}")
    if cooking:
        details.append(f"Cooking method: {cooking}")

    detail_text = "\n".join(details)
    return (
        "You are a meticulous nutritionist. Estimate the nutritional values for the food shown.\n"
        f"{detail_text}\n"
        "Respond with a JSON object containing exactly these keys and numeric values only:\n"
        '{"calories_kcal": float, "protein_g": float, "fat_g": float, "carbohydrate_g": float, "confidence": float}.\n'
        "The confidence should be between 0 and 1."
    )


def _extract_prediction(text: str) -> Dict[str, Optional[float]]:
    """Parse the JSON-like response into numeric nutrition fields."""
    import re

    json_blob: Optional[str] = None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        json_blob = match.group(0)

    parsed: Dict[str, Any] = {}
    if json_blob:
        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(json_blob.replace("'", '"'))
            except json.JSONDecodeError:
                parsed = {}

    result: Dict[str, Optional[float]] = {}
    for field in (*TARGET_FIELDS, "confidence"):
        value = parsed.get(field)
        if value is None:
            result[field] = None
            continue
        try:
            result[field] = float(value)
        except (TypeError, ValueError):
            result[field] = None
    return result


def _iter_dataset(
    split: str, limit: Optional[int], streaming: bool, seed: Optional[int]
) -> Iterator[Dict[str, Any]]:
    dataset = load_dataset("Codatta/MM-Food-100K", split=split, streaming=streaming)
    if isinstance(dataset, IterableDataset):
        iterator = iter(dataset.shuffle(seed=seed) if seed is not None else dataset)
    else:
        iterator = iter(dataset.shuffle(seed=seed) if seed is not None else dataset)

    count = 0
    for example in iterator:
        yield example
        count += 1
        if limit is not None and count >= limit:
            break


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    tokenizer, processor, model, device = _load_hf_model(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
    )
    eos_id = getattr(tokenizer, "eos_token_id", None)

    overall_tracker = MetricsTracker()
    per_type: Dict[str, MetricsTracker] = defaultdict(MetricsTracker)
    dump_writer = None
    if args.dump_jsonl:
        dump_path = Path(args.dump_jsonl)
        _ensure_dir(dump_path)
        dump_writer = dump_path.open("w", encoding="utf-8")

    total = 0
    skipped = 0
    for idx, example in enumerate(
        _iter_dataset(args.split, args.limit, args.streaming, args.seed)
    ):
        image_url = example.get("image_url")
        if not image_url:
            skipped += 1
            continue

        try:
            image = _fetch_image(image_url)
        except Exception as exc:  # pragma: no cover - network variability
            print(f"[warn] Failed to fetch image {image_url}: {exc}")
            skipped += 1
            continue

        question = _build_question(example)
        _, response = generate_response(
            tokenizer,
            processor,
            model,
            device,
            question=question,
            images=[image],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stop_token_id=eos_id,
        )
        pred = _extract_prediction(response)

        nutrition_raw = _safe_json_loads(example.get("nutritional_profile", {}))
        if not isinstance(nutrition_raw, dict):
            nutrition_raw = {}
        truth = {
            field: float(nutrition_raw.get(field, 0.0))
            for field in TARGET_FIELDS
        }

        overall_tracker.update(truth, pred)
        food_type = example.get("food_type", "unknown")
        per_type[food_type].update(truth, pred)

        record = SampleResult(
            index=idx,
            image_url=image_url,
            food_type=food_type,
            ground_truth=truth,
            prediction={field: pred.get(field) for field in TARGET_FIELDS},
            confidence=pred.get("confidence"),
            raw_text=response,
        )
        total += 1

        if dump_writer is not None:
            json.dump(
                {
                    "index": record.index,
                    "image_url": record.image_url,
                    "food_type": record.food_type,
                    "ground_truth": record.ground_truth,
                    "prediction": record.prediction,
                    "confidence": record.confidence,
                    "response": record.raw_text,
                },
                dump_writer,
            )
            dump_writer.write("\n")

        if args.log_every and total % args.log_every == 0:
            print(
                f"[info] Processed {total} samples "
                f"(skipped {skipped}); latest response: {record.raw_text[:120]}..."
            )

    if dump_writer is not None:
        dump_writer.close()

    overall_metrics = overall_tracker.as_dict()
    by_type = {
        food: tracker.as_dict() for food, tracker in sorted(per_type.items())
    }

    summary = {
        "total": total,
        "skipped": skipped,
        "overall": overall_metrics,
        "by_food_type": by_type,
    }

    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot nutrition evaluation on MM-Food-100K."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of samples."
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the dataset instead of downloading to disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional shuffle seed when using streaming datasets.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Disable trusting remote code when loading the model.",
    )
    parser.add_argument(
        "--dump_jsonl",
        default="",
        help="Optional path to save per-sample predictions as JSONL.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log progress every N processed samples.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
