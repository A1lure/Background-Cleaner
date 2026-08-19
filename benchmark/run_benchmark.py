from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from datetime import date
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from PIL import Image, ImageDraw
from rembg import new_session, remove  # type: ignore[import-untyped]


BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
INPUTS_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results" / "current"
RESULTS_PATH = RESULTS_DIR / "summary.json"
BENCHMARK_RESULTS_PATH = BASE_DIR / "benchmark_results.json"

MODELS = [
    "u2netp",
    "silueta",
    "u2net",
    "isnet-general-use",
]

SAMPLES = [
    ("photo_1_desk", INPUTS_DIR / "photo_1_desk.webp"),
    ("photo_2_vegetables", INPUTS_DIR / "photo_2_vegetables.jpg"),
    ("photo_3_portrait_full", INPUTS_DIR / "photo_3_portrait_full.jpg"),
    ("photo_4_neon_scene", INPUTS_DIR / "photo_4_neon_scene.jpg"),
    ("photo_5_portrait_close", INPUTS_DIR / "photo_5_portrait_close.webp"),
]


class SampleInfo(TypedDict):
    name: str
    path: str
    sha256: str
    width: int
    height: int
    format: str | None


class AlphaStats(TypedDict):
    foreground_ratio: float
    transparent_ratio: float
    soft_alpha_ratio: float
    mean_alpha: float


class SampleResult(AlphaStats):
    inference_s: float
    output_path: str
    preview_path: str
    output_kb: float


class ModelResult(TypedDict, total=False):
    load_s: float
    samples: dict[str, SampleResult]
    avg_inference_s: float
    error: str


class BenchmarkResult(TypedDict):
    benchmark_date: str
    environment: dict[str, str]
    notes: str
    samples: list[SampleInfo]
    models: dict[str, ModelResult]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alpha_stats(image: Image.Image) -> AlphaStats:
    alpha = np.array(image.convert("RGBA").split()[-1])
    total = alpha.size
    return {
        "foreground_ratio": round(float((alpha > 127).sum() / total), 4),
        "transparent_ratio": round(float((alpha < 16).sum() / total), 4),
        "soft_alpha_ratio": round(float(((alpha > 0) & (alpha < 255)).sum() / total), 4),
        "mean_alpha": round(float(alpha.mean()), 2),
    }


def save_preview(image: Image.Image, path: Path) -> None:
    image = image.convert("RGBA")
    tile = 24
    background = Image.new("RGBA", image.size, "white")
    draw = ImageDraw.Draw(background)

    for y in range(0, image.height, tile):
        for x in range(0, image.width, tile):
            if (x // tile + y // tile) % 2:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(220, 220, 220, 255))

    background.alpha_composite(image)
    background.convert("RGB").save(path, quality=92)


def make_contact_sheet(preview_paths: list[tuple[str, str, Path]], path: Path) -> None:
    if not preview_paths:
        return

    models = list(dict.fromkeys(model for model, _, _ in preview_paths))
    samples = list(dict.fromkeys(sample for _, sample, _ in preview_paths))
    cell_w, cell_h = 260, 240
    header_h = 28
    sheet = Image.new("RGB", (cell_w * len(samples), header_h + cell_h * len(models)), "white")
    draw = ImageDraw.Draw(sheet)

    for col, sample in enumerate(samples):
        draw.text((col * cell_w + 8, 7), sample, fill=(0, 0, 0))

    for row, model in enumerate(models):
        y0 = header_h + row * cell_h
        draw.text((8, y0 + 6), model, fill=(0, 0, 0))
        for col, sample in enumerate(samples):
            preview = next((p for m, s, p in preview_paths if m == model and s == sample), None)
            if preview is None:
                continue
            image = Image.open(preview).convert("RGB")
            image.thumbnail((cell_w - 16, cell_h - 34))
            x = col * cell_w + (cell_w - image.width) // 2
            y = y0 + 30 + (cell_h - 34 - image.height) // 2
            sheet.paste(image, (x, y))

    sheet.save(path, quality=92)


def run(models: list[str]) -> BenchmarkResult:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    samples: list[SampleInfo] = []
    for name, path in SAMPLES:
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        image = Image.open(path)
        samples.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256(path),
                "width": image.width,
                "height": image.height,
                "format": image.format,
            }
        )

    results: BenchmarkResult = {
        "benchmark_date": date.today().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "notes": "No ground-truth masks were provided, so this benchmark reports inference time and alpha-mask statistics, not IoU/MAE.",
        "samples": samples,
        "models": {},
    }

    preview_paths: list[tuple[str, str, Path]] = []

    for model_name in models:
        print(f"Loading {model_name}")
        started = time.perf_counter()
        try:
            session = new_session(model_name)
        except Exception as exc:
            results["models"][model_name] = {"error": str(exc)}
            print(f"  failed: {exc}")
            continue
        load_s = time.perf_counter() - started

        model_results: ModelResult = {
            "load_s": round(load_s, 3),
            "samples": {},
        }

        for sample in samples:
            sample_name = sample["name"]
            input_path = Path(sample["path"])
            output_path = RESULTS_DIR / f"{model_name}_{sample_name}.png"
            preview_path = RESULTS_DIR / f"{model_name}_{sample_name}_preview.jpg"

            started = time.perf_counter()
            output_bytes = cast(bytes, remove(input_path.read_bytes(), session=session))
            inference_s = time.perf_counter() - started
            output_path.write_bytes(output_bytes)

            output_image = Image.open(output_path).convert("RGBA")
            save_preview(output_image, preview_path)
            preview_paths.append((model_name, sample_name, preview_path))

            stats = alpha_stats(output_image)
            sample_result: SampleResult = {
                "inference_s": round(inference_s, 3),
                "output_path": str(output_path),
                "preview_path": str(preview_path),
                "output_kb": round(output_path.stat().st_size / 1024, 1),
                "foreground_ratio": stats["foreground_ratio"],
                "transparent_ratio": stats["transparent_ratio"],
                "soft_alpha_ratio": stats["soft_alpha_ratio"],
                "mean_alpha": stats["mean_alpha"],
            }
            model_results["samples"][sample_name] = sample_result
            print(f"  {sample_name}: {sample_result}")

        sample_times = [v["inference_s"] for v in model_results["samples"].values()]
        model_results["avg_inference_s"] = round(sum(sample_times) / len(sample_times), 3)
        results["models"][model_name] = model_results

        del session
        gc.collect()

    make_contact_sheet(preview_paths, RESULTS_DIR / "contact_sheet.jpg")
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    BENCHMARK_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    run(args.models)
