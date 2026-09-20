#!/usr/bin/env python3
"""Open-weight Qwen3-VL backend for NFS-CBM concept naming.

This script consumes intervention image pairs produced by ``pipeline.py
--dry-run``.  It runs Qwen3-VL locally and never calls an external API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import name_concepts as common


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
VERSION = "1.0.0"
DEFAULT_MAX_IMAGE_SIDE = 768
DEFAULT_MAX_NEW_TOKENS = 512


class QwenNamingError(common.NamingError):
    """A concise, user-facing local inference or output error."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path, max_image_side: int) -> tuple[Image.Image, dict[str, Any]]:
    try:
        source_size = path.stat().st_size
        if source_size > 20 * 1024 * 1024:
            raise QwenNamingError(f"Image exceeds the 20 MiB release limit: {path.name}")
        with Image.open(path) as original:
            if original.format not in {"PNG", "JPEG", "WEBP"}:
                raise QwenNamingError(f"Use a PNG, JPEG or WebP image: {path.name}")
            if getattr(original, "n_frames", 1) != 1:
                raise QwenNamingError(f"Animated inputs are not supported: {path.name}")
            if original.width * original.height > 10_240_000 or max(original.size) > 6000:
                raise QwenNamingError(f"Image dimensions exceed this release's limit: {path.name}")
            image = ImageOps.exif_transpose(original).convert("RGB")
            original_size = list(image.size)
            if max(image.size) > max_image_side:
                image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
            image = image.copy()
    except (OSError, Image.DecompressionBombError) as exc:
        raise QwenNamingError(f"Cannot read image: {path.name}") from exc
    return image, {
        "path": str(path),
        "source_sha256": file_sha256(path),
        "source_bytes": source_size,
        "original_size": original_size,
        "submitted_size": list(image.size),
    }


def discover_pair_jobs(
    pairs_root: Path, dataset_context: str, max_pairs: int
) -> list[dict[str, Any]]:
    if not pairs_root.is_dir():
        raise QwenNamingError(f"Pair directory not found: {pairs_root}")
    jobs: list[dict[str, Any]] = []
    for concept_dir in sorted(path for path in pairs_root.iterdir() if path.is_dir()):
        originals = sorted(concept_dir.glob("sample_*_original.png"))
        if not originals:
            continue
        pairs = []
        for before in originals[:max_pairs]:
            after = before.with_name(before.name.replace("_original.png", "_perturbed.png"))
            if not after.is_file():
                raise QwenNamingError(f"Missing perturbed image for: {before.name}")
            pairs.append({"before": before.resolve(), "after": after.resolve(),
                          "predicted_class": ""})
        jobs.append({"concept_id": concept_dir.name, "dataset_context": dataset_context,
                     "pairs": pairs})
    if not jobs:
        raise QwenNamingError(
            "No sample_*_original.png / sample_*_perturbed.png pairs were found."
        )
    return jobs


def build_messages(
    job: dict[str, Any], max_image_side: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    template = (ROOT / "prompt_template.txt").read_text(encoding="utf-8")
    prompt = template.replace("{dataset_context}", job["dataset_context"])
    prompt = prompt.replace("{concept_id}", job["concept_id"])
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if len(job["pairs"]) > 1:
        extension = (ROOT / "multisample_instruction.txt").read_text(encoding="utf-8")
        content.append({"type": "text", "text": extension})
    pair_metadata = []
    for index, pair in enumerate(job["pairs"], 1):
        before, before_meta = load_rgb(Path(pair["before"]), max_image_side)
        after, after_meta = load_rgb(Path(pair["after"]), max_image_side)
        if before_meta["original_size"] != after_meta["original_size"]:
            raise QwenNamingError(f"Pair {index}: before and after must have equal dimensions.")
        predicted = str(pair.get("predicted_class", "")).strip()
        label = f"Pair {index}."
        if predicted:
            label += (f" Predicted class: {predicted}. This is contextual metadata; "
                      "base the name on visible intervention changes.")
        content.extend([
            {"type": "text", "text": label + " Image A: reconstruction before intervention."},
            {"type": "image", "image": before},
            {"type": "text", "text": f"Pair {index}. Image B: reconstruction after perturbing one concept unit."},
            {"type": "image", "image": after},
        ])
        pair_metadata.append({"pair_index": index, "predicted_class": predicted,
                              "before": before_meta, "after": after_meta})
    messages = [{"role": "user", "content": content}]
    provenance = {
        "script_version": VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_context": job["dataset_context"],
        "concept_id": job["concept_id"],
        "pair_count": len(pair_metadata),
        "pairs": pair_metadata,
        "prompt_template_sha256": common.sha256(template.encode()),
        "text_blocks": [item["text"] for item in content if item["type"] == "text"],
    }
    return messages, provenance


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except ValueError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise QwenNamingError("Qwen3-VL response does not contain a JSON object.")
        try:
            value = json.loads(cleaned[start:end + 1])
        except ValueError as exc:
            raise QwenNamingError("Qwen3-VL response is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise QwenNamingError("Qwen3-VL response must be a JSON object.")
    return value


def normalize_result(value: dict[str, Any], concept_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"concept_id": concept_id}
    for field in common.TEXT_FIELDS:
        if field == "concept_id":
            continue
        item = value.get(field, "")
        if not isinstance(item, str):
            item = str(item)
        result[field] = item.strip()
    for field in common.LIST_FIELDS:
        item = value.get(field, [])
        if isinstance(item, str):
            item = [part.strip() for part in re.split(r"\n|;|\|", item) if part.strip()]
        if not isinstance(item, list):
            item = [str(item)]
        result[field] = [str(part).strip() for part in item if str(part).strip()]
    confidence = value.get("confidence", "low")
    if isinstance(confidence, (int, float)):
        confidence = "high" if confidence >= 0.75 else "medium" if confidence >= 0.45 else "low"
    confidence = str(confidence).lower().strip()
    result["confidence"] = confidence if confidence in {"high", "medium", "low"} else "low"
    if not result["best_name"]:
        candidates = result["candidate_names"]
        if candidates:
            result["best_name"] = candidates[0]
    if not result["best_name"]:
        raise QwenNamingError("Qwen3-VL returned an empty concept name.")
    if not result["candidate_names"]:
        result["candidate_names"] = [result["best_name"]]
    return result


class LocalQwenNamer:
    def __init__(self, args: argparse.Namespace):
        try:
            import torch
            import transformers
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise QwenNamingError(
                "Install the local backend with: python -m pip install "
                "-r concept_naming/requirements_qwen.txt"
            ) from exc
        if not torch.cuda.is_available():
            raise QwenNamingError("The released Qwen3-VL-8B FP16 setup requires a CUDA GPU.")
        if torch.cuda.device_count() != 1:
            raise QwenNamingError(
                "Expected one visible GPU. Use --gpu to select one physical GPU."
            )
        model_source = args.model
        local_path = Path(model_source).expanduser()
        local_only = local_path.is_dir()
        if local_only:
            model_source = str(local_path.resolve())
        model_kwargs: dict[str, Any] = {"local_files_only": local_only}
        if not local_only:
            model_kwargs["revision"] = args.revision
        self.torch = torch
        self.transformers_version = transformers.__version__
        self.processor = AutoProcessor.from_pretrained(model_source, **model_kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source,
            dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            **model_kwargs,
        )
        self.model.eval()
        self.device = torch.device("cuda:0")
        self.max_new_tokens = args.max_new_tokens
        self.runtime = {
            "requested_model": model_source,
            "model_id": MODEL_ID,
            "model_revision": args.revision,
            "dtype": "float16",
            "attention_implementation": "eager",
            "decoding": "greedy",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "max_new_tokens": args.max_new_tokens,
            "max_image_side": args.max_image_side,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "transformers_version": self.transformers_version,
            "cuda_runtime": torch.version.cuda,
        }

    def generate(self, messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str, dict[str, Any]]:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        self.torch.cuda.empty_cache()
        self.torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        elapsed = time.perf_counter() - start
        generated = output[0, input_tokens:]
        text = self.processor.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        metrics = {
            "input_tokens": input_tokens,
            "output_tokens": int(generated.numel()),
            "elapsed_seconds": round(elapsed, 4),
            "peak_gpu_memory_gib": round(
                self.torch.cuda.max_memory_allocated(self.device) / (1024 ** 3), 4
            ),
        }
        return extract_json(text), text, metrics


def load_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.pairs_root:
        if args.manifest or args.before or args.after or args.concept_id:
            raise QwenNamingError("Use --pairs-root by itself for pair discovery.")
        context = args.dataset_context or common.CONTEXTS.get(args.dataset, "")
        if not context:
            raise QwenNamingError("Use --dataset or --dataset-context with --pairs-root.")
        return discover_pair_jobs(args.pairs_root.resolve(), context, args.max_pairs)
    return common.load_jobs(args)


def prepare_job(
    job: dict[str, Any], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages, provenance = build_messages(job, args.max_image_side)
    provenance["settings"] = {
        "model": args.model,
        "revision": args.revision,
        "dtype": "float16",
        "attention_implementation": "eager",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "max_image_side": args.max_image_side,
    }
    provenance["request_sha256"] = common.sha256(common.json_bytes(provenance))
    return messages, provenance


def run(args: argparse.Namespace) -> int:
    jobs = load_jobs(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for job in jobs:
            _, provenance = prepare_job(job, args)
            print(f"Validated {job['concept_id']}: {provenance['pair_count']} image pair(s).")
        print("Dry run complete. Qwen3-VL was not loaded.")
        return 0

    namer: LocalQwenNamer | None = None
    records = []
    for job in jobs:
        messages, provenance = prepare_job(job, args)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", job["concept_id"])[:48] or "concept"
        folder = output / (safe_id + "_" + provenance["request_sha256"][:16])
        folder.mkdir(exist_ok=True)
        result_path = folder / "result.json"
        if result_path.is_file():
            if not args.resume:
                raise QwenNamingError(
                    "A matching result exists. Use --resume to reuse it, or a new --output."
                )
            saved = common.read_json(result_path)
            result = normalize_result(saved.get("result", {}), job["concept_id"])
            if saved.get("provenance", {}).get("request_sha256") != provenance["request_sha256"]:
                raise QwenNamingError("Cached result does not match the current request.")
            saved["result"] = result
            records.append(saved)
            common.write_summary(output, records)
            print(f"Reused {job['concept_id']}.")
            continue
        common.save_json(folder / "request_metadata.json", provenance)
        if namer is None:
            namer = LocalQwenNamer(args)
        raw, raw_text, metrics = namer.generate(messages)
        result = normalize_result(raw, job["concept_id"])
        (folder / "raw_output.txt").write_text(raw_text + "\n", encoding="utf-8")
        saved = {
            "result": result,
            "provenance": {
                **provenance,
                **namer.runtime,
                **metrics,
                "utc_time": datetime.now(timezone.utc).isoformat(),
            },
        }
        common.save_json(result_path, saved)
        records.append(saved)
        common.write_summary(output, records)
        print(f"Concept {job['concept_id']}: {result['best_name']} ({result['confidence']}).")
    print(f"Complete: named {len(records)} concept dimension(s) with local Qwen3-VL.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-root", type=Path,
                        help="pair folders produced by pipeline.py --dry-run")
    parser.add_argument("--before", type=Path, help="single-pair Image A")
    parser.add_argument("--after", type=Path, help="single-pair Image B")
    parser.add_argument("--concept-id", help="identifier of the perturbed concept")
    parser.add_argument("--manifest", type=Path,
                        help="the same JSON manifest format accepted by name_concepts.py")
    parser.add_argument("--dataset", choices=list(common.CONTEXTS))
    parser.add_argument("--dataset-context",
                        help="explicit dataset context, overriding --dataset")
    parser.add_argument("--predicted-class", default="")
    parser.add_argument("--model", default=MODEL_ID,
                        help="local weight directory or Hugging Face model id")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--gpu", default="0", help="physical GPU index")
    parser.add_argument("--max-image-side", type=int, default=DEFAULT_MAX_IMAGE_SIDE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-pairs", type=int, default=5,
                        help="maximum pairs per concept when using --pairs-root")
    parser.add_argument("--output", type=Path, default=Path("runs_qwen"))
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs without loading Qwen3-VL")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_image_side < 224 or args.max_image_side > 2048:
        parser.error("--max-image-side must be from 224 to 2048")
    if args.max_new_tokens < 64 or args.max_new_tokens > 2048:
        parser.error("--max-new-tokens must be from 64 to 2048")
    if args.max_pairs < 1 or args.max_pairs > 20:
        parser.error("--max-pairs must be from 1 to 20")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        return run(args)
    except (QwenNamingError, common.NamingError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed results remain available for --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
