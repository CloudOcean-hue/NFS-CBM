#!/usr/bin/env python3
"""Run NFS-CBM inference and name every concept dimension with GPT-5.4."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from concept_naming import name_concepts as namer

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class PipelineError(Exception):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(f"Cannot read configuration: {path}") from exc
    required = {"checkpoint_path", "dataset_root", "output_dir", "dataset_context",
                "image_size", "batch_size", "top_m", "perturbation", "device",
                "seed", "openai"}
    if not isinstance(config, dict) or set(config) != required:
        raise PipelineError(f"Configuration keys must be exactly: {sorted(required)}")
    for key in ("checkpoint_path", "dataset_root", "output_dir", "dataset_context", "device"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise PipelineError(f"{key} must be a nonempty string")
    for key in ("image_size", "batch_size", "top_m", "seed"):
        if type(config[key]) is not int:
            raise PipelineError(f"{key} must be an integer")
    if config["image_size"] < 32 or config["batch_size"] < 1 or config["top_m"] < 1:
        raise PipelineError("image_size, batch_size and top_m must be positive")
    if not isinstance(config["perturbation"], (int, float)) or config["perturbation"] == 0:
        raise PipelineError("perturbation must be a nonzero number")
    openai_config = dict(namer.DEFAULTS)
    if not isinstance(config["openai"], dict):
        raise PipelineError("openai must be an object")
    unknown = set(config["openai"]) - set(namer.DEFAULTS)
    if unknown:
        raise PipelineError(f"Unknown OpenAI settings: {sorted(unknown)}")
    openai_config.update(config["openai"])
    config["openai"] = openai_config
    config["checkpoint_path"] = resolve_path(config["checkpoint_path"], path)
    config["dataset_root"] = resolve_path(config["dataset_root"], path)
    config["output_dir"] = resolve_path(config["output_dir"], path)
    return config


def discover_images(root: Path) -> list[tuple[Path, str]]:
    if not root.is_dir():
        raise PipelineError(f"Dataset directory not found: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(root)
            class_name = relative.parts[0] if len(relative.parts) > 1 else "unknown"
            records.append((path, class_name))
    if not records:
        raise PipelineError(f"No PNG, JPEG or WebP images found under: {root}")
    return records


def write_dictionary(output: Path, records: list[dict[str, Any]]) -> None:
    records = sorted(records, key=lambda item: int(item["concept_index"]))
    namer.save_json(output / "concept_dictionary.json", records)
    fields = ["concept_index", "concept_id", "best_name", "confidence", "definition",
              "supporting_evidence", "uncertainty"]
    with (output / "concept_dictionary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record.get(field, "") for field in fields} for record in records)


def select_samples(model, loader, top_m: int, device: str, torch):
    selected = None
    with torch.inference_mode():
        for images, paths, classes in loader:
            images = images.to(device)
            logits, concepts = model(images)
            if concepts.ndim != 2 or logits.ndim != 2 or concepts.shape[0] != images.shape[0]:
                raise PipelineError("checkpoint forward output must be logits and [batch, concepts]")
            predicted = logits.argmax(dim=1).cpu().tolist()
            concepts_cpu = concepts.detach().float().cpu()
            if selected is None:
                selected = [[] for _ in range(concepts_cpu.shape[1])]
            elif len(selected) != concepts_cpu.shape[1]:
                raise PipelineError("concept dimension changed between batches")
            count = min(top_m, concepts_cpu.shape[0])
            values, indices = torch.topk(concepts_cpu, count, dim=0)
            for concept_index in range(concepts_cpu.shape[1]):
                heap = selected[concept_index]
                for rank in range(count):
                    row = int(indices[rank, concept_index])
                    candidate = (float(values[rank, concept_index]), str(paths[row]),
                                 str(classes[row]), int(predicted[row]))
                    if len(heap) < top_m:
                        heapq.heappush(heap, candidate)
                    elif candidate[0] > heap[0][0]:
                        heapq.heapreplace(heap, candidate)
    if selected is None:
        raise PipelineError("dataset loader produced no batches")
    return [sorted(heap, reverse=True) for heap in selected]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="generate every pair without calling OpenAI")
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed concept responses")
    args = parser.parse_args()
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        if not config["checkpoint_path"].is_file():
            raise PipelineError(f"Checkpoint not found: {config['checkpoint_path']}")

        try:
            import torch
            from PIL import Image
            from torch.utils.data import DataLoader, Dataset
            from torchvision.transforms import v2
            from torchvision.utils import save_image
            from model.nfs_cbm import load_inference_checkpoint
        except ImportError as exc:
            raise PipelineError("Install the pinned packages with: python -m pip install -r requirements.txt") from exc

        device = config["device"]
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise PipelineError("CUDA was requested but is unavailable; set device to cpu or provide CUDA")
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])

        image_records = discover_images(config["dataset_root"])
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((config["image_size"], config["image_size"]), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])

        class ImageDataset(Dataset):
            def __len__(self):
                return len(image_records)

            def __getitem__(self, index):
                path, class_name = image_records[index]
                with Image.open(path) as image:
                    tensor = transform(image.convert("RGB"))
                return tensor, str(path), class_name

        loader = DataLoader(ImageDataset(), batch_size=config["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=device.startswith("cuda"))
        model = load_inference_checkpoint(config["checkpoint_path"], device)
        selected = select_samples(model, loader, config["top_m"], device, torch)

        output = config["output_dir"]
        pairs_root = output / "pairs"
        responses_root = output / "responses"
        output.mkdir(parents=True, exist_ok=True)
        run_metadata = {
            "checkpoint_path": str(config["checkpoint_path"]),
            "checkpoint_sha256": file_sha256(config["checkpoint_path"]),
            "dataset_root": str(config["dataset_root"]),
            "image_count": len(image_records),
            "concept_count": len(selected),
            "top_m": config["top_m"],
            "perturbation": config["perturbation"],
            "seed": config["seed"],
            "openai_model": config["openai"]["model"],
            "reasoning_effort": config["openai"]["reasoning_effort"],
        }
        namer.save_json(output / "run_metadata.json", run_metadata)

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not args.dry_run and not api_key:
            raise PipelineError("Set OPENAI_API_KEY before a live run")

        dictionary: list[dict[str, Any]] = []
        for concept_index, samples in enumerate(selected):
            concept_id = f"concept_{concept_index:03d}"
            pair_dir = pairs_root / concept_id
            response_dir = responses_root / concept_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            response_dir.mkdir(parents=True, exist_ok=True)
            result_path = response_dir / "result.json"
            if args.resume and result_path.is_file():
                saved = namer.read_json(result_path)
                result = saved["result"]
                dictionary.append({"concept_index": concept_index, **result})
                write_dictionary(output, dictionary)
                continue

            pairs = []
            for sample_index, (_, image_path, class_name, predicted_id) in enumerate(samples, 1):
                with Image.open(image_path) as image:
                    image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
                with torch.inference_mode():
                    _, concepts = model(image_tensor)
                    perturbed = concepts.clone()
                    perturbed[:, concept_index] = torch.clamp(
                        perturbed[:, concept_index] + float(config["perturbation"]), min=0.0
                    )
                    original_image = model.reconstruct(image_tensor, concepts)
                    perturbed_image = model.reconstruct(image_tensor, perturbed)
                original_path = pair_dir / f"sample_{sample_index:02d}_original.png"
                perturbed_path = pair_dir / f"sample_{sample_index:02d}_perturbed.png"
                save_image(original_image.detach().cpu(), original_path)
                save_image(perturbed_image.detach().cpu(), perturbed_path)
                pairs.append({"before": original_path, "after": perturbed_path,
                              "predicted_class": f"class_{predicted_id}"})

            job = {"concept_id": concept_id, "dataset_context": config["dataset_context"],
                   "pairs": pairs}
            payload, provenance = namer.build_request(job, config["openai"])
            provenance.update({"concept_index": concept_index,
                               "selected_samples": [
                                   {"activation": sample[0], "source_image": sample[1],
                                    "dataset_class": sample[2], "predicted_class_id": sample[3]}
                                   for sample in samples
                               ],
                               "perturbation": config["perturbation"]})
            namer.save_json(response_dir / "request_metadata.json", provenance)
            if args.dry_run:
                print(f"Generated {concept_id}: {len(pairs)} pair(s)")
                continue

            raw, request_id = namer.call_openai(payload, api_key, config["openai"])
            namer.save_json(response_dir / "raw_response.json", raw)
            result = namer.parse_response(raw, concept_id)
            saved = {"result": result, "provenance": {
                **provenance, "request_id": request_id, "response_id": raw.get("id", ""),
                "returned_model": raw.get("model", ""), "usage": raw.get("usage", {})}}
            namer.save_json(result_path, saved)
            dictionary.append({"concept_index": concept_index, **result})
            write_dictionary(output, dictionary)
            print(f"Named {concept_id}: {result['best_name']} ({result['confidence']})")

        if args.dry_run:
            print(f"Dry run complete: generated pairs for {len(selected)} concepts")
        else:
            if len(dictionary) != len(selected):
                raise PipelineError("run ended without a name for every concept dimension")
            print(f"Complete: named all {len(dictionary)} concept dimensions")
        return 0
    except (PipelineError, namer.NamingError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
