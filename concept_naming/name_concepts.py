#!/usr/bin/env python3
"""GPT-5.4 backend for the NFS-CBM all-dimension naming pipeline.

Python 3.10+. Run --help for single-pair and manifest input modes.
The main entry point is pipeline.py. This module also retains a pair-level CLI.
"""
from __future__ import annotations

import argparse
import base64
import csv
import getpass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://api.openai.com/v1/responses"
VERSION = "1.1.0"
DEFAULT_MODEL = "gpt-5.4-2026-03-05"
DEFAULTS = {"model": DEFAULT_MODEL, "api_key": "", "image_detail": "high",
            "timeout_seconds": 180, "max_output_tokens": 4096, "max_retries": 3}
CONTEXTS = {
    "cub": "CUB-200-2011 contains fine-grained bird categories. The target objects are birds.",
    "cars": "Stanford Cars contains fine-grained car categories. The target objects are cars.",
    "imagenet100": "ImageNet-100 contains diverse object categories.",
}
TEXT_FIELDS = ["concept_id", "best_name", "main_changed_part", "main_attribute",
               "definition", "supporting_evidence", "uncertainty"]
LIST_FIELDS = ["candidate_names", "visual_changes", "ignored_incidental_changes"]
SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "string"} for k in TEXT_FIELDS},
        **{k: {"type": "array", "items": {"type": "string"}} for k in LIST_FIELDS},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": TEXT_FIELDS + LIST_FIELDS + ["confidence"],
    "additionalProperties": False,
}


class NamingError(Exception):
    """A concise, user-facing input, API or output error."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def save_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json_bytes(value) + b"\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NamingError(f"Cannot read JSON file: {path.name}") from exc


def load_config(path: Path | None) -> dict:
    config = dict(DEFAULTS)
    if path is None:
        local = ROOT / "config.local.json"
        path = local if local.is_file() else None
    if path is not None:
        extra = read_json(path)
        if not isinstance(extra, dict) or set(extra) - set(DEFAULTS):
            raise NamingError("Configuration must contain only keys listed in config.example.json.")
        config.update(extra)
    if config["model"] not in ("gpt-5.4", DEFAULT_MODEL):
        raise NamingError("Use gpt-5.4-2026-03-05 or gpt-5.4 for this release.")
    if config["image_detail"] not in ("high", "original"):
        raise NamingError("image_detail must be high or original.")
    for name, lower, upper in [("timeout_seconds", 1, 3600),
                               ("max_output_tokens", 512, 32768), ("max_retries", 0, 8)]:
        val = config[name]
        if type(val) is not int or not lower <= val <= upper:
            raise NamingError(f"{name} must be an integer from {lower} to {upper}.")
    if not isinstance(config["api_key"], str):
        raise NamingError("api_key must be a string.")
    return config


def load_jobs(args: argparse.Namespace) -> list[dict]:
    if args.manifest:
        if args.before or args.after or args.concept_id or args.dataset or args.dataset_context:
            raise NamingError("Use --manifest by itself; put concept metadata in the manifest.")
        jobs = read_json(args.manifest)
        base = args.manifest.resolve().parent
        if not isinstance(jobs, list) or not jobs:
            raise NamingError("The manifest must be a nonempty JSON list.")
    else:
        if not all([args.before, args.after, args.concept_id]):
            raise NamingError("Supply --before, --after, --concept-id and --dataset. See --help.")
        context = args.dataset_context or CONTEXTS.get(args.dataset, "")
        jobs = [{"concept_id": args.concept_id, "dataset_context": context,
                 "pairs": [{"before": str(args.before), "after": str(args.after),
                            "predicted_class": args.predicted_class}]}]
        base = Path.cwd()
    seen = set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) - {"concept_id", "dataset_context", "pairs"}:
            raise NamingError("Each concept requires concept_id, dataset_context and pairs.")
        for field in ("concept_id", "dataset_context"):
            if not isinstance(job.get(field), str) or not job[field].strip():
                raise NamingError(f"Each concept needs a nonempty {field} string.")
        key = (job["dataset_context"], job["concept_id"])
        if key in seen:
            raise NamingError("Repeated concept entry. Combine its samples in one pairs list.")
        seen.add(key)
        pairs = job.get("pairs")
        if not isinstance(pairs, list) or not 1 <= len(pairs) <= 20:
            raise NamingError("Supply 1 to 20 pairs per concept. This is a release input limit.")
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) - {"before", "after", "predicted_class"}:
                raise NamingError("Each pair requires before and after; predicted_class is optional.")
            for field in ("before", "after"):
                if not isinstance(pair.get(field), str) or not pair[field]:
                    raise NamingError(f"Missing image path: {field}.")
                pair[field] = (base / pair[field]).resolve()
            if not isinstance(pair.get("predicted_class", ""), str):
                raise NamingError("predicted_class must be a string.")
    return jobs


def encode_image(path: Path) -> tuple[str, dict]:
    """Normalize orientation and color; preserve pixel dimensions and full field of view."""
    try:
        if path.stat().st_size > 20 * 1024 * 1024:
            raise NamingError(f"Image exceeds the 20 MiB release limit: {path.name}")
        source = path.read_bytes()
        with Image.open(io.BytesIO(source)) as original:
            if original.format not in {"PNG", "JPEG", "WEBP"}:
                raise NamingError(f"Use a PNG, JPEG or WebP image: {path.name}")
            if getattr(original, "n_frames", 1) != 1:
                raise NamingError(f"Animated inputs are not supported: {path.name}")
            if original.width * original.height > 10_240_000 or max(original.size) > 6000:
                raise NamingError(f"Image dimensions exceed this release's limit: {path.name}")
            rgb = ImageOps.exif_transpose(original).convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
            encoded = buf.getvalue()
            meta = {"filename": path.name, "source_sha256": sha256(source),
                    "submitted_sha256": sha256(encoded), "width": rgb.width, "height": rgb.height}
    except (OSError, Image.DecompressionBombError) as exc:
        raise NamingError(f"Cannot read image: {path.name}") from exc
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii"), meta


def build_request(job: dict, config: dict) -> tuple[dict, dict]:
    template = (ROOT / "prompt_template.txt").read_text(encoding="utf-8")
    # Literal replacement preserves the JSON braces in the appendix template.
    prompt = template.replace("{dataset_context}", job["dataset_context"])
    prompt = prompt.replace("{concept_id}", job["concept_id"])
    content = [{"type": "input_text", "text": prompt}]
    if len(job["pairs"]) > 1:
        extension = (ROOT / "multisample_instruction.txt").read_text(encoding="utf-8")
        content.append({"type": "input_text", "text": extension})
    metadata = []
    for index, pair in enumerate(job["pairs"], 1):
        a, ma = encode_image(pair["before"])
        b, mb = encode_image(pair["after"])
        if (ma["width"], ma["height"]) != (mb["width"], mb["height"]):
            raise NamingError(f"Pair {index}: before and after must have equal dimensions.")
        label = f"Pair {index}."
        predicted = pair.get("predicted_class", "").strip()
        if predicted:
            label += (f" Predicted class: {predicted}. This is contextual metadata; "
                      "base the name on visible intervention changes.")
        content.extend([
            {"type": "input_text", "text": label + " Image A: reconstruction before intervention."},
            {"type": "input_image", "image_url": a, "detail": config["image_detail"]},
            {"type": "input_text", "text": f"Pair {index}. Image B: reconstruction after perturbing one concept unit."},
            {"type": "input_image", "image_url": b, "detail": config["image_detail"]},
        ])
        metadata.append({"pair_index": index, "predicted_class": predicted, "before": ma, "after": mb})
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["concept_id"] = {"type": "string", "enum": [job["concept_id"]]}
    payload = {"model": config["model"], "store": False,
               "reasoning": {"effort": "none"}, "temperature": 0,
               "max_output_tokens": config["max_output_tokens"],
               "input": [{"role": "user", "content": content}],
               "text": {"format": {"type": "json_schema", "name": "concept_name",
                                    "strict": True, "schema": schema}}}
    if len(json_bytes(payload)) > 45 * 1024 * 1024:
        raise NamingError("Encoded request exceeds 45 MiB. Export smaller input images.")
    # Save exactly the text sent to the model, with image payloads replaced by checksums.
    record = {"script_version": VERSION, "endpoint": ENDPOINT,
              "requested_model": config["model"], "dataset_context": job["dataset_context"],
              "concept_id": job["concept_id"], "pair_count": len(metadata), "pairs": metadata,
              "prompt_template_sha256": sha256(template.encode()),
              "text_blocks": [c["text"] for c in content if c["type"] == "input_text"],
              "settings": {k: v for k, v in payload.items() if k != "input"},
              "request_sha256": sha256(json_bytes(payload))}
    return payload, record


def call_openai(payload: dict, api_key: str, config: dict) -> tuple[dict, str]:
    request = urllib.request.Request(ENDPOINT, data=json_bytes(payload), method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    for attempt in range(config["max_retries"] + 1):
        try:
            with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
                body = json.load(response)
                return body, response.headers.get("x-request-id", "")
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < config["max_retries"]:
                delay = min(2 ** attempt, 30)
                try:
                    retry_after = float(exc.headers.get("retry-after", "0"))
                    if math.isfinite(retry_after):
                        delay = max(delay, min(max(retry_after, 0), 60))
                except (TypeError, ValueError):
                    pass
                time.sleep(delay)
                continue
            request_id = exc.headers.get("x-request-id", "unknown")
            hints = {401: "Check your OpenAI API key.",
                     403: "Check access to this model in your OpenAI project.",
                     404: "Check access to the configured GPT-5.4 model.",
                     429: "Check API quota and rate limits, then resume the run."}
            raise NamingError(f"OpenAI HTTP {exc.code}. " + hints.get(exc.code, "Check request settings.")
                              + f" Request ID: {request_id}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            # A timeout can follow server execution. Avoid automatic duplicate billing.
            raise NamingError("Connection failed or timed out. The request may have reached OpenAI; "
                              "check usage before rerunning.") from exc
        except (ValueError, OSError) as exc:
            raise NamingError("Could not decode the OpenAI response.") from exc
    raise NamingError("Retry limit reached.")


def parse_response(body: dict, concept_id: str) -> dict:
    if not isinstance(body, dict) or body.get("status") != "completed":
        raise NamingError("OpenAI did not complete the response. Inspect raw_response.json.")
    texts = []
    for item in body.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "refusal":
                raise NamingError("The model refused this request. Inspect raw_response.json.")
            if part.get("type") == "output_text":
                texts.append(part.get("text", ""))
    try:
        result = json.loads("".join(texts))
    except ValueError as exc:
        raise NamingError("Response is not valid JSON. Inspect raw_response.json.") from exc
    if not isinstance(result, dict) or set(result) != set(SCHEMA["required"]):
        raise NamingError("The response fields do not match the naming schema.")
    if any(not isinstance(result[k], str) for k in TEXT_FIELDS):
        raise NamingError("Naming text fields must be strings.")
    if any(not isinstance(result[k], list) or any(not isinstance(x, str) for x in result[k])
           for k in LIST_FIELDS):
        raise NamingError("Naming list fields must contain strings.")
    if result["concept_id"] != concept_id or result["confidence"] not in {"high", "medium", "low"}:
        raise NamingError("Invalid concept identifier or confidence value in response.")
    if not result["best_name"].strip():
        raise NamingError("The model returned an empty name.")
    return result


def write_summary(output: Path, records: list[dict]) -> None:
    fields = ["concept_id", "dataset_context", "best_name", "confidence", "pair_count",
              "definition", "supporting_evidence", "uncertainty"]
    with (output / "names.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {**rec["result"], "dataset_context": rec["provenance"]["dataset_context"],
                   "pair_count": rec["provenance"]["pair_count"]}
            writer.writerow({k: row[k] for k in fields})
    def cell(value: Any) -> str:
        return str(value).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")
    lines = ["# Concept naming results", "", "Names summarize observed intervention effects.", "",
             "| Concept | Dataset context | Name | Confidence | Image pairs |",
             "| --- | --- | --- | --- | --- |"]
    for rec in records:
        r, p = rec["result"], rec["provenance"]
        lines.append("| " + " | ".join(map(cell, [r["concept_id"], p["dataset_context"],
                     r["best_name"], r["confidence"], p["pair_count"]])) + " |")
    (output / "names.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    jobs = load_jobs(args)
    # Validate all inputs before any request can incur a charge.
    for job in jobs:
        build_request(job, config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    api_key = ""
    records = []
    for job in jobs:
        payload, record = build_request(job, config)
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", job["concept_id"])[:48] or "concept"
        folder = output / (safe_id + "_" + record["request_sha256"][:16])
        folder.mkdir(exist_ok=True)
        result_path = folder / "result.json"
        if result_path.exists():
            if not args.resume:
                raise NamingError("A matching result exists. Use --resume to reuse it, or a new --output.")
            saved = read_json(result_path)
            if saved.get("provenance", {}).get("request_sha256") != record["request_sha256"]:
                raise NamingError("Cached result does not match the request.")
            # Revalidate cached result before including it in the summary.
            parse_response({"status": "completed", "output": [{"content": [
                {"type": "output_text", "text": json.dumps(saved.get("result"))}]}]}, job["concept_id"])
            records.append(saved)
            print(f"Reused concept {job['concept_id']}.")
            continue
        save_json(folder / "request_metadata.json", record)
        if args.dry_run:
            print(f"Validated concept {job['concept_id']}: {record['pair_count']} image pair(s). No API call.")
            continue
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip() or config["api_key"].strip()
            if not api_key and sys.stdin.isatty():
                api_key = getpass.getpass("OpenAI API key (hidden): ").strip()
            if not api_key or api_key.startswith("YOUR_"):
                raise NamingError("Set OPENAI_API_KEY or fill api_key in config.local.json.")
        raw, request_id = call_openai(payload, api_key, config)
        save_json(folder / "raw_response.json", raw)
        result = parse_response(raw, job["concept_id"])
        record.update({"utc_time": datetime.now(timezone.utc).isoformat(),
                       "response_id": raw.get("id", ""), "request_id": request_id,
                       "returned_model": raw.get("model", ""), "usage": raw.get("usage", {}),
                       "python_version": sys.version.split()[0], "pillow_version": Image.__version__})
        saved = {"result": result, "provenance": record}
        save_json(result_path, saved)
        records.append(saved)
        write_summary(output, records)
        print(f"Concept {job['concept_id']}: {result['best_name']} ({result['confidence']}).")
    if records:
        write_summary(output, records)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", type=Path, help="Image A: reconstruction before intervention")
    parser.add_argument("--after", type=Path, help="Image B: reconstruction after concept perturbation")
    parser.add_argument("--concept-id", help="Identifier of the perturbed concept")
    parser.add_argument("--dataset", choices=list(CONTEXTS), help="Dataset-level object context")
    parser.add_argument("--dataset-context", help="Explicit dataset context, overriding --dataset")
    parser.add_argument("--predicted-class", default="", help="Class predicted by the trained classifier")
    parser.add_argument("--manifest", type=Path, help="JSON list of concepts and their ordered image pairs")
    parser.add_argument("--config", type=Path, help="Configuration file; default: adjacent config.local.json")
    parser.add_argument("--output", type=Path, default=Path("runs"), help="Output directory (default: runs)")
    parser.add_argument("--dry-run", action="store_true", help="Validate images and save metadata without an API call")
    parser.add_argument("--resume", action="store_true", help="Reuse completed results with the same request checksum")
    try:
        return run(parser.parse_args())
    except (NamingError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Completed results remain available for --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
