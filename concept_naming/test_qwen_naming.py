"""Offline contract tests for the open-weight Qwen3-VL naming backend."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import name_concepts_qwen as qwen


class QwenNamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_discovers_pipeline_pairs(self):
        folder = self.root / "concept_000"
        folder.mkdir()
        for suffix in ("original", "perturbed"):
            Image.new("RGB", (32, 32), "gray").save(folder / f"sample_01_{suffix}.png")
        jobs = qwen.discover_pair_jobs(self.root, "Synthetic birds.", 5)
        self.assertEqual(jobs[0]["concept_id"], "concept_000")
        self.assertEqual(len(jobs[0]["pairs"]), 1)

    def test_missing_pair_is_rejected(self):
        folder = self.root / "concept_000"
        folder.mkdir()
        Image.new("RGB", (32, 32), "gray").save(folder / "sample_01_original.png")
        with self.assertRaises(qwen.QwenNamingError):
            qwen.discover_pair_jobs(self.root, "Synthetic birds.", 5)

    def test_messages_preserve_image_order(self):
        before, after = self.root / "a.png", self.root / "b.png"
        Image.new("RGB", (32, 32), "black").save(before)
        Image.new("RGB", (32, 32), "white").save(after)
        job = {"concept_id": "concept_000", "dataset_context": "Synthetic birds.",
               "pairs": [{"before": before, "after": after, "predicted_class": ""}]}
        messages, metadata = qwen.build_messages(job, 768)
        content = messages[0]["content"]
        self.assertEqual([item["type"] for item in content].count("image"), 2)
        self.assertIn("Image A", content[1]["text"])
        self.assertIn("Image B", content[3]["text"])
        self.assertEqual(metadata["pair_count"], 1)
        self.assertNotEqual(metadata["pairs"][0]["before"]["source_sha256"],
                            metadata["pairs"][0]["after"]["source_sha256"])

    def test_messages_reject_mismatched_dimensions(self):
        before, after = self.root / "a.png", self.root / "b.png"
        Image.new("RGB", (32, 32), "black").save(before)
        Image.new("RGB", (31, 32), "white").save(after)
        job = {"concept_id": "concept_000", "dataset_context": "Synthetic birds.",
               "pairs": [{"before": before, "after": after, "predicted_class": ""}]}
        with self.assertRaises(qwen.QwenNamingError):
            qwen.build_messages(job, 768)

    def test_json_fences_and_numeric_confidence(self):
        value = {field: "synthetic" for field in qwen.common.TEXT_FIELDS}
        value.update({field: [] for field in qwen.common.LIST_FIELDS})
        value.update(concept_id="concept_000", best_name="synthetic concept",
                     candidate_names=["synthetic concept"], confidence=0.8)
        parsed = qwen.extract_json("```json\n" + json.dumps(value) + "\n```")
        result = qwen.normalize_result(parsed, "concept_000")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["best_name"], "synthetic concept")


if __name__ == "__main__":
    unittest.main()
