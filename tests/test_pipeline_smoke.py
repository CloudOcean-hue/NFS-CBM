import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn

from model.nfs_cbm import FactorizedSemanticBottleneck, NFSCBMInferenceModel


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(3, 4)

    def forward(self, images):
        return self.projection(self.pool(images).flatten(1))


class DummyCDG(nn.Module):
    def forward(self, images, concepts):
        shift = concepts.mean(dim=1).view(-1, 1, 1, 1) * 0.01
        return (images + shift).clamp(0.0, 1.0)


class PipelineSmokeTest(unittest.TestCase):
    def test_all_dimensions_generate_pairs_without_api(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            checkpoint = temp_path / "nfs_cbm_inference.pt"
            dataset = temp_path / "data" / "bird"
            output = temp_path / "output"
            dataset.mkdir(parents=True)
            for index, color in enumerate((40, 90, 140, 190)):
                Image.new("RGB", (48, 48), (color, 80, 120)).save(dataset / f"sample_{index}.png")

            model = NFSCBMInferenceModel(
                DummyBackbone(),
                FactorizedSemanticBottleneck(feature_dim=4, concept_dim=3),
                nn.Linear(3, 2),
                DummyCDG(),
            )
            model.export_inference_checkpoint(checkpoint)
            config = {
                "checkpoint_path": str(checkpoint),
                "dataset_root": str(dataset.parent),
                "output_dir": str(output),
                "dataset_context": "Synthetic bird fixture.",
                "image_size": 48,
                "batch_size": 2,
                "top_m": 2,
                "perturbation": 0.5,
                "device": "cpu",
                "seed": 42,
                "openai": {
                    "model": "gpt-5.4-2026-03-05",
                    "image_detail": "high",
                    "timeout_seconds": 30,
                    "max_output_tokens": 1024,
                    "max_retries": 0,
                },
            }
            config_path = temp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            process = subprocess.run(
                [sys.executable, str(root / "concept_naming" / "pipeline.py"),
                 "--config", str(config_path), "--dry-run"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            metadata = json.loads((output / "run_metadata.json").read_text())
            self.assertEqual(metadata["concept_count"], 3)
            self.assertEqual(len(list((output / "pairs").glob("*/*_original.png"))), 6)
            self.assertEqual(len(list((output / "pairs").glob("*/*_perturbed.png"))), 6)
            self.assertFalse((output / "concept_dictionary.json").exists())


if __name__ == "__main__":
    unittest.main()
