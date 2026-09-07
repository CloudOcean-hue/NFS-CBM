"""Core NFS-CBM inference components used by the naming pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FactorizedSemanticBottleneck(nn.Module):
    """Non-negative ridge-style factor estimator from Eqs. 2 and 3."""

    def __init__(self, feature_dim: int, concept_dim: int = 128, epsilon: float = 1e-4):
        super().__init__()
        if feature_dim < 1 or concept_dim < 1 or epsilon <= 0:
            raise ValueError("feature_dim, concept_dim and epsilon must be positive")
        self.feature_dim = feature_dim
        self.concept_dim = concept_dim
        self.epsilon = epsilon
        self.raw_basis = nn.Parameter(torch.empty(concept_dim, feature_dim))
        nn.init.orthogonal_(self.raw_basis)

    def basis(self) -> Tensor:
        return F.softplus(self.raw_basis)

    def forward(self, features: Tensor) -> Tensor:
        positive = F.softplus(features)
        basis = self.basis()
        gram = basis @ basis.transpose(0, 1)
        gram = gram + self.epsilon * torch.eye(
            self.concept_dim, dtype=gram.dtype, device=gram.device
        )
        cross = positive @ basis.transpose(0, 1)
        coefficients = torch.linalg.solve(gram, cross.transpose(0, 1)).transpose(0, 1)
        return F.softplus(coefficients)

    def decorrelation_loss(self) -> Tensor:
        gram = self.basis() @ self.basis().transpose(0, 1)
        eye = torch.eye(self.concept_dim, dtype=gram.dtype, device=gram.device)
        return torch.linalg.matrix_norm(gram * (1.0 - eye), ord="fro") ** 2


class NFSCBMInferenceModel(nn.Module):
    """Exportable NFS-CBM inference wrapper.

    ``backbone`` maps RGB images in [0, 1] to a two-dimensional feature tensor.
    Include the experiment's normalization inside ``backbone`` before export.
    ``cdg`` maps ``(images, concepts)`` to reconstructions in [0, 1].
    """

    def __init__(self, backbone: nn.Module, bottleneck: FactorizedSemanticBottleneck,
                 classifier: nn.Linear, cdg: nn.Module):
        super().__init__()
        if classifier.in_features != bottleneck.concept_dim:
            raise ValueError("classifier input must equal the concept dimension")
        self.backbone = backbone
        self.bottleneck = bottleneck
        self.classifier = classifier
        self.cdg = cdg

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.backbone(images)
        if features.ndim != 2:
            raise RuntimeError("backbone must return [batch, feature_dim]")
        concepts = self.bottleneck(features)
        return self.classifier(concepts), concepts

    @torch.jit.export
    def reconstruct(self, images: Tensor, concepts: Tensor) -> Tensor:
        return self.cdg(images, concepts).clamp(0.0, 1.0)

    @torch.jit.export
    def contributions(self, concepts: Tensor, class_ids: Tensor) -> Tensor:
        weights = self.classifier.weight.index_select(0, class_ids)
        return weights * concepts

    def export_inference_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(torch.jit.script(self.eval()), str(destination))
        return destination


def load_inference_checkpoint(path: str | Path, device: str = "cpu"):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"NFS-CBM checkpoint not found: {checkpoint}")
    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()
    if not hasattr(model, "reconstruct"):
        raise RuntimeError("checkpoint does not expose reconstruct(images, concepts)")
    return model
