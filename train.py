#!/usr/bin/env python3
"""Partial training entry for NFS-CBM.

The supporting model, data, loss, and checkpoint packages will be released
after publication. This file documents the training orchestration used by the
method and is not the complete reproducibility package.
"""
from argparse import ArgumentParser
from pathlib import Path
import random

import torch
from torch.optim import AdamW

# These supporting modules are part of the planned full release.
from nfs_cbm.data import build_dataloaders
from nfs_cbm.losses import classification_loss, diffusion_loss, factorization_loss
from nfs_cbm.model import build_nfs_cbm
from nfs_cbm.training import save_checkpoint


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/nfs_cbm.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/nfs_cbm"))
    parser.add_argument("--concept-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--condition-dropout", type=float, default=0.15)
    parser.add_argument("--lambda-diff", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=0.2)
    parser.add_argument("--lambda-nmf", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader, validation_loader = build_dataloaders(
        args.config, batch_size=args.batch_size
    )
    model = build_nfs_cbm(
        args.config,
        concept_dim=args.concept_dim,
        condition_dropout=args.condition_dropout,
    ).to(args.device)

    optimizer = AdamW(model.trainable_parameters(), weight_decay=0.01)
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)

            prediction, concepts, diffusion_state = model(images)
            loss_cls = classification_loss(prediction, labels)
            loss_nmf = factorization_loss(model.fsb, concepts)
            loss_diff = diffusion_loss(model.cdg, diffusion_state, concepts)
            loss = (
                args.lambda_diff * loss_diff
                + args.lambda_cls * loss_cls
                + args.lambda_nmf * loss_nmf
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()

        save_checkpoint(model, optimizer, validation_loader, epoch, args.output)


if __name__ == "__main__":
    train(parse_args())
