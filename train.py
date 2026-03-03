#!/usr/bin/env python3
"""
train.py — Main Training Script for Project-2 FSL Segmentation (3-Class)
=========================================================================

Usage:
    python train.py                              # Default config
    python train.py --config configs/custom.yaml # Custom config
    python train.py --epochs 10 --lr 1e-3        # Override specific params

All hyperparameters are loaded from YAML config.
CLI arguments override config file values.
"""

from __future__ import annotations


import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.tiger_dataset import TigerDataset
from data.episodic_sampler import EpisodicSampler
from data.augmentations import get_train_transforms, get_val_transforms
from models.segmentor import FewShotSegmentor
from models.losses import CombinedLoss
from engine.trainer import Trainer


def load_config(config_path: str = None) -> dict:
    """Load YAML config file and return flat dict."""
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default_path

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """Parse CLI arguments — these override config file values."""
    parser = argparse.ArgumentParser(description="Project-2 FSL Training")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--k_shot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--train_episodes", type=int, default=None)
    parser.add_argument("--val_episodes", type=int, default=None)
    parser.add_argument("--debug_shapes", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, warmup_factor, eta_min):
    """
    Warmup + cosine annealing scheduler.

    Epochs 0..warmup_epochs-1: linear warmup from warmup_factor × lr to lr
    Epochs warmup_epochs..total_epochs: cosine decay to eta_min
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1.0 - warmup_factor) * epoch / max(warmup_epochs, 1)
        else:
            import math
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return eta_min / optimizer.defaults['lr'] + (1 - eta_min / optimizer.defaults['lr']) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(config: dict) -> float:
    """
    Train model using the provided configuration.

    Args:
        config: Dictionary containing all configuration parameters.

    Returns:
        float: Best validation metric (mDice_soft) achieved.
    """
    training_cfg = config.get("training", {})
    episode_cfg = config.get("episode", {})
    dataset_cfg = config.get("dataset", {})
    num_classes = dataset_cfg.get("num_classes", 3)

    # ── Seed ────────────────────────────────────────────────────────────
    seed = training_cfg.get("seed", 42)
    set_seed(seed)
    device = training_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"
    print(f"Device: {device}, Seed: {seed}, Classes: {num_classes}")

    # ── Dataset ─────────────────────────────────────────────────────────
    data_root = config["data_root"]
    fold = dataset_cfg.get("fold", 0)
    min_labeled = dataset_cfg.get("min_labeled_frac", 0.9)

    train_ds = TigerDataset(data_root, split="train", fold=fold,
                            transform=get_train_transforms(), min_labeled_frac=min_labeled)
    val_ds = TigerDataset(data_root, split="val", fold=fold,
                          transform=get_val_transforms(), min_labeled_frac=min_labeled)

    # ── Episodic Samplers (3-class mixed-support) ───────────────────────
    train_sampler = EpisodicSampler(
        train_ds,
        n_way=episode_cfg.get("n_way", 3),
        k_shot=episode_cfg.get("k_shot", 5),
        episodes=episode_cfg.get("train_episodes", 250),
        seed=seed,
        min_pixels_per_class=episode_cfg.get("min_pixels_per_class", 100),
        patient_disjoint=episode_cfg.get("patient_disjoint", True),
        ignore_index=dataset_cfg.get("ignore_index", 255),
    )

    val_sampler = EpisodicSampler(
        val_ds,
        n_way=episode_cfg.get("n_way", 3),
        k_shot=episode_cfg.get("k_shot", 5),
        episodes=episode_cfg.get("val_episodes", 40),
        seed=seed + 1,
        min_pixels_per_class=episode_cfg.get("min_pixels_per_class", 100),
        patient_disjoint=episode_cfg.get("patient_disjoint", True),
        ignore_index=dataset_cfg.get("ignore_index", 255),
    )

    # ── Model ───────────────────────────────────────────────────────────
    prototype_cfg = config.get("prototype", {})
    gpa_cfg = config.get("gpa", {})
    decoder_cfg = config.get("decoder", {})
    projector_cfg = config.get("projector", {})
    backbone_cfg = config.get("backbone", {})

    model = FewShotSegmentor(
        vqvae_weights=config["vqvae_weights"],
        vqvae_modules_path=config["vqvae_modules_path"],
        feature_dim=backbone_cfg.get("feature_dim", 256),
        num_classes=num_classes,
        projector_blocks=projector_cfg.get("num_blocks", 2),
        multi_scale_grids=prototype_cfg.get("multi_scale_grids", [1, 4]),
        temperature=prototype_cfg.get("temperature", 0.1),
        learnable_temp=prototype_cfg.get("learnable_temp", True),
        aspp_rates=tuple(decoder_cfg.get("aspp_rates", [6, 12, 18])),
        use_attention=decoder_cfg.get("use_attention", True),
        norm_groups=decoder_cfg.get("norm_groups", 8),
        dropout=decoder_cfg.get("dropout", 0.1),
        freeze_backbone=backbone_cfg.get("freeze", True),
        # --- Prototype upgrades (Option B) ---
        use_task_adapt=prototype_cfg.get("use_task_adapt", False),
        adapt_layers=prototype_cfg.get("adapt_layers", 2),
        adapt_heads=prototype_cfg.get("adapt_heads", 8),
        adapt_dropout=prototype_cfg.get("adapt_dropout", 0.0),
        adapt_gamma_init=prototype_cfg.get("adapt_gamma_init", 0.0),
        adapt_use_episode_token=prototype_cfg.get("adapt_use_episode_token", True),

        use_codebook_contrastive=prototype_cfg.get("use_codebook_contrastive", False),
        codebook_size=backbone_cfg.get("codebook_size", 4096),
        topk_codes_per_class=prototype_cfg.get("topk_codes_per_class", 64),
        min_target_code_count=prototype_cfg.get("min_target_code_count", 1),
        code_eps=prototype_cfg.get("code_eps", 1e-6),
        code_context_mode=prototype_cfg.get("code_context_mode", "non_target"),
        min_selected_pixels_global=prototype_cfg.get("min_selected_pixels_global", 16),
        min_selected_pixels_cell=prototype_cfg.get("min_selected_pixels_cell", 4),
    ).to(device)

    # ── Loss (3-class, per-class focal weights) ─────────────────────────
    loss_cfg = config.get("loss", {})
    loss_fn = CombinedLoss(
        ce_weight=loss_cfg.get("ce_weight", 1.0),
        dice_weight=loss_cfg.get("dice_weight", 0.5),
        focal_weight=loss_cfg.get("focal_weight", 0.3),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        focal_class_weights=loss_cfg.get("focal_class_weights", [1.0, 1.0, 1.0]),
        ignore_index=dataset_cfg.get("ignore_index", 255),
    ).to(device)

    # ── Optimizer ───────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    lr = training_cfg.get("lr", 5e-4)
    wd = training_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    # ── Scheduler ───────────────────────────────────────────────────────
    epochs = training_cfg.get("epochs", 100)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=training_cfg.get("warmup_epochs", 3),
        total_epochs=epochs,
        warmup_factor=training_cfg.get("warmup_factor", 0.1),
        eta_min=training_cfg.get("eta_min", 1e-6),
    )

    # ── Train ───────────────────────────────────────────────────────────
    trainer_config = {
        "ignore_index": dataset_cfg.get("ignore_index", 255),
        "num_classes": num_classes,
        "grad_clip": training_cfg.get("grad_clip", 5.0),
        "print_every": config.get("logging", {}).get("print_every", 10),
    }

    trainer = Trainer(
        model=model,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_config,
        device=device,
        output_dir=config.get("output_dir", "./runs"),
    )

    trainer.train(num_epochs=epochs)

    return trainer.best_metric


def main():
    args = parse_args()
    config = load_config(args.config)

    # ── Apply CLI overrides ─────────────────────────────────────────────
    training_cfg = config.get("training", {})
    episode_cfg = config.get("episode", {})
    dataset_cfg = config.get("dataset", {})

    if args.epochs is not None: training_cfg["epochs"] = args.epochs
    if args.lr is not None: training_cfg["lr"] = args.lr
    if args.fold is not None: dataset_cfg["fold"] = args.fold
    if args.k_shot is not None: episode_cfg["k_shot"] = args.k_shot
    if args.seed is not None: training_cfg["seed"] = args.seed
    if args.output_dir is not None: config["output_dir"] = args.output_dir
    if args.device is not None: training_cfg["device"] = args.device
    if args.train_episodes is not None: episode_cfg["train_episodes"] = args.train_episodes
    if args.val_episodes is not None: episode_cfg["val_episodes"] = args.val_episodes

    train_model(config)


if __name__ == "__main__":
    main()
