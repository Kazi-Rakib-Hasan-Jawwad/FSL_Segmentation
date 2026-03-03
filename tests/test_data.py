#!/usr/bin/env python3
"""Quick data pipeline test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)  # Force flush

print("Step 1: Imports...", flush=True)
from data.tiger_dataset import TigerDataset
from data.episodic_sampler import EpisodicSampler
from data.augmentations import get_val_transforms
print("  OK", flush=True)

print("Step 2: Load dataset...", flush=True)
ds = TigerDataset('/home/rakib/data/tiger-group-256', split='train', fold=0, transform=get_val_transforms())

print("Step 3: Load 1 sample...", flush=True)
img, mask, meta = ds[0]
print(f"  Image: {img.shape}, Mask: {mask.shape}", flush=True)
print(f"  Meta: slide={meta.slide_id[:30]}, fracs={meta.class_fracs}", flush=True)

print("Step 4: Episode sampler...", flush=True)
sampler = EpisodicSampler(ds, n_way=1, k_shot=5, episodes=3, seed=42, patient_disjoint=True)
ep = sampler[0]
print(f"  Support: {ep.support_images.shape}, Query: {ep.query_image.shape}", flush=True)
print(f"  FG class: {ep.fg_class}, Mask vals: {ep.query_mask.unique().tolist()}", flush=True)

print("Step 5: Verify patient disjointness...", flush=True)
assert ep.query_slide_id not in ep.support_slide_ids
print("  OK", flush=True)

print("\nALL TESTS PASSED", flush=True)
