#!/usr/bin/env python3
"""Session 2: Data pipeline — load dataset and 1 sample."""
import sys
sys.path.insert(0, '/home/rakib/PycharmProjects/PytorchProject/project2_fsl')

print("S2: Importing data modules...", flush=True)
from data.tiger_dataset import TigerDataset
from data.augmentations import get_val_transforms
print("  Imports OK", flush=True)

print("S2: Loading dataset...", flush=True)
ds = TigerDataset('/home/rakib/data/tiger-group-256', split='train', fold=0,
                  transform=get_val_transforms())
print(f"  Loaded: {len(ds)} patches", flush=True)

print("S2: Loading 1 sample...", flush=True)
img, mask, meta = ds[0]
print(f"  Image: {img.shape}, dtype={img.dtype}", flush=True)
print(f"  Mask: {mask.shape}, dtype={mask.dtype}, unique={mask.unique().tolist()}", flush=True)
print(f"  Meta: slide={meta.slide_id[:40]}, fracs={meta.class_fracs}", flush=True)

print("SESSION 2 PASSED", flush=True)
