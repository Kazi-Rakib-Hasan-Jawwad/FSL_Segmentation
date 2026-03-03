#!/usr/bin/env python3
"""Session 3: Episodic sampler — build 3 episodes, check shapes + disjointness."""
import sys
sys.path.insert(0, '/home/rakib/PycharmProjects/PytorchProject/project2_fsl')

print("S3: Loading dataset...", flush=True)
from data.tiger_dataset import TigerDataset
from data.augmentations import get_val_transforms
ds = TigerDataset('/home/rakib/data/tiger-group-256', split='train', fold=0,
                  transform=get_val_transforms())
print(f"  {len(ds)} patches loaded", flush=True)

print("S3: Building episodic sampler...", flush=True)
from data.episodic_sampler import EpisodicSampler
sampler = EpisodicSampler(ds, n_way=1, k_shot=5, episodes=3, seed=42, patient_disjoint=True)
print(f"  Sampler OK: {len(sampler)} episodes", flush=True)

print("S3: Building episode 0...", flush=True)
ep = sampler[0]
print(f"  Support images: {ep.support_images.shape}", flush=True)
print(f"  Support masks:  {ep.support_masks.shape}", flush=True)
print(f"  Query image:    {ep.query_image.shape}", flush=True)
print(f"  Query mask:     {ep.query_mask.shape}", flush=True)
print(f"  FG class:       {ep.fg_class}", flush=True)
print(f"  Mask values:    {ep.query_mask.unique().tolist()}", flush=True)

print("S3: Checking patient disjointness...", flush=True)
assert ep.query_slide_id not in ep.support_slide_ids, \
    f"FAIL: query slide {ep.query_slide_id} found in support slides"
print(f"  Query slide:   {ep.query_slide_id[:40]}", flush=True)
print(f"  Support slides: {[s[:20] for s in ep.support_slide_ids]}", flush=True)
print("  Disjointness OK", flush=True)

print("SESSION 3 PASSED", flush=True)
