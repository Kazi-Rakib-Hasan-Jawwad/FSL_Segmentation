#!/usr/bin/env python3
"""Session 5: Loss on GPU — test with random logits + targets."""
import sys
sys.path.insert(0, '/home/rakib/PycharmProjects/PytorchProject/project2_fsl')
import torch

device = "cuda"
print("S5: Building loss function...", flush=True)
from models.losses import CombinedLoss
loss_fn = CombinedLoss(ce_weight=1.0, dice_weight=0.5, focal_weight=0.3).to(device)
print("  Loss on GPU", flush=True)

print("S5: Testing with random data...", flush=True)
logits = torch.randn(1, 2, 256, 256, device=device, requires_grad=True)
target = torch.randint(0, 2, (1, 256, 256), device=device)
target[0, :10, :10] = 255  # Some ignore pixels

result = loss_fn(logits, target)
print(f"  Total loss: {result['loss'].item():.4f}", flush=True)
print(f"  CE: {result['ce'].item():.4f}", flush=True)
print(f"  Dice: {result['dice'].item():.4f}", flush=True)
print(f"  Focal: {result['focal'].item():.4f}", flush=True)
print(f"  Loss device: {result['loss'].device}", flush=True)

# Test backward
result['loss'].backward()
assert logits.grad is not None, "No gradients!"
print(f"  Gradients OK: {logits.grad.shape}", flush=True)
assert not torch.isnan(result['loss']), "NaN loss!"
print("SESSION 5 PASSED", flush=True)
