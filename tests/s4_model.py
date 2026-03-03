#!/usr/bin/env python3
"""Session 4: Model on GPU — build model, forward pass with random data."""
import sys
sys.path.insert(0, '/home/rakib/PycharmProjects/PytorchProject/project2_fsl')
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"S4: Device = {device}", flush=True)

print("S4: Building model...", flush=True)
from models.segmentor import FewShotSegmentor
model = FewShotSegmentor(
    vqvae_weights="/home/rakib/PycharmProjects/PytorchProject/tiger_fsl_repo_cv_patient_protocol/model.safetensors",
    vqvae_modules_path="/home/rakib/PycharmProjects/PytorchProject/tiger_fsl_repo_cv_patient_protocol",
    feature_dim=256, num_classes=2,
).to(device)
print("  Model built and on GPU", flush=True)

# Verify all parameters are on correct device
for name, param in model.named_parameters():
    if param.device.type != device:
        print(f"  WARNING: {name} is on {param.device}!", flush=True)
        break
else:
    print("  All parameters on correct device", flush=True)

print("S4: Forward pass with random data...", flush=True)
support_imgs = torch.randn(5, 3, 256, 256, device=device)
support_masks = torch.zeros(5, 256, 256, dtype=torch.long, device=device)
support_masks[:, :128, :] = 1  # Top half = foreground
query_img = torch.randn(1, 3, 256, 256, device=device)

with torch.no_grad():
    output = model(support_imgs, support_masks, query_img)

logits = output["logits"]
print(f"  Logits shape: {logits.shape}", flush=True)
print(f"  Logits device: {logits.device}", flush=True)
print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]", flush=True)
print(f"  Prototypes: {[(k, v.shape) for k, v in output['prototypes'].items()]}", flush=True)
assert logits.shape == (1, 2, 256, 256), f"Expected (1,2,256,256), got {logits.shape}"
assert logits.device.type == device
print("SESSION 4 PASSED", flush=True)
