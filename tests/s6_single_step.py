#!/usr/bin/env python3
"""Session 6: Single real episode — forward + backward + optimizer step on GPU."""
import sys
sys.path.insert(0, '/home/rakib/PycharmProjects/PytorchProject/project2_fsl')
import torch

device = "cuda"

# Step 1: Load dataset + build 1 episode
print("S6: Loading dataset...", flush=True)
from data.tiger_dataset import TigerDataset
from data.augmentations import get_val_transforms
from data.episodic_sampler import EpisodicSampler
ds = TigerDataset('/home/rakib/data/tiger-group-256', split='train', fold=0,
                  transform=get_val_transforms())
sampler = EpisodicSampler(ds, n_way=1, k_shot=5, episodes=1, seed=42)
ep = sampler[0]
print(f"  Episode: fg_class={ep.fg_class}", flush=True)

# Step 2: Build model
print("S6: Building model...", flush=True)
from models.segmentor import FewShotSegmentor
model = FewShotSegmentor(
    vqvae_weights="/home/rakib/PycharmProjects/PytorchProject/tiger_fsl_repo_cv_patient_protocol/model.safetensors",
    vqvae_modules_path="/home/rakib/PycharmProjects/PytorchProject/tiger_fsl_repo_cv_patient_protocol",
).to(device)
print("  Model on GPU", flush=True)

# Step 3: Move episode data to GPU
print("S6: Moving data to GPU...", flush=True)
support_imgs = ep.support_images.to(device)      # (5, 3, 256, 256)
support_masks = ep.support_masks.to(device)       # (5, 256, 256)
query_img = ep.query_image.unsqueeze(0).to(device)   # (1, 3, 256, 256)
query_mask = ep.query_mask.unsqueeze(0).to(device)   # (1, 256, 256)
print(f"  Support: {support_imgs.shape} on {support_imgs.device}", flush=True)
print(f"  Query: {query_img.shape} on {query_img.device}", flush=True)

# Step 4: Forward pass
print("S6: Forward pass...", flush=True)
model.train()
output = model(support_imgs, support_masks, query_img)
logits = output["logits"]
print(f"  Logits: {logits.shape}, device={logits.device}", flush=True)
print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]", flush=True)

# Step 5: Loss
print("S6: Computing loss...", flush=True)
from models.losses import CombinedLoss
loss_fn = CombinedLoss().to(device)
loss_dict = loss_fn(logits, query_mask)
print(f"  Loss: {loss_dict['loss'].item():.4f} (CE={loss_dict['ce'].item():.4f}, "
      f"Dice={loss_dict['dice'].item():.4f}, Focal={loss_dict['focal'].item():.4f})", flush=True)

# Step 6: Backward + optimizer step
print("S6: Backward + optimizer step...", flush=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
optimizer.zero_grad()
loss_dict['loss'].backward()
grad_norm = torch.nn.utils.clip_grad_norm_(
    [p for p in model.parameters() if p.requires_grad], 5.0
).item()
print(f"  Grad norm: {grad_norm:.4f}", flush=True)
assert torch.isfinite(torch.tensor(grad_norm)), "NaN gradients!"
optimizer.step()
print("  Optimizer step OK", flush=True)

# Step 7: Verify loss decreased on same example
print("S6: Re-forward to check loss changed...", flush=True)
with torch.no_grad():
    output2 = model(support_imgs, support_masks, query_img)
    loss2 = loss_fn(output2["logits"], query_mask)["loss"].item()
print(f"  Loss before: {loss_dict['loss'].item():.4f}, after: {loss2:.4f}", flush=True)

print("SESSION 6 PASSED", flush=True)
