#!/usr/bin/env python3
"""Session 7: Short training — 3 epochs × 5 episodes with full engine."""
import sys
sys.path.insert(0, '/home/rakib/PycharmProjects/PytorchProject/project2_fsl')
import torch

device = "cuda"

# Step 1: Dataset + sampler
print("S7: Building data...", flush=True)
from data.tiger_dataset import TigerDataset
from data.augmentations import get_val_transforms
from data.episodic_sampler import EpisodicSampler

ds = TigerDataset('/home/rakib/data/tiger-group-256', split='train', fold=0,
                  transform=get_val_transforms())
val_ds = TigerDataset('/home/rakib/data/tiger-group-256', split='val', fold=0,
                      transform=get_val_transforms())

train_sampler = EpisodicSampler(ds, n_way=1, k_shot=5, episodes=5, seed=42)
val_sampler = EpisodicSampler(val_ds, n_way=1, k_shot=5, episodes=3, seed=43)
print(f"  Train: {len(train_sampler)} episodes, Val: {len(val_sampler)} episodes", flush=True)

# Step 2: Model
print("S7: Building model...", flush=True)
from models.segmentor import FewShotSegmentor
model = FewShotSegmentor(
    vqvae_weights="/home/rakib/PycharmProjects/PytorchProject/tiger_fsl_repo_cv_patient_protocol/model.safetensors",
    vqvae_modules_path="/home/rakib/PycharmProjects/PytorchProject/tiger_fsl_repo_cv_patient_protocol",
).to(device)

# Step 3: Loss + optimizer
from models.losses import CombinedLoss
loss_fn = CombinedLoss().to(device)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

# Step 4: Manual training loop (not using engine, to keep it simple)
print("S7: Training 3 epochs...", flush=True)
from torch.utils.data import DataLoader
from utils.metrics import compute_metrics

for epoch in range(3):
    model.train()
    train_sampler.reset_seed(epoch)
    loader = DataLoader(train_sampler, batch_size=None, shuffle=False, num_workers=0)
    
    epoch_loss = 0.0
    for step, ep in enumerate(loader):
        s_imgs = ep.support_images.to(device)
        s_masks = ep.support_masks.to(device)
        q_img = ep.query_image.unsqueeze(0).to(device)
        q_mask = ep.query_mask.unsqueeze(0).to(device)
        
        output = model(s_imgs, s_masks, q_img)
        loss_dict = loss_fn(output["logits"], q_mask)
        
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        ).item()
        optimizer.step()
        
        epoch_loss += loss_dict["loss"].item()
    
    avg_loss = epoch_loss / len(train_sampler)
    
    # Quick val
    model.eval()
    val_sampler.reset_seed(999)
    val_loader = DataLoader(val_sampler, batch_size=None, shuffle=False, num_workers=0)
    val_metrics_all = {}
    n_val = 0
    with torch.no_grad():
        for ep in val_loader:
            s_imgs = ep.support_images.to(device)
            s_masks = ep.support_masks.to(device)
            q_img = ep.query_image.unsqueeze(0).to(device)
            q_mask = ep.query_mask.unsqueeze(0).to(device)
            
            output = model(s_imgs, s_masks, q_img)
            metrics = compute_metrics(output["logits"], q_mask)
            for k, v in metrics.items():
                val_metrics_all[k] = val_metrics_all.get(k, 0) + v
            n_val += 1
    
    val_avg = {k: v / n_val for k, v in val_metrics_all.items()}
    print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_fgIoU={val_avg.get('fg_IoU',0):.4f}, "
          f"val_softDice={val_avg.get('soft_dice_fg',0):.4f}", flush=True)

print("SESSION 7 PASSED", flush=True)
