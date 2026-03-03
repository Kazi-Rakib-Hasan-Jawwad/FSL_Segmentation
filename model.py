import torch
import torch.nn as nn
import torch.optim as toptim
import lightning as pl
import torch.nn.functional as F
from modules.ema import EMAWarmUp
from modules.vqvae import Encoder, Decoder, VectorQuantizer
from argparse import Namespace
from huggingface_hub import PyTorchModelHubMixin

class VQVAE(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quantizer = VectorQuantizer(config)
        self.proj = nn.Conv2d(config.encoder.get("downsample_blocks")["channels"][-1],
                config.quantizer.get("embedding_dim"), kernel_size=1, stride=1, padding=0)
        self.proj_t = nn.Conv2d(config.quantizer.get("embedding_dim"),
                config.decoder.get("upsample_blocks")["channels"][0], kernel_size=1,stride=1, padding=0)

    def forward(self, X):
        if isinstance(X, tuple) or isinstance(X, list):
            image, _ = X
        elif isinstance(X, dict):
            image = X["image"]
        z_e = self.encoder(image)
        z_e = self.proj(z_e)
        z_p, min_embeddings = self.quantizer(z_e)
        output = self.decoder(self.proj_t(z_e + (z_p - z_e).detach()))
        e_mean = torch.mean(min_embeddings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return output, z_e, z_p, perplexity

class VQVAEPreTraining(pl.LightningModule, PyTorchModelHubMixin,
                      repo_url="https://huggingface.co/yuando/HSVQVAE"):
    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.model = VQVAE(config)
        if self.config.use_ema:
            self.ema = EMAWarmUp(self.model, **self.config.ema_kwargs)

    def forward(self, X):
        return self.model(X)
    
    def loss(self, X, outputs):
        if isinstance(X, tuple) or isinstance(X, list):
            image, _ = X
        elif isinstance(X, dict):
            image = X["image"]
        B, _, _, _ = image.shape
        X_hat, z_e, z_q, perplexity = outputs
        negll = F.mse_loss(image, X_hat)
        embed_loss = F.mse_loss(z_e.detach(), z_q)
        comm_loss = F.mse_loss(z_e, z_q.detach()) * self.config.beta
        loss = negll + embed_loss + comm_loss
        return { "loss" : loss, "negll" : negll, "embed_loss" : embed_loss, "comm_loss" : comm_loss, "perplexity" : perplexity }

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        results = self.loss(batch, outputs)
        for key, value in results.items():
            self.log(key, value.item())

        if batch_idx % self.config.log_image_every_n_steps == 0:
            self.logger.log_images(outputs[0].cpu(), self.global_step)
            if isinstance(batch, tuple) or isinstance(batch, list):
                self.logger.log_images(batch[0].cpu(), self.global_step)
            elif isinstance(batch, dict):
                self.logger.log_images(batch["image"].cpu(), self.global_step)
        return results["loss"]

    def on_before_zero_grad(self, *args, **kwargs):
        if self.config.use_ema:
            self.ema(self.model)

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        results = self.loss(batch, outputs)
        for key, value in results.items():
            self.log("-".join(["validation", key]), value.item())
        if batch_idx % self.config.log_image_every_n_steps == 0:
            self.logger.log_images(outputs[0].cpu(), self.global_step)
            if isinstance(batch, tuple) or isinstance(batch, list):
                self.logger.log_images(batch[0].cpu(), self.global_step)
            elif isinstance(batch, dict):
                self.logger.log_images(batch["image"].cpu(), self.global_step)
        return results["loss"]

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimiser = getattr(toptim, self.config.optimizer_name)(self.model.parameters(), **self.config.optimizer_kwargs)
        scheduler = getattr(toptim.lr_scheduler, self.config.scheduler_name)(optimiser, **self.config.scheduler_kwargs)

        return {"optimizer" : optimiser, "lr_scheduler" : scheduler, "monitor" : "validation-loss"}

