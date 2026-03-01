import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvTranspose3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(ConvTranspose3DBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv_transpose(x)))

class DesignVAE(nn.Module):
    def __init__(self, input_shape=(64, 64, 64), latent_dim=32):
        super(DesignVAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        
        # GPU Conv Layers (Work fine on Blackwell)
        self.encoder_gpu = nn.Sequential(
            Conv3DBlock(1, 16, stride=2),   # 64 -> 32
            Conv3DBlock(16, 32, stride=2),  # 32 -> 16
            Conv3DBlock(32, 64, stride=2),  # 16 -> 8
            Conv3DBlock(64, 128, stride=2), # 8 -> 4
        )
        self.decoder_gpu = nn.Sequential(
            ConvTranspose3DBlock(128, 64),  # 4 -> 8
            ConvTranspose3DBlock(64, 32),   # 8 -> 16
            ConvTranspose3DBlock(32, 16),   # 16 -> 32
            ConvTranspose3DBlock(16, 8),    # 32 -> 64
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
        )

        # CPU Linear Layers (Bypass Blackwell cuBLAS GEMM bug)
        self.fc_encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, 512),
            nn.ReLU(),
        ).cpu().float()
        
        self.fc_mu = nn.Linear(512, latent_dim).cpu().float()
        self.fc_logvar = nn.Linear(512, latent_dim).cpu().float()

        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 4 * 4 * 4),
            nn.ReLU(),
        ).cpu().float()

        self.performance_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 3)
        ).cpu().float()
        
        self.parameter_head = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 2)
        ).cpu().float()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.fc_encode.cpu().float()
        self.fc_mu.cpu().float()
        self.fc_logvar.cpu().float()
        self.fc_decode.cpu().float()
        self.performance_head.cpu().float()
        self.parameter_head.cpu().float()
        return self

    def cuda(self, device=None):
        return self.to('cuda')

    def encode(self, x):
        h = self.encoder_gpu(x)
        # Ensure conversion to float32 on CPU for linear layers
        h_cpu = self.fc_encode(h.cpu().float())
        mu = self.fc_mu(h_cpu)
        logvar = self.fc_logvar(h_cpu)
        return mu, logvar

    def decode_logits(self, z):
        z_cpu = z.cpu().float()
        h_cpu = self.fc_decode(z_cpu)
        # Convert back to GPU device/dtype
        target_device = next(self.decoder_gpu.parameters()).device
        h_gpu = h_cpu.to(target_device).to(torch.bfloat16)
        h_gpu = h_gpu.view(-1, 128, 4, 4, 4)
        return self.decoder_gpu(h_gpu)

    def decode(self, z):
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        x_recon = self.decode_logits(z)
        perf_pred = self.performance_head(z)
        params_pred = self.parameter_head(z)
        return x_recon, mu, logvar, perf_pred, params_pred

    def predict_performance(self, z):
        return self.performance_head(z.cpu().float())

    def predict_parameters(self, z):
        return self.parameter_head(z.cpu().float())

class VAETrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-3, beta=1.0):
        self.model = model.cuda()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scaler = GradScaler('cuda')
        self.beta = beta
        self.writer = SummaryWriter(log_dir='./logs')
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch, kl_weight=1.0):
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            geom = batch['geometry'].to(self.device, non_blocking=True)
            perf = batch['performance'].cpu().float()
            pars = batch['parameters'].cpu().float()
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Conv layers on GPU (BF16)
            with autocast('cuda', dtype=torch.bfloat16):
                x_rec, mu, logvar, p_pred, pr_pred = self.model(geom)
                
                # BCE on GPU
                loss_rec = F.binary_cross_entropy_with_logits(
                    x_rec, geom, pos_weight=torch.tensor([30.0], device=self.device)
                )
            
            # KL and Heads on CPU (Force FP32)
            loss_kl = -0.5 * torch.mean(1 + logvar.float() - mu.float().pow(2) - logvar.float().exp())
            loss_perf = F.mse_loss(p_pred.float(), perf)
            loss_pars = F.mse_loss(pr_pred.float(), pars)
            
            # Combine
            loss = loss_rec + self.beta * kl_weight * loss_kl.to(self.device) + \
                   0.1 * loss_perf.to(self.device) + 0.1 * loss_pars.to(self.device)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(f"E{epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item():.4f}")
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                geom = batch['geometry'].cuda()
                with autocast('cuda', dtype=torch.bfloat16):
                    x_rec, _, _, _, _ = self.model(geom)
                    loss = F.binary_cross_entropy_with_logits(x_rec, geom)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("vae_best.pth")
        return avg_loss

    def save_checkpoint(self, name):
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(), 'latent_dim': self.model.latent_dim}, f"checkpoints/{name}")

    def fit(self, epochs=100):
        for e in range(epochs):
            kl_weight = min(1.0, (e + 1) / 20)
            train_loss = self.train_epoch(e, kl_weight)
            val_loss = self.validate(e)
            logger.info(f"Epoch {e} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

if __name__ == "__main__":
    model = DesignVAE()
    print("Model ready for Blackwell Hybrid execution (BF16 GPU / FP32 CPU).")
