import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
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
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranspose3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(ConvTranspose3DBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                                 stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DesignVAE(nn.Module):
    def __init__(self, input_shape=(64, 64, 64), latent_dim=32):
        super(DesignVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Encoder: 4 Layers for 64^3 -> 4^3 bottleneck
        self.encoder = nn.Sequential(
            Conv3DBlock(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 64 -> 32
            Conv3DBlock(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32 -> 16
            Conv3DBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 16 -> 8
            Conv3DBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8 -> 4
        )

        self.fc_encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, 512),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 4 * 4 * 4),
            nn.ReLU(),
        )

        # Decoder: mirrors Encoder
        self.decoder = nn.Sequential(
            ConvTranspose3DBlock(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0), # 4 -> 8
            ConvTranspose3DBlock(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # 8 -> 16
            ConvTranspose3DBlock(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0),  # 16 -> 32
            ConvTranspose3DBlock(16, 8, kernel_size=4, stride=2, padding=1, output_padding=0),   # 32 -> 64
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
        )

        self.performance_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

        self.parameter_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # h_mm, r_mm
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.fc_encode(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Fused kernel is inference-only (no grad_fn); use it only inside
        # torch.no_grad() contexts (e.g. validation, generation, decode_latent).
        if mu.is_cuda and not torch.is_grad_enabled():
            try:
                from cuda_kernels.gpu_reparam import fused_reparameterize
                return fused_reparameterize(mu, logvar)
            except Exception:
                pass
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 128, 4, 4, 4)
        x_recon = torch.sigmoid(self.decoder(h))
        return x_recon

    def decode_logits(self, z):
        """Returns raw logits (for use with BCEWithLogitsLoss during training)."""
        h = self.fc_decode(z)
        h = h.view(-1, 128, 4, 4, 4)
        return self.decoder(h)

    def predict_performance(self, z):
        return self.performance_head(z)

    def predict_parameters(self, z):
        return self.parameter_head(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode_logits(z)  # logits for BCEWithLogitsLoss
        perf_pred = self.predict_performance(z)
        params_pred = self.predict_parameters(z)
        return x_recon, mu, logvar, z, perf_pred, params_pred


class VAELoss(nn.Module):
    def __init__(self, beta=1.0, recon_weight=1.0):
        super(VAELoss, self).__init__()
        self.beta = beta
        self.recon_weight = recon_weight
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.perf_loss = nn.MSELoss(reduction='mean')

    def forward(self, x_recon, x_true, mu, logvar,
                perf_pred, perf_true,
                params_pred, params_true,
                kl_weight=1.0):
        # Weighted BCE: solid voxels (~3% of data in 64^3) are up-weighted 30x
        pos_weight = torch.tensor([30.0], device=x_true.device)
        recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            x_recon, x_true, pos_weight=pos_weight, reduction='mean'
        )

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        perf_loss = self.perf_loss(perf_pred, perf_true)
        param_loss = self.mse_loss(params_pred, params_true)

        total_loss = (
            self.recon_weight * recon_loss +
            self.beta * kl_weight * kl_loss +
            0.1 * perf_loss +
            0.1 * param_loss
        )

        return total_loss, recon_loss, kl_loss, perf_loss, param_loss


class VAETrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-3, beta=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.criterion = VAELoss(beta=beta)
        self.writer = SummaryWriter(log_dir='./logs')
        self.best_val_loss = float('inf')
        # Optimized for Blackwell: Mixed Precision
        self.scaler = GradScaler()

    def train_epoch(self, epoch, kl_weight=1.0):
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_perf = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            geometry = batch['geometry'].to(self.device, non_blocking=True)
            perf_true = batch['performance'].to(self.device, non_blocking=True)
            params_true = batch['parameters'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision autocast (Forced to BF16 for Blackwell/RTX 5080)
            with autocast(dtype=torch.bfloat16):
                x_recon, mu, logvar, z, perf_pred, params_pred = self.model(geometry)

                loss, recon_loss, kl_loss, perf_loss, param_loss = self.criterion(
                    x_recon, geometry, mu, logvar,
                    perf_pred, perf_true,
                    params_pred, params_true,
                    kl_weight=kl_weight
                )

            # Scaling for mixed precision
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_perf += perf_loss.item()

            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                           f'Loss: {loss.item():.4f} | '
                           f'Recon: {recon_loss.item():.4f} | '
                           f'KL: {kl_loss.item():.4f}')

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('train/total_loss', avg_loss, epoch)
        self.writer.add_scalar('train/recon_loss', total_recon / len(self.train_loader), epoch)
        self.writer.add_scalar('train/kl_loss', total_kl / len(self.train_loader), epoch)
        self.writer.add_scalar('train/perf_loss', total_perf / len(self.train_loader), epoch)

        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                geometry = batch['geometry'].to(self.device, non_blocking=True)
                perf_true = batch['performance'].to(self.device, non_blocking=True)
                params_true = batch['parameters'].to(self.device, non_blocking=True)

                with autocast(dtype=torch.bfloat16):
                    x_recon, mu, logvar, z, perf_pred, params_pred = self.model(geometry)

                    loss, recon_loss, kl_loss, perf_loss, param_loss = self.criterion(
                        x_recon, geometry, mu, logvar,
                        perf_pred, perf_true,
                        params_pred, params_true
                    )

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('val/total_loss', avg_loss, epoch)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(f'vae_best.pth')
            logger.info(f'New best validation loss: {avg_loss:.4f}')

        return avg_loss

    def save_checkpoint(self, filename):
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'latent_dim': self.model.latent_dim,
            'input_shape': self.model.input_shape,
        }, f'checkpoints/{filename}')
        logger.info(f'Saved checkpoint: checkpoints/{filename}')

    def load_checkpoint(self, filename):
        checkpoint = torch.load(f'checkpoints/{filename}', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f'Loaded checkpoint: checkpoints/{filename}')

    def fit(self, epochs=100, start_epoch=0):
        # Only compile the forward/backward logic if torch 2.x is present
        # Note: Blackwell support is still being tuned in torch.compile
        # self.model = torch.compile(self.model)

        for epoch in range(start_epoch, epochs):
            kl_weight = min(1.0, (epoch + 1) / 20)

            train_loss = self.train_epoch(epoch, kl_weight=kl_weight)
            val_loss = self.validate(epoch)

            logger.info(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'vae_epoch_{epoch+1}.pth')

        self.writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train VAE for design generation")
    parser.add_argument("--dataset-path", type=str, default="./fem_data/fem_dataset.pt",
                       help="Path to processed FEM dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--beta", type=float, default=0.05, help="KL divergence weight")

    args = parser.parse_args()

    checkpoint = torch.load(args.dataset_path, weights_only=False)
    train_loader = checkpoint['train_loader']
    val_loader = checkpoint['val_loader']

    model = DesignVAE(input_shape=(64, 64, 64), latent_dim=args.latent_dim)

    trainer = VAETrainer(
        model,
        train_loader,
        val_loader,
        device=args.device,
        lr=args.lr,
        beta=args.beta
    )

    logger.info(f"Training 64^3 VAE on RTX 5080 with latent_dim={args.latent_dim}")
    trainer.fit(epochs=args.epochs)
