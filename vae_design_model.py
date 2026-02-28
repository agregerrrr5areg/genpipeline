import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self, input_shape=(32, 32, 32), latent_dim=16):
        super(DesignVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            Conv3DBlock(1, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv3DBlock(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            Conv3DBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.fc_encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 256),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4 * 4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            ConvTranspose3DBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvTranspose3DBlock(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTranspose3DBlock(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0),
            ConvTranspose3DBlock(16, 8, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Conv3d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
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
        h = h.view(-1, 64, 4, 4, 4)
        x_recon = self.decoder(h)
        return x_recon

    def predict_performance(self, z):
        return self.performance_head(z)

    def predict_parameters(self, z):
        return self.parameter_head(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
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
        recon_loss = self.mse_loss(x_recon, x_true)

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

    def train_epoch(self, epoch, kl_weight=1.0):
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_perf = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            geometry = batch['geometry'].to(self.device)
            perf_true = batch['performance'].to(self.device)
            params_true = batch['parameters'].to(self.device)

            self.optimizer.zero_grad()

            x_recon, mu, logvar, z, perf_pred, params_pred = self.model(geometry)

            loss, recon_loss, kl_loss, perf_loss, param_loss = self.criterion(
                x_recon, geometry, mu, logvar,
                perf_pred, perf_true,
                params_pred, params_true,
                kl_weight=kl_weight
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

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
                geometry = batch['geometry'].to(self.device)
                perf_true = batch['performance'].to(self.device)
                params_true = batch['parameters'].to(self.device)

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
        }, f'checkpoints/{filename}')
        logger.info(f'Saved checkpoint: checkpoints/{filename}')

    def load_checkpoint(self, filename):
        checkpoint = torch.load(f'checkpoints/{filename}', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f'Loaded checkpoint: checkpoints/{filename}')

    def fit(self, epochs=100, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            kl_weight = min(1.0, (epoch + 1) / 20)

            train_loss = self.train_epoch(epoch, kl_weight=kl_weight)
            val_loss = self.validate(epoch)

            logger.info(f'Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'vae_epoch_{epoch+1}.pth')

        self.writer.close()


def sample_latent_space(model, device, n_samples=100):
    z = torch.randn(n_samples, model.latent_dim).to(device)
    with torch.no_grad():
        samples = model.decode(z)
    return z.cpu().numpy(), samples.cpu().numpy()


def interpolate_designs(model, z1, z2, n_steps=10, device='cuda'):
    alphas = np.linspace(0, 1, n_steps)
    interpolated = []

    with torch.no_grad():
        for alpha in alphas:
            z_interp = torch.tensor(
                (1 - alpha) * z1 + alpha * z2,
                dtype=torch.float32
            ).to(device).unsqueeze(0)
            design = model.decode(z_interp)
            interpolated.append(design.cpu().numpy())

    return np.array(interpolated)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train VAE for design generation")
    parser.add_argument("--dataset-path", type=str, default="./fem_data/fem_dataset.pt",
                       help="Path to processed FEM dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--beta", type=float, default=1.0, help="KL divergence weight")

    args = parser.parse_args()

    checkpoint = torch.load(args.dataset_path, weights_only=False)
    train_loader = checkpoint['train_loader']
    val_loader = checkpoint['val_loader']

    model = DesignVAE(input_shape=(32, 32, 32), latent_dim=args.latent_dim)

    trainer = VAETrainer(
        model,
        train_loader,
        val_loader,
        device=args.device,
        lr=args.lr,
        beta=args.beta
    )

    logger.info(f"Training VAE with latent_dim={args.latent_dim}")
    trainer.fit(epochs=args.epochs)

    logger.info("Training complete!")
