import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from collections import namedtuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VAEOutput = namedtuple("VAEOutput", ["x_recon", "mu", "logvar", "perf_pred", "param_pred"])

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvTranspose3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv_transpose(x)))


class DesignVAE(nn.Module):
    """
    3D VAE for voxel grids.

    Encoder: D→D/2→D/4→D/8→D/16, channels 1→32→64→128→256
    Decoder: D/16→D/8→D/4→D/2→D, channels 256→128→64→32→16→1

    input_shape must be divisible by 2**4 = 16.

    All layers run on GPU. nn.Linear (2D matmul) is safe on Blackwell RTX 5080
    — the cuBLAS bug only affects strided batched GEMM (dim > 2, batch ≥ 2).
    """

    _ENC_CH    = 256   # bottleneck channels
    _N_CONV    = 4     # number of stride-2 conv layers
    _FC_HIDDEN = 1024

    def __init__(self, input_shape=(64, 64, 64), latent_dim=32, pos_weight: float = 30.0):
        super().__init__()
        self.latent_dim  = latent_dim
        self.input_shape = input_shape
        self.pos_weight  = pos_weight

        stride = 2 ** self._N_CONV
        if input_shape[0] % stride != 0:
            raise ValueError(
                f"input_shape[0]={input_shape[0]} must be divisible by {stride} "
                f"(2^{self._N_CONV} stride-2 conv layers)."
            )
        enc_sp = input_shape[0] // stride   # spatial dim at bottleneck
        self._enc_sp = enc_sp

        flat = self._ENC_CH * enc_sp ** 3

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            Conv3DBlock(1,   32,  stride=2),   # 64 → 32
            Conv3DBlock(32,  64,  stride=2),   # 32 → 16
            Conv3DBlock(64,  128, stride=2),   # 16 → 8
            Conv3DBlock(128, 256, stride=2),   # 8  → 4
        )
        self.fc_encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, self._FC_HIDDEN),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(self._FC_HIDDEN, latent_dim)
        self.fc_logvar = nn.Linear(self._FC_HIDDEN, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, self._FC_HIDDEN),
            nn.ReLU(),
            nn.Linear(self._FC_HIDDEN, flat),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            ConvTranspose3DBlock(256, 128),    # 4  → 8
            ConvTranspose3DBlock(128, 64),     # 8  → 16
            ConvTranspose3DBlock(64,  32),     # 16 → 32
            ConvTranspose3DBlock(32,  16),     # 32 → 64
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

        # ── Prediction heads ─────────────────────────────────────────────────
        self.performance_head = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64),         nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.parameter_head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    # ── Forward helpers ───────────────────────────────────────────────────────

    def encode(self, x):
        h = self.encoder(x)
        # Cast to FP32 before linear — BF16 GEMM backward crashes on Blackwell
        h = h.float()
        with torch.autocast('cuda', enabled=False):
            h = self.fc_encode(h)
            return self.fc_mu(h), self.fc_logvar(h)

    def decode_logits(self, z):
        with torch.autocast('cuda', enabled=False):
            h = self.fc_decode(z.float())
        h = h.view(-1, self._ENC_CH, self._enc_sp, self._enc_sp, self._enc_sp)
        return self.decoder(h)

    def decode(self, z):
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z   = mu + torch.randn_like(std) * std
        x_recon = self.decode_logits(z)
        with torch.autocast('cuda', enabled=False):
            perf_pred  = self.performance_head(z.float())
            param_pred = self.parameter_head(z.float())
        return VAEOutput(x_recon, mu, logvar, perf_pred, param_pred)

    def predict_performance(self, z):
        return self.performance_head(z.float())

    def predict_parameters(self, z):
        return self.parameter_head(z.float())


class VAETrainer:
    def __init__(self, model, train_loader, val_loader,
                 device='cuda', lr=3e-4, beta=1.0, epochs=500):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.beta         = beta

        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = OneCycleLR(
            self.optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader), epochs=epochs,
            pct_start=0.05, anneal_strategy='cos',
        )
        self.scaler     = GradScaler('cuda')
        self.writer     = SummaryWriter(log_dir='./logs')
        self.best_val   = float('inf')
        self._global_step = 0

    def train_epoch(self, epoch, kl_weight=1.0):
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            geom = batch['geometry'].to(self.device, non_blocking=True)
            perf = batch['performance'].to(self.device, non_blocking=True)
            pars = batch['parameters'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.bfloat16):
                out = self.model(geom)
                x_rec, mu, logvar, p_pred, pr_pred = out
                loss_rec = F.binary_cross_entropy_with_logits(
                    x_rec, geom,
                    pos_weight=torch.tensor([self.model.pos_weight], device=self.device),
                )
                # KL in float32 for stability
                mu_f  = mu.float();  lv_f = logvar.float()
                loss_kl   = -0.5 * torch.mean(1 + lv_f - mu_f.pow(2) - lv_f.exp())
                loss_perf = F.mse_loss(p_pred.float(),  perf.float())
                loss_pars = F.mse_loss(pr_pred.float(), pars.float())
                loss = (loss_rec
                        + self.beta * kl_weight * loss_kl
                        + 0.1 * loss_perf
                        + 0.1 * loss_pars)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            self._global_step += 1

            if batch_idx % 10 == 0:
                lr_now = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"E{epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"loss={loss.item():.4f} rec={loss_rec.item():.4f} "
                    f"kl={loss_kl.item():.4f} lr={lr_now:.2e}"
                )
                self.writer.add_scalar('train/loss',     loss.item(),     self._global_step)
                self.writer.add_scalar('train/rec',      loss_rec.item(), self._global_step)
                self.writer.add_scalar('train/kl',       loss_kl.item(),  self._global_step)
                self.writer.add_scalar('train/lr',       lr_now,          self._global_step)

        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                geom = batch['geometry'].to(self.device)
                with autocast('cuda', dtype=torch.bfloat16):
                    x_rec, _, _, _, _ = self.model(geom)
                    loss = F.binary_cross_entropy_with_logits(x_rec, geom)
                total_loss += loss.item()
        avg = total_loss / len(self.val_loader)
        self.writer.add_scalar('val/loss', avg, epoch)
        if avg < self.best_val:
            self.best_val = avg
            self.save_checkpoint("vae_best.pth")
            logger.info(f"  → new best val {avg:.4f}, checkpoint saved")
        return avg

    def save_checkpoint(self, name):
        Path('checkpoints').mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'latent_dim':       self.model.latent_dim,
            'input_shape':      self.model.input_shape,
        }, f"checkpoints/{name}")

    def fit(self, epochs=500):
        for e in range(epochs):
            # KL ramp: full weight by epoch 50
            kl_weight   = min(1.0, (e + 1) / 50)
            train_loss  = self.train_epoch(e, kl_weight)
            val_loss    = self.validate(e)
            logger.info(f"Epoch {e:3d} | train={train_loss:.4f} | val={val_loss:.4f} | kl_w={kl_weight:.2f}")


if __name__ == "__main__":
    model = DesignVAE(input_shape=(64, 64, 64), latent_dim=32).cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"DesignVAE: {params/1e6:.1f}M parameters — full GPU, BF16 autocast")
    x = torch.randn(2, 1, 64, 64, 64, device='cuda')
    with autocast('cuda', dtype=torch.bfloat16):
        out = model(x)
    print(f"Forward OK: x_recon={out[0].shape}, mu={out[1].shape}")
