# tiny_imagenet_trainable_fp16.py
import os, time, torch, logging
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# ----------------------------------------------------------
# Logging setup
# ----------------------------------------------------------
os.makedirs("logs", exist_ok=True)
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"logs/run_{run_name}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)
log = logging.getLogger()
log.info("===== Tiny-ImageNet Training Started =====")

# ----------------------------------------------------------
# Device setup
# ----------------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log.info(f"Using {device} ✅" if device.type == "mps" else "Using CPU ⚠️")

# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------
data_root = "/Users/adavya/Downloads/tiny-imagenet-200"

def load_tiny_imagenet_data(batch_size=128):
    tmp_tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    tmp = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tmp_tf)
    tmp_loader = DataLoader(tmp, batch_size=256, shuffle=False)

    mean, std, n = torch.zeros(3), torch.zeros(3), 0
    for imgs, _ in tmp_loader:
        mean += imgs.mean([0, 2, 3]); std += imgs.std([0, 2, 3]); n += 1
    mean, std = mean / n, std / n
    log.info(f"Mean: {mean}"); log.info(f"Std:  {std}")

    train_tf = v2.Compose([
        v2.RandomResizedCrop(64, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean.tolist(), std.tolist()),
    ])
    test_tf = v2.Compose([
        v2.Resize(64),
        v2.CenterCrop(64),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean.tolist(), std.tolist()),
    ])

    train = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val   = datasets.ImageFolder(os.path.join(data_root, "val"), transform=test_tf)
    log.info(f"Train samples: {len(train)} | Val samples: {len(val)}")
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=256)

# ----------------------------------------------------------
# Model (Improved TinyCNN ≤500K params)
# ----------------------------------------------------------
class ConvBNAct(nn.Module):
    """Conv + BN + h-swish"""
    def __init__(self, inp, oup, k=3, s=1, g=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inp, oup, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(oup),
            nn.Hardswish()
        )
    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excite with tiny MLP"""
    def __init__(self, c, r=8):
        super().__init__()
        hidden = max(8, c // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c, bias=False),
            nn.Hardsigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class DepthwiseBlock(nn.Module):
    """
    Depthwise-separable conv block
    ConvBNAct(depthwise k=5) -> ConvBNAct(pointwise 1x1) -> optional SE
    """
    def __init__(self, inp, oup, stride=1, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(inp, inp, k=5, s=stride, g=inp),  # depthwise
            ConvBNAct(inp, oup, k=1, s=1, g=1),         # pointwise
            SEBlock(oup) if use_se else nn.Identity()
        )
    def forward(self, x):
        return self.block(x)


class GlobalMixerBlock(nn.Module):
    """
    Lightweight Transformer-style block:
    - LayerNorm
    - Multihead Self-Attention (small heads)
    - MLP
    Residual connections included.
    """
    def __init__(self, dim, num_heads=2, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            bias=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, C)
        # Self-attention with residual
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        # MLP with residual
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class TinyHybridCNN(nn.Module):
    """
    CNN backbone (depthwise-SE) + tiny global mixer head.
    - Resolution agnostic.
    - Aims to stay near/under ~1MB FP16.
    - Stronger global reasoning than pure CNN.
    """
    def __init__(self, num_classes=200, dropout=0.25):
        super().__init__()

        # --- Convolutional feature extractor ---
        # We'll make this moderately wide so it's strong but still efficient.
        self.features = nn.Sequential(
            # Stage 1: 64 -> 32 (or 224 -> 112, etc.)
            ConvBNAct(3,   32, k=3, s=2),
            DepthwiseBlock(32,   64, stride=1),
            # Stage 2: 32 -> 16 (or 112 -> 56)
            DepthwiseBlock(64,   96, stride=2),
            DepthwiseBlock(96,  128, stride=1),
            # Stage 3: 16 -> 8 (or 56 -> 28)
            DepthwiseBlock(128, 176, stride=2),
            DepthwiseBlock(176, 224, stride=1),
            DepthwiseBlock(224, 256, stride=1),
        )

        # After this, spatial size is:
        # - If input is 64x64: downsampled by 2,2,2 → 8x8
        # - If input is 224x224: → 28x28
        # Channels: 256

        self.channel_dim = 256  # final C

        # --- Global mixer head (1 transformer-like block) ---
        self.mixer = GlobalMixerBlock(
            dim=self.channel_dim,
            num_heads=2,
            mlp_ratio=2.0,
            dropout=0.1,
        )

        # --- Classifier ---
        # We'll pool over tokens (mean) after mixing.
        self.classifier = nn.Sequential(
            nn.Linear(self.channel_dim, 320),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(320, num_classes),
        )

        # --- Init (safe on bias=None) ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN backbone
        x = self.features(x)           # (B, C=256, H', W')
        B, C, H, W = x.shape

        # Flatten into tokens
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, N=H*W, C)

        # Global token mixer
        x = self.mixer(x)              # (B, N, C)

        # Global average over tokens
        x = x.mean(dim=1)              # (B, C)

        # Classifier head
        x = self.classifier(x)         # (B, num_classes)
        return x


# ----------------------------------------------------------
# Eval
# ----------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval(); tot, correct, loss_sum = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        tot += x.size(0)
    return loss_sum / tot, correct / tot

# ----------------------------------------------------------
# Train
# ----------------------------------------------------------
def train_tiny_cnn(lr=0.00254, weight_decay=9.86e-6, dropout=0.25, epochs=50):
    log.info(f"Initializing training with lr={lr}, weight_decay={weight_decay}, dropout={dropout}")
    train_loader, val_loader = load_tiny_imagenet_data()
    model = TinyHybridCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim == 1 or n.endswith(".bias") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total parameters: {total_params:,}")
    best_acc = 0.0
    log.info(f"Training for {epochs} epochs...")

    for ep in range(1, epochs + 1):
        start = time.time()
        model.train(); running_loss = running_correct = total = 0
        num_batches = len(train_loader)

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="mps", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

            if i in {int(num_batches*0.25), int(num_batches*0.5), int(num_batches*0.75)}:
                pct = (i/num_batches)*100
                log.info(f"Epoch {ep} [{pct:.0f}%] - sample_loss {loss.item():.4f} | grad_norm {float(grad_norm):.2f}")

        tr_loss, tr_acc = running_loss/total, running_correct/total
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_tiny_cnn_w_att.pt")
            log.info(f"✅ New best model saved! ValAcc={val_acc*100:.2f}%")

        scheduler.step()
        log.info(f"Epoch {ep}/{epochs}: TrainLoss={tr_loss:.4f}, TrainAcc={tr_acc*100:.2f}%, "
                 f"ValLoss={val_loss:.4f}, ValAcc={val_acc*100:.2f}%, best={best_acc*100:.2f}% "
                 f"({time.time()-start:.2f}s)")

    log.info(f"Training complete! Best val acc: {best_acc*100:.2f}%")
    log.info(f"Logs saved to {log_path}")
    if device.type == "mps": torch.mps.empty_cache()

if __name__ == "__main__":
    train_tiny_cnn()
