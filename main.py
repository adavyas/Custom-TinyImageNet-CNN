# fashion_mnist_mps.py (optimized)
import random, os, copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# -----------------------------
# Device (force MPS if present)
# -----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(">> Using Apple MPS")
else:
    device = torch.device("cpu")
    print(">> MPS not available, using CPU")

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed)
set_seed(42)

# -----------------------------
# Data (+RandomErasing)
# -----------------------------
# -----------------------------
# Data (separate transforms for train vs val)
# -----------------------------
FMNIST_MEAN, FMNIST_STD = 0.2860406, 0.35302424

train_tf = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((FMNIST_MEAN,), (FMNIST_STD,)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((FMNIST_MEAN,), (FMNIST_STD,))
])

root = "./data"

# Make indices once, then build two dataset views with different transforms
N = len(datasets.FashionMNIST(root, train=True, download=True))
val_size = 10_000
torch_gen = torch.Generator().manual_seed(42)
perm = torch.randperm(N, generator=torch_gen)
val_idx = perm[:val_size]
train_idx = perm[val_size:]

base_train = datasets.FashionMNIST(root, train=True, download=True, transform=train_tf)
base_val   = datasets.FashionMNIST(root, train=True, download=True, transform=test_tf)
test_set   = datasets.FashionMNIST(root, train=False, download=True, transform=test_tf)

from torch.utils.data import Subset
train_set = Subset(base_train, train_idx)
val_set   = Subset(base_val,   val_idx)

# macOS-friendly DataLoaders
train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=0, pin_memory=False)

# -----------------------------
# Model (GELU + GAP head)
# -----------------------------
Act = nn.GELU  # change from ReLU -> GELU

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), Act(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), Act(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), Act(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), Act(),
            nn.MaxPool2d(2),  # 7x7

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), Act(),
            nn.Dropout(0.20),
        )
        # Global average pooling head (reduces params & overfitting)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), Act(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.pool(x)
        return self.head(x)

model = SmallCNN().to(device)

# -----------------------------
# EMA wrapper
# -----------------------------
class ModelEMA:
    def __init__(self, model, decay=0.997):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        # update parameters with decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=(1. - d))
        # IMPORTANT: keep BN buffers (running_mean/var) in sync (no decay)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)

ema = ModelEMA(model, decay=0.997)

# -----------------------------
# Loss / Optim / Scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# weight-decay hygiene: no decay for norms and biases
decay, no_decay = [], []
for n, p in model.named_parameters():
    if p.ndim == 1 or n.endswith(".bias"):
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 1e-4},
    {"params": no_decay, "weight_decay": 0.0},
], lr=3e-3)  # higher max LR for OneCycle

epochs = 40
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.1,  # ~10% warmup
    div_factor=10.0, final_div_factor=1e3
)

use_amp = (device.type == "mps")  # try AMP on MPS; disable if it regresses

# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(loader, use_ema=False):
    m = ema.ema if use_ema else model
    m.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = m(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum/total, correct/total

def train_one_epoch(loader):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if use_amp:
            # autocast sometimes neutral on MPS; safe to try
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        scheduler.step()
        ema.update(model)

        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)

    return loss_sum/total, correct/total

best_val = 0.0
epochs = epochs
os.makedirs("checkpoints", exist_ok=True)

for ep in range(1, epochs+1):
    tr_loss, tr_acc = train_one_epoch(train_loader)
    va_loss, va_acc = evaluate(val_loader, use_ema=True)  # evaluate EMA
    if va_acc > best_val:
        best_val = va_acc
        torch.save(ema.ema.state_dict(), "checkpoints/best_fmnist_mps.pt")

    # show OneCycle current LR
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {ep:02d}/{epochs} | "
          f"lr {current_lr:.2e} | "
          f"train {tr_loss:.4f}/{tr_acc*100:.2f}% | "
          f"val {va_loss:.4f}/{va_acc*100:.2f}% | best_val {best_val*100:.2f}%")

# -----------------------------
# Test with best EMA checkpoint
# -----------------------------
ema.ema.load_state_dict(torch.load("checkpoints/best_fmnist_mps.pt", map_location=device))
test_loss, test_acc = evaluate(test_loader, use_ema=True)
print(f"\nTest  loss {test_loss:.4f} | Test acc {test_acc*100:.2f}%")

if device.type == "mps":
    try:
        torch.mps.empty_cache()
    except Exception:
        pass
