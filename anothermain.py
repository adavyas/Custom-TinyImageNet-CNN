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
# Model
# ----------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=200, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64 → 32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32 → 16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # no pool here — keep 16×16 spatial size
        )

        # Global average pooling instead of flattening 8×8 maps
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),     # 16×16 → 1×1
            nn.Flatten(),                # → (batch, 128)
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

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
def train_tiny_cnn(lr=1e-3, weight_decay=1e-4, dropout=0.2, epochs=5):
    log.info(f"Initializing training with lr={lr}, weight_decay={weight_decay}, dropout={dropout}")
    train_loader, val_loader = load_tiny_imagenet_data()
    model = TinyCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim == 1 or n.endswith(".bias") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.6, step_size=10)

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
            torch.save(model.state_dict(), "best_tiny_cnn.pt")
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
