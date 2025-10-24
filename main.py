# tiny_imagenet_mps_logger.py
import os, time, logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# ----------------------------------------------------------
# Create logs folder + timestamped log file
# ----------------------------------------------------------
os.makedirs("logs", exist_ok=True)
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"logs/run_{run_name}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

log.info("===== Tiny-ImageNet Training Started =====")

# ----------------------------------------------------------
# Apple MPS device
# ----------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    log.info("Using Apple MPS ✅")
else:
    device = torch.device("cpu")
    log.warning("Using CPU ⚠️")

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
data_root = "/Users/adavya/Downloads/tiny-imagenet-200"

# ----------------------------------------------------------
# Compute mean/std using raw tensor loader (NO augment)
# ----------------------------------------------------------
tmp_tf = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

tmp_set = datasets.ImageFolder(os.path.join(data_root, "train"), transform=tmp_tf)
tmp_loader = DataLoader(tmp_set, batch_size=256, shuffle=False, num_workers=0)

log.info("Computing mean/std...")
mean = torch.zeros(3)
std = torch.zeros(3)
num_batches = 0

for imgs, _ in tmp_loader:
    num_batches += 1
    mean += imgs.mean(dim=[0, 2, 3])
    std += imgs.std(dim=[0, 2, 3])

mean /= num_batches
std /= num_batches

log.info(f"Mean: {mean}")
log.info(f"Std:  {std}")

# ----------------------------------------------------------
# Final transforms (WITH augmentation)
# ----------------------------------------------------------
train_tf = v2.Compose([
    v2.RandomResizedCrop(64, scale=(0.6, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(15),
    v2.ColorJitter(0.3, 0.3, 0.3, 0.05),
    v2.RandomGrayscale(p=0.10),
    v2.RandomErasing(p=0.25),
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

log.info("Loading datasets...")
train_set = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
val_set   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=test_tf)

log.info(f"Train samples: {len(train_set)}")
log.info(f"Val samples:   {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=256, shuffle=False)

# ----------------------------------------------------------
# Simple CNN
# ----------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            #nn.GroupNorm(32, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            #nn.GroupNorm(64, 128), 
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = TinyCNN().to(device)
log.info("Model initialized.")

# param count
total_params = sum(p.numel() for p in model.parameters())
log.info(f"Total parameters: {total_params:,}")

# ----------------------------------------------------------
# Optimizer / Loss / Scheduler
# ----------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.6, step_size=10)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


# ----------------------------------------------------------
# Training Loop
# ----------------------------------------------------------
epochs = 5
best_acc = 0.0

log.info(f"Training for {epochs} epochs...")

for ep in range(1, epochs + 1):
    start = time.time()
    model.train()
    running_loss, running_correct, total = 0, 0, 0
    num_batches = len(train_loader)

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if device.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

        # Progress checkpoints
        if i in {int(num_batches*0.25), int(num_batches*0.5), int(num_batches*0.75)}:
            pct = (i/num_batches)*100
            log.info(f"Epoch {ep} [{pct:.0f}%] - sample_loss {loss.item():.4f} | grad_norm {float(grad_norm):.2f}")

    # compute epoch statistics
    tr_loss = running_loss / total
    tr_acc = running_correct / total
    val_loss, val_acc = evaluate(val_loader)

    # best checkpoint
    if val_acc > best_acc:
        improvement = (val_acc - best_acc) * 100
        best_acc = val_acc
        torch.save(model.state_dict(), "best_tiny_cnn.pt")
        log.info(f"✅ New best model saved! +{improvement:.2f}% gain")

    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    duration = time.time() - start

    log.info(f"EPOCH {ep}/{epochs}: "
             f"TrainLoss={tr_loss:.4f}, TrainAcc={tr_acc*100:.2f}%, "
             f"ValLoss={val_loss:.4f}, ValAcc={val_acc*100:.2f}%, "
             f"LR={lr:.5f}, time={duration:.2f}s, best={best_acc*100:.2f}%")

log.info("Training complete!")
log.info(f"Best validation accuracy: {best_acc*100:.2f}%")
log.info(f"Logs saved to {log_path}")

if device.type == "mps":
    torch.mps.empty_cache()
