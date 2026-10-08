"""
train_cifar.py — Small ViT training on image classification benchmarks.

Similar datasets to CIFAR-10, in increasing order of difficulty/size:
  mnist        - 28x28 greyscale digits, 10 classes, 60k train (trivial)
  fashion_mnist- 28x28 greyscale clothing, 10 classes, 60k train (easy)
  cifar10      - 32x32 colour, 10 classes, 50k train (~minutes)
  cifar100     - 32x32 colour, 100 classes, 50k train (harder, same speed)
  svhn         - 32x32 colour digits (street view), 73k train
  cinic10      - 32x32 colour, 10 classes, 90k train (CIFAR + ImageNet subset)
  tiny_imagenet- 64x64 colour, 200 classes, 100k train (~10x CIFAR)

Usage:
    python src/vision/train_cifar.py                         # CIFAR-10, full rank
    python src/vision/train_cifar.py --dataset cifar100
    python src/vision/train_cifar.py --rank 8                # low-rank r=8
    python src/vision/train_cifar.py --dataset cifar100 --rank 16
    python src/vision/train_cifar.py --debug                 # 100 steps
"""

import os, math, time, argparse, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Hyperparameters ───────────────────────────────────────────────────────────

BATCH_SIZE   = 128
EPOCHS       = 20
LR           = 1e-3
WARMUP_FRAC  = 0.05
WEIGHT_DECAY = 0.05
GRAD_CLIP    = 1.0
LOG_EVERY    = 50
VAL_EVERY    = 200
OUT_DIR      = "outputs_cifar"

# ── Dataset registry ──────────────────────────────────────────────────────────
# name -> (torchvision_class, n_classes, img_size, n_channels, mean, std)

DATASETS = {
    "mnist": (
        datasets.MNIST, 10, 28, 1,
        (0.1307,), (0.3081,)
    ),
    "fashion_mnist": (
        datasets.FashionMNIST, 10, 28, 1,
        (0.2860,), (0.3530,)
    ),
    "cifar10": (
        datasets.CIFAR10, 10, 32, 3,
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    ),
    "cifar100": (
        datasets.CIFAR100, 100, 32, 3,
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    ),
    "svhn": (
        datasets.SVHN, 10, 32, 3,
        (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    ),
}

# ── ViT ───────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into patches and linearly embed."""
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                        # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)        # (B, N, E)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale    = self.head_dim ** -0.5
        self.qkv  = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)   # each: (B, H, N, D)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = attn.transpose(1, 2).reshape(B, N, E)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, embed_dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = Attention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Small ViT for CIFAR-scale images.
    Default: patch=4, embed=192, depth=9, heads=3 (~3M params on CIFAR-10).
    """
    def __init__(self, img_size, in_channels, n_classes,
                 patch_size=4, embed_dim=192, depth=9, n_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches        = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(embed_dim, n_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                             # (B, N, E)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, E)
        x   = torch.cat([cls, x], dim=1)                    # (B, N+1, E)
        x   = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])                              # cls token
        return self.head(x)

# ── Low-rank ──────────────────────────────────────────────────────────────────

class LowRankLinear(nn.Module):
    """
    W = A @ B.T, stored as separate factors.
    Init: both A and B drawn from N(0, 1/√r) so W has the same
    variance as a standard Kaiming init at the given rank.
    (Zero-B init is fine for fine-tuning/LoRA but kills gradients
    in a network trained from scratch.)
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        r = min(rank, in_features, out_features)
        self.A = nn.Parameter(torch.empty(out_features, r))
        self.B = nn.Parameter(torch.empty(in_features,  r))
        # Both factors initialised so W = A@B.T has std ≈ 1/√in_features
        std = (in_features * r) ** -0.25
        nn.init.normal_(self.A, std=std)
        nn.init.normal_(self.B, std=std)

    def forward(self, x):
        return x @ self.B @ self.A.T


def replace_with_low_rank(model, rank):
    """Replace all nn.Linear layers (except head) with LowRankLinear."""
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if name == "head":          # keep classifier full-rank
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, name,
                        LowRankLinear(child.in_features, child.out_features, rank))
    return model


def materialize_low_rank(model):
    """Replace LowRankLinear with nn.Linear(W=A@B.T) in-place. Used before saving."""
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LowRankLinear):
                W = (child.A @ child.B.T).detach()
                linear = nn.Linear(child.B.shape[0], child.A.shape[0], bias=False,
                                   device=W.device, dtype=W.dtype)
                linear.weight = nn.Parameter(W)
                setattr(parent, name, linear)
    return model

# ── Data ──────────────────────────────────────────────────────────────────────

def make_loaders(dataset_name):
    cls, n_classes, img_size, n_channels, mean, std = DATASETS[dataset_name]

    # Augmentation for training
    if n_channels == 1:
        train_tf = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # SVHN uses split="train"/"test" instead of train=True/False
    if dataset_name == "svhn":
        train_ds = cls("data", split="train", download=True, transform=train_tf)
        val_ds   = cls("data", split="test",  download=True, transform=val_tf)
    else:
        train_ds = cls("data", train=True,  download=True, transform=train_tf)
        val_ds   = cls("data", train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)
    return train_loader, val_loader, n_classes, img_size, n_channels

# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
        loss_sum += loss.item() * y.size(0)
    model.train()
    return loss_sum / total, correct / total

# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step, total, warmup, peak, min_lr):
    if step < warmup:
        return peak * step / warmup
    t = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * t))

# ── Save ──────────────────────────────────────────────────────────────────────

def save_checkpoint(model, path):
    import copy
    m = copy.deepcopy(model).cpu()
    materialize_low_rank(m)
    os.makedirs(path, exist_ok=True)
    torch.save(m.state_dict(), os.path.join(path, "model.pt"))

# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    import random
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run_dir: explicit override, else auto-named with seed suffix
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_name = f"LR_r{args.rank}" if args.rank else "FR"
        if args.seed != 0:
            run_name += f"_seed{args.seed}"
        run_dir = os.path.join(OUT_DIR, args.dataset, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # CSV log
    log_path = os.path.join(run_dir, "log.csv")
    log_file = open(log_path, "w", newline="")
    log      = csv.writer(log_file)
    log.writerow(["step", "epoch", "train_loss", "lr", "val_loss", "val_acc",
                  "best_val_acc", "elapsed_s"])

    print(f"Dataset: {args.dataset} | Condition: {run_name} | Device: {device}")

    # Data
    train_loader, val_loader, n_classes, img_size, n_channels = make_loaders(args.dataset)

    # Model — low-rank BEFORE .to(device)
    model = ViT(img_size=img_size, in_channels=n_channels, n_classes=n_classes)
    if args.rank:
        model = replace_with_low_rank(model, args.rank)
        print(f"Low-rank: r={args.rank}")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Optimizer
    decay    = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim  < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.95), eps=1e-8,
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    if args.debug:
        total_steps = 100
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))
    min_lr       = LR * 0.1
    print(f"Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}")

    best_val_acc = 0.0
    step, t0     = 0, time.time()
    model.train()

    for epoch in range(EPOCHS):
        for x, y in train_loader:
            if step >= total_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = get_lr(step, total_steps, warmup_steps, LR, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            if step % LOG_EVERY == 0:
                print(f"step {step:5d} | epoch {epoch:3d} | loss {loss.item():.4f}"
                      f" | lr {lr:.2e} | {time.time()-t0:.0f}s")
                log.writerow([step, epoch, f"{loss.item():.4f}", f"{lr:.2e}",
                              "", "", f"{best_val_acc:.4f}", f"{time.time()-t0:.0f}"])
                log_file.flush()

            if not args.debug and step > 0 and step % VAL_EVERY == 0:
                val_loss, val_acc = evaluate(model, val_loader, device)
                print(f"  val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
                      f" (best {best_val_acc:.4f})")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_checkpoint(model, os.path.join(run_dir, "best"))
                    print(f"  New best — saved: {run_dir}/best/")
                log.writerow([step, epoch, f"{loss.item():.4f}", f"{lr:.2e}",
                              f"{val_loss:.4f}", f"{val_acc:.4f}",
                              f"{best_val_acc:.4f}", f"{time.time()-t0:.0f}"])
                log_file.flush()

            step += 1

        # Save checkpoint at end of every epoch for dynamics analysis
        if not args.debug:
            save_checkpoint(model, os.path.join(run_dir, "checkpoints", f"epoch_{epoch:02d}"))

        if step >= total_steps:
            break

    # Final evaluation
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"\nFinal | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
          f" | best {best_val_acc:.4f}")
    log.writerow(["final", "-", "-", "-", f"{val_loss:.4f}", f"{val_acc:.4f}",
                  f"{best_val_acc:.4f}", f"{time.time()-t0:.0f}"])
    log_file.close()

    if not args.debug:
        save_checkpoint(model, os.path.join(run_dir, "final"))
        print(f"Saved: {run_dir}/final/")

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10",
                   choices=list(DATASETS), help="Dataset to train on")
    p.add_argument("--rank",    type=int,  default=None,  help="Low-rank r (None = full rank)")
    p.add_argument("--debug",   action="store_true",       help="100 steps only, no saves")
    p.add_argument("--seed",    type=int,  default=0,      help="Random seed")
    p.add_argument("--run_dir", type=str,  default=None,   help="Override output directory")
    train(p.parse_args())