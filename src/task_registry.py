"""
task_registry.py — Task registry for the rank dynamics pipeline.

Maps a task name (from config) to:
  - make_loaders(cfg)  → train_loader, val_loader, task_meta
  - make_model(cfg)    → nn.Module (full-rank)
  - replace_with_low_rank(model, rank) → same model with LowRankLinear
  - materialize_low_rank(model) → same model with nn.Linear

task_meta is a dict with keys needed by downstream scripts:
  n_classes (vision) or vocab_size (NLP)
  img_size, in_channels (vision only)
  n_layers  — number of transformer blocks (for ablation k sweep)
  loss_fn   — 'cross_entropy_cls' | 'cross_entropy_lm'
  probe_hook — 'cls_token' | 'last_token' | 'mean_token'

Adding a new task:
  1. Register it in TASK_REGISTRY below.
  2. Implement make_loaders_<task> and make_model_<task>.
  3. Add a config YAML in configs/.

Currently registered:
  cifar10              — ViT on CIFAR-10 (32×32, 10 classes)
  cifar100             — ViT on CIFAR-100
  babylm_strict_small  — GPT-2 on BabyLM-2026-Strict-Small (10M tokens)
"""

import os, math, importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Low-rank utilities (shared across tasks) ───────────────────────────────

class LowRankLinear(nn.Module):
    """
    W = A @ B.T  where A: (out, r), B: (in, r).

    was_conv1d: if True, the original layer was HF Conv1D whose weight is
    stored as (in, out) — transpose when materialising back to a state dict
    that can be loaded into GPT2LMHeadModel.
    """
    def __init__(self, in_features, out_features, rank, was_conv1d=False):
        super().__init__()
        self.A         = nn.Parameter(torch.empty(out_features, rank))
        self.B         = nn.Parameter(torch.empty(in_features,  rank))
        self.was_conv1d = was_conv1d
        nn.init.normal_(self.A, std=1.0 / math.sqrt(rank))
        nn.init.normal_(self.B, std=1.0 / math.sqrt(rank))

    def forward(self, x):
        return x @ (self.A @ self.B.T).T

    def extra_repr(self):
        return (f"in={self.B.shape[0]}, out={self.A.shape[0]}, "
                f"rank={self.A.shape[1]}, conv1d={self.was_conv1d}")


def _get_linear_dims(module):
    """
    Return (in_features, out_features) for both nn.Linear and HF Conv1D.

    HF GPT-2 uses transformers.pytorch_utils.Conv1D whose weight is shaped
    (in_features, out_features) — the transpose of nn.Linear's (out, in).
    Both implement y = x @ W (Conv1D) or y = x @ W.T (Linear), so the
    effective linear map has the same in/out dims — we just read them
    differently depending on the class.
    """
    cls_name = type(module).__name__
    if cls_name == 'Conv1D':
        # Conv1D.weight: (in_features, out_features)
        in_f, out_f = module.weight.shape
    else:
        # nn.Linear.weight: (out_features, in_features)
        out_f, in_f = module.weight.shape
    return in_f, out_f


def _is_replaceable(name, module):
    """
    Return True if this module should be replaced with LowRankLinear.
    Targets nn.Linear and HF Conv1D; skips embedding/head layers and
    skip any layer where replacing would INCREASE parameters.
    """
    cls_name = type(module).__name__
    if cls_name not in ('Linear', 'Conv1D'):
        return False
    # Skip classifier / language model head
    leaf = name.split('.')[-1]
    if leaf in ('lm_head', 'head'):
        return False
    # Skip embedding projections (weight tying — lm_head already excluded)
    if 'embed' in name:
        return False
    return True


def replace_with_low_rank(model, rank, verbose=True):
    """
    Replace all eligible linear layers with LowRankLinear IN PLACE.

    Modifies the model's module tree directly via setattr — no copy is made.
    Returns the same model object for convenience, but the caller does not
    need to capture the return value since the modification is in-place.

    Handles both nn.Linear (vision ViT) and transformers.Conv1D (HF GPT-2).
    Skips lm_head, embeddings, and any layer where the rank factorisation
    would not reduce parameter count.

    For BabyLMWrapper, operates on the inner hf_model so that module name
    paths resolve correctly against the HF model's attribute tree.
    """
    # Unwrap BabyLMWrapper — navigate names relative to hf_model
    root = model.hf_model if hasattr(model, 'hf_model') else model

    n_replaced     = 0
    n_skipped_rank = 0
    params_before  = sum(p.numel() for p in root.parameters())

    for name, module in list(root.named_modules()):
        if not _is_replaceable(name, module):
            continue

        in_f, out_f = _get_linear_dims(module)

        # Only replace if rank factorisation reduces parameters
        # LR params: rank*(in+out)  vs  FR params: in*out
        lr_params = rank * (in_f + out_f)
        fr_params = in_f * out_f
        if lr_params >= fr_params:
            if verbose:
                print(f"  [SKIP no-reduction] {name}: ({in_f}×{out_f}) "
                      f"fr={fr_params:,} lr={lr_params:,} at r={rank}")
            n_skipped_rank += 1
            continue

        # Navigate to parent within root
        parts  = name.split('.')
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child = parts[-1]

        was_conv1d = (type(module).__name__ == 'Conv1D')
        new = LowRankLinear(in_f, out_f, rank, was_conv1d=was_conv1d)
        setattr(parent, child, new)
        n_replaced += 1
        if verbose:
            print(f"  [REPLACED] {name}: ({in_f}×{out_f}) "
                  f"fr={fr_params:,} → lr={lr_params:,} "
                  f"({100*(1-lr_params/fr_params):.0f}% reduction)")

    params_after = sum(p.numel() for p in root.parameters())
    print(f"replace_with_low_rank: {n_replaced} layers replaced, "
          f"{n_skipped_rank} skipped (rank too large). "
          f"Params: {params_before:,} → {params_after:,} "
          f"({100*(1-params_after/params_before):.1f}% reduction)")
    return model


def materialize_low_rank(model):
    """
    Replace all LowRankLinear with nn.Linear for checkpoint saving.
    Reconstructs W = A @ B.T and stores weight in the correct orientation:
      - nn.Linear (vision): weight (out, in)   — store as-is
      - Conv1D   (GPT-2):   weight (in, out)   — store transposed
    Unwraps BabyLMWrapper so name paths resolve correctly.
    """
    root = model.hf_model if hasattr(model, 'hf_model') else model
    for name, module in list(root.named_modules()):
        if not isinstance(module, LowRankLinear):
            continue
        parts  = name.split('.')
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child = parts[-1]
        W = (module.A @ module.B.T).detach()    # always (out, in)
        if module.was_conv1d:
            # Conv1D.weight must be (in, out) — transpose back
            W = W.T                              # now (in, out)
            new = nn.Linear(module.A.shape[0], module.B.shape[0], bias=False)
            new.weight.data.copy_(W)
        else:
            new = nn.Linear(module.B.shape[0], module.A.shape[0], bias=False)
            new.weight.data.copy_(W)
        setattr(parent, child, new)
    return model


# ── Vision task: CIFAR-10 / CIFAR-100 ─────────────────────────────────────

def _make_loaders_cifar(cfg, n_classes, dataset_cls):
    from torchvision import datasets, transforms
    mean = (0.4914, 0.4822, 0.4465) if n_classes == 10 else \
           (0.5071, 0.4867, 0.4408)
    std  = (0.2470, 0.2435, 0.2616) if n_classes == 10 else \
           (0.2675, 0.2565, 0.2761)
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    bs    = cfg.get('batch_size', 128)
    nw    = cfg.get('num_workers', 4)
    train = DataLoader(
        dataset_cls('data', train=True,  download=True, transform=tf_train),
        batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val   = DataLoader(
        dataset_cls('data', train=False, download=True, transform=tf_val),
        batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    meta  = {'n_classes': n_classes, 'img_size': 32, 'in_channels': 3,
             'n_layers': cfg.get('model', {}).get('n_layer', 9),
             'loss_fn': 'cross_entropy_cls', 'probe_hook': 'cls_token'}
    return train, val, meta


def make_loaders_cifar10(cfg):
    from torchvision.datasets import CIFAR10
    return _make_loaders_cifar(cfg, 10, CIFAR10)


def make_loaders_cifar100(cfg):
    from torchvision.datasets import CIFAR100
    return _make_loaders_cifar(cfg, 100, CIFAR100)


def make_model_cifar(cfg):
    """Build a ViT from train_cifar.py using model config."""
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "train_cifar",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision", "train_cifar.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ViT(img_size=32, in_channels=3,
                   n_classes=cfg.get('n_classes', 10))


# ── NLP task: BabyLM Strict-Small ─────────────────────────────────────────
#
# Architecture and tokenizer loaded directly from the BabyLM 2026 baseline:
#   BabyLM-community/BabyLM-2026-Baseline-Strict-Small-Interaction
#
# This ensures our FR and LR models use exactly the same architecture spec
# as the official baseline, making comparisons meaningful.

BABYLM_MODEL_ID = "BabyLM-community/BabyLM-2026-Baseline-Strict-Small-Interaction"
BABYLM_DATA_ID  = "BabyLM-community/BabyLM-2026-Strict-Small"


def _load_babylm_config_and_tokenizer(block_size_override=None):
    """
    Load the official BabyLM baseline config and tokenizer from HuggingFace.
    Returns (hf_config, tokenizer, block_size).
    """
    from transformers import AutoConfig, AutoTokenizer
    print(f"Loading config from {BABYLM_MODEL_ID}...")
    hf_config = AutoConfig.from_pretrained(BABYLM_MODEL_ID)
    print(f"  model_type={hf_config.model_type}  "
          f"n_layer={hf_config.n_layer}  "
          f"n_head={hf_config.n_head}  "
          f"n_embd={hf_config.n_embd}  "
          f"vocab_size={hf_config.vocab_size}")
    print(f"Loading tokenizer from {BABYLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BABYLM_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    # block_size: use n_positions from HF config unless overridden
    block_size = block_size_override or getattr(hf_config, 'n_positions',
                 getattr(hf_config, 'n_ctx', 1024))
    return hf_config, tokenizer, block_size


class BabyLMDataset(Dataset):
    """
    Fixed-length block dataset for causal LM training.
    Each item is (input_ids[:-1], input_ids[1:]) — standard next-token prediction.
    Loaded from a pre-tokenised tensor cache when available.
    """
    def __init__(self, chunks: torch.Tensor):
        # chunks: (N, block_size+1) long tensor
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


def _tokenise_to_chunks(texts, tokenizer, block_size):
    """Tokenise a list of texts and split into fixed-length blocks."""
    all_ids = []
    eos = tokenizer.eos_token_id
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.extend(ids + [eos])
    # Stack into (N, block_size+1) tensor
    total = len(all_ids)
    n_chunks = (total - 1) // block_size
    chunks = torch.zeros(n_chunks, block_size + 1, dtype=torch.long)
    for i in range(n_chunks):
        chunks[i] = torch.tensor(
            all_ids[i * block_size: i * block_size + block_size + 1],
            dtype=torch.long)
    return chunks


def _cache_path(cache_dir, split, block_size, model_id):
    """Deterministic cache filename encoding all parameters that affect content."""
    import hashlib
    key = hashlib.md5(f"{model_id}_{split}_{block_size}".encode()).hexdigest()[:8]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"babylm_{split}_bs{block_size}_{key}.pt")


def make_loaders_babylm_strict_small(cfg):
    """
    Downloads BabyLM-2026-Strict-Small, tokenises with the official BabyLM
    tokenizer, and caches the result to disk. On subsequent calls the cache
    is loaded directly — tokenisation is skipped entirely.

    Cache location: cfg.get('cache_dir', 'data/babylm_cache')
    Cache is keyed on (model_id, split, block_size) so changing any of these
    automatically triggers a fresh tokenisation.
    """
    from datasets import load_dataset

    model_cfg_yaml  = cfg.get('model', {})
    block_size_yaml = model_cfg_yaml.get('block_size', None)
    bs         = cfg.get('batch_size', 64)
    nw         = cfg.get('num_workers', 4)
    cache_dir  = cfg.get('cache_dir', 'data/babylm_cache')

    hf_config, tokenizer, block_size = _load_babylm_config_and_tokenizer(
        block_size_override=block_size_yaml)

    train_cache = _cache_path(cache_dir, 'train', block_size, BABYLM_MODEL_ID)
    val_cache   = _cache_path(cache_dir, 'val',   block_size, BABYLM_MODEL_ID)

    if os.path.exists(train_cache) and os.path.exists(val_cache):
        print(f"Loading tokenised data from cache:")
        print(f"  train: {train_cache}")
        print(f"  val:   {val_cache}")
        train_chunks = torch.load(train_cache, weights_only=True)
        val_chunks   = torch.load(val_cache,   weights_only=True)
        print(f"  Blocks: {len(train_chunks):,} train / {len(val_chunks):,} val")
    else:
        print(f"Cache not found — tokenising {BABYLM_DATA_ID}...")
        ds    = load_dataset(BABYLM_DATA_ID, split="train", trust_remote_code=True)
        texts = ds["text"]

        n_val       = max(1000, int(0.1 * len(texts)))
        val_texts   = texts[:n_val]
        train_texts = texts[n_val:]

        print(f"  Tokenising {len(train_texts):,} train docs...")
        train_chunks = _tokenise_to_chunks(train_texts, tokenizer, block_size)
        print(f"  Tokenising {len(val_texts):,} val docs...")
        val_chunks   = _tokenise_to_chunks(val_texts,   tokenizer, block_size)

        print(f"  Saving cache: {train_cache}")
        torch.save(train_chunks, train_cache)
        print(f"  Saving cache: {val_cache}")
        torch.save(val_chunks,   val_cache)
        print(f"  Blocks: {len(train_chunks):,} train / {len(val_chunks):,} val")

    train_ds = BabyLMDataset(train_chunks)
    val_ds   = BabyLMDataset(val_chunks)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    meta = {
        'vocab_size': hf_config.vocab_size,
        'block_size': block_size,
        'n_layers':   hf_config.n_layer,
        'n_embd':     hf_config.n_embd,
        'n_head':     hf_config.n_head,
        'loss_fn':    'cross_entropy_lm',
        'probe_hook': 'last_token',
        'tokenizer':  tokenizer,
        'hf_config':  hf_config,
    }
    return train_loader, val_loader, meta


class BabyLMWrapper(nn.Module):
    """
    Thin wrapper around HuggingFace GPT2LMHeadModel that exposes self.blocks
    (the transformer blocks) in the format expected by the rank dynamics pipeline.

    The HF model is instantiated from the official BabyLM baseline config with
    random weights so architecture is identical to the baseline. forward()
    delegates to the HF model and extracts logits from CausalLMOutput.

    self.blocks mirrors model.transformer.h — the ModuleList of GPT2Block
    objects — giving probe_dynamics.py and ablate_early_layers.py the same
    API as the vision ViT.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
        # Expose transformer blocks under self.blocks for probe / ablation API
        self.blocks   = hf_model.transformer.h

    def forward(self, idx):
        return self.hf_model(idx).logits   # (B, T, vocab_size)

    # Delegate module / parameter methods to the HF model
    def parameters(self, recurse=True):
        return self.hf_model.parameters(recurse=recurse)

    def named_modules(self, *args, **kwargs):
        return self.hf_model.named_modules(*args, **kwargs)

    def modules(self):
        return self.hf_model.modules()

    def state_dict(self, *args, **kwargs):
        return self.hf_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.hf_model.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        self.hf_model.train(mode)
        return self

    def eval(self):
        self.hf_model.eval()
        return self

    def to(self, *args, **kwargs):
        self.hf_model.to(*args, **kwargs)
        return self


def make_model_babylm_strict_small(cfg):
    """
    Instantiate a GPT2LMHeadModel from the official BabyLM baseline config
    with uninitialised weights (weights will be loaded from checkpoint or
    trained from scratch). Architecture is defined entirely by the HF config:
      BabyLM-community/BabyLM-2026-Baseline-Strict-Small-Interaction

    The YAML config.model.block_size can shorten the context window for
    debugging; all other architecture params come from the HF config.

    cfg may contain '_quiet': True to suppress the instantiation message,
    used when building a shell model to load checkpoint weights into.
    """
    from transformers import GPT2LMHeadModel

    model_cfg_yaml  = cfg.get('model', {})
    block_size_yaml = model_cfg_yaml.get('block_size', None)
    role = cfg.get('_role', 'train')   # 'train' | 'probe' | 'ref'

    hf_config, _, _ = _load_babylm_config_and_tokenizer(
        block_size_override=block_size_yaml)

    if block_size_yaml:
        hf_config.n_positions = block_size_yaml
        hf_config.n_ctx       = block_size_yaml

    role_msg = {
        'train': 'training from scratch — weights will be randomly initialised',
        'probe': 'probing — weights will be loaded from checkpoint',
        'ref':   'reference — weights will be loaded from checkpoint each epoch',
    }.get(role, 'weights will be loaded separately')
    print(f"Instantiating GPT2LMHeadModel [{role_msg}]...")
    hf_model = GPT2LMHeadModel(hf_config)
    n_params  = sum(p.numel() for p in hf_model.parameters())
    print(f"  Parameters: {n_params:,}")

    return BabyLMWrapper(hf_model)


# ── Registry ───────────────────────────────────────────────────────────────

TASK_REGISTRY = {
    'cifar10': {
        'make_loaders': make_loaders_cifar10,
        'make_model':   make_model_cifar,
    },
    'cifar100': {
        'make_loaders': make_loaders_cifar100,
        'make_model':   make_model_cifar,
    },
    'babylm_strict_small': {
        'make_loaders': make_loaders_babylm_strict_small,
        'make_model':   make_model_babylm_strict_small,
    },
}


def get_task(name):
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. "
            f"Registered tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]


def load_config(path):
    """Load a YAML config file, return as dict."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)