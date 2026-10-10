import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ClipDataset(Dataset):
    """Clips of `clip_len` cached feature frames, `frame_gap` apart, within one episode."""

    def __init__(self, paths, clip_len, frame_gap):
        self.paths = list(paths)
        self.clip_len = clip_len
        self.frame_gap = frame_gap
        span = (clip_len - 1) * frame_gap
        lengths = [np.load(p, mmap_mode="r").shape[0] for p in self.paths]
        self.index = [(i, s) for i, n in enumerate(lengths) for s in range(n - span)]
        if not self.index:
            raise RuntimeError("No clips: episodes are shorter than (rollout_k + 1) frames at this frame_gap.")
        self._arrays = {}  # opened lazily so each DataLoader worker gets its own memmaps

    def __len__(self):
        return len(self.index)

    def _array(self, i):
        if i not in self._arrays:
            self._arrays[i] = np.load(self.paths[i], mmap_mode="r")
        return self._arrays[i]

    def __getitem__(self, idx):
        i, s = self.index[idx]
        stop = s + (self.clip_len - 1) * self.frame_gap + 1
        clip = self._array(i)[s:stop:self.frame_gap]
        return torch.from_numpy(np.ascontiguousarray(clip))  # [T, N, D] fp16


class SlotInit(nn.Module):
    """Slots sampled from a learned Gaussian. Sampling is required: identical
    initial slots receive identical updates and never separate."""

    def __init__(self, num_slots, dim):
        super().__init__()
        self.num_slots = num_slots
        self.mu = nn.Parameter(torch.empty(1, 1, dim))
        self.log_sigma = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.mu)
        nn.init.xavier_uniform_(self.log_sigma)

    def forward(self, batch_size, generator=None):
        noise = torch.randn(batch_size, self.num_slots, self.mu.shape[-1],
                            device=self.mu.device, generator=generator)
        return self.mu + self.log_sigma.exp() * noise


class SlotAttention(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(inplace=True), nn.Linear(4 * dim, dim))

    def forward(self, inputs, slots, iters):
        """inputs [B, N, D], slots [B, K, D] -> slots [B, K, D], attn [B, K, N]
        (attn is the softmax over slots, i.e. a per-token soft assignment)."""
        B, K, D = slots.shape
        inputs = self.norm_inputs(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        attn = None
        for _ in range(iters):
            q = self.to_q(self.norm_slots(slots))
            attn = torch.einsum("bkd,bnd->bkn", q, k).mul(self.scale).softmax(dim=1)
            weights = attn + self.eps
            weights = weights / weights.sum(dim=-1, keepdim=True)  # weighted mean over tokens
            updates = torch.einsum("bkn,bnd->bkd", weights, v)
            slots = self.gru(updates.reshape(B * K, D), slots.reshape(B * K, D)).view(B, K, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


class MLPDecoder(nn.Module):
    """DINOSAUR-style decoder: each slot predicts every token's feature plus an
    alpha logit; alphas are softmaxed across slots and used to mix predictions."""

    def __init__(self, slot_dim, feat_dim, num_tokens, hidden, num_layers):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, 1, num_tokens, slot_dim) * 0.02)
        layers, d = [], slot_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers.append(nn.Linear(d, feat_dim + 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, slots):
        """slots [B, K, S] -> recon [B, N, D], masks [B, K, N]"""
        out = self.mlp(slots.unsqueeze(2) + self.pos)       # [B, K, N, D+1]
        feats, alpha = out[..., :-1].float(), out[..., -1].float()
        masks = alpha.softmax(dim=1)
        recon = (masks.unsqueeze(-1) * feats).sum(dim=1)
        return recon, masks


class SlotContrastModel(nn.Module):
    """Operates on cached DINO features. Slots of frame t+1 are initialized by a
    transformer predictor from the slots of frame t, so slot k keeps tracking
    the same object over time."""

    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.cfg = cfg
        S = cfg.slot_dim
        self.input_mlp = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, S), nn.ReLU(inplace=True), nn.Linear(S, S))
        self.slot_init = SlotInit(cfg.num_slots, S)
        self.slot_attention = SlotAttention(S)
        self.predictor = nn.TransformerEncoderLayer(
            S, nhead=4, dim_feedforward=4 * S, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = MLPDecoder(S, feat_dim, cfg.num_tokens, cfg.decoder_hidden, cfg.decoder_layers)

    # ---- building blocks (shared by training, eval and visualization) ----
    def first_step(self, feats_t, generator=None):
        """S_0 = group(init, x_0). feats_t [B, N, D] -> slots [B, K, S], attn [B, K, N]."""
        x = self.input_mlp(feats_t)
        return self.slot_attention(x, self.slot_init(x.shape[0], generator), self.cfg.slot_iters_first)

    def next_step(self, feats_t, prev_slots):
        """S_t = group(predictor(S_{t-1}), x_t)."""
        x = self.input_mlp(feats_t)
        return self.slot_attention(x, self.predictor(prev_slots), self.cfg.slot_iters_next)

    def encode(self, feats, slots=None, generator=None):
        """Full rollout (any length). feats [B, T, N, D]; `slots` continues a previous
        sequence. Returns slots [B, T, K, S], attn [B, T, K, N]."""
        all_slots, all_attn = [], []
        for t in range(feats.shape[1]):
            if slots is None:
                slots, attn = self.first_step(feats[:, t], generator)
            else:
                slots, attn = self.next_step(feats[:, t], slots)
            all_slots.append(slots)
            all_attn.append(attn)
        return torch.stack(all_slots, 1), torch.stack(all_attn, 1)

    def decode(self, slots):
        """slots [B, T, K, S] -> recon [B, T, N, D], masks [B, T, K, N].
        Frame by frame to keep decoder memory bounded for long rollouts."""
        outs = [self.decoder(slots[:, t]) for t in range(slots.shape[1])]
        return torch.stack([o[0] for o in outs], 1), torch.stack([o[1] for o in outs], 1)

    def forward(self, feats, generator=None):
        """Full rollout with every frame decoded (used for evaluation)."""
        slots, attn = self.encode(feats, generator=generator)
        recon, masks = self.decode(slots)
        return {"slots": slots, "attn": attn, "recon": recon, "masks": masks}

    def forward_truncated(self, feats, k, generator=None):
        """Training rollout over k transitions with gradients only on the last one.

        feats [B, >=k+1, N, D].
          S_0           first_step(x_0)          graph kept only if first_frame_weight > 0
          S_1..S_{k-1}  next_step, no_grad       burn-in through the model's own predictions
          S_k           next_step(x_k, S_{k-1}.detach())   <- the only temporal backprop
        """
        if not 1 <= k < feats.shape[1]:
            raise ValueError(f"k={k} needs at least k+1 frames, got {feats.shape[1]}.")
        train_first = self.cfg.first_frame_weight > 0
        with contextlib.nullcontext() if train_first else torch.no_grad():
            s0, _ = self.first_step(feats[:, 0], generator)

        slots = s0
        with torch.no_grad():
            for t in range(1, k):
                slots, _ = self.next_step(feats[:, t], slots)
        prev = slots.detach()

        last, _ = self.next_step(feats[:, k], prev)
        r_last, m_last = self.decoder(last)
        r0 = self.decoder(s0)[0] if train_first else None
        return {"s0": s0, "r0": r0, "prev": prev, "last": last, "r_last": r_last, "masks_last": m_last}

class DinoEncoder(nn.Module):
    """Frozen DINOv2: uint8 RGB frames -> normalized patch tokens [B, N, D]."""

    def __init__(self, name):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.model.requires_grad_(False).eval()
        self.dim = self.model.embed_dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):  # always frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, frames_uint8):
        x = frames_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - self.mean) / self.std
        return self.model.forward_features(x)["x_norm_patchtokens"]
