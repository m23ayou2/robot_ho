# -*- coding: utf-8 -*-
"""
OC-Flow — flow-matching policy on frozen SlotContrast slots (Unitree G1 dataset)
===============================================================================

Uses the slot model trained with ocslots_colab.py (checkpoint: checkpoints/slots.pt).

Pipeline
--------
1. Load the frozen slot model from `slot_ckpt` (architecture rebuilt from the config
   stored in the checkpoint, strict loading). Dataset / camera / fps / image size come
   from that config.
2. Episodes: by default EVERY episode of the dataset. The slot model's own held-out
   episodes (eval + viz of its splits.json) stay in the policy eval set; a fraction of
   the episodes the slot model never saw is added to it. Saved as policy_splits.json.
3. Slots: for each episode, slots are propagated over the WHOLE episode (first_step, then
   next_step — the recurrence used at inference) and cached as slots [T, K, S] + mask
   geometry [T, K, 5]. DINOv2 features are read from the slot-training cache when they
   exist; otherwise frames are decoded and encoded in memory, one video file at a time,
   so the ~100 MB/episode feature files are not needed (keep_features=False).
4. Actions + proprioception from the LeRobot parquet files (30 Hz), aligned with the
   10 fps slot frames.
5. Action targets are DELTAS w.r.t. the last command:  a[j : j+H] - a[j-1]
   (action_mode="delta"). An output of 0 is exactly "hold the last command", so the
   policy only has to learn the motion. The last command is also an input token.
6. Rectified-flow policy: object tokens (slot + geometry, no slot-index embedding,
   slot dropout) + state token + last-command token -> transformer scene encoder ->
   DiT-style action expert with AdaLN-Zero on flow time.
   Grippers (gripper_dims) are handled separately (gripper_mode="classify"): their discrete
   command levels are found in the training data (1-D k-means), and a classification head
   predicts the level of each gripper at every step of the chunk (cross-entropy). The flow
   then only models the arm joints. Alternatives: "absolute" (gripper value predicted
   absolutely inside the flow, no deltas) or "delta" (grippers treated like the arm).
7. Rollout training (rollout_k > 0, truncated backprop; with rollout_always=True on every batch
   from the first epoch, otherwise switched on once an evaluation shows the policy beating hold,
   then ramped in): before the trained step, the EMA policy is run for k replans (every `replan_every` slot frames) WITHOUT gradients, each
   replan using its OWN previous output as the last command:
       anchor_0 = recorded a[j_0 - 1]
       anchor_m = last executed action of the chunk predicted at replan m-1   (no_grad)
   The gradient only flows through the last replan, whose target is the TRUE future
   a[j_k : j_k+H] expressed relative to the drifted anchor_k — so the policy learns to
   correct its own drift (what happens on the robot). k ~ U{1..rollout_k} per batch.
8. Evaluation (EMA weights) reports the policy's L1 on the executed steps next to two
   baselines computed on the same samples: hold the last command, hold the state — plus
   the same L1 after a rollout of rollout_k replans on its own commands (l1_exec_roll).
   Early stopping on `best_metric`; best.pt / latest.pt in `out_dir`.

Run on a GPU runtime. In Colab: one cell, or split at the "# %%" markers.
If you run outside Colab, set `data_root` / `work_dir` (default: values stored in
the slot checkpoint, i.e. /content/...).
"""

# %% [0] Imports & config ======================================================
import copy
import glob
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, fields
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class PolicyConfig:
    # ---- inputs / outputs ----
    slot_ckpt: str = os.path.join("checkpoints", "slots.pt")
    out_dir: str = os.path.join("checkpoints", "policy")
    data_root: Optional[str] = None      # None -> value stored in the slot checkpoint
    work_dir: Optional[str] = None       # None -> value stored in the slot checkpoint
    action_key: str = "action"
    state_key: str = "observation.state"

    # ---- episodes ----
    num_episodes: Optional[int] = None   # None -> every episode of the dataset
    new_eval_fraction: float = 0.1       # of the episodes the slot model never saw
    keep_features: bool = False          # also save DINOv2 features of new episodes (~100 MB each)
    delete_videos: bool = False          # delete each video file once its episodes are processed

    # ---- observation ----
    slot_seed: int = 0                   # slot-init seed; reuse the same one on the robot
    n_hist: int = 2                      # slot frames per observation (t - gap, t)
    hist_gap: int = 1                    # in slot frames (10 fps -> 0.1 s)
    use_slots: bool = True               # ablation switches
    use_geometry: bool = True            # centroid / spread / area / velocity of each slot mask
    use_state: bool = True               # proprioception token
    use_prev_action: bool = True         # last-command token
    use_rel_state: bool = True           # token: (state history - last command) / delta scale — makes
                                         # drift of the command w.r.t. the measured robot directly visible
    min_slot_area: float = 0.01          # slots covering less of the image are masked out

    # ---- action chunk ----
    action_mode: str = "delta"           # "delta": a[j:j+H] - a[j-1] | "absolute": a[j:j+H]
    horizon: int = 32                    # steps at the dataset fps (30 Hz -> ~1.07 s)
    exec_steps: int = 10                 # steps executed before replanning (evaluated)

    # ---- grippers (handled separately from the arm) ----
    gripper_dims: list = field(default_factory=lambda: [14, 15])   # action dims of the grippers; [] = none
    gripper_mode: str = "classify"       # "classify": discrete level per step from a classification head
                                         # "absolute": absolute value inside the flow | "delta": like the arm
    gripper_max_levels: int = 6          # max discrete levels per gripper (1-D k-means on the train data)
    gripper_level_tol: float = 0.03      # fewest levels whose mean |value - level| < tol * gripper range
    gripper_loss_weight: float = 1.0     # weight of the gripper cross-entropy vs the flow loss
    gripper_label_smoothing: float = 0.05

    # ---- model ----
    d_model: int = 256
    heads: int = 4
    enc_layers: int = 2
    dec_layers: int = 4
    fourier_bands: int = 6
    dropout: float = 0.1                 # scene encoder
    flow_dropout: float = 0.1            # action expert

    # ---- regularization ----
    slot_dropout: float = 0.1            # drop each slot token with this probability (training)
    input_noise: float = 0.05            # Gaussian noise on normalized state / last command (training)

    # ---- flow matching ----
    flow_steps: int = 10                 # Euler steps at inference / eval

    # ---- rollout training (truncated backprop, gradient only on the last replan) ----
    rollout_k: int = 3                   # no-grad replans on the policy's own commands before the
                                         # trained one; 0 = off (standard single-step training)
    rollout_always: bool = True          # every batch, from epoch 1, is trained after a rollout (k >= 1);
                                         # False -> adaptive: start once the policy beats hold, then ramp
                                         # in up to rollout_prob (the 4 settings below)
    rollout_clean_loss: bool = True      # with a rollout, ALSO train the same batch on the recorded last
                                         # command (loss = mean of both) so the basic motion keeps being learned
    rollout_sample: bool = True          # k ~ U{1..rollout_k} per batch so every horizon is trained
    rollout_prob: float = 0.5            # fraction of batches trained after a rollout (others: k = 0)
    rollout_start_epoch: int = 4         # rollouts start at the first evaluation, from this epoch on,
    rollout_min_gain: float = 0.0        # where the policy beats hold by this gain (a policy worse than
                                         # hold drifts so far that rollout targets are noise) ...
    rollout_ramp_epochs: int = 6         # ... then rollout_prob ramps up linearly over these epochs
    replan_every: int = 3                # slot frames between replans (3 -> 0.3 s -> 9 actions executed)
    rollout_flow_steps: int = 4          # Euler steps of the no-grad burn-in samples (speed)
    eval_rollout_k: int = 3              # eval: replans on own commands before scoring (l1_exec_roll);
                                         # independent of rollout_k so runs with/without rollouts compare

    # ---- training ----
    batch_size: int = 256
    num_epochs: int = 100
    eval_every: int = 2
    patience: int = 5                    # evaluations without improvement before stopping; 0 = off
    lr: float = 3e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 500
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    eval_stride: int = 3                 # evaluate on every n-th frame of the eval episodes
    eval_samples: int = 4                # samples per observation (best-of-n, spread)
    best_metric: str = "auto"            # "auto" (= l1_exec_roll if rollout_k > 0, else l1_exec) |
                                         # "l1_exec" | "l1_exec_roll" | "mse_exec" | "flow"
    num_workers: int = 2
    seed: int = 42
    resume: bool = True

    def __post_init__(self):
        if self.n_hist < 1 or self.hist_gap < 1:
            raise ValueError("n_hist and hist_gap must be >= 1.")
        if not 1 <= self.exec_steps <= self.horizon:
            raise ValueError("exec_steps must be in [1, horizon].")
        if self.action_mode not in ("delta", "absolute"):
            raise ValueError("action_mode must be 'delta' or 'absolute'.")
        if self.best_metric == "auto":
            self.best_metric = "l1_exec_roll" if self.rollout_k > 0 else "l1_exec"
        if self.best_metric not in ("l1_exec", "l1_exec_roll", "mse_exec", "flow"):
            raise ValueError("best_metric must be 'auto', 'l1_exec', 'l1_exec_roll', 'mse_exec' or 'flow'.")
        if self.best_metric == "l1_exec_roll" and self.eval_rollout_k < 1:
            raise ValueError("best_metric 'l1_exec_roll' needs eval_rollout_k >= 1.")
        if min(self.rollout_k, self.eval_rollout_k) < 0 or self.replan_every < 1 or not 0 <= self.rollout_prob <= 1:
            raise ValueError("rollout_k >= 0, replan_every >= 1 and 0 <= rollout_prob <= 1.")
        self.gripper_dims = sorted({int(d) for d in (self.gripper_dims or [])})
        if self.gripper_mode not in ("classify", "absolute", "delta"):
            raise ValueError("gripper_mode must be 'classify', 'absolute' or 'delta'.")
        if not (self.use_slots or self.use_state or self.use_prev_action):
            raise ValueError("The policy needs at least one input (slots, state or last command).")

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


OBS_KEYS = ("slots", "geom", "state", "prev_action", "state_hist_raw", "anchor_raw")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no GPU found — training will be slow.")
    return device


def amp_dtype(device):
    """bf16 on Ampere+, fp16 on older GPUs (T4), disabled on CPU."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def autocast(device):
    dtype = amp_dtype(device)
    return torch.autocast(device.type, dtype=dtype, enabled=dtype is not None)


# %% [1] Frozen slot model (same classes / parameter names as ocslots_colab.py) =
class SlotCfg(SimpleNamespace):
    """The slot-training Config, rebuilt from the dict stored in the checkpoint."""

    @property
    def grid_hw(self):
        return self.img_h // self.patch_size, self.img_w // self.patch_size

    @property
    def num_tokens(self):
        gh, gw = self.grid_hw
        return gh * gw

    @property
    def cache_dir(self):
        return os.path.join(self.work_dir, "cache")


class SlotInit(nn.Module):
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
        B, K, D = slots.shape
        inputs = self.norm_inputs(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        attn = None
        for _ in range(iters):
            q = self.to_q(self.norm_slots(slots))
            attn = torch.einsum("bkd,bnd->bkn", q, k).mul(self.scale).softmax(dim=1)
            weights = attn + self.eps
            weights = weights / weights.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", weights, v)
            slots = self.gru(updates.reshape(B * K, D), slots.reshape(B * K, D)).view(B, K, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


class MLPDecoder(nn.Module):
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
        out = self.mlp(slots.unsqueeze(2) + self.pos)
        feats, alpha = out[..., :-1].float(), out[..., -1].float()
        masks = alpha.softmax(dim=1)
        recon = (masks.unsqueeze(-1) * feats).sum(dim=1)
        return recon, masks


class SlotContrastModel(nn.Module):
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

    def first_step(self, feats_t, generator=None):
        x = self.input_mlp(feats_t)
        return self.slot_attention(x, self.slot_init(x.shape[0], generator), self.cfg.slot_iters_first)

    def next_step(self, feats_t, prev_slots):
        x = self.input_mlp(feats_t)
        return self.slot_attention(x, self.predictor(prev_slots), self.cfg.slot_iters_next)


def file_hash(path, n=12):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def load_slot_model(pc, device):
    if not os.path.exists(pc.slot_ckpt):
        raise FileNotFoundError(f"Slot checkpoint not found: {pc.slot_ckpt}")
    ckpt = torch.load(pc.slot_ckpt, map_location="cpu", weights_only=False)
    scfg = SlotCfg(**ckpt["config"])
    if pc.data_root:
        scfg.data_root = pc.data_root
    if pc.work_dir:
        scfg.work_dir = pc.work_dir
    model = SlotContrastModel(scfg, ckpt["feat_dim"])
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    print(f"Slot model: {pc.slot_ckpt} (epoch {ckpt.get('epoch', '?')}) | K={scfg.num_slots}, "
          f"slot_dim={scfg.slot_dim}, grid={scfg.grid_hw}, {scfg.target_fps} fps")
    return model, scfg, ckpt["feat_dim"], file_hash(pc.slot_ckpt)


# %% [2] Dataset: metadata, splits, actions ====================================
def hf_download(repo_id, root, patterns):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id, repo_type="dataset", local_dir=root, allow_patterns=patterns)


def load_metadata(scfg):
    root = scfg.data_root
    info_path = os.path.join(root, "meta", "info.json")
    pattern = os.path.join(root, "meta", "episodes", "**", "*.parquet")
    if not os.path.exists(info_path) or not glob.glob(pattern, recursive=True):
        hf_download(scfg.repo_id, root, ["meta/info.json", "meta/episodes/*"])
    with open(info_path) as f:
        info = json.load(f)
    files = sorted(glob.glob(pattern, recursive=True))
    episodes = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    return info, episodes


def load_slot_splits(scfg, ids):
    """The slot model's split (splits.json); recomputed identically if missing."""
    path = os.path.join(scfg.work_dir, "splits.json")
    if os.path.exists(path):
        with open(path) as f:
            splits = json.load(f)
        if sorted(sum(splits.values(), [])) == sorted(ids):
            return splits
        print("splits.json does not match the slot-training episodes — recomputing.")
    ids = list(ids)
    rng = random.Random(scfg.seed)
    rng.shuffle(ids)
    viz = [ids.pop()]
    n_eval = max(1, round(scfg.eval_fraction * len(ids)))
    splits = {"train": sorted(ids[n_eval:]), "eval": sorted(ids[:n_eval]), "viz": viz}
    os.makedirs(scfg.work_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def policy_splits(pc, scfg, all_ids):
    """Train / eval episodes for the policy.
    eval = slot model's held-out episodes (eval + viz) + `new_eval_fraction` of the
    episodes the slot model never saw; train = everything else."""
    slot_ids = all_ids[:scfg.num_episodes]
    slot_splits = load_slot_splits(scfg, slot_ids)
    n = len(all_ids) if pc.num_episodes is None else max(pc.num_episodes, len(slot_ids))
    seen = set(slot_ids)
    new = [e for e in all_ids[:n] if e not in seen]
    random.Random(pc.seed).shuffle(new)
    n_eval = round(pc.new_eval_fraction * len(new))
    held_out = slot_splits["eval"] + slot_splits.get("viz", [])
    splits = {"train": sorted(slot_splits["train"] + new[n_eval:]),
              "eval": sorted(held_out + new[:n_eval]),
              "eval_slot_heldout": sorted(held_out)}
    os.makedirs(pc.out_dir, exist_ok=True)
    with open(os.path.join(pc.out_dir, "policy_splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def data_files(info, episodes, ids):
    """Relative parquet paths holding the selected episodes (LeRobot v3), or None."""
    sel = episodes[episodes["episode_index"].isin(ids)]
    if not {"data/chunk_index", "data/file_index"} <= set(sel.columns):
        return None
    try:
        return sorted({info["data_path"].format(chunk_index=int(r["data/chunk_index"]),
                                                 file_index=int(r["data/file_index"]))
                       for _, r in sel.iterrows()})
    except (KeyError, IndexError):
        return None


def load_actions(pc, scfg, info, episodes, ids):
    """{episode: (actions [L, A], states [L, S])} at the dataset fps, sorted by frame."""
    root = scfg.data_root
    rels = data_files(info, episodes, ids)
    if rels is None:
        files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
        if not files:
            hf_download(scfg.repo_id, root, ["data/*"])
            files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    else:
        missing = [r for r in rels if not os.path.exists(os.path.join(root, r))]
        if missing:
            print(f"Downloading {len(missing)} data file(s)...")
            hf_download(scfg.repo_id, root, missing)
        files = [os.path.join(root, r) for r in rels]

    wanted, out = set(ids), {}
    cols = ["episode_index", "frame_index", pc.action_key, pc.state_key]
    for path in files:
        names = pq.read_schema(path).names
        absent = [c for c in cols if c not in names]
        if absent:
            raise KeyError(f"{path} lacks {absent}. Available columns: {names}")
        df = pd.read_parquet(path, columns=cols)
        df = df[df["episode_index"].isin(wanted)]
        for ep, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            out[int(ep)] = (np.stack(g[pc.action_key].to_numpy()).astype(np.float32),
                            np.stack(g[pc.state_key].to_numpy()).astype(np.float32))
    lost = sorted(wanted - set(out))
    if lost:
        raise RuntimeError(f"No action rows found for episodes {lost[:10]}...")
    return out


# %% [3] Video decoding & DINOv2 ===============================================
def feature_path(scfg, episode_index):
    return os.path.join(scfg.cache_dir, f"ep_{episode_index:06d}_feats.npy")


def episode_table(scfg, info, episodes):
    prefix = f"videos/{scfg.camera}/"
    needed = ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]
    missing = [prefix + c for c in needed if prefix + c not in episodes.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Columns: {[c for c in episodes.columns if scfg.camera in c]}")
    rows = []
    for _, ep in episodes.iterrows():
        rows.append({
            "episode_index": int(ep["episode_index"]),
            "video": info["video_path"].format(video_key=scfg.camera,
                                               chunk_index=int(ep[prefix + "chunk_index"]),
                                               file_index=int(ep[prefix + "file_index"])),
            "start": float(ep[prefix + "from_timestamp"]),
            "end": float(ep[prefix + "to_timestamp"]),
        })
    return pd.DataFrame(rows)


def decode_segment(path, start, end, fps, width, height):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.4f}", "-i", path, "-t", f"{end - start:.4f}",
           "-vf", f"fps={fps},scale={width}:{height}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}:\n{proc.stderr.decode()[-2000:]}")
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    size = height * width * 3
    if frames.size == 0 or frames.size % size:
        raise RuntimeError(f"Decoded {frames.size} bytes from {path} [{start:.2f}, {end:.2f}]s.")
    return frames.reshape(-1, height, width, 3)


class DinoEncoder(nn.Module):
    """Frozen DINOv2: uint8 RGB frames [B, H, W, 3] -> patch tokens [B, N, D]."""

    def __init__(self, name):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", name)
        self.model.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, frames_uint8):
        x = frames_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        return self.model.forward_features((x - self.mean) / self.std)["x_norm_patchtokens"]


@torch.no_grad()
def encode_frames(encoder, frames, device, batch_size=64):
    """uint8 frames [T, H, W, 3] (numpy) -> DINOv2 features [T, N, D] fp16 on `device`."""
    out = []
    for s in range(0, len(frames), batch_size):
        batch = torch.from_numpy(np.array(frames[s:s + batch_size])).to(device)  # copy: buffer is read-only
        with autocast(device):
            out.append(encoder(batch).half())
    return torch.cat(out)


# %% [4] Slot extraction (propagated over whole episodes, cached) ==============
def grid_coords(gh, gw):
    """Patch-center coordinates in [-1, 1], [N, 2] as (x, y), row-major like DINO."""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, gh), torch.linspace(-1, 1, gw), indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], -1)


def slot_geometry(masks, coords):
    """masks [..., K, N] (sum to 1 over K) -> [..., K, 5] = (cx, cy, sx, sy, area)."""
    mass = masks.sum(-1)
    w = masks / mass.clamp_min(1e-6)[..., None]
    mean = w @ coords
    var = w @ coords.pow(2) - mean.pow(2)
    area = mass / masks.shape[-1]
    return torch.cat([mean, var.clamp_min(1e-6).sqrt(), area[..., None]], -1)


@torch.no_grad()
def rollout_slots(slot_model, feats, device, seed, chunk=64):
    """feats [T, N, D] (numpy or tensor) -> slots [T, K, S] fp16, geometry [T, K, 5] fp32 (numpy)."""
    if torch.is_tensor(feats):
        x = feats.to(device).float()
    else:
        x = torch.from_numpy(np.array(feats, dtype=np.float32)).to(device)  # copy: memmaps are read-only
    gen = torch.Generator(device=device).manual_seed(seed)
    out, slots = [], None
    with autocast(device):
        for t in range(len(x)):
            if slots is None:
                slots, _ = slot_model.first_step(x[t:t + 1], gen)
            else:
                slots, _ = slot_model.next_step(x[t:t + 1], slots)
            out.append(slots)
        all_slots = torch.cat(out).float()
        masks = torch.cat([slot_model.decoder(all_slots[s:s + chunk])[1].float()
                           for s in range(0, len(all_slots), chunk)])
    geom = slot_geometry(masks, grid_coords(*slot_model.cfg.grid_hw).to(device))
    return all_slots.half().cpu().numpy(), geom.cpu().numpy()


def ensure_slots(pc, scfg, slot_model, slot_hash, info, episodes, ids, device):
    """Slots for every episode in `ids`. Order of preference: slot cache -> DINO feature
    cache -> decode video + encode in memory (one video file at a time)."""
    cache = os.path.join(scfg.work_dir, "policy_cache", f"slots_{slot_hash}_seed{pc.slot_seed}")
    os.makedirs(cache, exist_ok=True)
    slot_file = lambda ep: os.path.join(cache, f"ep_{ep:06d}.npz")

    def save(ep, slots, geom):
        tmp = slot_file(ep) + ".tmp.npz"
        np.savez(tmp, slots=slots, geom=geom)
        os.replace(tmp, slot_file(ep))

    todo = [e for e in ids if not os.path.exists(slot_file(e))]
    from_feats = [e for e in todo if os.path.exists(feature_path(scfg, e))]
    from_video = [e for e in todo if not os.path.exists(feature_path(scfg, e))]
    print(f"Slots: {len(ids) - len(todo)} cached, {len(from_feats)} from DINO cache, "
          f"{len(from_video)} from video")

    for n, ep in enumerate(from_feats):
        save(ep, *rollout_slots(slot_model, np.load(feature_path(scfg, ep), mmap_mode="r"), device, pc.slot_seed))
        print(f"  [{n + 1}/{len(from_feats)}] episode {ep} (cached features)")

    if from_video:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found (Colab: !apt-get -y install ffmpeg).")
        table = episode_table(scfg, info, episodes[episodes["episode_index"].isin(from_video)])
        encoder = DinoEncoder(scfg.dino_model).to(device)
        if pc.keep_features:
            os.makedirs(scfg.cache_dir, exist_ok=True)
        done = 0
        for video, group in table.groupby("video", sort=True):
            local = os.path.join(scfg.data_root, video)
            if not os.path.exists(local):
                print(f"Downloading {video} ({len(group)} episodes)...")
                hf_download(scfg.repo_id, scfg.data_root, [video])
            for r in group.itertuples():
                frames = decode_segment(local, r.start, r.end, scfg.target_fps, scfg.img_w, scfg.img_h)
                feats = encode_frames(encoder, frames, device)
                if pc.keep_features:
                    np.save(feature_path(scfg, r.episode_index), feats.cpu().numpy())
                save(r.episode_index, *rollout_slots(slot_model, feats, device, pc.slot_seed))
                done += 1
                print(f"  [{done}/{len(from_video)}] episode {r.episode_index}: {len(frames)} frames")
            if pc.delete_videos:
                os.remove(local)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {}
    for ep in ids:
        with np.load(slot_file(ep)) as d:
            result[ep] = (d["slots"], d["geom"])
    return result


# %% [5] Action coding & training data =========================================
def _like(arr, x):
    return torch.as_tensor(arr, device=x.device, dtype=x.dtype) if torch.is_tensor(x) else arr


class QuantileNormalizer:
    """Maps each dimension's [q01, q99] to [-1, 1]. Works on numpy arrays and tensors."""

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, np.float32)
        self.hi = np.asarray(hi, np.float32)
        self.span = np.maximum(self.hi - self.lo, 1e-3)

    @classmethod
    def fit(cls, data, q=0.01):
        return cls(np.quantile(data, q, axis=0), np.quantile(data, 1 - q, axis=0))

    def encode(self, x):
        return 2 * (x - _like(self.lo, x)) / _like(self.span, x) - 1

    def decode(self, x):
        return (x + 1) / 2 * _like(self.span, x) + _like(self.lo, x)

    def state_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["lo"], d["hi"])


class DeltaScaler:
    """Per-dimension scale for deltas (no offset: 0 stays 0 = hold the last command).
    scale = q99(|delta|), floored at 1% of the dimension's absolute range."""

    def __init__(self, scale):
        self.scale = np.asarray(scale, np.float32)

    @classmethod
    def fit(cls, deltas, abs_span, q=0.99):
        s = np.quantile(np.abs(deltas.reshape(-1, deltas.shape[-1])), q, axis=0)
        return cls(np.maximum(s, np.maximum(0.01 * abs_span, 1e-6)))

    def encode(self, x):
        return x / _like(self.scale, x)

    def decode(self, x):
        return x * _like(self.scale, x)

    def state_dict(self):
        return {"scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["scale"])


def fit_gripper_levels(values, max_levels=6, tol=0.03):
    """Discrete command levels of one gripper: 1-D k-means with the fewest levels (2..max_levels)
    whose mean |value - nearest level| is below tol * range. Farthest-point initialization so
    every plateau (open / half / closed ...) gets its own level."""
    v = np.asarray(values, np.float64)
    span = max(float(v.max() - v.min()), 1e-6)
    centers = np.array([v.min(), v.max()])
    for k in range(2, max_levels + 1):
        c = np.array([v.min(), v.max()])
        while len(c) < k:
            c = np.append(c, v[np.abs(v[:, None] - c[None]).min(1).argmax()])
        for _ in range(100):
            lab = np.abs(v[:, None] - c[None]).argmin(1)
            new = np.array([v[lab == i].mean() if (lab == i).any() else c[i] for i in range(len(c))])
            if np.allclose(new, c):
                break
            c = new
        centers = np.unique(np.round(c, 5))
        if np.abs(v[:, None] - centers[None]).min(1).mean() < tol * span:
            break
    return centers.astype(np.float32)


class GripperCoder:
    """Gripper commands <-> discrete level indices (nearest level)."""

    def __init__(self, dims, levels):
        self.dims = list(dims)
        self.n_levels = [len(l) for l in levels]
        self.table = np.full((len(dims), max(self.n_levels)), np.nan, np.float32)   # [G, L_max]
        for g, l in enumerate(levels):
            self.table[g, :len(l)] = l

    @property
    def valid(self):                       # [G, L_max] bool
        return ~np.isnan(self.table)

    def encode(self, chunk):
        """raw chunk [..., H, A] (tensor) -> level index [..., H, G] (long)."""
        tab = torch.as_tensor(self.table, device=chunk.device)
        dist = (chunk[..., self.dims, None].float() - tab).abs()
        return torch.nan_to_num(dist, nan=float("inf")).argmin(-1)

    def decode(self, classes):
        """level index [..., H, G] -> raw gripper commands [..., H, G]."""
        tab = torch.as_tensor(self.table, device=classes.device)
        return tab[torch.arange(len(self.dims), device=classes.device), classes]

    def state_dict(self):
        return {"dims": self.dims, "levels": [self.table[g, :n].tolist() for g, n in enumerate(self.n_levels)]}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["dims"], d["levels"])


class ActionCodec:
    """Raw action chunk <-> policy targets (torch tensors).
    Flow target, for the "flow dims" (all dims, minus the grippers when they are classified):
      delta:    DeltaScaler(chunk - anchor),  anchor = last command a[j-1]
      absolute: QuantileNormalizer(chunk)
      gripper_mode "absolute": the gripper dims are always absolute (no delta) in the flow.
    gripper_mode "classify": gripper dims are decoded from level classes (GripperCoder)."""

    def __init__(self, mode, abs_norm, delta_scaler, grip_dims=(), grip_mode="delta", grip=None):
        self.mode, self.abs_norm, self.delta = mode, abs_norm, delta_scaler
        self.A = len(abs_norm.lo)
        self.grip_dims = list(grip_dims) if grip_mode != "delta" else []
        self.grip_mode = grip_mode if self.grip_dims else "delta"
        self.grip = grip if self.grip_mode == "classify" else None
        if self.grip_mode == "classify" and grip is None:
            raise ValueError("gripper_mode 'classify' needs a GripperCoder.")
        self.flow_dims = [d for d in range(self.A) if not (self.grip_mode == "classify" and d in self.grip_dims)]
        abs_dims = self.grip_dims if self.grip_mode == "absolute" else []
        self.abs_pos = [i for i, d in enumerate(self.flow_dims) if self.mode == "absolute" or d in abs_dims]

    @property
    def flow_dim(self):
        return len(self.flow_dims)

    def encode(self, chunk, anchor):
        """raw chunk [..., H, A], anchor [..., A] -> flow target [..., H, F]."""
        f = self.flow_dims
        absolute = self.abs_norm.encode(chunk)[..., f]
        if self.mode == "absolute":
            return absolute
        t = self.delta.encode(chunk - anchor[..., None, :])[..., f]
        if self.abs_pos:
            t[..., self.abs_pos] = absolute[..., self.abs_pos]
        return t

    def decode(self, target, anchor, grip_classes=None):
        """flow target [..., H, F] (+ gripper level classes [..., H, G]) -> raw chunk [..., H, A].
        Dims that are not predicted (classified grippers without classes) hold the anchor."""
        f = self.flow_dims
        full = anchor.unsqueeze(-2).expand(*anchor.shape[:-1], target.shape[-2], self.A).clone()
        lo = torch.as_tensor(self.abs_norm.lo, device=target.device)[f]
        span = torch.as_tensor(self.abs_norm.span, device=target.device)[f]
        scale = torch.as_tensor(self.delta.scale, device=target.device)[f]
        vals = (target + 1) / 2 * span + lo if self.mode == "absolute" \
            else anchor[..., None, f] + target * scale
        if self.abs_pos and self.mode == "delta":
            vals[..., self.abs_pos] = ((target + 1) / 2 * span + lo)[..., self.abs_pos]
        full[..., f] = vals.to(full.dtype)
        if self.grip is not None and grip_classes is not None:
            full[..., self.grip_dims] = self.grip.decode(grip_classes).to(full.dtype)
        return full


def frame_map(n_slot_frames, L, ratio):
    """Usable slot frames i and their dataset frame j = round(i * ratio) (< L)."""
    usable = min(n_slot_frames, int(math.floor((L - 1) / ratio)) + 1)
    return usable


def fit_delta_scaler(episodes, slot_data, actions, pc, ratio, abs_norm, stride=5):
    deltas = []
    for ep in episodes:
        acts = actions[ep][0]
        L = len(acts)
        i = np.arange(0, frame_map(len(slot_data[ep][0]), L, ratio), stride)
        j = np.minimum(np.round(i * ratio).astype(int), L - 1)
        idx = np.minimum(j[:, None] + np.arange(pc.horizon), L - 1)
        deltas.append(acts[idx] - acts[np.maximum(j - 1, 0)][:, None])
    return DeltaScaler.fit(np.concatenate(deltas), abs_norm.span)


class ChunkDataset(Dataset):
    """One sample per slot frame i: slots/geometry AND state of frames (i-(H-1)g ... i) (state at
    the matching dataset frames j = round(i * src_fps / slot_fps)), last command a[j-1], and the
    target chunk a[j : j+horizon] (padded with the last action), encoded by `codec`."""

    def __init__(self, episodes, slot_data, actions, codec, abs_norm, state_norm, pc, ratio,
                 stride=1, with_raw=False, rollout_window=0):
        self.pc, self.ratio, self.codec, self.with_raw = pc, ratio, codec, with_raw
        self.K = rollout_window  # replans kept before frame i (every replan_every frames) for rollouts
        self.data, self.index = {}, []
        for ep in episodes:
            slots, geom = slot_data[ep]
            acts, states = actions[ep]
            L = len(acts)
            usable = frame_map(len(slots), L, ratio)
            if abs(len(slots) - L / ratio) > 3:
                print(f"  note: episode {ep}: {len(slots)} slot frames vs {L} action rows "
                      f"(expected ~{L / ratio:.0f}); using {usable}.")
            self.data[ep] = {
                "slots": torch.from_numpy(slots[:usable]),
                "geom": torch.from_numpy(geom[:usable]).float(),
                "acts": torch.from_numpy(acts),
                "states": torch.from_numpy(states),
                "state_n": torch.from_numpy(np.clip(state_norm.encode(states), -3, 3)),
                "prev_n": torch.from_numpy(np.clip(abs_norm.encode(acts), -3, 3)),
            }
            self.index += [(ep, i) for i in range(0, usable, stride)]
        if not self.index:
            raise RuntimeError("Empty dataset.")

    def __len__(self):
        return len(self.index)

    def _hist(self, i):
        H, g = self.pc.n_hist, self.pc.hist_gap
        return [max(0, i - (H - 1 - h) * g) for h in range(H)]

    def __getitem__(self, idx):
        ep, i = self.index[idx]
        d = self.data[ep]
        L = len(d["acts"])
        to_j = lambda f: min(int(round(f * self.ratio)), L - 1)
        j = to_j(i)
        a_idx = torch.arange(j, j + self.pc.horizon).clamp_max(L - 1)
        p = max(j - 1, 0)
        chunk, anchor = d["acts"][a_idx], d["acts"][p]
        hist = self._hist(i)
        item = {"slots": d["slots"][hist].float(), "geom": d["geom"][hist],
                "state": d["state_n"][[to_j(f) for f in hist]],   # [n_hist, S]: measured motion
                "prev_action": d["prev_n"][p],
                "state_hist_raw": d["states"][[to_j(f) for f in hist]],
                "anchor_raw": anchor,
                "target": self.codec.encode(chunk, anchor).float(),
                "chunk_raw": chunk}
        if self.codec.grip is not None:
            item["grip_target"] = self.codec.grip.encode(chunk)          # [horizon, G] level indices
        if self.K > 0:
            # replans m = 0..K ending at frame i (frames before the episode start are clamped to 0)
            frames = [max(0, i - (self.K - m) * self.pc.replan_every) for m in range(self.K + 1)]
            js = [to_j(f) for f in frames]
            item.update(
                r_slots=torch.stack([d["slots"][self._hist(f)] for f in frames]).float(),
                r_geom=torch.stack([d["geom"][self._hist(f)] for f in frames]),
                r_state=torch.stack([d["state_n"][[to_j(x) for x in self._hist(f)]] for f in frames]),
                r_state_raw=torch.stack([d["states"][[to_j(x) for x in self._hist(f)]] for f in frames]),
                r_anchor_raw=d["acts"][[max(x - 1, 0) for x in js]],
                r_j=torch.tensor(js))
        if self.with_raw:
            item.update(state_raw=d["states"][j])
        return item


# %% [6] Policy model ==========================================================
def fourier(x, bands):
    """[..., C] -> [..., C * 2 * bands]."""
    freqs = (2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)) * math.pi
    ang = x[..., None] * freqs
    return torch.cat([ang.sin(), ang.cos()], -1).flatten(-2)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def token_mlp(in_dim, d):
    return nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))


class TimeEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, tau):
        half = self.d // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device) / half)
        ang = 1000.0 * tau[:, None].float() * freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))


class ObjectTokenizer(nn.Module):
    """One token per (frame, slot). No slot-index embedding -> permutation-invariant."""

    def __init__(self, pc, slot_dim):
        super().__init__()
        self.pc = pc
        b = pc.fourier_bands
        in_dim = slot_dim + (5 * 2 * b + 2 * 2 * b if pc.use_geometry else 0)
        self.norm = nn.LayerNorm(slot_dim)
        self.mlp = token_mlp(in_dim, pc.d_model)
        self.frame_emb = nn.Parameter(torch.zeros(pc.n_hist, 1, pc.d_model))

    def forward(self, slots, geom):
        """slots [B, H, K, S], geom [B, H, K, 5] -> tokens [B, H*K, d], pad [B, H*K]."""
        parts = [self.norm(slots)]
        if self.pc.use_geometry:
            vel = torch.zeros_like(geom[..., :2])
            vel[:, 1:] = geom[:, 1:, :, :2] - geom[:, :-1, :, :2]  # slot k = same object over time
            parts += [fourier(geom, self.pc.fourier_bands), fourier(5.0 * vel, self.pc.fourier_bands)]
        tokens = self.mlp(torch.cat(parts, -1)) + self.frame_emb
        area = geom[..., 4]
        pad = area < self.pc.min_slot_area
        if self.training and self.pc.slot_dropout > 0:
            pad = pad | (torch.rand(pad.shape, device=pad.device) < self.pc.slot_dropout)
        pad = pad.scatter(-1, area.argmax(-1, keepdim=True), False)  # always keep the largest slot
        return tokens.flatten(1, 2), pad.flatten(1, 2)


class FlowBlock(nn.Module):
    """Self-attention over the chunk, cross-attention to the scene, MLP; AdaLN-Zero on c."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, ctx, ctx_pad, c):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.ada(c)[:, None].chunk(9, -1)
        h = modulate(self.norm1(x), b1, s1)
        x = x + g1 * self.self_attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), b2, s2)
        x = x + g2 * self.cross_attn(h, ctx, ctx, key_padding_mask=ctx_pad, need_weights=False)[0]
        h = modulate(self.norm3(x), b3, s3)
        return x + g3 * self.mlp(h)


class GripperHead(nn.Module):
    """Learned per-step queries cross-attend to the scene tokens -> level logits per gripper and step."""

    def __init__(self, pc, n_grip, n_levels):
        super().__init__()
        d = pc.d_model
        self.n_grip, self.n_levels = n_grip, n_levels
        self.step_queries = nn.Parameter(torch.randn(pc.horizon, d) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, pc.heads, 4 * d, dropout=pc.flow_dropout, batch_first=True, norm_first=True),
            2)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_grip * n_levels)

    def forward(self, enc):
        B = enc["ctx"].shape[0]
        x = self.decoder(self.step_queries.expand(B, -1, -1), enc["ctx"], memory_key_padding_mask=enc["pad"])
        return self.out(self.norm(x)).view(B, -1, self.n_grip, self.n_levels)   # [B, H, G, L_max]


def repeat_enc(enc, n):
    return enc if n == 1 else {k: v.repeat_interleave(n, 0) for k, v in enc.items()}


class OCFlowPolicy(nn.Module):
    """Rectified flow over (delta) arm action chunks + gripper level classification,
    conditioned on slot / state / last-command tokens."""

    def __init__(self, pc, slot_dim, state_dim, action_dim, rel_scale=None, flow_dim=None, grip_table=None):
        super().__init__()
        self.pc = pc
        self.action_dim = action_dim
        self.flow_dim = flow_dim or action_dim          # dims modelled by the flow (arm [+ grippers])
        d = pc.d_model
        self.use_rel = pc.use_rel_state and state_dim == action_dim
        if self.use_rel:   # scale of the delta targets, so drift and targets share units
            scale = torch.ones(action_dim) if rel_scale is None else torch.as_tensor(rel_scale, dtype=torch.float32)
            self.register_buffer("rel_scale", scale)
            self.rel_in = token_mlp(pc.n_hist * state_dim, d)
        self.objects = ObjectTokenizer(pc, slot_dim) if pc.use_slots else None
        self.state_in = token_mlp(pc.n_hist * state_dim, d) if pc.use_state else None  # state history
        self.prev_in = token_mlp(action_dim, d) if pc.use_prev_action else None
        self.type_emb = nn.Embedding(4, d)  # 0 object, 1 state, 2 last command, 3 state - last command
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, pc.heads, 4 * d, dropout=pc.dropout, batch_first=True, norm_first=True),
            pc.enc_layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)

        self.time = TimeEmbedding(d)
        self.cond = nn.Linear(d, d)
        self.act_in = nn.Linear(self.flow_dim, d)
        self.act_pos = nn.Parameter(torch.randn(pc.horizon, d) * 0.02)
        self.blocks = nn.ModuleList([FlowBlock(d, pc.heads, pc.flow_dropout) for _ in range(pc.dec_layers)])
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, self.flow_dim)
        for layer in (self.final_ada[1], self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.grip_head = None
        if grip_table is not None:                       # classified grippers: levels [G, L_max], NaN = padding
            table = torch.as_tensor(np.asarray(grip_table, np.float32))
            self.register_buffer("grip_valid", ~torch.isnan(table))
            self.register_buffer("grip_table", torch.nan_to_num(table, nan=1e9))
            self.grip_head = GripperHead(pc, *table.shape)
            # logit bonus for the CURRENT gripper level (from the last command): the head starts as
            # "keep the current level" (like a zero delta for the arm) and learns when to switch
            self.grip_hold_bias = nn.Parameter(torch.tensor(3.0))

    def encode_obs(self, obs):
        B = obs["state"].shape[0]
        tokens, pads = [], []
        if self.objects is not None:
            t, p = self.objects(obs["slots"], obs["geom"])
            tokens.append(t + self.type_emb.weight[0])
            pads.append(p)
        for key, net, k in (("state", self.state_in, 1), ("prev_action", self.prev_in, 2)):
            if net is not None:
                tokens.append((net(obs[key].flatten(1)) + self.type_emb.weight[k])[:, None])
                pads.append(torch.zeros(B, 1, dtype=torch.bool, device=obs[key].device))
        if self.use_rel:
            rel = (obs["state_hist_raw"] - obs["anchor_raw"][:, None]) / self.rel_scale
            tokens.append((self.rel_in(rel.clamp(-10, 10).flatten(1)) + self.type_emb.weight[3])[:, None])
            pads.append(torch.zeros(B, 1, dtype=torch.bool, device=rel.device))
        tokens, pad = torch.cat(tokens, 1), torch.cat(pads, 1)
        ctx = self.enc_norm(self.encoder(tokens, src_key_padding_mask=pad))
        keep = (~pad).to(ctx.dtype)[..., None]
        return {"ctx": ctx, "pad": pad, "summary": (ctx * keep).sum(1) / keep.sum(1)}

    def velocity(self, a_tau, tau, enc):
        c = self.time(tau) + self.cond(enc["summary"])
        x = self.act_in(a_tau) + self.act_pos
        for blk in self.blocks:
            x = blk(x, enc["ctx"], enc["pad"], c)
        shift, scale = self.final_ada(c)[:, None].chunk(2, -1)
        return self.out(modulate(self.final_norm(x), shift, scale))

    def current_grip_level(self, anchor_raw):
        """Level index [B, G] closest to the last gripper command."""
        x = anchor_raw[:, self.pc.gripper_dims].float()
        return (x[..., None] - self.grip_table).abs().argmin(-1)

    def grip_logits(self, enc, anchor_raw):
        logits = self.grip_head(enc).float()
        current = F.one_hot(self.current_grip_level(anchor_raw), logits.shape[-1]).float()
        logits = logits + self.grip_hold_bias * current[:, None]
        return logits.masked_fill(~self.grip_valid, -1e4)        # padded levels never chosen

    def grip_loss(self, logits, target):
        """Cross-entropy with label smoothing spread over the VALID levels of each gripper only."""
        valid = self.grip_valid.float()                           # [G, L]
        logp = logits.log_softmax(-1).masked_fill(~self.grip_valid, 0.0)
        eps = self.pc.gripper_label_smoothing
        q = (1 - eps) * F.one_hot(target, logits.shape[-1]).float() + eps * valid / valid.sum(-1, keepdim=True)
        return -(q * logp).sum(-1).mean()

    def loss(self, batch, generator=None):
        """Rectified flow on the flow dims: a_tau = (1 - tau) eps + tau a, target velocity a - eps
        (+ gripper cross-entropy). Returns (total, {"flow": ..., "grip_ce": ...})."""
        obs = {k: batch[k] for k in OBS_KEYS}
        if self.training and self.pc.input_noise > 0:
            for k in ("state", "prev_action"):
                obs[k] = obs[k] + self.pc.input_noise * torch.randn_like(obs[k])
        target = batch["target"]
        enc = self.encode_obs(obs)
        eps = torch.randn(target.shape, device=target.device, generator=generator)
        tau = torch.sigmoid(torch.randn(target.shape[0], device=target.device, generator=generator))
        t = tau[:, None, None]
        a_tau = (1 - t) * eps + t * target
        flow = F.mse_loss(self.velocity(a_tau, tau, enc).float(), (target - eps).float())
        parts = {"flow": flow}
        total = flow
        if self.grip_head is not None:
            parts["grip_ce"] = self.grip_loss(self.grip_logits(enc, obs["anchor_raw"]), batch["grip_target"])
            total = total + self.pc.gripper_loss_weight * parts["grip_ce"]
        return total, parts

    @torch.no_grad()
    def sample(self, obs, steps=None, generator=None, enc=None):
        """Euler integration from noise (tau = 0) to a normalized flow target chunk (tau = 1)."""
        steps = steps or self.pc.flow_steps
        enc = enc if enc is not None else self.encode_obs(obs)
        B = enc["ctx"].shape[0]
        a = torch.randn(B, self.pc.horizon, self.flow_dim, device=enc["ctx"].device, generator=generator)
        for i in range(steps):
            tau = torch.full((B,), i / steps, device=a.device)
            a = a + self.velocity(a, tau, enc).float() / steps
        return a

    @torch.no_grad()
    def predict(self, obs, n_samples=1, steps=None, generator=None):
        """flow_samples [B, n, H, F], flow (sample mean) [B, H, F],
        grip (most likely level) [B, H, G] and grip_logits — the last two if grippers are classified."""
        enc = self.encode_obs(obs)
        B = enc["ctx"].shape[0]
        samples = self.sample(None, steps=steps, generator=generator, enc=repeat_enc(enc, n_samples))
        samples = samples.view(B, n_samples, *samples.shape[1:])
        out = {"flow_samples": samples, "flow": samples.mean(1)}
        if self.grip_head is not None:
            out["grip_logits"] = self.grip_logits(enc, obs["anchor_raw"])
            out["grip"] = out["grip_logits"].argmax(-1)
        return out


@torch.no_grad()
def predict_raw(policy, codec, obs, steps=None, n_samples=1, generator=None):
    """obs (incl. anchor_raw = last command) -> raw action chunk [B, H, A] and the prediction dict."""
    pr = policy.predict(obs, n_samples=n_samples, steps=steps, generator=generator)
    return codec.decode(pr["flow"], obs["anchor_raw"].float(), pr.get("grip")), pr


# %% [7] Rollout on the policy's own commands ==================================
@torch.no_grad()
def own_anchor_rollout(policy, batch, k, pc, codec, steps, n_samples=1, generator=None):
    """Run k replans without gradients, each using the policy's OWN previous output as the
    last command, and return a batch for the replan that follows (frame i):
    prev_action = encoded own anchor, target = true chunk relative to the own anchor.
    Requires the r_* keys of ChunkDataset (rollout_k > 0) and k <= rollout_k."""
    K = batch["r_j"].shape[1] - 1
    B = batch["r_j"].shape[0]
    rows = torch.arange(B, device=batch["r_j"].device)
    own = batch["r_anchor_raw"][:, K - k].float()             # recorded command before the window
    was_training = policy.training
    policy.eval()
    for m in range(K - k, K):
        obs = {"slots": batch["r_slots"][:, m], "geom": batch["r_geom"][:, m], "state": batch["r_state"][:, m],
               "prev_action": codec.abs_norm.encode(own).clamp(-3, 3),
               "state_hist_raw": batch["r_state_raw"][:, m], "anchor_raw": own}
        chunk, _ = predict_raw(policy, codec, obs, steps=steps, n_samples=n_samples, generator=generator)
        chunk = chunk.float()                                      # raw units, [B, H, A]
        span = (batch["r_j"][:, m + 1] - batch["r_j"][:, m]).clamp(0, pc.horizon)
        executed = chunk[rows, (span - 1).clamp(min=0)]            # last action executed before replan m+1
        own = torch.where((span > 0)[:, None], executed, own)      # no time passed -> keep the anchor
    policy.train(was_training)
    out = {key: batch[key] for key in OBS_KEYS}
    out["prev_action"] = codec.abs_norm.encode(own).clamp(-3, 3)
    out["anchor_raw"] = own
    out["target"] = codec.encode(batch["chunk_raw"].float(), own).clamp(-10, 10)
    out["own_anchor_raw"] = own
    if "grip_target" in batch:            # gripper levels are absolute: target unchanged by the drift
        out["grip_target"] = batch["grip_target"]
    return out


def rollout_prob_at(pc, epoch, start):
    """`start` = first epoch with rollouts (None until an evaluation unlocked them)."""
    if pc.rollout_k and pc.rollout_always:
        return 1.0
    if pc.rollout_k == 0 or start is None or epoch < start:
        return 0.0
    ramp = min(1.0, (epoch - start + 1) / max(1, pc.rollout_ramp_epochs))
    return pc.rollout_prob * ramp


def sample_k(pc, p):
    if random.random() >= p:
        return 0
    return random.randint(1, pc.rollout_k) if pc.rollout_sample else pc.rollout_k


# %% [8] Training ==============================================================
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1 - d)
        for e, b in zip(self.model.buffers(), model.buffers()):
            e.copy_(b)


def build_optimizer(model, pc):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "emb" not in name and "act_pos" not in name else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": pc.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=pc.lr, betas=(0.9, 0.95))


def build_scheduler(optimizer, pc, total_steps):
    def lr_lambda(step):
        if step < pc.warmup_steps:
            return (step + 1) / pc.warmup_steps
        progress = min(1.0, (step - pc.warmup_steps) / max(1, total_steps - pc.warmup_steps))
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(policy, ema, loader, optimizer, scheduler, scaler, pc, codec, device, epoch, step,
                    rollout_start=None):
    policy.train()
    total, n, k_sum = 0.0, 0, 0
    part_sums = defaultdict(float)
    p_roll = rollout_prob_at(pc, epoch, rollout_start)
    for batch in loader:
        batch = to_device(batch, device)
        k = sample_k(pc, p_roll)
        with autocast(device):
            if k > 0:   # burn-in on own commands with the EMA (= deployed) weights, no grad;
                        # the gradient only flows through the last replan
                rolled = own_anchor_rollout(ema.model, batch, k, pc, codec, pc.rollout_flow_steps)
                loss, parts = policy.loss(rolled)
                if pc.rollout_clean_loss:
                    loss_c, parts_c = policy.loss(batch)
                    loss = 0.5 * (loss + loss_c)
                    parts = {key: 0.5 * (parts[key] + parts_c[key]) for key in parts}
            else:
                loss, parts = policy.loss(batch)
        k_sum += k
        for key, v in parts.items():
            part_sums[key] += v.item()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(policy.parameters(), pc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(policy, step)
        step += 1
        total += loss.item()
        n += 1
    parts = {key: v / n for key, v in part_sums.items()}
    grip = f" | gripper CE {parts['grip_ce']:.4f}" if "grip_ce" in parts else ""
    print(f"epoch {epoch:4d} | flow loss {parts['flow']:.4f}{grip} | rollout p {p_roll:.2f}, mean k {k_sum / n:.2f} | "
          f"lr {scheduler.get_last_lr()[0]:.2e}")
    return {"loss": total / n, **parts, "k": k_sum / n}, step


@torch.no_grad()
def evaluate(policy, loader, pc, codec, device):
    """On the first `exec_steps` steps (what is executed before replanning), in raw action units:
       l1_exec         policy (mean of `eval_samples` samples)
       l1_hold_action  baseline: repeat the last command a[j-1]
       l1_hold_state   baseline: command = current state (if state and action have the same size)
       gain            1 - l1_exec / l1_hold_action   (> 0: the policy beats holding)
       l1_exec_roll    policy after `eval_rollout_k` replans on its OWN commands (drift), same truth
       gain_roll       1 - l1_exec_roll / l1_hold_action
       *_arm / *_grip  the same L1s restricted to the arm / gripper dims
       grip_acc        fraction of executed steps where the predicted gripper level is right
    In normalized flow-target units: flow loss, mse_mean / mse_best (whole chunk), mse_exec,
    spread (std across samples; ~0 means the flow collapsed to one deterministic answer)."""
    policy.eval()
    gen = torch.Generator(device=device).manual_seed(0)
    n_s, E = pc.eval_samples, pc.exec_steps
    sums, count = defaultdict(float), 0
    dim_sums, dim_count = defaultdict(lambda: 0.0), 0
    for batch in loader:
        batch = to_device(batch, device)
        B = batch["target"].shape[0]
        with autocast(device):
            _, parts = policy.loss(batch, generator=gen)
            chunk_raw, pr = predict_raw(policy, codec, {k: batch[k] for k in OBS_KEYS},
                                        n_samples=n_s, generator=gen)
        target = batch["target"].float()
        samples = pr["flow_samples"].float()
        mean = pr["flow"].float()

        tgt_raw = batch["chunk_raw"][:, :E].float()
        pred_raw = chunk_raw[:, :E].float()
        err = {"policy": (pred_raw - tgt_raw).abs(),
               "hold_action": (batch["anchor_raw"].float()[:, None] - tgt_raw).abs()}
        if batch["state_raw"].shape[-1] == tgt_raw.shape[-1]:
            err["hold_state"] = (batch["state_raw"].float()[:, None] - tgt_raw).abs()
        if pc.eval_rollout_k > 0:
            with autocast(device):
                rolled = own_anchor_rollout(policy, batch, pc.eval_rollout_k, pc, codec, pc.flow_steps,
                                            n_samples=n_s, generator=gen)
                chunk_r, _ = predict_raw(policy, codec, {k: rolled[k] for k in OBS_KEYS},
                                         n_samples=n_s, generator=gen)
            err["roll"] = (chunk_r[:, :E].float() - tgt_raw).abs()
        for k, v in err.items():
            dim_sums[k] = dim_sums[k] + v.sum((0, 1)).cpu()
        dim_count += B * E

        metrics = {
            **{key: v.item() for key, v in parts.items()},
            "mse_mean": ((mean - target) ** 2).mean().item(),
            "mse_best": ((samples - target[:, None]) ** 2).mean((2, 3)).min(1).values.mean().item(),
            "mse_exec": ((mean[:, :E] - target[:, :E]) ** 2).mean().item(),
            "spread": samples.std(1).mean().item() if n_s > 1 else 0.0,
        }
        if "grip" in pr:
            metrics["grip_acc"] = (pr["grip"][:, :E] == batch["grip_target"][:, :E]).float().mean().item()
        for k, v in metrics.items():
            sums[k] += v * B
        count += B

    out = {k: v / count for k, v in sums.items()}
    per_dim = {k: (v / dim_count).numpy() for k, v in dim_sums.items()}
    out["l1_exec"] = float(per_dim["policy"].mean())
    out["l1_hold_action"] = float(per_dim["hold_action"].mean())
    out["l1_hold_state"] = float(per_dim["hold_state"].mean()) if "hold_state" in per_dim else float("nan")
    out["gain"] = 1.0 - out["l1_exec"] / max(out["l1_hold_action"], 1e-12)
    if "roll" in per_dim:
        out["l1_exec_roll"] = float(per_dim["roll"].mean())
        out["gain_roll"] = 1.0 - out["l1_exec_roll"] / max(out["l1_hold_action"], 1e-12)
    grip = [d for d in pc.gripper_dims if d < len(per_dim["policy"])]
    arm = [d for d in range(len(per_dim["policy"])) if d not in grip]
    for name, dims in (("arm", arm), ("grip", grip)):
        if dims:
            out[f"l1_exec_{name}"] = float(per_dim["policy"][dims].mean())
            out[f"l1_hold_{name}"] = float(per_dim["hold_action"][dims].mean())
            if "roll" in per_dim:
                out[f"l1_exec_roll_{name}"] = float(per_dim["roll"][dims].mean())
    out["per_dim"] = {k: v.round(6).tolist() for k, v in per_dim.items()}
    return out


def save_checkpoint(path, pc, policy, ema, optimizer, scheduler, scaler, state, meta):
    tmp = path + ".tmp"
    torch.save({
        "config": asdict(pc),
        "meta": meta,                      # dims, normalizers, slot-model info, fps, splits
        "model": ema.model.state_dict(),   # EMA weights: use these for inference
        "raw_model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        **state,                           # epoch, step, history, best, bad_evals
    }, tmp)
    os.replace(tmp, path)


def build_codec(pc, meta):
    grip = GripperCoder.from_state_dict(meta["gripper"]) if meta.get("gripper") else None
    return ActionCodec(pc.action_mode, QuantileNormalizer.from_state_dict(meta["action_norm"]),
                       DeltaScaler.from_state_dict(meta["delta_scale"]),
                       grip_dims=pc.gripper_dims, grip_mode=pc.gripper_mode, grip=grip)


def load_policy(path, device):
    """Rebuild the policy (EMA weights), action codec and normalizers from a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    pc, meta = PolicyConfig.from_dict(ckpt["config"]), ckpt["meta"]
    codec = build_codec(pc, meta)
    policy = OCFlowPolicy(pc, meta["slot_dim"], meta["state_dim"], meta["action_dim"],
                          rel_scale=meta["delta_scale"]["scale"], flow_dim=codec.flow_dim,
                          grip_table=codec.grip.table if codec.grip is not None else None)
    policy.load_state_dict(ckpt["model"], strict=True)
    policy.to(device).eval()
    return policy, codec, QuantileNormalizer.from_state_dict(meta["state_norm"]), pc, meta


def make_loader(ds, pc, shuffle):
    return DataLoader(ds, batch_size=pc.batch_size, shuffle=shuffle, drop_last=shuffle and len(ds) > pc.batch_size,
                      num_workers=pc.num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=pc.num_workers > 0)


def fmt_eval(ev):
    keys = ["l1_exec", "l1_exec_roll", "l1_hold_action", "l1_hold_state", "gain", "gain_roll",
            "l1_exec_arm", "l1_exec_roll_arm", "l1_hold_arm", "l1_exec_grip", "l1_exec_roll_grip", "l1_hold_grip",
            "grip_acc", "mse_exec", "mse_mean", "mse_best", "spread", "flow", "grip_ce"]
    return " | ".join(f"{k} {ev[k]:.4f}" for k in keys if k in ev)


def per_dim_table(ev):
    table = pd.DataFrame(ev["per_dim"])
    table["gain"] = 1 - table["policy"] / table["hold_action"].clip(lower=1e-12)
    if "roll" in table:
        table["gain_roll"] = 1 - table["roll"] / table["hold_action"].clip(lower=1e-12)
    table.index.name = "action dim"
    table["part"] = ["gripper" if d in ev.get("_grip_dims", []) else "arm" for d in table.index]
    return table


def main(pc):
    seed_everything(pc.seed)
    device = get_device()
    os.makedirs(pc.out_dir, exist_ok=True)

    # 1. Frozen slot model, episodes, splits
    slot_model, scfg, feat_dim, slot_hash = load_slot_model(pc, device)
    info, episodes = load_metadata(scfg)
    all_ids = episodes["episode_index"].astype(int).tolist()
    splits = policy_splits(pc, scfg, all_ids)
    ids = sorted(splits["train"] + splits["eval"])
    selected = episodes[episodes["episode_index"].isin(ids)]
    print(f"Episodes: {len(splits['train'])} train / {len(splits['eval'])} eval "
          f"({len(splits['eval_slot_heldout'])} held out from slot training) | dataset {info['fps']} fps")

    # 2-3. Propagated slots (cached) and actions
    slot_data = ensure_slots(pc, scfg, slot_model, slot_hash, info, selected, ids, device)
    del slot_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    actions = load_actions(pc, scfg, info, selected, ids)

    # 4. Normalization, fitted on the training episodes only
    train_eps, eval_eps = splits["train"], splits["eval"]
    ratio = info["fps"] / scfg.target_fps
    if (pc.rollout_k or pc.eval_rollout_k) and pc.replan_every * ratio > pc.horizon:
        raise ValueError(f"replan_every={pc.replan_every} slot frames is longer than the {pc.horizon}-step chunk.")
    abs_norm = QuantileNormalizer.fit(np.concatenate([actions[e][0] for e in train_eps]))
    state_norm = QuantileNormalizer.fit(np.concatenate([actions[e][1] for e in train_eps]))
    delta_scale = fit_delta_scaler(train_eps, slot_data, actions, pc, ratio, abs_norm)
    action_dim, state_dim = actions[ids[0]][0].shape[1], actions[ids[0]][1].shape[1]
    bad = [d for d in pc.gripper_dims if d >= action_dim]
    if bad:
        raise ValueError(f"gripper_dims {bad} out of range for {action_dim}-D actions.")
    grip = None
    if pc.gripper_dims and pc.gripper_mode == "classify":
        train_acts = np.concatenate([actions[e][0] for e in train_eps])
        levels = [fit_gripper_levels(train_acts[::2, d], pc.gripper_max_levels, pc.gripper_level_tol)
                  for d in pc.gripper_dims]
        grip = GripperCoder(pc.gripper_dims, levels)
        for d, l in zip(pc.gripper_dims, levels):
            v = train_acts[:, d]
            dev = np.abs(v[:, None] - l[None]).min(1)
            print(f"Gripper dim {d}: {len(l)} levels {np.round(l, 3).tolist()} | mean |value - level| "
                  f"{dev.mean():.4f} ({dev.mean() / max(np.ptp(v), 1e-6):.1%} of range)")
    codec = ActionCodec(pc.action_mode, abs_norm, delta_scale, grip_dims=pc.gripper_dims,
                        grip_mode=pc.gripper_mode, grip=grip)
    slot_dim = slot_data[ids[0]][0].shape[-1]

    train_ds = ChunkDataset(train_eps, slot_data, actions, codec, abs_norm, state_norm, pc, ratio,
                            rollout_window=pc.rollout_k)
    eval_ds = ChunkDataset(eval_eps, slot_data, actions, codec, abs_norm, state_norm, pc, ratio,
                           stride=pc.eval_stride, with_raw=True, rollout_window=pc.eval_rollout_k)
    train_loader, eval_loader = make_loader(train_ds, pc, True), make_loader(eval_ds, pc, False)
    print(f"Samples: {len(train_ds):,} train / {len(eval_ds):,} eval | action {action_dim}-D, "
          f"state {state_dim}-D | {pc.action_mode} chunk of {pc.horizon} steps = {pc.horizon / info['fps']:.2f} s")
    if pc.gripper_dims:
        print(f"Grippers {pc.gripper_dims}: {codec.grip_mode} | flow models {codec.flow_dim} dims")
    if pc.rollout_k and pc.rollout_always:
        print(f"Rollout training on every batch from epoch 1: k ~ U{{1..{pc.rollout_k}}} replans every "
              f"{pc.replan_every / scfg.target_fps:.1f} s on own commands (no grad), gradient on the last replan only")
    elif pc.rollout_k:
        print(f"Rollout training once an eval (epoch >= {pc.rollout_start_epoch}) beats hold, ramp {pc.rollout_ramp_epochs} epochs: "
              f"k ~ U{{1..{pc.rollout_k}}} replans every {pc.replan_every / scfg.target_fps:.1f} s on own commands "
              f"for up to {pc.rollout_prob:.0%} of batches; gradient on the last replan only")

    meta = {"slot_ckpt": pc.slot_ckpt, "slot_hash": slot_hash, "slot_config": vars(scfg).copy(),
            "feat_dim": feat_dim, "slot_dim": slot_dim, "state_dim": state_dim, "action_dim": action_dim,
            "dataset_fps": info["fps"], "slot_fps": scfg.target_fps, "splits": splits,
            "action_norm": abs_norm.state_dict(), "delta_scale": delta_scale.state_dict(),
            "state_norm": state_norm.state_dict(), "gripper": grip.state_dict() if grip is not None else None}

    # 5. Model / optimization
    policy = OCFlowPolicy(pc, slot_dim, state_dim, action_dim, rel_scale=delta_scale.scale,
                          flow_dim=codec.flow_dim, grip_table=grip.table if grip is not None else None).to(device)
    ema = EMA(policy, pc.ema_decay)
    optimizer = build_optimizer(policy, pc)
    scheduler = build_scheduler(optimizer, pc, pc.num_epochs * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype(device) == torch.float16)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")

    latest, best_path = os.path.join(pc.out_dir, "latest.pt"), os.path.join(pc.out_dir, "best.pt")
    state = {"epoch": 0, "step": 0, "history": [], "best": float("inf"), "bad_evals": 0, "stopped": False,
             "rollout_start": None, "selection_reset": False}
    if pc.resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        if ckpt["config"] == asdict(pc) and ckpt["meta"]["slot_hash"] == slot_hash \
                and ckpt["meta"]["splits"] == splits:
            policy.load_state_dict(ckpt["raw_model"])
            ema.model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            state = {k: ckpt.get(k, v) for k, v in state.items()}
            print(f"Resumed from epoch {state['epoch']}.")
        else:
            print("Config, slot model or splits changed since latest.pt — training from scratch.")
    if pc.rollout_k and pc.rollout_always and state["rollout_start"] is None:
        state["rollout_start"] = 1

    # 6. Train / evaluate with early stopping
    epoch = state["epoch"]
    while epoch < pc.num_epochs and not state["stopped"]:
        epoch += 1
        tr, state["step"] = train_one_epoch(policy, ema, train_loader, optimizer, scheduler, scaler,
                                            pc, codec, device, epoch, state["step"], state["rollout_start"])
        entry = {"epoch": epoch, "train": tr}
        if epoch % pc.eval_every == 0 or epoch == pc.num_epochs:
            ev = evaluate(ema.model, eval_loader, pc, codec, device)
            ev["_grip_dims"] = pc.gripper_dims
            entry["eval"] = ev
            print("  eval (EMA): " + fmt_eval(ev))
            # with rollouts, best.pt and early stopping only count once rollouts are fully ramped in:
            # earlier checkpoints never trained on their own commands and drift on the robot
            ramped = not pc.rollout_k or pc.rollout_always or (
                state["rollout_start"] is not None and epoch >= state["rollout_start"] + pc.rollout_ramp_epochs - 1)
            if pc.rollout_k and ramped and not state["selection_reset"]:
                # first evaluation with rollouts fully ramped in: pre-rollout checkpoints no longer count
                state["best"], state["bad_evals"], state["selection_reset"] = float("inf"), 0, True
                if not pc.rollout_always:
                    print("  rollouts fully ramped in: best checkpoint selection restarts")
            if ev[pc.best_metric] < state["best"]:
                state["best"], state["bad_evals"] = ev[pc.best_metric], 0
                save_checkpoint(best_path, pc, policy, ema, optimizer, scheduler, scaler,
                                {**state, "epoch": epoch, "history": state["history"] + [entry]}, meta)
                print(f"  new best ({pc.best_metric} = {state['best']:.4f})"
                      + ("" if ramped else "  [pre-rollout; replaced once rollouts are ramped in]"))
            elif ramped:
                state["bad_evals"] += 1
                if pc.patience and state["bad_evals"] >= pc.patience:
                    state["stopped"] = True
                    print(f"  early stop: no improvement in {pc.patience} evaluations")
            if (pc.rollout_k and not pc.rollout_always and state["rollout_start"] is None
                    and epoch >= pc.rollout_start_epoch and ev["gain"] > pc.rollout_min_gain):
                state["rollout_start"] = epoch + 1
                print(f"  policy beats hold (gain {ev['gain']:+.1%}) -> rollout training from epoch {epoch + 1}")
        state["history"].append(entry)
        state["epoch"] = epoch
        save_checkpoint(latest, pc, policy, ema, optimizer, scheduler, scaler, state, meta)

    history = state["history"]
    with open(os.path.join(pc.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)

    if pc.rollout_k and state["rollout_start"] is None:
        print("\nWARNING: rollouts never started (the policy never beat hold) — best.pt is NOT rollout-trained.")
    elif pc.rollout_k and not state["selection_reset"]:
        print("\nWARNING: training ended before rollouts were fully ramped in — best.pt is NOT rollout-trained. "
              "Increase num_epochs or patience.")
    evals = [h for h in history if "eval" in h]
    if pc.rollout_k and state["rollout_start"] is not None:
        evals = [h for h in evals if h["epoch"] >= state["rollout_start"] + pc.rollout_ramp_epochs - 1] or evals
    if evals:
        best = min(evals, key=lambda h: h["eval"][pc.best_metric])
        print(f"\nBest epoch {best['epoch']}: " + fmt_eval(best["eval"]))
        print("Per action dimension (L1 on executed steps, raw units):")
        print(per_dim_table(best["eval"]).round(4).to_string())
    return history


def plot_history(history, pc):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ev = [h for h in history if "eval" in h]
    if not ev:
        return
    e_ep = [h["epoch"] for h in ev]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    axes[0].plot([h["epoch"] for h in history], [h["train"]["flow"] for h in history], label="train flow")
    axes[0].plot(e_ep, [h["eval"]["flow"] for h in ev], "o-", label="eval flow (EMA)")
    if "grip_ce" in history[0]["train"]:
        axes[0].plot([h["epoch"] for h in history], [h["train"]["grip_ce"] for h in history], label="train gripper CE")
        axes[0].plot(e_ep, [h["eval"]["grip_ce"] for h in ev], "o-", label="eval gripper CE")
    axes[0].set_title("losses")
    axes[1].plot(e_ep, [h["eval"]["l1_exec"] for h in ev], "o-", label="policy")
    if "l1_exec_roll" in ev[0]["eval"]:
        axes[1].plot(e_ep, [h["eval"]["l1_exec_roll"] for h in ev], "s-", label=f"policy after {pc.eval_rollout_k} own replans")
    axes[1].plot(e_ep, [h["eval"]["l1_hold_action"] for h in ev], "--", label="hold last command")
    axes[1].plot(e_ep, [h["eval"]["l1_hold_state"] for h in ev], ":", label="hold state")
    axes[1].set_title(f"L1 on the first {pc.exec_steps} steps (action units)")
    axes[2].plot(e_ep, [h["eval"]["spread"] for h in ev], "o-", label="spread")
    axes[2].plot(e_ep, [h["eval"]["mse_best"] / max(h["eval"]["mse_mean"], 1e-12) for h in ev], "o-",
                 label="mse_best / mse_mean")
    axes[2].set_title("sample diversity")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(pc.out_dir, "history.png"), dpi=150)
    plt.show()


# %% [9] Inference helper (robot / replay) ======================================
class OCFlowRunner:
    """Frame -> DINOv2 -> propagated slots -> action chunk (raw action units).

    Call `observe(frame, state)` at the slot fps the model was trained with (10 fps: every 3rd
    camera frame at 30 Hz) with the robot state measured at that frame, then
    `act(last_action)` whenever a new chunk is needed. `last_action` is the last command
    sent to the robot (the delta anchor).
    `frame` is uint8 RGB at any resolution; it is resized like during training."""

    def __init__(self, policy_ckpt, device, encoder=None):
        self.device = device
        self.policy, self.codec, self.state_norm, self.pc, meta = load_policy(policy_ckpt, device)
        pc_slots = PolicyConfig.from_dict({**asdict(self.pc), "slot_ckpt": meta["slot_ckpt"]})
        self.slot_model, self.scfg, _, slot_hash = load_slot_model(pc_slots, device)
        if slot_hash != meta["slot_hash"]:
            print("WARNING: slot checkpoint differs from the one the policy was trained on.")
        self.encoder = encoder or DinoEncoder(self.scfg.dino_model).to(device)
        self.coords = grid_coords(*self.scfg.grid_hw).to(device)
        self.reset()

    def reset(self):
        self.slots = None
        self.gen = torch.Generator(device=self.device).manual_seed(self.pc.slot_seed)
        self.hist = deque(maxlen=(self.pc.n_hist - 1) * self.pc.hist_gap + 1)

    @torch.no_grad()
    def observe(self, frame_rgb, state):
        import cv2
        img = cv2.resize(np.ascontiguousarray(frame_rgb), (self.scfg.img_w, self.scfg.img_h),
                         interpolation=cv2.INTER_AREA)
        with autocast(self.device):
            feats = self.encoder(torch.from_numpy(img)[None].to(self.device)).float()
            if self.slots is None:
                self.slots, _ = self.slot_model.first_step(feats, self.gen)
            else:
                self.slots, _ = self.slot_model.next_step(feats, self.slots)
            masks = self.slot_model.decoder(self.slots.float())[1].float()
        s = torch.as_tensor(np.asarray(state, np.float32), device=self.device)
        self.hist.append((self.slots[0].float(), slot_geometry(masks, self.coords)[0],
                          self.state_norm.encode(s).clamp(-3, 3), s))

    @torch.no_grad()
    def act(self, last_action, steps=None):
        if not self.hist:
            raise RuntimeError("Call observe() first.")
        H, g = self.pc.n_hist, self.pc.hist_gap
        items = list(self.hist)
        picks = [items[max(0, len(items) - 1 - (H - 1 - h) * g)] for h in range(H)]
        last = torch.as_tensor(np.asarray(last_action, np.float32), device=self.device)
        obs = {"slots": torch.stack([p[0] for p in picks])[None],
               "geom": torch.stack([p[1] for p in picks])[None],
               "state": torch.stack([p[2] for p in picks])[None],
               "state_hist_raw": torch.stack([p[3] for p in picks])[None],
               "anchor_raw": last[None],
               "prev_action": self.codec.abs_norm.encode(last).clamp(-3, 3)[None]}
        with autocast(self.device):
            chunk, _ = predict_raw(self.policy, self.codec, obs, steps=steps)
        return chunk[0].float().cpu().numpy()


# %% [10] Run ===================================================================
if __name__ == "__main__":
    PCFG = PolicyConfig()
    HISTORY = main(PCFG)
    plot_history(HISTORY, PCFG)
    print("Best policy:", os.path.join(PCFG.out_dir, "best.pt"))