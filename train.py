import contextlib
import glob
import json
import math
import os
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, fields

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset

from src.models import ClipDataset
from src.models import SlotInit
from src.models import SlotAttention
from src.models import MLPDecoder
from src.models import SlotContrastModel
from src.models import SlotContrastModel
from src.models import DinoEncoder

from src.config import Config

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
        print("WARNING: no GPU found — Runtime > Change runtime type > GPU.")
    return device


def amp_dtype(device):
    """bf16 on Ampere+, fp16 on older GPUs (T4), disabled on CPU."""
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def download_metadata(cfg):
    snapshot_download(
        cfg.repo_id,
        repo_type="dataset",
        local_dir=cfg.data_root,
        allow_patterns=["meta/info.json", "meta/episodes/*"],
    )
    with open(os.path.join(cfg.data_root, "meta", "info.json")) as f:
        info = json.load(f)

    files = sorted(glob.glob(os.path.join(cfg.data_root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    if not files:
        raise RuntimeError("No episode metadata found under meta/episodes/.")
    episodes = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)

    cameras = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    print(f"Dataset {info.get('codebase_version')} | {info['total_episodes']} episodes | "
          f"{info['fps']} fps | cameras: {cameras}")
    if cfg.camera not in cameras:
        raise ValueError(f"Camera '{cfg.camera}' not in dataset. Pick one of {cameras}.")
    return info, episodes


def episode_table(cfg, info, episodes):
    """One row per selected episode with its video file and time window.

    LeRobot v3 packs many episodes into each mp4, so an episode is a
    (file, from_timestamp, to_timestamp) segment — not a whole file.
    """
    prefix = f"videos/{cfg.camera}/"
    needed = ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]
    missing = [prefix + c for c in needed if prefix + c not in episodes.columns]
    if missing:
        cam_cols = [c for c in episodes.columns if cfg.camera in c]
        raise KeyError(f"Missing columns {missing}. Columns for this camera: {cam_cols}")

    rows = []
    for _, ep in episodes.head(cfg.num_episodes).iterrows():
        rel = info["video_path"].format(
            video_key=cfg.camera,
            chunk_index=int(ep[prefix + "chunk_index"]),
            file_index=int(ep[prefix + "file_index"]),
        )
        rows.append({
            "episode_index": int(ep["episode_index"]),
            "video": rel,
            "start": float(ep[prefix + "from_timestamp"]),
            "end": float(ep[prefix + "to_timestamp"]),
            "length": int(ep["length"]) if "length" in ep else None,
        })
    return pd.DataFrame(rows)


def download_videos(cfg, table):
    files = sorted(table["video"].unique())
    print(f"Downloading {len(files)} video file(s) for {len(table)} episodes...")
    snapshot_download(cfg.repo_id, repo_type="dataset", local_dir=cfg.data_root, allow_patterns=files)


def make_splits(cfg, table):
    """Episode-level split (no frame leakage). Saved so reruns are stable."""
    path = os.path.join(cfg.work_dir, "splits.json")
    ids = table["episode_index"].tolist()
    if os.path.exists(path):
        with open(path) as f:
            splits = json.load(f)
        if sorted(sum(splits.values(), [])) == sorted(ids):
            return splits
        print("Episode selection changed — recomputing splits.")

    if len(ids) < 3:
        raise ValueError("Need at least 3 episodes (train, eval, visualization).")
    rng = random.Random(cfg.seed)
    rng.shuffle(ids)
    viz = [ids.pop()]
    n_eval = max(1, round(cfg.eval_fraction * len(ids)))
    splits = {"train": sorted(ids[n_eval:]), "eval": sorted(ids[:n_eval]), "viz": viz}

    os.makedirs(cfg.work_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found: run `!apt-get -y install ffmpeg`.")
    decoders = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True).stdout
    if "av1" not in decoders:
        print("WARNING: this ffmpeg has no AV1 decoder; the dataset videos are AV1.")


def decode_segment(path, start, end, fps, width, height):
    """Decode [start, end) seconds of a video to uint8 RGB frames [T, H, W, 3]."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{start:.4f}", "-i", path, "-t", f"{end - start:.4f}",
        "-vf", f"fps={fps},scale={width}:{height}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}:\n{proc.stderr.decode()[-2000:]}")
    frames = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = height * width * 3
    if frames.size == 0 or frames.size % frame_size:
        raise RuntimeError(f"Decoded {frames.size} bytes from {path} [{start:.2f}, {end:.2f}]s.")
    return frames.reshape(-1, height, width, 3)


def feature_path(cfg, episode_index):
    return os.path.join(cfg.cache_dir, f"ep_{episode_index:06d}_feats.npy")


def display_path(cfg, episode_index):
    return os.path.join(cfg.cache_dir, f"ep_{episode_index:06d}_display.npy")


def save_npy_atomic(path, array):
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def build_feature_cache(cfg, info, table, splits, device, batch_size=64):
    """Decode every selected episode once and store DINO features as fp16 .npy."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    todo = [r for r in table.itertuples() if not os.path.exists(feature_path(cfg, r.episode_index))]
    viz_ids = set(splits["viz"])
    todo_display = [r for r in table.itertuples()
                    if r.episode_index in viz_ids and not os.path.exists(display_path(cfg, r.episode_index))]
    if not todo and not todo_display:
        print("Feature cache complete.")
        return

    check_ffmpeg()
    encoder = DinoEncoder(cfg.dino_model).to(device) if todo else None
    dtype = amp_dtype(device)

    for i, r in enumerate(todo):
        video = os.path.join(cfg.data_root, r.video)
        frames = decode_segment(video, r.start, r.end, cfg.target_fps, cfg.img_w, cfg.img_h)
        if r.length:
            expected = r.length / info["fps"] * cfg.target_fps
            if abs(len(frames) - expected) > 3:
                print(f"  note: episode {r.episode_index} gave {len(frames)} frames, expected ~{expected:.0f}")

        feats = []
        for s in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[s:s + batch_size]).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                feats.append(encoder(batch).half().cpu())
        feats = torch.cat(feats).numpy()
        assert feats.shape[1] == cfg.num_tokens, f"Expected {cfg.num_tokens} tokens, got {feats.shape[1]}"
        save_npy_atomic(feature_path(cfg, r.episode_index), feats)
        print(f"[{i + 1}/{len(todo)}] episode {r.episode_index}: {len(frames)} frames -> {feats.shape}")

    for r in todo_display:
        video = os.path.join(cfg.data_root, r.video)
        frames = decode_segment(video, r.start, r.end, cfg.target_fps, cfg.display_w, cfg.display_h)
        save_npy_atomic(display_path(cfg, r.episode_index), frames)

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()


def slot_contrastive_loss(slots, temperature):
    """SlotContrast loss. slots [B, T, K, S].

    For each t, slot k of sample b at t must match slot k of sample b at t+1
    against ALL other B*K slots in the batch. Targets are the fixed diagonal —
    not the argmax of the similarities themselves, which would make the loss
    self-confirming and give no temporal signal.
    """
    B, T, K, S = slots.shape
    if T < 2:
        return slots.new_zeros(())
    z = F.normalize(slots.float(), dim=-1)
    target = torch.arange(B * K, device=slots.device)
    losses = []
    for t in range(T - 1):
        logits = z[:, t].reshape(B * K, S) @ z[:, t + 1].reshape(B * K, S).T / temperature
        losses.append(0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)))
    return torch.stack(losses).mean()


def truncated_view(full, k, cfg):
    """Pick the terms of the horizon-k objective out of a full rollout. In no-grad
    evaluation this gives exactly the values `forward_truncated` computes."""
    return {
        "r0": full["recon"][:, 0] if cfg.first_frame_weight > 0 else None,
        "prev": full["slots"][:, k - 1],
        "last": full["slots"][:, k],
        "r_last": full["recon"][:, k],
    }


def compute_losses(out, feats0, featsk, cfg):
    """loss = recon(S_k) + λ·contrast(S_{k-1}, S_k) + w0·recon(S_0).
    S_{k-1} is detached, so the contrastive term pulls S_k toward the identity
    its slots carried after the burn-in."""
    recon_last = F.mse_loss(out["r_last"], featsk)
    contrast = slot_contrastive_loss(torch.stack([out["prev"], out["last"]], 1), cfg.temperature)
    recon_first = F.mse_loss(out["r0"], feats0) if out["r0"] is not None else recon_last.new_zeros(())
    loss = recon_last + cfg.contrast_weight * contrast + cfg.first_frame_weight * recon_first
    return loss, {"loss": loss.item(), "recon_last": recon_last.item(),
                  "contrast": contrast.item(), "recon_first": recon_first.item()}


@torch.no_grad()
def mask_stats(masks):
    """masks [B, T, K, N]. Active slots (>2% of tokens) and fraction of tokens
    keeping the same slot between consecutive frames."""
    hard = masks.argmax(dim=2)                                     # [B, T, N]
    K = masks.shape[2]
    usage = F.one_hot(hard, K).float().mean(dim=2)                 # [B, T, K]
    active = (usage > 0.02).float().sum(-1).mean().item()
    stable = (hard[:, 1:] == hard[:, :-1]).float().mean().item() if hard.shape[1] > 1 else float("nan")
    return active, stable


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and not name.endswith(("pos", "mu", "log_sigma")):
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )


def build_scheduler(optimizer, cfg, total_steps):
    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sample_rollout_k(cfg):
    return random.randint(1, cfg.rollout_k) if cfg.rollout_sample else cfg.rollout_k


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg, device, epoch):
    model.train()
    dtype = amp_dtype(device)
    totals = defaultdict(float)
    for step, feats in enumerate(loader):
        feats = feats.to(device, non_blocking=True).float()  # [B, rollout_k + 1, N, D]
        k = sample_rollout_k(cfg)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            out = model.forward_truncated(feats, k)
        loss, parts = compute_losses(out, feats[:, 0], feats[:, k], cfg)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for key, v in parts.items():
            totals[key] += v
        totals["k"] += k
        if step % 50 == 0:
            print(f"  epoch {epoch} step {step:5d}/{len(loader)} k={k:2d}  loss={parts['loss']:.4f} "
                  f"recon_k={parts['recon_last']:.4f} recon_0={parts['recon_first']:.4f} "
                  f"contrast={parts['contrast']:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
    return {key: v / len(loader) for key, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    """Horizon-k objective (k = rollout_k, fixed) on eval clips, plus the recon
    error at every rollout step 0..k to show error accumulation."""
    model.eval()
    dtype = amp_dtype(device)
    gen = torch.Generator(device=device).manual_seed(0)  # same slot init every eval
    k = cfg.rollout_k
    totals, curve, n = defaultdict(float), torch.zeros(k + 1), 0
    for feats in loader:
        feats = feats.to(device, non_blocking=True).float()
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            full = model(feats, generator=gen)
        _, parts = compute_losses(truncated_view(full, k, cfg), feats[:, 0], feats[:, k], cfg)
        active, stable = mask_stats(full["masks"])
        for key, v in {**parts, "active_slots": active, "temporal_stability": stable}.items():
            totals[key] += v
        curve += ((full["recon"].float() - feats) ** 2).mean(dim=(0, 2, 3)).cpu()
        n += 1
    result = {key: v / n for key, v in totals.items()}
    result["step_curve"] = (curve / n).tolist()
    return result


@torch.no_grad()
def evaluate_long_horizon(model, paths, cfg, device, chunk=16):
    """Roll the slots through `eval_horizon` frames of every eval episode (far
    beyond the training horizon) and measure drift and collapse."""
    model.eval()
    dtype = amp_dtype(device)
    arrays = [np.load(p, mmap_mode="r") for p in paths]
    T = min(cfg.eval_horizon, min(len(a) for a in arrays))
    feats = torch.from_numpy(np.stack([np.asarray(a[:T]) for a in arrays])).to(device).float()  # [E, T, N, D]
    gen = torch.Generator(device=device).manual_seed(0)

    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        slots, _ = model.encode(feats, generator=gen)
    errs, masks = [], []
    for s in range(0, T, chunk):
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            recon, m = model.decode(slots[:, s:s + chunk])
        errs.append(((recon.float() - feats[:, s:s + chunk]) ** 2).mean(dim=(0, 2, 3)).cpu())
        masks.append(m.float().cpu())
    curve, masks = torch.cat(errs), torch.cat(masks, 1)

    tail = max(2, T // 10)
    _, stability = mask_stats(masks)
    active_end, _ = mask_stats(masks[:, -tail:])
    return {
        "long_recon": curve.mean().item(),
        "long_recon_start": curve[:tail].mean().item(),
        "long_recon_end": curve[-tail:].mean().item(),
        "long_stability": stability,
        "long_active_end": active_end,
        "long_curve": curve.tolist(),
        "long_T": T,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, history, best_eval, feat_dim):
    """Only the trainable model is saved (DINO is frozen and reloaded from the hub)."""
    tmp = path + ".tmp"
    torch.save({
        "config": asdict(model.cfg),
        "feat_dim": feat_dim,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_eval": best_eval,
    }, tmp)
    os.replace(tmp, path)


def load_model(path, device):
    """Rebuild the model from the config stored in the checkpoint, so the
    architecture always matches the weights (strict loading)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = SlotContrastModel(cfg, ckpt["feat_dim"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"Loaded {path} (epoch {ckpt['epoch']}, best eval {ckpt['best_eval']:.4f})")
    return model, ckpt


def make_loader(dataset, cfg, shuffle):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )


def train(cfg, splits, device):
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    train_ds = ClipDataset([feature_path(cfg, e) for e in splits["train"]], cfg.clip_len, cfg.frame_gap)
    eval_ds = ClipDataset([feature_path(cfg, e) for e in splits["eval"]], cfg.clip_len, cfg.frame_gap)
    print(f"rollout_k={cfg.rollout_k} (sampled={cfg.rollout_sample}) -> {cfg.clip_len} frames per sample")
    print(f"Train: {len(splits['train'])} episodes, {len(train_ds):,} clips | "
          f"Eval: {len(splits['eval'])} episodes, {len(eval_ds):,} clips")
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    eval_loader = make_loader(eval_ds, cfg, shuffle=False)

    feat_dim = np.load(train_ds.paths[0], mmap_mode="r").shape[-1]
    model = SlotContrastModel(cfg, feat_dim).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, cfg.num_epochs * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype(device) == torch.float16)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    latest = os.path.join(cfg.ckpt_dir, "latest.pt")
    best_path = os.path.join(cfg.ckpt_dir, "best.pt")
    start_epoch, history, best_eval = 0, [], float("inf")
    if cfg.resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        if ckpt["config"] == asdict(cfg):
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch, history, best_eval = ckpt["epoch"], ckpt["history"], ckpt["best_eval"]
            print(f"Resumed from epoch {start_epoch}.")
        else:
            print("Config changed since latest.pt — training from scratch.")

    eval_paths = [feature_path(cfg, e) for e in splits["eval"]]
    for epoch in range(start_epoch + 1, cfg.num_epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, device, epoch)
        ev = evaluate(model, eval_loader, cfg, device)
        ev.update(evaluate_long_horizon(model, eval_paths, cfg, device))
        history.append({"epoch": epoch, "train": tr, "eval": ev})
        print(f"Epoch {epoch}: train loss {tr['loss']:.4f} (mean k {tr['k']:.1f}) | "
              f"eval@k={cfg.rollout_k} loss {ev['loss']:.4f} "
              f"(recon_k {ev['recon_last']:.4f}, recon_0 {ev['recon_first']:.4f}, contrast {ev['contrast']:.4f})\n"
              f"  long horizon ({ev['long_T']} frames): recon {ev['long_recon_start']:.4f} -> "
              f"{ev['long_recon_end']:.4f} | stability {ev['long_stability']:.3f} | "
              f"active slots at end {ev['long_active_end']:.1f}/{cfg.num_slots}")

        score = ev["long_recon"] if cfg.best_metric == "long_recon" else ev["loss"]
        if score < best_eval:
            best_eval = score
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, history, best_eval, feat_dim)
            print("  new best checkpoint")
        save_checkpoint(latest, model, optimizer, scheduler, scaler, epoch, history, best_eval, feat_dim)

    return history


def plot_history(history, cfg):
    """Top: losses per epoch (train uses sampled k, eval uses k = rollout_k).
    Bottom: error accumulation of the last epoch, within the training horizon
    and over the long-horizon rollout."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, key in zip(axes[0], ["loss", "recon_last", "contrast"]):
        ax.plot(epochs, [h["train"][key] for h in history], label="train")
        ax.plot(epochs, [h["eval"][key] for h in history], label=f"eval (k={cfg.rollout_k})")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, [h["eval"]["long_recon_start"] for h in history], label="start of rollout")
    ax.plot(epochs, [h["eval"]["long_recon_end"] for h in history], label="end of rollout")
    ax.set_title(f"long-horizon recon ({history[-1]['eval']['long_T']} frames)")
    ax.set_xlabel("epoch")
    ax.legend()

    last = history[-1]["eval"]
    ax = axes[1, 1]
    ax.plot(range(len(last["step_curve"])), last["step_curve"], marker="o")
    ax.set_title("recon error vs rollout step (eval clips)")
    ax.set_xlabel("step t (S_0 ... S_k)")

    ax = axes[1, 2]
    ax.plot(np.arange(len(last["long_curve"])), last["long_curve"])
    ax.axvline(cfg.rollout_k, color="k", ls="--", lw=1, label=f"rollout_k = {cfg.rollout_k}")
    ax.set_title("recon error vs frame (long rollout)")
    ax.set_xlabel("frame")
    ax.legend()
    plt.tight_layout()
    plt.show()


SLOT_COLORS = np.array([
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (170, 110, 40),
], dtype=np.float32)


@torch.no_grad()
def episode_masks(model, feats, device, chunk=32):
    """Run the model over a whole episode, carrying slots across frames.
    feats [T, N, D] -> decoder masks [T, K, N] (on CPU)."""
    gen = torch.Generator(device=device).manual_seed(0)
    dtype = amp_dtype(device)
    slots, masks = None, []
    for s in range(0, len(feats), chunk):
        x = torch.from_numpy(np.asarray(feats[s:s + chunk])).to(device).float().unsqueeze(0)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            seq, _ = model.encode(x, slots=slots, generator=gen)
            _, m = model.decoder(seq[0])
        slots = seq[:, -1]
        masks.append(m.float().cpu())
    return torch.cat(masks)


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_frame(frame, masks, frame_idx, ncols=3):
    """frame [H, W, 3] uint8, masks [K, H, W] in [0, 1] (sum to 1 over K)."""
    K = masks.shape[0]
    base = frame.astype(np.float32)
    colors = SLOT_COLORS[np.arange(K) % len(SLOT_COLORS)]

    seg = colors[masks.argmax(0)]
    panels = [label(frame.copy(), f"RGB  t={frame_idx}"),
              label((0.5 * base + 0.5 * seg).astype(np.uint8), "Segmentation")]
    for k in range(K):
        m = masks[k][..., None]
        panel = base * (0.15 + 0.85 * m) + colors[k] * 0.35 * m
        panels.append(label(np.clip(panel, 0, 255).astype(np.uint8), f"Slot {k}"))

    while len(panels) % ncols:
        panels.append(np.zeros_like(frame))
    rows = [np.hstack(panels[i:i + ncols]) for i in range(0, len(panels), ncols)]
    return np.vstack(rows)


def visualize_episode(cfg, model, episode_index, device, out_dir=None):
    out_dir = out_dir or os.path.join(cfg.work_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    feats = np.load(feature_path(cfg, episode_index), mmap_mode="r")
    frames = np.load(display_path(cfg, episode_index), mmap_mode="r")
    T = min(len(feats), len(frames), cfg.viz_max_frames)

    masks = episode_masks(model, feats[:T], device)                       # [T, K, N]
    gh, gw = cfg.grid_hw
    masks = masks.view(T, -1, gh, gw)
    masks = F.interpolate(masks, size=(cfg.display_h, cfg.display_w), mode="bilinear", align_corners=False)
    masks = masks.numpy()

    hard = masks.argmax(1)
    stability = float((hard[1:] == hard[:-1]).mean()) if T > 1 else float("nan")
    print(f"Episode {episode_index}: {T} frames, temporal stability {stability:.3f}")

    video_path = os.path.join(out_dir, f"episode_{episode_index:06d}_slots.mp4")
    first = render_frame(frames[0], masks[0], 0)
    h, w = first.shape[:2]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(cfg.target_fps), "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", video_path],
        stdin=subprocess.PIPE,
    )
    for t in range(T):
        proc.stdin.write(render_frame(frames[t], masks[t], t).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the visualization video.")

    np.save(os.path.join(out_dir, f"episode_{episode_index:06d}_masks.npy"), masks.astype(np.float16))

    picks = np.linspace(0, T - 1, 3).astype(int)
    fig, axes = plt.subplots(1, len(picks), figsize=(6 * len(picks), 6))
    for ax, t in zip(np.atleast_1d(axes), picks):
        ax.imshow(render_frame(frames[t], masks[t], t))
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return video_path


if __name__ == "__main__":
    CFG = Config()
    seed_everything(CFG.seed)
    DEVICE = get_device()
    os.makedirs(CFG.work_dir, exist_ok=True)

    # 1. Data
    INFO, EPISODES = download_metadata(CFG)
    TABLE = episode_table(CFG, INFO, EPISODES)
    download_videos(CFG, TABLE)
    SPLITS = make_splits(CFG, TABLE)
    print({k: len(v) for k, v in SPLITS.items()}, "episodes per split")

    # 2. Features (cached; reruns skip this)
    build_feature_cache(CFG, INFO, TABLE, SPLITS, DEVICE)

    # 3. Train
    HISTORY = train(CFG, SPLITS, DEVICE)
    plot_history(HISTORY, CFG)

    # 4. Visualize the held-out episode with the best checkpoint
    MODEL, _ = load_model(os.path.join(CFG.ckpt_dir, "best.pt"), DEVICE)
    VIDEO = visualize_episode(CFG, MODEL, SPLITS["viz"][0], DEVICE)
    try:
        from IPython.display import Video, display
        display(Video(VIDEO, embed=True, width=960))
    except ImportError:
        print("Saved:", VIDEO)