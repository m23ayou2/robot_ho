"""Train ACT-with-slots + flow matching on the G1 dataset.

Pipeline (reuses the slot run's episodes, splits and DINO feature cache):
  1. download the low-dim parquet data (state + action) for those episodes
  2. align it with the 10 fps feature frames
  3. run the frozen slot model over every episode (causal propagation, the same
     way it would run on the robot) and cache slots + mask geometry
  4. train FlowMatchingPolicy, keep the checkpoint with the lowest eval action L1
  5. replay the held-out episode open-loop (teacher-forced observations) with
     temporal ensembling and with receding-horizon execution

Run `train.py` first; set PolicyConfig.obs_tokens = "dino" for the no-slots control.
"""
import json
import os
from collections import defaultdict
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from src.config import Config, PolicyConfig
from src.dataset import PolicyDataset
from src.policy import FlowMatchingPolicy
from train import (amp_dtype, build_scheduler, download_metadata, feature_path, get_device,
                   load_model, save_npy_atomic, seed_everything)


# ---------------------------------------------------------------- data ----

def load_lowdim(cfg, info, episodes, ids, cache_dir):
    """Per-episode state [L, S] and action [L, A] at the source fps (cached as .npz)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = lambda e: os.path.join(cache_dir, f"ep_{e:06d}_lowdim.npz")
    todo = [e for e in ids if not os.path.exists(path(e))]
    if todo:
        rows = episodes.set_index("episode_index").loc[todo]
        files = sorted({info["data_path"].format(chunk_index=int(r["data/chunk_index"]),
                                                 file_index=int(r["data/file_index"]))
                        for _, r in rows.iterrows()})
        print(f"Downloading {len(files)} data file(s) for {len(todo)} episodes...")
        snapshot_download(cfg.repo_id, repo_type="dataset", local_dir=cfg.data_root, allow_patterns=files)
        cols = ["episode_index", "frame_index", "observation.state", "action"]
        df = pd.concat([pd.read_parquet(os.path.join(cfg.data_root, f), columns=cols) for f in files])
        for e, g in df[df.episode_index.isin(todo)].groupby("episode_index"):
            g = g.sort_values("frame_index")
            np.savez(path(int(e)), state=np.stack(g["observation.state"].to_numpy()).astype(np.float32),
                     action=np.stack(g["action"].to_numpy()).astype(np.float32))
        missing = [e for e in todo if not os.path.exists(path(e))]
        if missing:
            raise RuntimeError(f"No low-dim rows found for episodes {missing}.")
    out = {}
    for e in ids:
        z = np.load(path(e))
        out[e] = (z["state"], z["action"])
    return out


@torch.no_grad()
def slot_tokens(model, feats, device, chunk=64):
    """Propagate slots through a whole episode. feats [T, N, D] -> [T, K, S + 5]:
    slot vectors plus centroid (cx, cy), spread (sx, sy) and area of each slot's
    attention mask on the patch grid, so the policy keeps explicit position."""
    gh, gw = model.cfg.grid_hw
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, gh, device=device),
                            torch.linspace(-1, 1, gw, device=device), indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    gen = torch.Generator(device=device).manual_seed(0)
    dtype = amp_dtype(device)
    slots, out = None, []
    for s in range(0, len(feats), chunk):
        x = torch.from_numpy(np.asarray(feats[s:s + chunk])).to(device).float()[None]
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            seq, attn = model.encode(x, slots=slots, generator=gen)
        slots = seq[:, -1]
        attn = attn[0].float()                                         # [t, K, N]
        w = attn / attn.sum(-1, keepdim=True).clamp_min(1e-6)
        cx, cy = (w * xs).sum(-1), (w * ys).sum(-1)
        sx = ((w * (xs - cx[..., None]) ** 2).sum(-1)).clamp_min(0).sqrt()
        sy = ((w * (ys - cy[..., None]) ** 2).sum(-1)).clamp_min(0).sqrt()
        geom = torch.stack([cx, cy, sx, sy, attn.mean(-1)], -1)
        out.append(torch.cat([seq[0].float(), geom], -1).cpu())
    return torch.cat(out).numpy()


def build_obs_cache(pcfg, cfg, ids, device):
    """Path of the per-episode observation tokens. Slot caches are keyed by the
    slot checkpoint's epoch so a retrained slot model never reuses stale slots."""
    if pcfg.obs_tokens == "dino":
        return {e: feature_path(cfg, e) for e in ids}
    ckpt = torch.load(pcfg.slot_checkpoint, map_location="cpu", weights_only=False)
    cache_dir = os.path.join(pcfg.work_dir, "cache", f"slots_epoch{ckpt['epoch']}")
    os.makedirs(cache_dir, exist_ok=True)
    paths = {e: os.path.join(cache_dir, f"ep_{e:06d}_slots.npy") for e in ids}
    todo = [e for e in ids if not os.path.exists(paths[e])]
    if todo:
        model, _ = load_model(pcfg.slot_checkpoint, device)
        for i, e in enumerate(todo):
            tokens = slot_tokens(model, np.load(feature_path(cfg, e), mmap_mode="r"), device)
            save_npy_atomic(paths[e], tokens.astype(np.float32))
            print(f"[{i + 1}/{len(todo)}] slots for episode {e}: {tokens.shape}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return paths


def make_episodes(cfg, ids, obs_paths, lowdim, fps):
    """Align low-dim data with the observation frames (decoded at target_fps)."""
    episodes = []
    for e in ids:
        state, action = lowdim[e]
        T = np.load(obs_paths[e], mmap_mode="r").shape[0]
        src = np.round(np.arange(T) * fps / cfg.target_fps).astype(np.int64)
        T = int((src < len(action)).sum())  # drop frames ffmpeg produced past the data
        src = src[:T]
        episodes.append({"episode_index": e, "obs_path": obs_paths[e], "src": src,
                         "state": state[src], "actions": action})
    return episodes


def normalization_stats(episodes, pcfg, stride):
    """State stats, and stats of the flow target: absolute actions, or for
    action_mode="relative" the offsets a[src[t] + j*stride] - state[t] over the chunk."""
    state = np.concatenate([ep["state"] for ep in episodes])
    if pcfg.action_mode == "relative":
        steps = stride * np.arange(pcfg.chunk_size)
        action = np.concatenate([
            (ep["actions"][np.minimum(ep["src"][:, None] + steps, len(ep["actions"]) - 1)]
             - ep["state"][:, None]).reshape(-1, ep["actions"].shape[1]) for ep in episodes])
    else:
        action = np.concatenate([ep["actions"] for ep in episodes])
    clamp = lambda s: np.maximum(s, 1e-2)
    return state.mean(0), clamp(state.std(0)), action.mean(0), clamp(action.std(0))


def make_loader(ds, pcfg, shuffle):
    return DataLoader(ds, batch_size=pcfg.batch_size, shuffle=shuffle, drop_last=shuffle,
                      num_workers=pcfg.num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=pcfg.num_workers > 0)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ------------------------------------------------------------ training ----

def build_optimizer(model, pcfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.endswith(("_pos", "_emb")) else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": pcfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=pcfg.lr)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, pcfg, device, epoch):
    model.train()
    dtype = amp_dtype(device)
    total = 0.0
    for step, batch in enumerate(loader):
        b = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss = model.loss(b["obs"], b["state"], b["actions"], b["is_pad"])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), pcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
        if step % 100 == 0:
            print(f"  epoch {epoch} step {step:5d}/{len(loader)} flow loss {loss.item():.4f} "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")
    return {"loss": total / len(loader)}


@torch.no_grad()
def evaluate(model, loader, device):
    """Flow loss plus L1 (raw joint units) of sampled chunks against the demo.
    `hold_l1` is the trivial policy "stay at the current state"; a useful policy
    must beat it clearly."""
    model.eval()
    dtype = amp_dtype(device)
    gen = torch.Generator(device=device).manual_seed(0)
    totals, n = defaultdict(float), 0
    for batch in loader:
        b = to_device(batch, device)
        keep = (~b["is_pad"]).float()
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss = model.loss(b["obs"], b["state"], b["actions"], b["is_pad"], generator=gen)
            pred = model.sample(b["obs"], b["state"], generator=gen)
        err = (pred.float() - b["actions"]).abs().mean(-1)                    # [B, C]
        hold = (b["state"][:, None] - b["actions"]).abs().mean(-1)
        totals["hold_l1_first"] += hold[:, 0].mean().item()
        totals["loss"] += loss.item()
        totals["l1_first"] += err[:, 0].mean().item()
        totals["l1_chunk"] += ((err * keep).sum() / keep.sum()).item()
        totals["hold_l1_chunk"] += ((hold * keep).sum() / keep.sum()).item()
        n += 1
    return {k: v / n for k, v in totals.items()}


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, history, best, dims):
    tmp = path + ".tmp"
    torch.save({"config": asdict(model.cfg), "dims": dims, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "history": history, "best": best}, tmp)
    os.replace(tmp, path)


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FlowMatchingPolicy(PolicyConfig.from_dict(ckpt["config"]), **ckpt["dims"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"Loaded {path} (epoch {ckpt['epoch']}, best eval L1 {ckpt['best']:.4f})")
    return model, ckpt


def train(pcfg, train_eps, eval_eps, stride, device):
    os.makedirs(pcfg.ckpt_dir, exist_ok=True)
    train_ds = PolicyDataset(train_eps, pcfg.obs_history, pcfg.chunk_size, stride)
    eval_ds = PolicyDataset(eval_eps, pcfg.obs_history, pcfg.chunk_size, stride)
    train_loader, eval_loader = make_loader(train_ds, pcfg, True), make_loader(eval_ds, pcfg, False)
    sample = train_ds[0]
    dims = {"obs_dim": sample["obs"].shape[-1], "tokens_per_frame": sample["obs"].shape[-2],
            "state_dim": sample["state"].numel(), "action_dim": sample["actions"].shape[-1]}
    print(f"Train: {len(train_eps)} episodes, {len(train_ds):,} samples | "
          f"Eval: {len(eval_eps)} episodes, {len(eval_ds):,} samples | dims {dims}")

    model = FlowMatchingPolicy(pcfg, **dims).to(device)
    model.set_normalization(*normalization_stats(train_eps, pcfg, stride))
    optimizer = build_optimizer(model, pcfg)
    scheduler = build_scheduler(optimizer, pcfg, pcfg.num_epochs * len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype(device) == torch.float16)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    latest, best_path = os.path.join(pcfg.ckpt_dir, "latest.pt"), os.path.join(pcfg.ckpt_dir, "best.pt")
    start_epoch, history, best = 0, [], float("inf")
    if pcfg.resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        if ckpt["config"] == asdict(pcfg):
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch, history, best = ckpt["epoch"], ckpt["history"], ckpt["best"]
            print(f"Resumed from epoch {start_epoch}.")
        else:
            print("Config changed since latest.pt — training from scratch.")

    for epoch in range(start_epoch + 1, pcfg.num_epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, pcfg, device, epoch)
        ev = evaluate(model, eval_loader, device)
        history.append({"epoch": epoch, "train": tr, "eval": ev})
        print(f"Epoch {epoch}: train flow {tr['loss']:.4f} | eval flow {ev['loss']:.4f} | "
              f"L1 first {ev['l1_first']:.4f} (hold {ev['hold_l1_first']:.4f}), "
              f"chunk {ev['l1_chunk']:.4f} (hold {ev['hold_l1_chunk']:.4f})")
        if ev["l1_chunk"] < best:
            best = ev["l1_chunk"]
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, history, best, dims)
            print("  new best checkpoint")
        save_checkpoint(latest, model, optimizer, scheduler, scaler, epoch, history, best, dims)
    return history


# ---------------------------------------------------------- inference ----

@torch.no_grad()
def predict_chunks(model, ep, stride, device, batch_size=128):
    """A chunk for every frame of an episode, all from the SAME initial noise so
    overlapping chunks come from one mode and can be averaged. [T, C, A]."""
    ds = PolicyDataset([ep], model.cfg.obs_history, model.cfg.chunk_size, stride)
    gen = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(1, model.cfg.chunk_size, model.action_mean.numel(), device=device, generator=gen)
    dtype = amp_dtype(device)
    chunks = []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        b = to_device(batch, device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            pred = model.sample(b["obs"], b["state"], noise=noise.expand(len(b["obs"]), -1, -1))
        chunks.append(pred.float().cpu())
    return torch.cat(chunks).numpy()


def temporal_ensemble(chunks, m):
    """ACT Algorithm 2: average every prediction made for step t, w_i = exp(-m i), i = 0 oldest."""
    T, C, _ = chunks.shape
    out = np.empty((T, chunks.shape[-1]), np.float32)
    for t in range(T):
        starts = range(max(0, t - C + 1), t + 1)                  # oldest first
        preds = np.stack([chunks[s, t - s] for s in starts])
        w = np.exp(-m * np.arange(len(preds)))[:, None]
        out[t] = (w * preds).sum(0) / w.sum()
    return out


def receding_horizon(chunks, h):
    """Predict every h steps and execute the first h actions of each chunk."""
    T = len(chunks)
    return np.concatenate([chunks[s, :h] for s in range(0, T, h)])[:T]


def replay_episode(model, ep, stride, device, out_dir, joint_names=None):
    """Open-loop replay on recorded observations (no robot in the loop): this
    measures how well the executed actions track the demonstration, not task success."""
    os.makedirs(out_dir, exist_ok=True)
    chunks = predict_chunks(model, ep, stride, device)
    gt = ep["actions"][ep["src"]]
    runs = {"temporal ensemble": temporal_ensemble(chunks, model.cfg.ensemble_m),
            f"receding horizon (h={model.cfg.exec_horizon})": receding_horizon(chunks, model.cfg.exec_horizon)}
    metrics = {k: float(np.abs(v - gt).mean()) for k, v in runs.items()}
    metrics["hold state"] = float(np.abs(ep["state"] - gt).mean())
    print(f"Replay episode {ep['episode_index']}: mean |a - a_demo| " +
          ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    A = gt.shape[1]
    ncols = 4
    nrows = int(np.ceil(A / ncols))
    t = np.arange(len(gt)) / 10.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.2 * nrows), sharex=True, squeeze=False)
    for j, ax in enumerate(axes.ravel()):
        if j >= A:
            ax.axis("off")
            continue
        ax.plot(t, gt[:, j], "k", lw=1.5, label="demo")
        for k, v in runs.items():
            ax.plot(t, v[:, j], lw=1, label=k)
        ax.set_title(joint_names[j] if joint_names else f"joint {j}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"Open-loop replay, episode {ep['episode_index']} "
                 f"({model.cfg.obs_tokens} tokens, {model.cfg.action_mode} actions)")
    fig.tight_layout()
    name = f"replay_ep{ep['episode_index']}"
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def plot_history(history, out_dir):
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(epochs, [h["train"]["loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["eval"]["loss"] for h in history], label="eval")
    axes[0].set_title("flow-matching loss")
    for key, label in [("l1_first", "first action"), ("l1_chunk", "whole chunk"), ("hold_l1_chunk", "hold state")]:
        axes[1].plot(epochs, [h["eval"][key] for h in history], label=label, ls="--" if "hold" in key else "-")
    axes[1].set_title("eval L1 of sampled actions (raw units)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "history.png"), dpi=150)
    plt.close(fig)


def prepare(pcfg, device):
    """Episodes (train / eval / viz) aligned with observation tokens, plus the action stride."""
    slot_ckpt = torch.load(pcfg.slot_checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(slot_ckpt["config"])
    del slot_ckpt
    with open(os.path.join(cfg.work_dir, "splits.json")) as f:
        splits = json.load(f)
    ids = sorted(sum(splits.values(), []))
    info, episodes = download_metadata(cfg)
    lowdim = load_lowdim(cfg, info, episodes, ids, os.path.join(pcfg.work_dir, "cache", "lowdim"))
    obs_paths = build_obs_cache(pcfg, cfg, ids, device)
    fps = info["fps"]
    stride = int(round(fps / cfg.target_fps))  # chunk step j == observation frame t + j
    by_id = {ep["episode_index"]: ep for ep in make_episodes(cfg, ids, obs_paths, lowdim, fps)}
    joint_names = info["features"]["action"].get("names")
    joint_names = joint_names[0] if joint_names and isinstance(joint_names[0], list) else joint_names
    return {k: [by_id[e] for e in v] for k, v in splits.items()}, stride, joint_names


if __name__ == "__main__":
    PCFG = PolicyConfig()
    seed_everything(PCFG.seed)
    DEVICE = get_device()
    os.makedirs(PCFG.run_dir, exist_ok=True)

    EPISODES, STRIDE, JOINTS = prepare(PCFG, DEVICE)
    HISTORY = train(PCFG, EPISODES["train"], EPISODES["eval"], STRIDE, DEVICE)
    OUT = os.path.join(PCFG.run_dir, "results")
    os.makedirs(OUT, exist_ok=True)
    plot_history(HISTORY, OUT)

    MODEL, _ = load_policy(os.path.join(PCFG.ckpt_dir, "best.pt"), DEVICE)
    for ep in EPISODES["viz"] + EPISODES["eval"][:2]:
        replay_episode(MODEL, ep, STRIDE, DEVICE, OUT, JOINTS)
    print("Results saved to", os.path.abspath(OUT))
