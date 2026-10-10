import json
import os
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")            # figures are only saved to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from train import *


VIZ = {
    "checkpoint": os.path.join(Config().ckpt_dir, "best.pt"),
    "max_frames": 300,       # frames of the episode to process
    "n_cols": 8,             # frames shown in the time-strip figures
    "n_pca_cols": 4,
    "init_seeds": 16,        # samples averaged in the init-similarity figure
    "extra_eval_episodes": 3,  # also draw the qualitative figure for a few eval episodes
    "out_dir": "results",      # every figure, the video and the metrics are saved here
}

DEVICE = get_device()
MODEL, _ = load_model(VIZ["checkpoint"], DEVICE)
CFG = MODEL.cfg
print(f"Model trained with rollout_k={CFG.rollout_k} (sampled={CFG.rollout_sample}, "
      f"first_frame_weight={CFG.first_frame_weight})")
with open(os.path.join(CFG.work_dir, "splits.json")) as f:
    SPLITS = json.load(f)
OUT_DIR = VIZ["out_dir"]
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = np.array([
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
    (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128), (128, 128, 0),
], dtype=np.float32)


def slot_colors(k):
    return PALETTE[np.arange(k) % len(PALETTE)]


@torch.no_grad()
def rollout(model, feats, device, mode="propagate", seed=0, chunk=32):
    """Run the model over an episode of cached features [T, N, D].

    mode="propagate":   S_0 = first_step, then S_t = next_step(S_{t-1}) for every
                        frame — the same recurrence the model is trained on with
                        rollout_k, but continued for the whole episode.
    mode="independent": first_step on every frame (fresh init, no temporal link).
    Returns CPU tensors: slots [T,K,S], attn [T,K,N], masks [T,K,N] (decoder alphas),
    recon_err [T] (feature-reconstruction MSE per frame).
    """
    dtype = amp_dtype(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    x_all = torch.from_numpy(np.asarray(feats)).to(device).float()
    T = len(x_all)

    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        if mode == "propagate":
            slots, attn = model.encode(x_all[None], generator=gen)
            slots, attn = slots[0], attn[0]
        elif mode == "independent":
            parts = [model.first_step(x_all[s:s + chunk], gen) for s in range(0, T, chunk)]
            slots = torch.cat([p[0] for p in parts])
            attn = torch.cat([p[1] for p in parts])
        else:
            raise ValueError(mode)
        slots, attn = slots.float(), attn.float()
        recon, masks = [], []
        for s in range(0, T, chunk):
            r, m = model.decoder(slots[s:s + chunk])
            recon.append(r.float())
            masks.append(m.float())
    recon, masks = torch.cat(recon), torch.cat(masks)
    recon_err = ((recon - x_all) ** 2).mean(dim=(1, 2))
    return {"slots": slots.cpu(), "attn": attn.cpu(), "masks": masks.cpu(), "recon_err": recon_err.cpu()}


def track_by_matching(out):
    """Link slots of consecutive frames by Hungarian matching on soft-mask IoU."""
    masks, slots, attn = out["masks"].clone(), out["slots"].clone(), out["attn"].clone()
    for t in range(1, len(masks)):
        prev, cur = masks[t - 1], masks[t]
        inter = prev @ cur.T
        union = prev.sum(1, keepdim=True) + cur.sum(1)[None] - inter
        _, perm = linear_sum_assignment(-(inter / union.clamp_min(1e-6)).numpy())
        masks[t], slots[t], attn[t] = cur[perm], slots[t][perm], attn[t][perm]
    return {**out, "slots": slots, "attn": attn, "masks": masks}


def load_display_frames(cfg, episode_index):
    """Display-resolution RGB frames; decoded and cached on first use."""
    path = display_path(cfg, episode_index)
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    info, episodes = download_metadata(cfg)  # served from the local cache
    row = episode_table(cfg, info, episodes).set_index("episode_index").loc[episode_index]
    frames = decode_segment(os.path.join(cfg.data_root, row.video), row.start, row.end,
                            cfg.target_fps, cfg.display_w, cfg.display_h)
    save_npy_atomic(path, frames)
    return frames


def upsample(mask_kn, cfg):
    """[K, N] soft masks -> [K, H, W] at display resolution."""
    gh, gw = cfg.grid_hw
    m = mask_kn.view(1, -1, gh, gw).float()
    return F.interpolate(m, size=(cfg.display_h, cfg.display_w), mode="bilinear", align_corners=False)[0].numpy()


def overlay(frame, hard, k, alpha=0.55):
    """Paper-style overlay: colored segments + white boundaries."""
    out = (1 - alpha) * frame.astype(np.float32) + alpha * slot_colors(k)[hard]
    edge = np.zeros(hard.shape, bool)
    edge[:-1] |= hard[:-1] != hard[1:]
    edge[:, :-1] |= hard[:, :-1] != hard[:, 1:]
    out[edge] = 255
    return out.clip(0, 255).astype(np.uint8)


def overlay_at(frames, result, t, cfg):
    m = upsample(result["masks"][t], cfg)
    return overlay(frames[t], m.argmax(0), m.shape[0])


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def stability(masks):
    """Fraction of patches keeping the same slot between consecutive frames."""
    hard = masks.argmax(1)
    return (hard[1:] == hard[:-1]).float().mean().item()


def active_slots(masks, min_frac=0.01):
    """Per-frame number of slots that own at least `min_frac` of the patches."""
    hard = masks.argmax(1)
    area = F.one_hot(hard, masks.shape[1]).float().mean(1)  # [T, K]
    return (area >= min_frac).sum(1).numpy()


def fig_qualitative(frames, results, cols, name, title):
    """~Fig. 4: rows = input + methods, columns = time steps."""
    rows = [("Input", None)] + list(results.items())
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols), 1.85 * len(rows)), squeeze=False)
    for r, (label, res) in enumerate(rows):
        for c, t in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(frames[t] if res is None else overlay_at(frames, res, t, CFG))
            clean_axis(ax)
            if r == 0:
                ax.set_title(f"t = {t}  ({t / CFG.target_fps:.1f}s)", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle(f"{title}  —  trained with rollout_k = {CFG.rollout_k}", fontsize=12)
    fig.tight_layout()
    save(fig, name)


def fig_slot_decomposition(frames, result, cols, name):
    """Input row + one row per slot showing its soft mask over time."""
    K = result["masks"].shape[1]
    colors = slot_colors(K) / 255
    fig, axes = plt.subplots(K + 1, len(cols), figsize=(2.1 * len(cols), 1.6 * (K + 1)), squeeze=False)
    for c, t in enumerate(cols):
        m = upsample(result["masks"][t], CFG)
        base = frames[t].astype(np.float32) / 255
        axes[0, c].imshow(frames[t])
        axes[0, c].set_title(f"t = {t}", fontsize=9)
        for k in range(K):
            soft = m[k][..., None]
            axes[k + 1, c].imshow((base * (0.12 + 0.88 * soft)).clip(0, 1))
    for r in range(K + 1):
        for c in range(len(cols)):
            clean_axis(axes[r, c])
    axes[0, 0].set_ylabel("Input", fontsize=10)
    for k in range(K):
        axes[k + 1, 0].set_ylabel(f"Slot {k}", fontsize=10, color=colors[k], fontweight="bold")
    fig.suptitle("Per-slot masks over time (propagated slots)", fontsize=12)
    fig.tight_layout()
    save(fig, name)


def temporal_similarity(slots):
    """Mean cosine similarity between slot i at t and slot j at t+1 -> [K, K]."""
    z = F.normalize(slots, dim=-1)
    return torch.einsum("tid,tjd->ij", z[:-1], z[1:]).div(len(z) - 1).numpy()


def fig_temporal_similarity(results, name):
    """~Fig. 2a: a strong diagonal means slot k stays the same object over time."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.4 * len(results), 4), squeeze=False)
    for ax, (label, res) in zip(axes[0], results.items()):
        S = temporal_similarity(res["slots"])
        K = len(S)
        diag, off = np.diag(S).mean(), S[~np.eye(K, dtype=bool)].mean()
        im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{label}\ndiag {diag:.2f} | off-diag {off:.2f}", fontsize=10)
        ax.set_xlabel("slot at t+1")
        ax.set_ylabel("slot at t")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="cosine similarity")
    save(fig, name)


@torch.no_grad()
def fig_init_similarity(model, feats0, name, n_seeds):
    """~Fig. S1: similarity among initial slots S0 and among first-frame slots S1."""
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    x = torch.from_numpy(np.asarray(feats0)).to(DEVICE).float()[None].expand(n_seeds, -1, -1)
    s0 = model.slot_init(n_seeds, gen)
    s1, _ = model.slot_attention(model.input_mlp(x), s0, model.cfg.slot_iters_first)

    def mean_sim(s):
        z = F.normalize(s.float(), dim=-1)
        return (z @ z.transpose(1, 2)).mean(0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4))
    for ax, (label, S) in zip(axes, [("Initial slots $S_0$", mean_sim(s0)), ("First-frame slots $S_1$", mean_sim(s1))]):
        K = len(S)
        im = ax.imshow(S, cmap="viridis", vmin=-1, vmax=1)
        off = S[~np.eye(K, dtype=bool)].mean()
        ax.set_title(f"{label}\nmean off-diag {off:.2f}", fontsize=10)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="cosine similarity")
    save(fig, name)


def pca_rgb(x, gh, gw, out_hw):
    """x [n, N, D] -> [n, H, W, 3] using a PCA shared across all n frames."""
    n, N, D = x.shape
    X = x.reshape(-1, D).float()
    X = X - X.mean(0)
    _, _, V = torch.linalg.svd(X, full_matrices=False)
    P = (X @ V[:3].T).numpy()
    lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
    P = ((P - lo) / (hi - lo + 1e-8)).clip(0, 1).reshape(n, gh, gw, 3)
    return np.stack([cv2.resize(p, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST) for p in P])


def fig_pca(frames, feats, result, cols, name):
    """~Fig. S2: frozen DINOv2 features vs dense features induced by the slots
    (each patch gets the mask-weighted mix of slot vectors)."""
    gh, gw = CFG.grid_hw
    hw = (CFG.display_h, CFG.display_w)
    idx = torch.as_tensor(cols)
    dino = torch.from_numpy(np.asarray(feats[cols])).float()
    slot_dense = torch.einsum("tkn,tks->tns", result["masks"][idx], result["slots"][idx])
    rows = [("Input", [frames[t] for t in cols]),
            ("DINOv2 PCA", pca_rgb(dino, gh, gw, hw)),
            ("Slot features PCA", pca_rgb(slot_dense, gh, gw, hw))]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.8 * len(cols), 2.2 * len(rows)), squeeze=False)
    for r, (label, imgs) in enumerate(rows):
        for c, t in enumerate(cols):
            axes[r, c].imshow(imgs[c])
            clean_axis(axes[r, c])
            if r == 0:
                axes[r, c].set_title(f"t = {t}", fontsize=9)
        axes[r, 0].set_ylabel(label, fontsize=10)
    fig.tight_layout()
    save(fig, name)


def fig_active_slots(counts_by_method, K, name):
    """~Fig. 7: how many slots are actually used per frame."""
    bins = np.arange(1, K + 1)
    width = 0.8 / len(counts_by_method)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (label, counts) in enumerate(counts_by_method.items()):
        freq = np.array([(counts == b).mean() for b in bins])
        ax.bar(bins + (i - (len(counts_by_method) - 1) / 2) * width, freq, width, label=label)
    ax.set_xticks(bins)
    ax.set_xlabel("active slots per frame (≥1% of patches)")
    ax.set_ylabel("fraction of frames")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, name)


def fig_occupancy(results, name):
    """Slot area over time; with consistent slots the bands stay put."""
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 2.4 * len(results)), sharex=True, squeeze=False)
    for ax, (label, res) in zip(axes[:, 0], results.items()):
        hard = res["masks"].argmax(1)
        K = res["masks"].shape[1]
        area = F.one_hot(hard, K).float().mean(1).numpy().T  # [K, T]
        ax.stackplot(np.arange(area.shape[1]) / CFG.target_fps, area,
                     colors=slot_colors(K) / 255, labels=[f"slot {k}" for k in range(K)])
        ax.set_ylim(0, 1)
        ax.set_ylabel("area")
        ax.set_title(f"{label} — temporal stability {stability(res['masks']):.3f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(ncol=K, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    fig.tight_layout()
    save(fig, name)


def fig_error_accumulation(curves, actives, rollout_k, fps, name):
    """Recon error and active slots vs rollout length, averaged over episodes.
    The dashed line is the training horizon rollout_k: a model that only learned
    short rollouts drifts (error up) or collapses (active slots down) after it."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for label in curves:
        T = min(len(c) for c in curves[label])
        t = np.arange(T) / fps
        err = np.stack([c[:T] for c in curves[label]])
        act = np.stack([a[:T] for a in actives[label]]).astype(np.float32)
        smooth = np.ones(5) / 5
        line, = axes[0].plot(t, err.mean(0), label=label)
        axes[0].fill_between(t, np.percentile(err, 25, 0), np.percentile(err, 75, 0),
                             color=line.get_color(), alpha=0.15, lw=0)
        axes[1].plot(t, np.convolve(act.mean(0), smooth, mode="same"), label=label)
    for ax in axes:
        ax.axvline(rollout_k / fps, color="k", ls="--", lw=1)
        ax.text(rollout_k / fps, 1.0, f" rollout_k={rollout_k}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8)
        ax.set_xlabel("time since slot initialization (s)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("feature recon MSE")
    axes[0].set_title("Error accumulation over the rollout (mean, IQR band)", fontsize=10)
    axes[1].set_ylabel("active slots")
    axes[1].set_ylim(0, CFG.num_slots + 0.5)
    axes[1].set_title("Slot usage over the rollout (collapse = drop)", fontsize=10)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, name)


def write_comparison_video(frames, results, path, fps):
    """input | method 1 | method 2 ..., H.264 so it plays in the notebook."""
    labels = ["Input"] + list(results)
    T = len(next(iter(results.values()))["masks"])
    h, w = CFG.display_h, CFG.display_w * len(labels)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
        stdin=subprocess.PIPE,
    )
    for t in range(T):
        # np.array makes a writable copy (memmap / decoded frames are read-only) for putText
        panels = [np.array(frames[t])] + [overlay_at(frames, r, t, CFG) for r in results.values()]
        for p, text in zip(panels, labels):
            cv2.putText(p, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(p, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panels[0], f"t={t}", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        proc.stdin.write(np.hstack(panels).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the comparison video.")
    return path


def run_methods(episode_index, max_frames):
    feats = np.load(feature_path(CFG, episode_index), mmap_mode="r")[:max_frames]
    results = {
        "SlotContrast (ours)": rollout(MODEL, feats, DEVICE, "propagate", seed=0),
        "Per-frame + matching": track_by_matching(rollout(MODEL, feats, DEVICE, "independent", seed=0)),
    }
    return feats, results


EP = SPLITS["viz"][0]
FRAMES = load_display_frames(CFG, EP)
FEATS, RESULTS = run_methods(EP, min(VIZ["max_frames"], len(FRAMES)))
T = len(FEATS)
COLS = np.linspace(0, T - 1, VIZ["n_cols"]).round().astype(int)
PCA_COLS = np.linspace(0, T - 1, VIZ["n_pca_cols"]).round().astype(int)
OURS = RESULTS["SlotContrast (ours)"]

print(f"Episode {EP}: {T} frames")
METRICS = {"checkpoint": VIZ["checkpoint"], "episode": int(EP), "frames": int(T), "rollout_k": CFG.rollout_k,
           "methods": {}}
for label, res in RESULTS.items():
    S = temporal_similarity(res["slots"])
    m = {"temporal_stability": stability(res["masks"]),
         "slot_sim_diag": float(np.diag(S).mean()),
         "active_slots": float(active_slots(res["masks"]).mean()),
         "recon_upto_k": float(res["recon_err"][:CFG.rollout_k + 1].mean()),
         "recon_after_k": float(res["recon_err"][CFG.rollout_k + 1:].mean())}
    METRICS["methods"][label] = m
    print(f"  {label:24s} temporal stability {m['temporal_stability']:.3f} | "
          f"slot sim diag {m['slot_sim_diag']:.3f} | active slots {m['active_slots']:.2f} | "
          f"recon@<=k {m['recon_upto_k']:.4f} -> recon@>k {m['recon_after_k']:.4f}")
with open(os.path.join(OUT_DIR, f"metrics_ep{EP}.json"), "w") as f:
    json.dump(METRICS, f, indent=2)

fig_qualitative(FRAMES, RESULTS, COLS, f"fig_qualitative_ep{EP}", f"Held-out episode {EP}")
fig_slot_decomposition(FRAMES, OURS, COLS, f"fig_slot_decomp_ep{EP}")
fig_temporal_similarity(RESULTS, f"fig_temporal_sim_ep{EP}")
fig_init_similarity(MODEL, FEATS[0], "fig_init_sim", VIZ["init_seeds"])
fig_pca(FRAMES, FEATS, OURS, PCA_COLS, f"fig_pca_ep{EP}")
fig_occupancy(RESULTS, f"fig_occupancy_ep{EP}")

# Active-slot distribution over the held-out + eval episodes.
# Active-slot distribution + error accumulation over the held-out + eval episodes.
counts = {label: [active_slots(res["masks"])] for label, res in RESULTS.items()}
curves = {label: [res["recon_err"].numpy()] for label, res in RESULTS.items()}
for ep in SPLITS["eval"]:
    _, res_ep = run_methods(ep, VIZ["max_frames"])
    for label, res in res_ep.items():
        counts[label].append(active_slots(res["masks"]))
        curves[label].append(res["recon_err"].numpy())
fig_active_slots({k: np.concatenate(v) for k, v in counts.items()}, CFG.num_slots, "fig_active_slots")
fig_error_accumulation(curves, counts, CFG.rollout_k, CFG.target_fps, "fig_error_accum")

# Same qualitative figure on a few eval episodes.
for ep in SPLITS["eval"][:VIZ["extra_eval_episodes"]]:
    frames_ep = load_display_frames(CFG, ep)
    feats_ep, res_ep = run_methods(ep, min(VIZ["max_frames"], len(frames_ep)))
    cols_ep = np.linspace(0, len(feats_ep) - 1, VIZ["n_cols"]).round().astype(int)
    fig_qualitative(frames_ep, res_ep, cols_ep, f"fig_qualitative_ep{ep}", f"Eval episode {ep}")

VIDEO = write_comparison_video(FRAMES, RESULTS, os.path.join(OUT_DIR, f"episode_{EP:06d}_compare.mp4"),
                               CFG.target_fps)
print("Figures, video and metrics saved to", os.path.abspath(OUT_DIR))