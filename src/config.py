
from dataclasses import asdict, dataclass, fields
import os
import contextlib

@dataclass
class Config:
    # ---- data ----
    repo_id: str = "unitreerobotics/G1_Dex1_MountCamera_Dataset"
    data_root: str = "./results/slots/g1_data"
    work_dir: str = "./results/slots/ocslots"
    camera: str = "observation.images.cam_left_high"  # head camera
    num_episodes: int = 40        # of 351 available; raise for better results
    target_fps: float = 10.0      # source is 30 fps; consecutive frames are near-identical
    img_h: int = 210              # 4:3 like the 640x480 source, multiples of 14
    img_w: int = 280
    eval_fraction: float = 0.15
    display_w: int = 320          # visualization panel size
    display_h: int = 240

    # ---- model ----
    dino_model: str = "dinov2_vits14"
    patch_size: int = 14
    num_slots: int = 7
    slot_dim: int = 256
    slot_iters_first: int = 3     # first frame of a clip
    slot_iters_next: int = 2      # later frames start from predicted slots
    decoder_hidden: int = 1024
    decoder_layers: int = 3

    # ---- temporal rollout (truncated backprop) ----
    rollout_k: int = 20            # transitions per training sample: S_0 -> S_1 -> ... -> S_k
    rollout_sample: bool = True   # draw k ~ U{1..rollout_k} per batch so every horizon is trained
    first_frame_weight: float = 1.0  # weight of recon(S_0); 0 = gradients ONLY on the last step
    eval_horizon: int = 100       # frames rolled out per eval episode to measure drift / collapse
    best_metric: str = "long_recon"  # "long_recon" (long-horizon recon error) or "loss" (horizon-k objective)

    # ---- training ----
    frame_gap: int = 1            # in subsampled frames
    batch_size: int = 32
    num_epochs: int = 15
    lr: float = 4e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    temperature: float = 0.1
    contrast_weight: float = 0.5
    num_workers: int = 2
    seed: int = 42
    resume: bool = True

    # ---- visualization ----
    viz_max_frames: int = 300

    def __post_init__(self):
        if self.rollout_k < 1:
            raise ValueError("rollout_k must be >= 1 (S_0 -> S_1 is the shortest rollout).")
        if self.best_metric not in ("long_recon", "loss"):
            raise ValueError("best_metric must be 'long_recon' or 'loss'.")

    @property
    def clip_len(self):
        """Frames loaded per training sample: x_0 .. x_k."""
        return self.rollout_k + 1

    @classmethod
    def from_dict(cls, d):
        """Build from a saved config, ignoring keys from older versions."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

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

    @property
    def ckpt_dir(self):
        return os.path.join(self.work_dir, "checkpoints")


@dataclass
class PolicyConfig:
    """ACT-style action-chunking policy whose image encoder (ResNet in ACT) is
    replaced by the frozen slot model and whose CVAE objective is replaced by
    flow matching. Data paths, episodes and splits come from the slot checkpoint."""

    # ---- inputs ----
    slot_checkpoint: str = "./results/slots/ocslots/checkpoints/best.pt"
    work_dir: str = "./results/slots/policy"
    obs_tokens: str = "slots"     # "slots" (object-centric) or "dino" (raw patch tokens: the control)
    slot_geometry: bool = True    # append centroid / spread / area of each slot's attention mask
    obs_history: int = 1          # observation frames per sample (ACT uses 1)

    # ---- actions ----
    chunk_size: int = 20          # actions per chunk, at target_fps (20 @ 10 fps = 2 s, like ACT's 100 @ 50 Hz)
    action_mode: str = "relative" # "relative" to the current state, or "absolute" joint targets (ACT)

    # ---- model ----
    d_model: int = 256
    nhead: int = 8
    enc_layers: int = 4
    dec_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1

    # ---- flow matching ----
    flow_steps: int = 10          # Euler steps at inference
    tau_sampling: str = "beta"    # "beta" (pi0: emphasizes noisy times) or "uniform"

    # ---- inference ----
    ensemble_m: float = 0.01      # ACT temporal ensemble weight exp(-m * i), i = 0 oldest
    exec_horizon: int = 5         # actions executed per chunk (receding horizon, closed-loop rollouts)

    # ---- closed-loop rollout training ----
    # Unroll the policy over rollout_k chunks: predict a chunk (no grad), execute
    # exec_horizon actions, feed the resulting state back in, predict again. The
    # flow loss is applied at every decision point with the policy's own state, so
    # it learns to recover from its own drift (images stay the recorded ones).
    rollout_k: int = 4            # decision points per sample; 1 = plain teacher-forced training
    rollout_sample: bool = False  # draw k ~ U{1..rollout_k} per batch
    rollout_start_epoch: int = 5  # teacher-forced warm-up: an untrained policy's rollouts are noise
    rollout_flow_steps: int = 5   # Euler steps for the (no-grad) rollout samples; cheaper than flow_steps
    eval_horizon: int = 200       # frames of closed-loop rollout per eval episode
    best_metric: str = "closed_loop"  # "closed_loop" (closed-loop L1) or "l1_chunk" (teacher-forced L1)

    # ---- training ----
    batch_size: int = 64
    num_epochs: int = 60
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    num_workers: int = 2
    seed: int = 42
    resume: bool = True

    def __post_init__(self):
        if self.obs_tokens not in ("slots", "dino"):
            raise ValueError("obs_tokens must be 'slots' or 'dino'.")
        if self.action_mode not in ("absolute", "relative"):
            raise ValueError("action_mode must be 'absolute' or 'relative'.")
        if self.tau_sampling not in ("beta", "uniform"):
            raise ValueError("tau_sampling must be 'beta' or 'uniform'.")
        if self.rollout_k < 1:
            raise ValueError("rollout_k must be >= 1.")
        if self.best_metric not in ("closed_loop", "l1_chunk"):
            raise ValueError("best_metric must be 'closed_loop' or 'l1_chunk'.")
        if not 1 <= self.exec_horizon <= self.chunk_size:
            raise ValueError("exec_horizon must be in [1, chunk_size].")

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    @property
    def run_dir(self):
        return os.path.join(self.work_dir, f"{self.obs_tokens}_{self.action_mode}_rk{self.rollout_k}")

    @property
    def ckpt_dir(self):
        return os.path.join(self.run_dir, "checkpoints")
