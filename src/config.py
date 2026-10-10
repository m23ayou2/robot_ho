
from dataclasses import asdict, dataclass, fields
import os
import contextlib

@dataclass
class Config:
    # ---- data ----
    repo_id: str = "unitreerobotics/G1_Dex1_MountCamera_Dataset"
    data_root: str = "/results/slots/g1_data"
    work_dir: str = "/results/slots/ocslots"
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
    rollout_k: int = 10            # transitions per training sample: S_0 -> S_1 -> ... -> S_k
    rollout_sample: bool = False   # draw k ~ U{1..rollout_k} per batch so every horizon is trained
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
