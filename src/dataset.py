import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import contextlib


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


class PolicyDataset(Dataset):
    """(observation history, state, action chunk) samples, one per observation frame.

    Each episode is a dict with
      obs_path  .npy of per-frame tokens [T, K, F] (slots) or [T, N, D] (DINO patches)
      state     [T, S]  proprioception at the observation frames
      actions   [L, A]  actions at the source rate (L source frames)
      src       [T]     source-frame index of each observation frame
    Chunk step j of frame t is source action src[t] + j * stride; steps past the
    end repeat the last action and are flagged in `is_pad`.
    """

    def __init__(self, episodes, obs_history, chunk_size, stride):
        self.episodes = episodes
        self.obs_history = obs_history
        self.chunk_size = chunk_size
        self.stride = stride
        self.index = [(i, t) for i, ep in enumerate(episodes) for t in range(len(ep["src"]))]
        self._arrays = {}  # opened lazily so each DataLoader worker gets its own memmaps

    def __len__(self):
        return len(self.index)

    def _obs(self, i):
        if i not in self._arrays:
            self._arrays[i] = np.load(self.episodes[i]["obs_path"], mmap_mode="r")
        return self._arrays[i]

    def __getitem__(self, idx):
        return self.sample(*self.index[idx])

    def sample(self, i, t):
        """Sample at observation frame t of episode i."""
        ep = self.episodes[i]
        hist = np.clip(np.arange(t - self.obs_history + 1, t + 1), 0, None)
        obs = np.asarray(self._obs(i)[hist], dtype=np.float32)
        steps = ep["src"][t] + self.stride * np.arange(self.chunk_size)
        is_pad = steps >= len(ep["actions"])
        actions = ep["actions"][np.minimum(steps, len(ep["actions"]) - 1)]
        return {
            "obs": torch.from_numpy(obs),
            "state": torch.from_numpy(ep["state"][t].astype(np.float32)),
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "is_pad": torch.from_numpy(is_pad),
        }



class RolloutPolicyDataset(PolicyDataset):
    """`rollout_k` consecutive decision points t0, t0 + h, ..., t0 + (rollout_k - 1) h
    of one episode (h = exec_horizon), each a PolicyDataset sample, stacked on dim 0.
    Only start frames whose whole rollout lies inside the episode are used."""

    def __init__(self, episodes, obs_history, chunk_size, stride, rollout_k, exec_horizon):
        super().__init__(episodes, obs_history, chunk_size, stride)
        self.rollout_k = rollout_k
        self.exec_horizon = exec_horizon
        span = (rollout_k - 1) * exec_horizon
        self.index = [(i, t) for i, ep in enumerate(episodes) for t in range(len(ep["src"]) - span)]
        if not self.index:
            raise RuntimeError("No rollouts: episodes are shorter than (rollout_k - 1) * exec_horizon + 1 frames.")

    def __getitem__(self, idx):
        i, t0 = self.index[idx]
        steps = [self.sample(i, t0 + r * self.exec_horizon) for r in range(self.rollout_k)]
        return {k: torch.stack([s[k] for s in steps]) for k in steps[0]}  # each [R, ...]
