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

