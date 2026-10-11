import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GEOMETRY_DIM = 5  # cx, cy, sx, sy, area of a slot's attention mask


def sinusoid(x, dim):
    """x [...] (any real values) -> [..., dim] sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x.float().unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FlowMatchingPolicy(nn.Module):
    """ACT with two swaps:
      * perception: the ResNet18 feature maps become slot tokens (or raw DINO
        patch tokens for the control run), fed to the same transformer encoder
        together with the proprioceptive state;
      * objective: the CVAE (encoder + z + L1) becomes conditional flow matching.
        The decoder's queries are the noisy action chunk fused with the flow time
        (as in pi0) instead of fixed position embeddings, and it predicts the
        velocity a - eps.

    Flow convention: x_tau = (1 - tau) * eps + tau * a, tau = 0 noise -> 1 data.
    State/action normalization lives in buffers so deployment can't diverge.
    With action_mode="relative" the chunk is modeled as a - state (the current
    proprioception), and action_mean/std are statistics of that offset.
    """

    def __init__(self, cfg, obs_dim, tokens_per_frame, state_dim, action_dim):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.geometry = cfg.obs_tokens == "slots" and cfg.slot_geometry
        feat_dim = obs_dim - GEOMETRY_DIM if self.geometry else obs_dim

        # ---- observation encoder ----
        self.obs_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
        self.geom_proj = nn.Linear(GEOMETRY_DIM, d) if self.geometry else None
        # Slots are a set whose index carries no meaning across episodes: no
        # per-token position. Patch tokens keep a learned 2D position.
        self.token_pos = (nn.Parameter(torch.randn(1, 1, tokens_per_frame, d) * 0.02)
                          if cfg.obs_tokens == "dino" else None)
        self.hist_emb = nn.Parameter(torch.randn(1, cfg.obs_history, 1, d) * 0.02)
        self.state_proj = nn.Linear(state_dim, d)
        self.type_emb = nn.Parameter(torch.randn(2, d) * 0.02)  # 0 = observation, 1 = state
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, cfg.nhead, cfg.ff_dim, cfg.dropout, batch_first=True, norm_first=True),
            cfg.enc_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)

        # ---- flow-matching action decoder ----
        self.action_in = nn.Linear(action_dim, d)
        self.action_time = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("chunk_pos", sinusoid(torch.arange(cfg.chunk_size), d)[None], persistent=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, cfg.nhead, cfg.ff_dim, cfg.dropout, batch_first=True, norm_first=True),
            cfg.dec_layers, norm=nn.LayerNorm(d))
        self.action_out = nn.Linear(d, action_dim)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

        for name, size in [("state_mean", state_dim), ("state_std", state_dim),
                           ("action_mean", action_dim), ("action_std", action_dim)]:
            self.register_buffer(name, torch.zeros(size) if name.endswith("mean") else torch.ones(size))

    # ---- normalization ----
    def set_normalization(self, state_mean, state_std, action_mean, action_std):
        for name, v in [("state_mean", state_mean), ("state_std", state_std),
                        ("action_mean", action_mean), ("action_std", action_std)]:
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32))

    def norm_action(self, a, state):
        """Raw actions [B, C, A] -> normalized flow targets."""
        if self.cfg.action_mode == "relative":
            a = a - state[:, None]
        return (a - self.action_mean) / self.action_std

    def denorm_action(self, a, state):
        a = a * self.action_std + self.action_mean
        return a + state[:, None] if self.cfg.action_mode == "relative" else a

    # ---- building blocks ----
    def encode(self, obs, state):
        """obs [B, H, K, F], state [B, state_dim] (raw) -> memory [B, H*K + 1, d]."""
        B, H, K, _ = obs.shape
        if self.geometry:
            tok = self.obs_proj(obs[..., :-GEOMETRY_DIM]) + self.geom_proj(obs[..., -GEOMETRY_DIM:])
        else:
            tok = self.obs_proj(obs)
        if self.token_pos is not None:
            tok = tok + self.token_pos
        tok = (tok + self.hist_emb[:, -H:] + self.type_emb[0]).reshape(B, H * K, -1)
        s = self.state_proj((state - self.state_mean) / self.state_std) + self.type_emb[1]
        return self.encoder(torch.cat([tok, s[:, None]], dim=1))

    def velocity(self, memory, x_tau, tau):
        """memory [B, M, d], x_tau [B, C, A] (normalized), tau [B] -> v [B, C, A]."""
        a = self.action_in(x_tau)
        t = sinusoid(tau * 1000.0, a.shape[-1])[:, None].expand_as(a)
        q = self.action_time(torch.cat([a, t], dim=-1)) + self.chunk_pos[:, :a.shape[1]]
        return self.action_out(self.decoder(q, memory))

    def sample_tau(self, n, device, generator=None):
        if self.cfg.tau_sampling == "uniform":
            return torch.rand(n, device=device, generator=generator)
        # pi0: Beta(1.5, 1) on 1 - tau/0.999, i.e. more weight on noisy (small) tau.
        # torch.distributions has no generator argument, so draw it from a uniform:
        # if U ~ U(0,1) then U^(1/1.5) ~ Beta(1.5, 1).
        u = torch.rand(n, device=device, generator=generator).pow(1.0 / 1.5)
        return 0.999 * (1.0 - u)

    # ---- training / inference ----
    def loss(self, obs, state, actions, is_pad, generator=None):
        """Flow-matching loss on raw actions [B, C, A]; padded steps are ignored."""
        a = self.norm_action(actions, state)
        eps = torch.randn(a.shape, device=a.device, generator=generator)
        tau = self.sample_tau(a.shape[0], a.device, generator)
        x_tau = (1 - tau)[:, None, None] * eps + tau[:, None, None] * a
        v = self.velocity(self.encode(obs, state), x_tau, tau)
        err = F.mse_loss(v.float(), (a - eps).float(), reduction="none").mean(-1)  # [B, C]
        keep = (~is_pad).float()
        return (err * keep).sum() / keep.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, obs, state, noise=None, steps=None, generator=None):
        """Euler integration from noise to an action chunk. Returns raw actions [B, C, A].
        Pass the same `noise` across calls to keep consecutive chunks in one mode."""
        steps = steps or self.cfg.flow_steps
        memory = self.encode(obs, state)
        B = obs.shape[0]
        if noise is None:
            noise = torch.randn(B, self.cfg.chunk_size, self.action_mean.numel(),
                                device=obs.device, generator=generator)
        x = noise.to(obs.device)
        dt = 1.0 / steps
        for i in range(steps):
            tau = torch.full((B,), i * dt, device=obs.device)
            x = x + dt * self.velocity(memory, x, tau).float()
        return self.denorm_action(x, state)
