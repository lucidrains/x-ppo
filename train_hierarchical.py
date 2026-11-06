# /// script
# dependencies = [
#   "adam-atan2-pytorch",
#   "assoc-scan",
#   "einops",
#   "ema-pytorch",
#   "fire",
#   "gymnasium[box2d,other]",
#   "hl-gauss-pytorch>=0.1.7",
#   "imageio[ffmpeg]",
#   "mean-conc-beta>=0.2.7",
#   "memmap-replay-buffer>=0.1.2",
#   "numpy",
#   "pandas",
#   "seaborn",
#   "torch",
#   "torch-einops-utils>=0.2.5",
#   "tqdm"
# ]
# ///

from __future__ import annotations
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path

import fire
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch import nn, Tensor, tensor, cat, stack, zeros, full
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter
from torch.distributions import Categorical

from einops import repeat, einsum, rearrange
from einops.layers.torch import Rearrange

import gymnasium as gym

from mean_conc_beta import Beta as MeanConcBeta

from x_ppo import (
    hpo_actor_loss,
    calc_gae,
    clipped_value_loss_fn,
    dones_to_masks
)

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)

LOG_2_PI = float(np.log(2. * np.pi))

ENTROPY_SCHEDULE = (0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002)
LR_SCALE_SCHEDULE = (1.0, 0.85, 0.50, 0.20, 0.10, 0.05)
STATE_HISTORY_LR_MULT = 0.02

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

@contextmanager
def frozen(modules):
    params = [p for module in modules for p in module.parameters()]
    flags = [p.requires_grad for p in params]

    for p in params:
        p.requires_grad = False

    try:
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad = flag

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def categorical_entropy(probs):
    return -(probs * log(probs)).sum(dim = -1)

def normalized_categorical_mutual_info(probs):
    num_classes = probs.shape[-1]

    if num_classes <= 1 or probs.shape[-2] <= 1:
        return probs.sum() * 0.

    marginal = probs.mean(dim = -2)
    mutual_info = categorical_entropy(marginal) - categorical_entropy(probs).mean(dim = -1)
    return mutual_info.mean() / float(np.log(num_classes))

def resolve_hierarchies(hierarchies: tuple[int, ...] | int) -> tuple[int, ...]:
    if isinstance(hierarchies, int):
        return tuple(2 ** i for i in range(hierarchies))
    return tuple(int(stride) for stride in hierarchies)

def in_flight_mask(states: Tensor) -> Tensor:
    if states.shape[-1] < 8:
        return torch.ones(states.shape[0], dtype = torch.bool, device = states.device)
    return (states[:, 6] + states[:, 7]) < 1.5

# regularizing auxiliary losses

def counterfactual_decoder_binding_loss(action_probs):
    return 1. - normalized_categorical_mutual_info(action_probs)

def continuous_binding_loss(means, stds):
    var = stds.detach().square()
    var_cond = var.mean(dim = -2)
    var_mix = var_cond + means.var(dim = -2, unbiased = False)
    mi = 0.5 * torch.log1p(var_mix / var_cond.clamp_min(1e-8)).mean(dim = -1)
    return (1. - mi / (mi + 1.)).mean()

def composed_action_coupling_loss(code_probs, action_probs, target_actions):
    composed_probs = einsum(code_probs, action_probs, 'n k, n k a -> n a')
    return F.nll_loss(log(composed_probs), target_actions.long())

def continuous_action_coupling_loss(code_probs, means, stds, target_actions):
    q = ((target_actions[:, None] - means) / stds).square()
    log_comp = (-0.5 * q - stds.log() - 0.5 * LOG_2_PI).sum(dim = -1)
    log_mix = torch.logsumexp(log(code_probs) + log_comp, dim = -1)
    return -log_mix.mean()

# simba residual block - https://arxiv.org/abs/2410.09754

class SimBaBlock(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.net = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim)
        )
        self.norm = nn.RMSNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

# discrete latent readout and hypernetwork decoder

class DiscreteLatentReadout(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        num_discrete = 4,
        temperature = 0.7,
        logit_cap = 3.
    ):
        super().__init__()
        self.num_discrete = num_discrete
        self.temperature = temperature
        self.logit_cap = logit_cap

        self.norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_discrete)
        nn.init.orthogonal_(self.to_logits.weight, gain = 0.1)
        nn.init.zeros_(self.to_logits.bias)

        self.prototypes = Parameter(torch.empty(num_discrete, dim_latent))
        nn.init.orthogonal_(self.prototypes, gain = (dim_latent ** 0.5) * 0.5)

    def forward(self, x):
        raw_logits = self.to_logits(self.norm(x))
        return self.logit_cap * torch.tanh(raw_logits / self.logit_cap)

    def sample(self, logits, actions = None):
        probs = (logits / self.temperature).softmax(dim = -1)

        if not exists(actions) or (actions < 0).any():
            sampled = Categorical(probs = probs).sample() if self.training else probs.argmax(dim = -1)
            actions = torch.where(actions < 0, sampled, actions) if exists(actions) else sampled

        one_hot = F.one_hot(actions.long(), self.num_discrete).float()

        # straight-through estimator

        one_hot = one_hot + probs - probs.detach()

        latents = einsum(one_hot, self.prototypes, '... k, k d -> ... d')
        return latents, actions, probs

class HyperNetworkDecoder(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        low_rank = 8,
        num_time_embeds = 8,
        init_gain = 0.01
    ):
        super().__init__()
        self.num_time_embeds = max(1, num_time_embeds)
        self.time_emb = nn.Embedding(self.num_time_embeds, dim)
        self.latent_time_emb = nn.Embedding(self.num_time_embeds, dim_latent)
        nn.init.normal_(self.time_emb.weight, std = 0.01)
        nn.init.normal_(self.latent_time_emb.weight, std = 0.01)

        self.decoder = nn.Sequential(
            nn.RMSNorm(dim_latent),
            nn.Linear(dim_latent, dim_latent * 2),
            nn.SiLU(),
            nn.Linear(dim_latent * 2, 2 * low_rank * dim, bias = False)
        )

        nn.init.orthogonal_(self.decoder[-1].weight, gain = init_gain)
        self.to_hyper_network_weights = Rearrange('... (two d r) -> two ... d r', two = 2, r = low_rank)

    def forward(self, x, latent, time_indices = None):
        if exists(time_indices):
            clamped_indices = time_indices.clamp(0, self.num_time_embeds - 1)
            x = x + self.time_emb(clamped_indices)
            latent = latent + self.latent_time_emb(clamped_indices)

        weights_in, weights_out = self.to_hyper_network_weights(self.decoder(latent))
        return einsum(x, weights_in, weights_out, '... i, ... i r, ... o r -> ... o')

# composable hierarchy level

class HierarchyLevel(Module):
    def __init__(
        self,
        state_dim,
        dim,
        dim_latent,
        *,
        is_bottom = False,
        is_top = False,
        num_actions = 4,
        continuous = False,
        num_internal_actions = 4,
        low_rank = 8,
        num_time_embeds = 4,
        use_state_history = True,
        state_history_len = 4,
        beta_init_conc = 5.,
        beta_eps = 1e-5,
        dropout = 0.
    ):
        super().__init__()
        self.is_bottom = is_bottom
        self.is_top = is_top
        self.continuous = continuous
        self.state_history_len = state_history_len

        self.proj_in = nn.Linear(state_dim, dim)
        self.stem = SimBaBlock(dim, dropout = dropout)

        if not is_top:
            self.from_higher = HyperNetworkDecoder(
                dim,
                dim_latent,
                low_rank = low_rank,
                num_time_embeds = num_time_embeds
            )
            self.steered_norm = nn.RMSNorm(dim)

        self.upper = SimBaBlock(dim, dropout = dropout)

        if is_bottom:
            out_dim = 2 * num_actions if continuous else num_actions
            self.action_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.SiLU(),
                nn.Linear(dim // 2, out_dim)
            )
            if continuous:
                nn.init.orthogonal_(self.action_head[-1].weight, gain = 0.01)
                nn.init.zeros_(self.action_head[-1].bias)
                self.beta = MeanConcBeta(
                    bounds = (-1., 1.),
                    init_conc = beta_init_conc,
                    unimodal = True,
                    eps = beta_eps
                )
        else:
            self.readout = DiscreteLatentReadout(dim, dim_latent, num_discrete = num_internal_actions)
            self.residual_proj = nn.Linear(dim, dim_latent, bias = False)
            nn.init.zeros_(self.residual_proj.weight)
            self.register_buffer('gate', tensor(0.))

        self.has_state_history = not is_bottom and use_state_history

        if self.has_state_history:
            chunk_dim = state_history_len * state_dim
            self.state_chunk_mlp = nn.Sequential(
                Rearrange('... k d -> ... (k d)'),
                nn.RMSNorm(chunk_dim),
                nn.Linear(chunk_dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, dim)
            )
            nn.init.zeros_(self.state_chunk_mlp[-1].weight)
            nn.init.zeros_(self.state_chunk_mlp[-1].bias)

    def encode_stem(self, state):
        return self.stem(self.proj_in(state))

    def to_logits(self, features):
        if not self.is_bottom:
            return self.readout(features)

        out = self.action_head(features)
        return rearrange(out, '... (a c) -> ... a c', c = 2) if self.continuous else out

    def sample_action(self, out):
        return self.beta.sample(out) if self.continuous else Categorical(logits = out).sample()

    def action_log_prob(self, out, actions):
        return self.beta.log_prob(out, actions) if self.continuous else Categorical(logits = out).log_prob(actions)

    def action_entropy(self, out):
        return self.beta.entropy(out) if self.continuous else Categorical(logits = out).entropy()

    def action_mean(self, out):
        return self.beta.mean(out) if self.continuous else out.argmax(dim = -1)

    def forward(
        self,
        state,
        latent_from_higher = None,
        higher_gate = None,
        actions = None,
        time_indices = None,
        past_states = None
    ):
        hidden = self.encode_stem(state)

        if self.has_state_history and exists(past_states):
            hidden = hidden + 0.02 * self.state_chunk_mlp(past_states)

        if not self.is_top:
            if exists(latent_from_higher) and (not exists(higher_gate) or higher_gate > 0.):
                hidden = hidden + self.from_higher(hidden, latent_from_higher, time_indices = time_indices)

            hidden = self.steered_norm(hidden)

        upper_hidden = self.upper(hidden)
        logits = self.to_logits(upper_hidden)

        if self.is_bottom:
            return logits

        latent, actions, probs = self.readout.sample(logits, actions = actions)
        latent = self.gate * (latent + self.residual_proj(upper_hidden))
        return latent, actions, logits, probs

# networks

class HierarchicalActor(Module):
    def __init__(
        self,
        state_dim = 8,
        hidden_dim = 64,
        num_actions = 4,
        continuous = False,
        beta_init_conc = 5.,
        num_internal_actions = 4,
        dim_latent = 32,
        low_rank = 8,
        hierarchies: tuple[int, ...] = (1, 4, 8),
        phase_shifts: tuple[int, ...] | None = None,
        use_state_history = True,
        dropout = 0.
    ):
        super().__init__()
        assert len(hierarchies) >= 1 and hierarchies[0] == 1, 'hierarchies must start with base stride 1, e.g. (1, 4, 8)'

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.continuous = continuous
        self.num_internal_actions = num_internal_actions
        self.dim_latent = dim_latent
        self.hierarchies = tuple(hierarchies)
        self.num_levels = len(self.hierarchies)
        self.num_internal_levels = self.num_levels - 1
        self.use_state_history = use_state_history and self.num_levels > 1

        phase_shifts = default(phase_shifts, (0,) * self.num_levels)
        assert len(phase_shifts) == self.num_levels
        self.phase_shifts = tuple(phase_shifts)

        dim = hidden_dim * 2
        levels = []

        for level_index in range(self.num_levels):
            is_bottom = (level_index == 0)
            is_top = (level_index == self.num_levels - 1)
            num_time_embeds = max(1, self.hierarchies[level_index + 1] // self.hierarchies[level_index]) if not is_top else 1
            hist_len = max(1, self.hierarchies[level_index] // self.hierarchies[level_index - 1]) if not is_bottom else 1

            levels.append(HierarchyLevel(
                state_dim,
                dim,
                dim_latent,
                is_bottom = is_bottom,
                is_top = is_top,
                num_actions = num_actions,
                continuous = continuous,
                beta_init_conc = beta_init_conc,
                num_internal_actions = num_internal_actions,
                low_rank = low_rank,
                num_time_embeds = num_time_embeds,
                use_state_history = self.use_state_history,
                state_history_len = hist_len,
                dropout = dropout
            ))

        self.levels = ModuleList(levels)

    def init_past_states(self, batch_size: int, device: torch.device) -> list[Tensor | None]:
        return [
            zeros(batch_size, level.state_history_len, self.state_dim, device = device)
            if level.has_state_history else None
            for level in self.levels
        ]

    def update_past_states(self, past_states, state, timestep):
        if not self.use_state_history:
            return

        for level_index in range(1, self.num_levels):
            stride = self.hierarchies[level_index - 1]
            shift = self.phase_shifts[level_index - 1]

            if not divisible_by(timestep - shift, stride):
                continue

            past_states[level_index] = cat((
                past_states[level_index][:, 1:],
                state[:, None]
            ), dim = 1)

    def reset_past_states(self, past_states, done):
        if not self.use_state_history:
            return

        for level_index in range(1, self.num_levels):
            past_states[level_index] = past_states[level_index].masked_fill(done[:, None, None], 0.)

    def level_time_indices(self, level_index: int, timesteps: Tensor | int | None):
        if not exists(timesteps) or level_index >= self.num_levels - 1:
            return None

        stride_current = self.hierarchies[level_index]
        stride_higher = self.hierarchies[level_index + 1]
        shift_higher = self.phase_shifts[level_index + 1]
        return ((timesteps - shift_higher) % stride_higher) // stride_current

    def counterfactual_probe(
        self,
        state,
        level_index = 0,
        time_indices = None,
        temperature = 0.5,
        bind_receiver = False
    ):
        assert 0 <= level_index < self.num_levels - 1

        if state.ndim == 1:
            state = rearrange(state, '... -> 1 ...')

        num_codes = self.num_internal_actions
        receiver = self.levels[level_index]
        sender = self.levels[level_index + 1]

        hidden = receiver.encode_stem(state).detach()

        repeated_hidden = repeat(hidden, 'b d -> (b k) d', k = num_codes)
        repeated_time_indices = repeat(time_indices, 'b -> (b k)', k = num_codes) if exists(time_indices) else None
        repeated_prototypes = repeat(sender.readout.prototypes, 'k d -> (b k) d', b = hidden.shape[0])

        receiver_head = receiver.action_head if receiver.is_bottom else receiver.readout
        receiver_modules = (receiver.steered_norm, receiver.upper, receiver_head)
        context = nullcontext() if bind_receiver else frozen(receiver_modules)

        with context:
            steered = receiver.steered_norm(
                repeated_hidden + receiver.from_higher(
                    repeated_hidden, repeated_prototypes, time_indices = repeated_time_indices
                )
            )
            steered_out = receiver.to_logits(receiver.upper(steered))

        if receiver.is_bottom and receiver.continuous:
            means = receiver.beta.mean(steered_out)
            stds = receiver.beta(steered_out).stddev
            return rearrange(means, '(b k) a -> b k a', k = num_codes), rearrange(stds, '(b k) a -> b k a', k = num_codes), None

        with torch.no_grad():
            base_out = receiver.to_logits(receiver.upper(receiver.steered_norm(hidden)))

        delta_logits = steered_out - repeat(base_out, 'b a -> (b k) a', k = num_codes)
        probs = (delta_logits / temperature).softmax(dim = -1)
        return None, None, rearrange(probs, '(b k) a -> b k a', k = num_codes)

    def forward_level(
        self,
        level_index: int,
        state: Tensor,
        latent_from_higher: Tensor | None = None,
        actions: Tensor | None = None,
        time_indices: Tensor | None = None,
        past_states: Tensor | None = None
    ):
        level = self.levels[level_index]
        higher_gate = self.levels[level_index + 1].gate if level_index < self.num_levels - 1 else None
        return level(
            state,
            latent_from_higher = latent_from_higher,
            higher_gate = higher_gate,
            actions = actions,
            time_indices = time_indices,
            past_states = past_states
        )

    def forward(
        self,
        state: Tensor,
        actions: Tensor | None = None,
        timesteps: Tensor | None = None,
        past_states: list[Tensor | None] | None = None
    ):
        no_batch = state.ndim == 1
        if no_batch:
            state = rearrange(state, '... -> 1 ...')

        internal_actions_per_level = actions.unbind(dim = -1) if exists(actions) else (None,) * self.num_internal_levels

        latent_above = None
        internals_top_down = []

        for level_index in reversed(range(self.num_levels)):
            time_indices = self.level_time_indices(level_index, timesteps)
            level_past = past_states[level_index] if exists(past_states) else None

            if level_index == 0:
                action_logits = self.forward_level(0, state, latent_from_higher = latent_above, time_indices = time_indices)
            else:
                level_action = internal_actions_per_level[level_index - 1]
                latent_above, sampled_action, level_logits, _ = self.forward_level(
                    level_index, state, latent_from_higher = latent_above,
                    actions = level_action, time_indices = time_indices, past_states = level_past
                )
                internals_top_down.append((level_logits, sampled_action))

        if no_batch:
            action_logits = rearrange(action_logits, '1 ... -> ...')

        return action_logits, list(reversed(internals_top_down))

class Critic(Module):
    def __init__(
        self,
        state_dim = 8,
        hidden_dim = 192
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# evaluation and video recording

@torch.no_grad()
def record_best_episode_video(
    actor: HierarchicalActor,
    env_name: str = 'LunarLander-v3',
    continuous: bool = False,
    video_folder: str = './lunar-recording',
    video_prefix: str = 'hierarchical-best',
    max_timesteps: int = 600,
    num_eval_episodes: int = 8,
    seed: int = 100
):
    actor.eval()
    env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_timesteps, continuous = continuous)

    best_return = -float('inf')
    best_frames = []
    best_seed = seed

    for episode_index in range(num_eval_episodes):
        episode_seed = seed + episode_index
        obs, _ = env.reset(seed = episode_seed)
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        held_latents = [zeros(1, actor.dim_latent, device = device) for _ in range(actor.num_levels)]
        past_states_buf = actor.init_past_states(1, device)
        frames = [env.render()]
        episode_return = 0.

        for timestep in range(max_timesteps):
            timestep_tensor = full((1,), timestep, dtype = torch.long, device = device)

            for level_index in reversed(range(1, actor.num_levels)):
                stride = actor.hierarchies[level_index]
                shift = actor.phase_shifts[level_index]
                if timestep == 0 or divisible_by(timestep - shift, stride):
                    time_indices = actor.level_time_indices(level_index, timestep_tensor)
                    latent_above = held_latents[level_index + 1] if (level_index + 1) < actor.num_levels else None
                    held_latents[level_index], _, _, _ = actor.forward_level(
                        level_index, state, latent_from_higher = latent_above,
                        time_indices = time_indices, past_states = past_states_buf[level_index]
                    )

            base_time_indices = actor.level_time_indices(0, timestep_tensor)
            latent_from_level_one = held_latents[1] if actor.num_levels > 1 else None
            action_out = actor.forward_level(0, state, latent_from_higher = latent_from_level_one, time_indices = base_time_indices)

            actor.update_past_states(past_states_buf, state, timestep)

            action = actor.levels[0].action_mean(action_out)
            action = action.squeeze(0).cpu().numpy() if actor.continuous else action.item()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            frames.append(env.render())

            if terminated or truncated:
                break

            state = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        if episode_return > best_return:
            best_return = episode_return
            best_frames = frames
            best_seed = episode_seed

    env.close()

    out_dir = Path(video_folder).resolve()
    out_dir.mkdir(parents = True, exist_ok = True)
    mp4_path = out_dir / f'{video_prefix}.mp4'
    gif_path = out_dir / f'{video_prefix}.gif'

    imageio.mimsave(str(mp4_path), best_frames, fps = 50, macro_block_size = 1)
    imageio.mimsave(str(gif_path), best_frames[::2], fps = 25)

    print(f'\n[video] best eval return: {best_return:+.1f} (seed={best_seed}) -> {mp4_path}')
    return best_return, str(mp4_path), str(gif_path)

def plot_return_comparison(
    csv_paths: list[str | Path],
    labels: list[str],
    out_path: str | Path = './lunar-recording/lunar_returns_comparison.png',
    title: str = 'LunarLander-v3: hierarchical vs flat baselines'
):
    frames = []

    for csv_path, label in zip(csv_paths, labels):
        path = Path(csv_path)
        if not path.exists():
            continue

        frame = pd.read_csv(path)
        if frame.empty:
            continue

        frames.append(frame.assign(
            label = f'{label} (peak {frame.recent_return_32.max():+.1f})',
            env_steps = frame.env_steps / 1e3
        ))

    if not frames:
        return None

    traces = pd.concat(frames, ignore_index = True)

    sns.set_theme(style = 'whitegrid', context = 'talk')
    fig, ax = plt.subplots(figsize = (11., 6.), dpi = 160)

    sns.lineplot(data = traces, x = 'env_steps', y = 'update_return', hue = 'label', alpha = 0.2, linewidth = 0.8, errorbar = None, ax = ax, legend = False)
    sns.lineplot(data = traces, x = 'env_steps', y = 'recent_return_32', hue = 'label', linewidth = 2.4, errorbar = None, ax = ax)

    ax.axhline(200., color = 'green', linestyle = '--', linewidth = 1.1, label = 'solved (+200)')
    ax.axhline(0., color = 'gray', linestyle = ':', linewidth = 0.9)
    ax.set(xlabel = 'environment steps (k)', ylabel = 'episode return', title = title)
    ax.legend(loc = 'lower right', fontsize = 9)

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(out_path, bbox_inches = 'tight')
    plt.close(fig)
    print(f'[plot] saved -> {out_path}')
    return str(out_path)

# main

def main(
    env_name = 'LunarLander-v3',
    continuous = False,
    beta_init_conc = 5.,
    hierarchies: tuple[int, ...] | int = (1, 4, 8),
    phase_shifts: tuple[int, ...] | None = None,
    num_cycles = 4,
    num_phases: int | None = None,
    cycle_length: int | None = None,
    updates_per_phase = 10,
    simultaneous = False,
    use_state_history = True,
    always_train_base_level = True,
    reset_latents_on_done = True,
    num_envs = 16,
    rollout_steps = 384,
    max_episode_steps = 600,
    actor_hidden_dim = 64,
    critic_hidden_dim = 192,
    num_internal_actions = 4,
    dim_latent = 32,
    low_rank = 8,
    epochs = 6,
    minibatch_envs = 4,
    lr = 1.0e-3,
    critic_lr = 2.5e-3,
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.2,
    internal_policy_loss_weight = 0.25,
    binding_loss_weight = 0.10,
    coupling_loss_weight = 0.10,
    max_grad_norm = 0.5,
    seed = 42,
    render = True,
    video_folder = './lunar-recording',
    video_prefix = 'hierarchical-best',
    save_path = './ppo_hierarchical.pt',
    log_csv = None,
    compare_with_flat = False,
    comparison_plot_path = './lunar-recording/lunar_returns_comparison.png'
):
    hierarchies = resolve_hierarchies(hierarchies)
    num_levels = len(hierarchies)
    total_phases = default(num_phases, num_cycles * num_levels)
    cycle_length = default(cycle_length, num_levels)

    if compare_with_flat:
        out_dir = Path(video_folder)
        hier_csv = out_dir / 'hierarchical_returns.csv'
        flat64_csv = out_dir / 'flat_dim64_returns.csv'
        flat_csv = out_dir / 'flat_returns.csv'
        flat_hidden_dim = int(round(actor_hidden_dim * num_levels ** 0.5))

        common = dict(
            env_name = env_name, continuous = continuous, beta_init_conc = beta_init_conc,
            num_phases = total_phases, cycle_length = cycle_length, updates_per_phase = updates_per_phase,
            num_envs = num_envs, rollout_steps = rollout_steps, max_episode_steps = max_episode_steps,
            critic_hidden_dim = critic_hidden_dim, epochs = epochs, minibatch_envs = minibatch_envs,
            lr = lr, critic_lr = critic_lr, gamma = gamma, lam = lam, eps_clip = eps_clip,
            max_grad_norm = max_grad_norm, seed = seed, video_folder = video_folder
        )

        runs = (
            (f'hierarchical {hierarchies} dim={actor_hidden_dim}', dict(
                hierarchies = hierarchies, phase_shifts = phase_shifts, simultaneous = simultaneous,
                use_state_history = use_state_history, always_train_base_level = always_train_base_level,
                reset_latents_on_done = reset_latents_on_done, actor_hidden_dim = actor_hidden_dim,
                num_internal_actions = num_internal_actions, dim_latent = dim_latent, low_rank = low_rank,
                internal_policy_loss_weight = internal_policy_loss_weight, binding_loss_weight = binding_loss_weight,
                coupling_loss_weight = coupling_loss_weight, render = render, video_prefix = video_prefix,
                save_path = save_path, log_csv = str(hier_csv)
            )),
            (f'flat (1,) dim={actor_hidden_dim}', dict(
                hierarchies = (1,), actor_hidden_dim = actor_hidden_dim, use_state_history = False,
                render = False, save_path = str(out_dir / 'flat_dim64.pt'), log_csv = str(flat64_csv)
            )),
            (f'flat (1,) dim={flat_hidden_dim}', dict(
                hierarchies = (1,), actor_hidden_dim = flat_hidden_dim, use_state_history = False,
                render = False, save_path = str(out_dir / 'flat.pt'), log_csv = str(flat_csv)
            ))
        )

        for run_index, (name, overrides) in enumerate(runs, 1):
            print(f'=== [{run_index}/{len(runs)}] {name} ===', flush = True)
            main(**common, **overrides)

        plot_return_comparison(
            [hier_csv, flat64_csv, flat_csv],
            [name for name, _ in runs],
            out_path = comparison_plot_path,
            title = f'{env_name}: hierarchical vs flat baselines'
        )
        return

    phase_shifts = tuple(int(shift) for shift in phase_shifts) if exists(phase_shifts) else (0,) * num_levels

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make_vec(
        env_name, num_envs = num_envs, vectorization_mode = 'sync',
        max_episode_steps = max_episode_steps, continuous = continuous
    )

    state_dim = env.single_observation_space.shape[0]
    num_actions = env.single_action_space.shape[0] if continuous else int(env.single_action_space.n)

    actor = HierarchicalActor(
        state_dim = state_dim, hidden_dim = actor_hidden_dim, num_actions = num_actions,
        continuous = continuous, beta_init_conc = beta_init_conc,
        num_internal_actions = num_internal_actions, dim_latent = dim_latent,
        low_rank = low_rank, hierarchies = hierarchies, phase_shifts = phase_shifts,
        use_state_history = use_state_history
    ).to(device)

    base_level = actor.levels[0]
    critic = Critic(state_dim, critic_hidden_dim).to(device)

    # level-wise parameter partitioning: each higher level l >= 1 owns its encoder/readout and downward hypernetwork

    level_params = []
    level_optimizers = []
    level_lrs = [
        lr * (0.85 if num_levels >= 3 and level_index == num_levels - 1 else 1.)
        for level_index in range(num_levels)
    ]

    for level_index, level in enumerate(actor.levels):
        params = [*level.proj_in.parameters(), *level.stem.parameters(), *level.upper.parameters()]

        if not level.is_top:
            params.extend([*level.steered_norm.parameters()])

        if level.is_bottom:
            params.extend([*level.action_head.parameters()])
        else:
            params.extend([
                *level.readout.parameters(),
                *level.residual_proj.parameters(),
                *actor.levels[level_index - 1].from_higher.parameters()
            ])

        level_lr = level_lrs[level_index]
        history_params = [*level.state_chunk_mlp.parameters()] if level.has_state_history else []

        level_params.append([*params, *history_params])

        param_groups = [{'params': params, 'lr_mult': 1.}]

        if history_params:
            param_groups.append({'params': history_params, 'lr_mult': STATE_HISTORY_LR_MULT})

        level_optimizers.append(torch.optim.AdamW(param_groups, lr = level_lr))

    opt_all = torch.optim.AdamW(actor.parameters(), lr = lr)
    opt_critic = torch.optim.AdamW(critic.parameters(), lr = critic_lr)

    phases = [('all', updates_per_phase) for _ in range(total_phases)] if simultaneous else [
        (phase_index % num_levels, updates_per_phase) for phase_index in range(total_phases)
    ]

    obs, _ = env.reset(seed = seed)
    state = torch.from_numpy(obs).float().to(device)
    running_episode_return = np.zeros(num_envs)

    held_internal_actions = [zeros(num_envs, dtype = torch.long, device = device) for _ in range(num_levels)]
    held_internal_log_probs = [zeros(num_envs, device = device) for _ in range(num_levels)]
    held_latents = [zeros(num_envs, dim_latent, device = device) for _ in range(num_levels)]
    past_states_buf = actor.init_past_states(num_envs, device)

    recent_returns = []
    best_rolling_return = -float('inf')
    best_actor_state = deepcopy(actor.state_dict())
    start_time = time.time()
    total_env_steps = 0

    if exists(log_csv):
        Path(log_csv).parent.mkdir(parents = True, exist_ok = True)
        with open(log_csv, 'w') as f:
            f.write('phase_index,phase_label,update_index,env_steps,recent_return_32,update_return,best_rolling_return\n')

    for phase_index, (phase, num_updates) in enumerate(phases):
        actor.train()
        critic.train()

        max_active_level = (num_levels - 1) if phase == 'all' else int(phase)
        for level_index in range(1, max_active_level + 1):
            actor.levels[level_index].gate.fill_(1.)

        cycle_index = phase_index // max(1, cycle_length)

        entropy_coef = ENTROPY_SCHEDULE[min(cycle_index, len(ENTROPY_SCHEDULE) - 1)]
        lr_scale = LR_SCALE_SCHEDULE[min(cycle_index, len(LR_SCALE_SCHEDULE) - 1)]
        aux_scale = lr_scale ** 2

        for level_index, optimizer in enumerate(level_optimizers):
            co_scale = 0.15 if (level_index == 0 and isinstance(phase, int) and phase > 0 and always_train_base_level) else 1.

            for param_group in optimizer.param_groups:
                param_group['lr'] = level_lrs[level_index] * lr_scale * co_scale * param_group.get('lr_mult', 1.)

        phase_returns = []
        landed_flags = []
        start_return_32 = float(np.mean(recent_returns[-32:])) if recent_returns else -200.
        phase_label = f'level {phase} (dt={hierarchies[phase]})' if isinstance(phase, int) else 'all'

        for update_index in range(num_updates):
            rollout_states, rollout_actions, rollout_log_probs = [], [], []
            rollout_rewards, rollout_terminated, rollout_truncated = [], [], []
            rollout_internal_actions = [[] for _ in range(num_levels)]
            rollout_internal_log_probs = [[] for _ in range(num_levels)]
            rollout_time_indices = [[] for _ in range(num_levels)]
            rollout_past_states = [[] for _ in range(num_levels)]
            episode_returns = []

            for timestep in range(rollout_steps):
                timestep_tensor = full((num_envs,), timestep, dtype = torch.long, device = device)
                step_time_indices = [actor.level_time_indices(level_index, timestep_tensor) for level_index in range(num_levels)]

                with torch.no_grad():
                    for level_index in reversed(range(1, num_levels)):
                        stride = hierarchies[level_index]
                        shift = phase_shifts[level_index]
                        if divisible_by(timestep - shift, stride):
                            latent_above = held_latents[level_index + 1] if (level_index + 1) < num_levels else None
                            latent, sampled_code, _, probs = actor.forward_level(
                                level_index, state, latent_from_higher = latent_above,
                                time_indices = step_time_indices[level_index],
                                past_states = past_states_buf[level_index]
                            )
                            held_latents[level_index] = latent
                            held_internal_actions[level_index] = sampled_code
                            held_internal_log_probs[level_index] = Categorical(probs = probs).log_prob(sampled_code)

                    latent_from_level_one = held_latents[1] if num_levels > 1 else None
                    level_out = actor.forward_level(0, state, latent_from_higher = latent_from_level_one, time_indices = step_time_indices[0])
                    action = base_level.sample_action(level_out)
                    action_log_prob = base_level.action_log_prob(level_out, action)

                if actor.use_state_history:
                    for level_index in range(1, num_levels):
                        rollout_past_states[level_index].append(past_states_buf[level_index].clone())

                    actor.update_past_states(past_states_buf, state, timestep)

                next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = terminated | truncated

                if reset_latents_on_done and done.any():
                    actor.reset_past_states(past_states_buf, torch.from_numpy(done).bool().to(device))

                running_episode_return += reward

                for env_index in range(num_envs):
                    if done[env_index]:
                        episode_returns.append(float(running_episode_return[env_index]))
                        recent_returns.append(float(running_episode_return[env_index]))
                        landed_flags.append(1. if reward[env_index] >= 50. else 0.)
                        running_episode_return[env_index] = 0.

                rollout_states.append(state)
                rollout_actions.append(action)
                rollout_log_probs.append(action_log_prob)
                rollout_rewards.append(torch.from_numpy(reward / 100.).float().to(device))
                rollout_terminated.append(torch.from_numpy(terminated).bool().to(device))
                rollout_truncated.append(torch.from_numpy(truncated).bool().to(device))

                for level_index in range(num_levels):
                    if level_index >= 1:
                        rollout_internal_actions[level_index].append(held_internal_actions[level_index].clone())
                        rollout_internal_log_probs[level_index].append(held_internal_log_probs[level_index].clone())
                    if exists(step_time_indices[level_index]):
                        rollout_time_indices[level_index].append(step_time_indices[level_index])

                state = torch.from_numpy(next_obs).float().to(device)

            total_env_steps += rollout_steps * num_envs

            if episode_returns:
                phase_returns.extend(episode_returns)
                current_rolling_return = float(np.mean(recent_returns[-32:]))
                if current_rolling_return > best_rolling_return:
                    best_rolling_return = current_rolling_return
                    best_actor_state = deepcopy(actor.state_dict())

            if exists(log_csv):
                recent_return_32 = float(np.mean(recent_returns[-32:])) if recent_returns else float('nan')
                update_return = float(np.mean(episode_returns)) if episode_returns else float('nan')
                with open(log_csv, 'a') as f:
                    f.write(f'{phase_index + 1},{phase_label},{update_index + 1},{total_env_steps},{recent_return_32:.4f},{update_return:.4f},{best_rolling_return:.4f}\n')

            states = stack(rollout_states, dim = 1)
            actions = stack(rollout_actions, dim = 1)
            old_action_log_probs = stack(rollout_log_probs, dim = 1)
            rewards = stack(rollout_rewards, dim = 1)
            terminated_flags = stack(rollout_terminated, dim = 1)
            truncated_flags = stack(rollout_truncated, dim = 1)
            _, done_masks = dones_to_masks(terminated = terminated_flags, truncated = truncated_flags)

            internal_actions = {level_index: stack(rollout_internal_actions[level_index], dim = 1) for level_index in range(1, num_levels)}
            old_internal_log_probs = {level_index: stack(rollout_internal_log_probs[level_index], dim = 1) for level_index in range(1, num_levels)}
            time_indices = {level_index: stack(rollout_time_indices[level_index], dim = 1) for level_index in range(num_levels - 1)}
            past_states_stacked = (
                {level_index: stack(rollout_past_states[level_index], dim = 1) for level_index in range(1, num_levels)}
                if actor.use_state_history else {}
            )

            with torch.no_grad():
                values = critic(states)
                next_value = critic(state)
                gae_per_level = calc_gae(
                    rewards,
                    values,
                    masks = done_masks,
                    next_value = next_value,
                    gamma = gamma,
                    lam = lam,
                    chunk_size = hierarchies,
                    use_accelerated = False,
                    return_advantages = True
                )

            for epoch_index in range(epochs):
                perm = torch.randperm(num_envs, device = device)

                for start in range(0, num_envs, minibatch_envs):
                    batch_indices = perm[start:start + minibatch_envs]
                    batch_size = batch_indices.shape[0]

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_action_log_probs[batch_indices]
                    batch_old_values = values[batch_indices]
                    batch_base_returns = gae_per_level[0].returns[batch_indices]
                    batch_base_advantages = gae_per_level[0].advantages[batch_indices]

                    flat_states = rearrange(batch_states, 'b t d -> (b t) d')
                    flat_actions = rearrange(batch_actions, 'b t ... -> (b t) ...')
                    flat_old_log_probs = rearrange(batch_old_log_probs, 'b t -> (b t)')
                    flat_base_advantages = rearrange(batch_base_advantages, 'b t -> (b t)')
                    flat_base_returns = rearrange(batch_base_returns, 'b t -> (b t)')
                    flat_old_values = rearrange(batch_old_values, 'b t -> (b t)')
                    flat_base_time_indices = rearrange(time_indices[0][batch_indices], 'b t -> (b t)') if num_levels > 1 else None

                    # top-down chunked forward pass across all hierarchy levels

                    latents_for_lower = {}
                    probs_per_level = {}
                    chunk_states_per_level = {}
                    chunk_actions_per_level = {}
                    chunk_old_log_probs_per_level = {}

                    for level_index in reversed(range(1, num_levels)):
                        stride = hierarchies[level_index]
                        chunk_states = rearrange(batch_states[:, 0::stride], 'b c d -> (b c) d')
                        chunk_internal_actions = rearrange(internal_actions[level_index][batch_indices][:, 0::stride], 'b c -> (b c)')
                        chunk_old_log_probs = rearrange(old_internal_log_probs[level_index][batch_indices][:, 0::stride], 'b c -> (b c)')
                        chunk_time_indices = rearrange(time_indices[level_index][batch_indices][:, 0::stride], 'b c -> (b c)') if level_index < (num_levels - 1) else None
                        chunk_past_states = (
                            rearrange(past_states_stacked[level_index][batch_indices][:, 0::stride], 'b c k d -> (b c) k d')
                            if actor.use_state_history else None
                        )

                        chunk_states_per_level[level_index] = chunk_states
                        chunk_actions_per_level[level_index] = chunk_internal_actions
                        chunk_old_log_probs_per_level[level_index] = chunk_old_log_probs

                        latent_above = latents_for_lower.get(level_index + 1, None)
                        need_grad = (phase == 'all') or (isinstance(phase, int) and phase >= level_index)

                        with (nullcontext() if need_grad else torch.no_grad()):
                            chunk_latents, _, _, chunk_probs = actor.forward_level(
                                level_index, chunk_states, latent_from_higher = latent_above,
                                actions = chunk_internal_actions, time_indices = chunk_time_indices, past_states = chunk_past_states
                            )

                        probs_per_level[level_index] = chunk_probs
                        repeat_factor = max(1, stride // hierarchies[level_index - 1])
                        latents_for_lower[level_index] = repeat(
                            rearrange(chunk_latents, '(b c) d -> b c d', b = batch_size), 'b c d -> (b c r) d', r = repeat_factor
                        )

                    level_out = actor.forward_level(0, flat_states, latent_from_higher = latents_for_lower.get(1, None), time_indices = flat_base_time_indices)
                    if continuous and phase != 0 and phase != 'all':
                        level_out = stack((level_out[..., 0], level_out[..., 1].detach()), dim = -1)

                    base_log_probs = base_level.action_log_prob(level_out, flat_actions)
                    base_entropy = base_level.action_entropy(level_out).mean()
                    base_policy_loss = hpo_actor_loss(base_log_probs, flat_old_log_probs, flat_base_advantages, eps_clip = eps_clip, normalize_advantages = True).mean()

                    if phase == 0:
                        loss = base_policy_loss - entropy_coef * base_entropy

                        if num_levels > 1 and actor.levels[1].gate > 0.:
                            in_flight = (flat_states[:128, 6] + flat_states[:128, 7]) < 1.5 if state_dim >= 8 else torch.ones(min(128, flat_states.shape[0]), dtype = torch.bool, device = device)
                            if in_flight.any():
                                cf_means, cf_stds, cf_probs = actor.counterfactual_probe(
                                    flat_states[:128][in_flight], level_index = 0,
                                    time_indices = flat_base_time_indices[:128][in_flight], bind_receiver = True
                                )
                                binding_aux = continuous_binding_loss(cf_means, cf_stds) if continuous else counterfactual_decoder_binding_loss(cf_probs)
                                loss = loss + 0.5 * binding_loss_weight * aux_scale * binding_aux

                        level_optimizers[0].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(level_params[0], max_grad_norm)
                        level_optimizers[0].step()

                    elif isinstance(phase, int) and 1 <= phase < num_levels:
                        level_index = phase
                        lower_index = level_index - 1
                        stride = hierarchies[level_index]

                        chunk_states = chunk_states_per_level[level_index]
                        chunk_internal_actions = chunk_actions_per_level[level_index]
                        chunk_old_log_probs = chunk_old_log_probs_per_level[level_index]
                        chunk_probs = probs_per_level[level_index]
                        chunk_advantages = rearrange(gae_per_level[level_index].advantages[batch_indices], 'b c -> (b c)')

                        internal_dist = Categorical(probs = chunk_probs)
                        level_policy_loss = hpo_actor_loss(internal_dist.log_prob(chunk_internal_actions), chunk_old_log_probs, chunk_advantages, eps_clip = eps_clip, normalize_advantages = True).mean()

                        sub_count = min(128, chunk_states.shape[0])
                        in_flight = (chunk_states[:sub_count, 6] + chunk_states[:sub_count, 7]) < 1.5 if state_dim >= 8 else torch.ones(sub_count, dtype = torch.bool, device = device)
                        active_slice = in_flight if in_flight.any() else slice(None)
                        binding_states = chunk_states[:sub_count][active_slice]

                        lower_is_continuous = continuous and lower_index == 0

                        binding_means, binding_stds, binding_probs = actor.counterfactual_probe(
                            binding_states, level_index = lower_index,
                            time_indices = zeros(binding_states.shape[0], dtype = torch.long, device = device),
                            bind_receiver = False
                        )
                        binding_loss = continuous_binding_loss(binding_means, binding_stds) if lower_is_continuous else counterfactual_decoder_binding_loss(binding_probs)

                        with torch.no_grad():
                            cf_means, cf_stds, cf_probs = actor.counterfactual_probe(
                                chunk_states, level_index = lower_index,
                                time_indices = zeros(chunk_states.shape[0], dtype = torch.long, device = device),
                                bind_receiver = False
                            )

                        if lower_index == 0:
                            target_actions = rearrange(batch_actions[:, 0::stride], 'b c ... -> (b c) ...')
                        else:
                            target_actions = rearrange(internal_actions[lower_index][batch_indices][:, 0::stride], 'b c -> (b c)')

                        positive_mask = chunk_advantages > 0.

                        if not positive_mask.any():
                            coupling_loss = zeros((), device = device)
                        elif lower_is_continuous:
                            coupling_loss = continuous_action_coupling_loss(
                                chunk_probs[positive_mask], cf_means[positive_mask], cf_stds[positive_mask], target_actions[positive_mask]
                            )
                        else:
                            coupling_loss = composed_action_coupling_loss(
                                chunk_probs[positive_mask], cf_probs[positive_mask], target_actions[positive_mask]
                            )

                        if lower_index == 0:
                            lower_policy_loss = base_policy_loss
                            lower_entropy = base_entropy
                        else:
                            lower_dist = Categorical(probs = probs_per_level[lower_index])
                            lower_advantages = rearrange(gae_per_level[lower_index].advantages[batch_indices], 'b c -> (b c)')
                            lower_policy_loss = 0.5 * base_policy_loss + 0.5 * hpo_actor_loss(
                                lower_dist.log_prob(chunk_actions_per_level[lower_index]),
                                chunk_old_log_probs_per_level[lower_index],
                                lower_advantages,
                                eps_clip = eps_clip,
                                normalize_advantages = True
                            ).mean()
                            lower_entropy = lower_dist.entropy().mean()

                        loss = (
                            lower_policy_loss +
                            internal_policy_loss_weight * level_policy_loss -
                            entropy_coef * (lower_entropy + internal_dist.entropy().mean()) +
                            binding_loss_weight * aux_scale * binding_loss +
                            coupling_loss_weight * aux_scale * coupling_loss
                        )

                        co_step_base = always_train_base_level and (epoch_index == epochs - 1)
                        level_optimizers[level_index].zero_grad()
                        if co_step_base:
                            level_optimizers[0].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(level_params[level_index], max_grad_norm)
                        level_optimizers[level_index].step()
                        if co_step_base:
                            nn.utils.clip_grad_norm_(level_params[0], max_grad_norm)
                            level_optimizers[0].step()

                    elif phase == 'all':
                        loss = base_policy_loss - entropy_coef * base_entropy
                        for level_index in range(1, num_levels):
                            internal_dist = Categorical(probs = probs_per_level[level_index])
                            chunk_advantages = rearrange(gae_per_level[level_index].advantages[batch_indices], 'b c -> (b c)')
                            level_policy_loss = hpo_actor_loss(
                                internal_dist.log_prob(chunk_actions_per_level[level_index]),
                                chunk_old_log_probs_per_level[level_index],
                                chunk_advantages,
                                eps_clip = eps_clip,
                                normalize_advantages = True
                            ).mean()
                            loss = loss + internal_policy_loss_weight * level_policy_loss - entropy_coef * internal_dist.entropy().mean()

                        opt_all.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                        opt_all.step()

                    predicted_values = critic(flat_states)
                    value_loss = clipped_value_loss_fn(predicted_values, flat_old_values, flat_base_returns, clip = None)
                    opt_critic.zero_grad()
                    value_loss.backward()
                    opt_critic.step()

        end_return_32 = float(np.mean(recent_returns[-32:])) if recent_returns else -200.
        phase_mean_return = float(np.mean(phase_returns)) if phase_returns else end_return_32
        landed_percent = 100. * float(np.mean(landed_flags)) if landed_flags else 0.
        print(
            f'Phase {phase_index + 1:2d} [{phase_label:14s}]: '
            f'mean = {phase_mean_return:+6.1f} | start32 = {start_return_32:+6.1f} -> end32 = {end_return_32:+6.1f} '
            f'(delta = {end_return_32 - start_return_32:+6.1f}) | land = {landed_percent:4.1f}% [{time.time() - start_time:.1f}s]',
            flush = True
        )

    env.close()

    actor.load_state_dict(best_actor_state)
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict(),
        'hierarchies': hierarchies,
        'continuous': continuous
    }, save_path)

    if render:
        record_best_episode_video(
            actor, env_name = env_name, continuous = continuous,
            video_folder = video_folder, video_prefix = video_prefix, max_timesteps = max_episode_steps
        )

if __name__ == '__main__':
    fire.Fire(main)
