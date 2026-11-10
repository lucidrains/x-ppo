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
            with torch.random.fork_rng():
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
        return self.beta.entropy(out).sum(dim = -1) if self.continuous else Categorical(logits = out).entropy()

    def action_mean(self, out):
        return self.beta.mean(out) if self.continuous else out.argmax(dim = -1)

    def straight_through_actions(self, out, actions):
        if self.continuous:
            means = self.beta.mean(out)
            return actions + means - means.detach()

        probs = out.softmax(dim = -1)
        one_hot = F.one_hot(actions.long(), probs.shape[-1]).float()
        return one_hot + probs - probs.detach()

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
        past_states = None,
        temperature = 0.5,
        bind_receiver = False
    ):
        assert 0 <= level_index < self.num_levels - 1

        if state.ndim == 1:
            state = rearrange(state, '... -> 1 ...')

        num_codes = self.num_internal_actions
        receiver = self.levels[level_index]
        sender = self.levels[level_index + 1]

        with torch.no_grad():
            hidden = receiver.encode_stem(state)
            if receiver.has_state_history and exists(past_states):
                hidden = hidden + 0.02 * receiver.state_chunk_mlp(past_states)

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

    def counterfactual_base_action_chunks(
        self,
        state_chunks: Tensor,
        level_index: int
    ) -> tuple[Tensor, Tensor]:
        assert 1 <= level_index < self.num_levels
        batch_chunks, chunk_len, _ = state_chunks.shape
        num_codes = self.num_internal_actions
        dev = state_chunks.device

        sender = self.levels[level_index]
        target_codes = repeat(torch.arange(num_codes, device = dev), 'k -> (b k)', b = batch_chunks)
        curr_latents = repeat(sender.readout.prototypes, 'k d -> (b k) 1 d', b = batch_chunks)
        rep_states = repeat(state_chunks, 'b s d -> (b k) s d', k = num_codes)

        for curr_level in reversed(range(level_index)):
            level_mod = self.levels[curr_level]
            stride_curr = self.hierarchies[curr_level]
            stride_higher = self.hierarchies[curr_level + 1]
            sub_steps = max(1, stride_higher // stride_curr)

            sub_states = rep_states[:, 0::stride_curr]
            num_sub = sub_states.shape[1]
            latents_expanded = repeat(curr_latents, 'bk c d -> bk (c r) d', r = sub_steps)[:, :num_sub]
            t_idx = torch.arange(num_sub, device = dev) % sub_steps
            t_idx_flat = repeat(t_idx, 's -> (bk s)', bk = sub_states.shape[0])

            flat_sub_states = rearrange(sub_states, 'bk s d -> (bk s) d')
            flat_latents = rearrange(latents_expanded, 'bk s d -> (bk s) d')

            hidden = level_mod.encode_stem(flat_sub_states)
            hidden = level_mod.steered_norm(hidden + level_mod.from_higher(hidden, flat_latents, time_indices = t_idx_flat))
            upper_hidden = level_mod.upper(hidden)
            logits = level_mod.to_logits(upper_hidden)

            if curr_level == 0:
                action_repr = level_mod.beta.mean(logits) if self.continuous else logits.softmax(dim = -1)
                action_chunks = rearrange(action_repr, '(bk s) a -> bk s a', bk = sub_states.shape[0], s = chunk_len)
                return action_chunks, target_codes

            probs = (logits / level_mod.readout.temperature).softmax(dim = -1)
            soft_latent = einsum(probs, level_mod.readout.prototypes, 'n k, k d -> n d')
            soft_latent = soft_latent + level_mod.residual_proj(upper_hidden)
            curr_latents = rearrange(soft_latent, '(bk s) d -> bk s d', bk = sub_states.shape[0], s = num_sub)

        raise RuntimeError('unreachable')

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

# underpowered DIAYN action-chunk discriminator with shrink-and-perturb

class ActionChunkDiscriminator(Module):
    def __init__(
        self,
        chunk_len: int,
        num_actions: int,
        num_codes: int,
        hidden_dim: int = 16
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.num_actions = num_actions
        self.num_codes = num_codes
        self.log_num_codes = float(np.log(num_codes))

        in_dim = (chunk_len + 1) * num_actions
        self.net = nn.Sequential(
            nn.RMSNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, num_codes, bias = False)
        )

    def forward(self, action_chunks: Tensor) -> Tensor:
        flat_chunks = rearrange(action_chunks, 'b s a -> b (s a)')
        mean_actions = action_chunks.mean(dim = 1)
        logits = self.net(cat((flat_chunks, mean_actions), dim = -1))
        return logits - logits.mean(dim = -1, keepdim = True)

    @torch.no_grad()
    def shrink_and_perturb_(self, shrink: float = 0.85, perturb: float = 0.02):
        with torch.random.fork_rng():
            for param in self.parameters():
                noise = torch.randn_like(param)
                if param.ndim >= 2:
                    nn.init.orthogonal_(noise, gain = 1.)
                param.mul_(shrink).add_(noise, alpha = perturb)

    @torch.no_grad()
    def intrinsic_reward(self, action_chunks: Tensor, codes: Tensor) -> Tensor:
        log_probs = F.log_softmax(self(action_chunks), dim = -1)
        chosen_log_q = log_probs.gather(-1, codes[:, None]).squeeze(-1)
        return ((chosen_log_q + self.log_num_codes) / self.log_num_codes).clamp(-1., 1.)

    def update_(
        self,
        optimizer: torch.optim.Optimizer,
        real_chunks: Tensor,
        real_codes: Tensor,
        cf_chunks: Tensor,
        cf_codes: Tensor,
        *,
        steps: int = 4,
        shrink: float = 0.85,
        perturb: float = 0.02
    ):
        self.shrink_and_perturb_(shrink = shrink, perturb = perturb)

        counts = torch.bincount(real_codes, minlength = self.num_codes).float().clamp_min(1.)
        class_weights = (counts.sum() / (self.num_codes * counts)).clamp(0.25, 4.)

        for _ in range(steps):
            loss = (
                0.5 * F.cross_entropy(self(real_chunks), real_codes, weight = class_weights) +
                0.5 * F.cross_entropy(self(cf_chunks), cf_codes)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

# latent-action differentiation evaluation and visualization

@torch.no_grad()
def evaluate_latent_action_differentiation(
    actor: HierarchicalActor,
    env_name: str = 'LunarLander-v3',
    continuous: bool = False,
    num_eval_episodes: int = 16,
    max_timesteps: int = 600,
    seed: int = 200,
    label: str = 'hierarchical'
) -> dict:
    actor.eval()
    env = gym.make(env_name, max_episode_steps = max_timesteps, continuous = continuous)

    num_levels, num_codes, num_actions = actor.num_levels, actor.num_internal_actions, actor.num_actions
    log_k = float(np.log(num_codes))

    level_chunks = {l: [] for l in range(1, num_levels)}
    level_codes = {l: [] for l in range(1, num_levels)}
    step_codes = {l: [] for l in range(1, num_levels)}
    step_actions, episode_returns = [], []
    sample_trajectory = None

    for episode_index in range(num_eval_episodes):
        obs, _ = env.reset(seed = seed + episode_index)
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        held_latents = [zeros(1, actor.dim_latent, device = device) for _ in range(num_levels)]
        held_codes = [0] * num_levels
        past_states_buf = actor.init_past_states(1, device)

        ep_actions, ep_states = [], []
        ep_codes = {l: [] for l in range(1, num_levels)}
        ep_return = 0.

        for timestep in range(max_timesteps):
            timestep_tensor = full((1,), timestep, dtype = torch.long, device = device)

            for level_index in reversed(range(1, num_levels)):
                stride, shift = actor.hierarchies[level_index], actor.phase_shifts[level_index]
                if timestep == 0 or divisible_by(timestep - shift, stride):
                    time_indices = actor.level_time_indices(level_index, timestep_tensor)
                    latent_above = held_latents[level_index + 1] if (level_index + 1) < num_levels else None
                    latent, sampled_code, _, _ = actor.forward_level(
                        level_index, state, latent_from_higher = latent_above,
                        time_indices = time_indices, past_states = past_states_buf[level_index]
                    )
                    held_latents[level_index] = latent
                    held_codes[level_index] = int(sampled_code.item())

            action_out = actor.forward_level(
                0, state, latent_from_higher = held_latents[1] if num_levels > 1 else None,
                time_indices = actor.level_time_indices(0, timestep_tensor)
            )
            actor.update_past_states(past_states_buf, state, timestep)

            action = actor.levels[0].action_mean(action_out)
            act_np = action.squeeze(0).cpu().numpy() if continuous else int(action.item())
            act_vec = act_np.astype(np.float32) if continuous else np.eye(num_actions, dtype = np.float32)[act_np]

            ep_states.append(obs.copy())
            ep_actions.append(act_vec)
            step_actions.append(act_np)

            for level_index in range(1, num_levels):
                ep_codes[level_index].append(held_codes[level_index])
                step_codes[level_index].append(held_codes[level_index])

            obs, reward, terminated, truncated, _ = env.step(act_np)
            ep_return += float(reward)

            if terminated or truncated:
                break

            state = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        episode_returns.append(ep_return)

        if not exists(sample_trajectory) or ep_return > sample_trajectory['return']:
            sample_trajectory = dict(
                states = np.asarray(ep_states),
                actions = np.asarray(ep_actions),
                codes = {l: np.asarray(ep_codes[l]) for l in range(1, num_levels)},
                return_ = ep_return
            )
            sample_trajectory['return'] = ep_return

        ep_actions_arr = np.asarray(ep_actions, dtype = np.float32)
        for level_index in range(1, num_levels):
            stride = actor.hierarchies[level_index]
            for start_t in range(0, (len(ep_actions_arr) // stride) * stride, stride):
                level_chunks[level_index].append(ep_actions_arr[start_t:start_t + stride])
                level_codes[level_index].append(ep_codes[level_index][start_t])

    env.close()

    results = dict(label = label, mean_eval_return = float(np.mean(episode_returns)), levels = {}, sample_trajectory = sample_trajectory)
    step_actions_arr = np.asarray(step_actions)

    print(f'\n=== Latent-Action Differentiation Report: {label} (Eval Return: {results["mean_eval_return"]:+.1f}) ===')

    for level_index in range(1, num_levels):
        stride = actor.hierarchies[level_index]
        chunks_np = np.asarray(level_chunks[level_index], dtype = np.float32)
        codes_np = np.asarray(level_codes[level_index], dtype = np.int64)
        step_codes_np = np.asarray(step_codes[level_index], dtype = np.int64)

        code_counts = np.bincount(codes_np, minlength = num_codes).astype(np.float64)
        code_usage = code_counts / max(1., code_counts.sum())
        usage_entropy = -np.sum(code_usage * np.log(np.clip(code_usage, 1e-12, 1.))) / log_k

        cond_action_dist = np.full((num_codes, num_actions), 1. / num_actions, dtype = np.float64)
        for k in range(num_codes):
            mask = (step_codes_np == k)
            if not mask.any():
                continue
            if continuous:
                cond_action_dist[k] = chunks_np[codes_np == k].mean(axis = (0, 1)) if (codes_np == k).any() else 0.
            else:
                counts_k = np.bincount(step_actions_arr[mask], minlength = num_actions).astype(np.float64)
                cond_action_dist[k] = counts_k / max(1., counts_k.sum())

        norm_mi_logk, norm_mi_ha = 0., 0.
        if not continuous:
            joint = np.zeros((num_codes, num_actions), dtype = np.float64)
            np.add.at(joint, (step_codes_np, step_actions_arr), 1.)
            joint /= max(1., joint.sum())
            p_z, p_a = joint.sum(axis = 1, keepdims = True), joint.sum(axis = 0, keepdims = True)
            mi_za = float(np.sum(joint * np.log(np.clip(joint, 1e-12, 1.) / np.clip(p_z @ p_a, 1e-12, 1.))))
            h_a = float(-np.sum(p_a * np.log(np.clip(p_a, 1e-12, 1.))))
            norm_mi_logk, norm_mi_ha = mi_za / log_k, mi_za / max(1e-8, h_a)

        active_codes = [k for k in range(num_codes) if code_usage[k] > 0.02]
        tv_dists = [
            0.5 * float(np.abs(cond_action_dist[i] - cond_action_dist[j]).sum())
            if (code_usage[i] > 0.01 and code_usage[j] > 0.01) else 0.
            for i in range(num_codes) for j in range(i + 1, num_codes)
        ]
        mean_tv = float(np.mean(tv_dists)) if tv_dists else 0.

        with torch.enable_grad(), torch.random.fork_rng():
            torch.manual_seed(0)
            chunks_t = torch.from_numpy(chunks_np).float().to(device)
            codes_t = torch.from_numpy(codes_np).long().to(device)
            perm = torch.randperm(len(codes_t), device = device)
            preds_all = torch.zeros_like(codes_t)

            for fold in range(2):
                val_idx, train_idx = perm[fold::2], perm[(1 - fold)::2]
                probe = ActionChunkDiscriminator(stride, num_actions, num_codes, hidden_dim = 16).to(device)
                probe_opt = torch.optim.AdamW(probe.parameters(), lr = 1e-2, weight_decay = 1e-3)
                counts_tr = torch.bincount(codes_t[train_idx], minlength = num_codes).float().clamp_min(1.)
                weights = (counts_tr.sum() / (num_codes * counts_tr)).clamp(0.2, 5.)

                for _ in range(80):
                    loss_p = F.cross_entropy(probe(chunks_t[train_idx]), codes_t[train_idx], weight = weights)
                    probe_opt.zero_grad()
                    loss_p.backward()
                    probe_opt.step()

                with torch.no_grad():
                    preds_all[val_idx] = probe(chunks_t[val_idx]).argmax(dim = -1)

            preds_np = preds_all.cpu().numpy()

        raw_acc = float(np.mean(preds_np == codes_np))
        balanced_acc = float(np.mean([
            float(np.mean(preds_np[codes_np == k] == k)) if (codes_np == k).sum() >= 3 else 0.
            for k in range(num_codes)
        ]))

        conf_mat = np.zeros((num_codes, num_codes), dtype = np.float64)
        np.add.at(conf_mat, (codes_np, preds_np), 1.)
        conf_mat /= max(1., conf_mat.sum())
        pz_t, pz_p = conf_mat.sum(axis = 1, keepdims = True), conf_mat.sum(axis = 0, keepdims = True)
        chunk_mi = float(np.sum(conf_mat * np.log(np.clip(conf_mat, 1e-12, 1.) / np.clip(pz_t @ pz_p, 1e-12, 1.)))) / log_k

        results['levels'][level_index] = dict(
            stride = stride, code_usage = code_usage, usage_entropy = float(usage_entropy),
            cond_action_dist = cond_action_dist, norm_mi_logk = norm_mi_logk, norm_mi_ha = norm_mi_ha,
            mean_tv = mean_tv, chunk_raw_acc = raw_acc, chunk_balanced_acc = balanced_acc,
            chunk_mi = chunk_mi, num_active_codes = len(active_codes)
        )

        print(
            f'  Level {level_index} (dt={stride}): '
            f'active_codes={len(active_codes)}/{num_codes} | usage_H={100. * usage_entropy:5.1f}% | '
            f'chunk_balanced_acc={100. * balanced_acc:5.1f}% (chance=25.0%) | '
            f'chunk_MI={100. * chunk_mi:5.1f}% | step_MI/H(A)={100. * norm_mi_ha:5.1f}% | '
            f'pairwise_TV={100. * mean_tv:5.1f}%'
        )
        action_names = ('No-op', 'Left', 'Main', 'Right') if (not continuous and num_actions == 4) else tuple(f'a{i}' for i in range(num_actions))
        for k in range(num_codes):
            probs_str = ', '.join(f'{action_names[a]}={100. * cond_action_dist[k, a]:4.1f}%' for a in range(num_actions))
            print(f'    z={k} (used {100. * code_usage[k]:4.1f}%): [{probs_str}]')

    return results

def plot_diayn_differentiation_comparison(
    base_eval: dict,
    diayn_eval: dict,
    out_path: str | Path = './lunar-recording/diayn_action_differentiation.png'
):
    sns.set_theme(style = 'whitegrid', context = 'notebook')
    fig, axes = plt.subplots(2, 3, figsize = (16., 9.5), dpi = 160)

    action_names = ['No-op', 'Left', 'Main', 'Right']
    levels = sorted(base_eval['levels'].keys())[:2]

    for row_idx, level_index in enumerate(levels):
        stride = base_eval['levels'][level_index]['stride']
        for col_idx, (eval_dict, title_tag) in enumerate(((base_eval, 'Baseline Hierarchy (no DIAYN)'), (diayn_eval, 'Hierarchy + DIAYN Discriminator'))):
            ax = axes[row_idx, col_idx]
            lvl = eval_dict['levels'][level_index]
            mat, usage = lvl['cond_action_dist'] * 100., lvl['code_usage'] * 100.
            y_labels = [f'z={k} ({usage[k]:.0f}%)' for k in range(mat.shape[0])]

            sns.heatmap(
                mat, annot = True, fmt = '.1f', cmap = 'YlGnBu', vmin = 0., vmax = 80.,
                xticklabels = action_names, yticklabels = y_labels, cbar = False, ax = ax
            )
            ax.set_title(
                f'{title_tag} — L{level_index} (dt={stride})\n'
                f'Bal Acc: {100. * lvl["chunk_balanced_acc"]:.1f}% | TV: {100. * lvl["mean_tv"]:.1f}% | Usage H: {100. * lvl["usage_entropy"]:.1f}%',
                fontsize = 10.5, fontweight = 'bold'
            )
            ax.set_xlabel('Base Primitive Action P(a | z) [%]')
            ax.set_ylabel(f'Level {level_index} Latent Code z')

    ax_bar = axes[0, 2]
    metrics = [
        (f'L{l} {tag}', base_eval['levels'][l][key], diayn_eval['levels'][l][key])
        for l in levels
        for tag, key in (('Chunk\nBal Acc', 'chunk_balanced_acc'), ('Action\nTV Diff', 'mean_tv'), ('Code\nUsage H', 'usage_entropy'))
    ]
    x_pos, width = np.arange(len(metrics)), 0.36
    ax_bar.bar(x_pos - width / 2, [100. * m[1] for m in metrics], width, label = 'Baseline (no DIAYN)', color = '#4c72b0')
    ax_bar.bar(x_pos + width / 2, [100. * m[2] for m in metrics], width, label = '+ DIAYN (Shrink & Perturb)', color = '#dd8452')
    ax_bar.axhline(25., color = 'gray', linestyle = '--', linewidth = 1., label = '4-Code Chance (25%)')
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([m[0] for m in metrics], fontsize = 9)
    ax_bar.set_ylim(0., 105.)
    ax_bar.set_ylabel('Percentage (%)')
    ax_bar.set_title('Latent-to-Base-Action Explainability & Diversity', fontsize = 10.5, fontweight = 'bold')
    ax_bar.legend(loc = 'upper right', fontsize = 8.5)

    ax_traj = axes[1, 2]
    traj = diayn_eval['sample_trajectory']
    if exists(traj) and len(traj['states']) > 0:
        t_steps = np.arange(len(traj['states']))
        prim_actions = traj['actions'].argmax(axis = -1)
        codes_l1 = traj['codes'][1]
        palette = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728')
        for k in range(4):
            mask = (codes_l1 == k)
            if mask.any():
                ax_traj.scatter(t_steps[mask], prim_actions[mask] + (k - 1.5) * 0.05, s = 18, color = palette[k], label = f'z1={k}', alpha = 0.85)
        ax_traj.set_yticks([0, 1, 2, 3])
        ax_traj.set_yticklabels(action_names)
        ax_traj.set_xlabel('Episode Timestep t')
        ax_traj.set_ylabel('Executed Base Action a_t')
        ax_traj.set_title(f'DIAYN Eval Rollout: Action vs Active Latent z1 (Return {traj["return"]:+.1f})', fontsize = 10.5, fontweight = 'bold')
        ax_traj.legend(loc = 'upper right', ncol = 2, fontsize = 8.5)

    plt.tight_layout()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(out_path, bbox_inches = 'tight')
    plt.close(fig)
    print(f'[plot] saved differentiation comparison -> {out_path}')
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
    always_train_base_level = False,
    reset_latents_on_done = True,
    use_diayn = False,
    diayn_disc_hidden_dim = 16,
    diayn_disc_lr = 3.0e-3,
    diayn_disc_steps = 4,
    diayn_reward_weight = 0.08,
    diayn_loss_weight = 0.12,
    diayn_shrink = 0.85,
    diayn_perturb = 0.02,
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
    compare_diayn = False,
    return_actor = False,
    comparison_plot_path = './lunar-recording/lunar_returns_comparison.png',
    diayn_plot_path = './lunar-recording/diayn_action_differentiation.png'
):
    hierarchies = resolve_hierarchies(hierarchies)
    num_levels = len(hierarchies)
    total_phases = default(num_phases, num_cycles * num_levels)
    cycle_length = default(cycle_length, num_levels)

    if compare_diayn:
        out_dir = Path(video_folder)
        base_csv = out_dir / 'hier_baseline_returns.csv'
        diayn_csv = out_dir / 'hier_diayn_returns.csv'
        base_ckpt = out_dir / 'ppo_hier_baseline.pt'
        diayn_ckpt = out_dir / 'ppo_hier_diayn.pt'

        common = dict(
            env_name = env_name, continuous = continuous, beta_init_conc = beta_init_conc,
            hierarchies = hierarchies, phase_shifts = phase_shifts,
            num_phases = total_phases, cycle_length = cycle_length, updates_per_phase = updates_per_phase,
            simultaneous = simultaneous, use_state_history = use_state_history,
            always_train_base_level = always_train_base_level, reset_latents_on_done = reset_latents_on_done,
            num_envs = num_envs, rollout_steps = rollout_steps, max_episode_steps = max_episode_steps,
            actor_hidden_dim = actor_hidden_dim, critic_hidden_dim = critic_hidden_dim,
            num_internal_actions = num_internal_actions, dim_latent = dim_latent, low_rank = low_rank,
            epochs = epochs, minibatch_envs = minibatch_envs, lr = lr, critic_lr = critic_lr,
            gamma = gamma, lam = lam, eps_clip = eps_clip,
            internal_policy_loss_weight = internal_policy_loss_weight,
            binding_loss_weight = binding_loss_weight, coupling_loss_weight = coupling_loss_weight,
            diayn_disc_hidden_dim = diayn_disc_hidden_dim, diayn_disc_lr = diayn_disc_lr,
            diayn_disc_steps = diayn_disc_steps, diayn_reward_weight = diayn_reward_weight,
            diayn_loss_weight = diayn_loss_weight, diayn_shrink = diayn_shrink, diayn_perturb = diayn_perturb,
            max_grad_norm = max_grad_norm, seed = seed, video_folder = video_folder, return_actor = True
        )

        print(f'=== [1/2] Baseline Hierarchical {hierarchies} (use_diayn=False) ===', flush = True)
        base_actor = main(
            **common, use_diayn = False, render = False,
            save_path = str(base_ckpt), log_csv = str(base_csv)
        )
        base_eval = evaluate_latent_action_differentiation(
            base_actor, env_name = env_name, continuous = continuous,
            max_timesteps = max_episode_steps, label = 'Baseline Hierarchy (no DIAYN)'
        )

        print(f'\n=== [2/2] DIAYN Hierarchical {hierarchies} (use_diayn=True, shrink={diayn_shrink}, perturb={diayn_perturb}) ===', flush = True)
        diayn_actor = main(
            **common, use_diayn = True, render = render, video_prefix = 'hierarchical-diayn-best',
            save_path = str(diayn_ckpt), log_csv = str(diayn_csv)
        )
        diayn_eval = evaluate_latent_action_differentiation(
            diayn_actor, env_name = env_name, continuous = continuous,
            max_timesteps = max_episode_steps, label = 'Hierarchy + DIAYN Discriminator'
        )

        plot_return_comparison(
            [base_csv, diayn_csv],
            [f'hierarchical {hierarchies} (no DIAYN)', f'hierarchical {hierarchies} + DIAYN'],
            out_path = out_dir / 'diayn_returns_comparison.png',
            title = f'{env_name}: Baseline Hierarchy vs DIAYN Action-Chunk Discriminator'
        )
        plot_diayn_differentiation_comparison(
            base_eval, diayn_eval, out_path = diayn_plot_path
        )
        return

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
                reset_latents_on_done = reset_latents_on_done, use_diayn = use_diayn,
                actor_hidden_dim = actor_hidden_dim,
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

    discriminators = {}
    disc_optimizers = {}
    if use_diayn and num_levels > 1:
        with torch.random.fork_rng():
            for level_index in range(1, num_levels):
                disc = ActionChunkDiscriminator(
                    chunk_len = hierarchies[level_index],
                    num_actions = num_actions,
                    num_codes = num_internal_actions,
                    hidden_dim = diayn_disc_hidden_dim
                ).to(device)
                discriminators[level_index] = disc
                disc_optimizers[level_index] = torch.optim.AdamW(disc.parameters(), lr = diayn_disc_lr, weight_decay = 1e-3)

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

        for param_group in opt_all.param_groups:
            param_group['lr'] = lr * lr_scale

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
                        if timestep == 0 or divisible_by(timestep - shift, stride):
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
                actor_gae_per_level = gae_per_level

            if use_diayn and isinstance(phase, int) and 1 <= phase < num_levels:
                active_level = phase
                active_stride = hierarchies[active_level]
                disc = discriminators[active_level]

                with torch.no_grad():
                    action_vecs = actions if continuous else F.one_hot(actions.long(), num_actions).float()
                    rollout_action_chunks = rearrange(action_vecs, 'b (c s) a -> (b c) s a', s = active_stride)
                    rollout_code_chunks = rearrange(internal_actions[active_level][:, 0::active_stride], 'b c -> (b c)')
                    state_chunks_all = rearrange(states, 'b (c s) d -> (b c) s d', s = active_stride)
                    cf_sub_chunks, cf_sub_codes = actor.counterfactual_base_action_chunks(state_chunks_all[:64], active_level)

                disc.update_(
                    disc_optimizers[active_level],
                    rollout_action_chunks, rollout_code_chunks,
                    cf_sub_chunks, cf_sub_codes,
                    steps = diayn_disc_steps, shrink = diayn_shrink, perturb = diayn_perturb
                )

                with torch.no_grad():
                    chunk_diayn_reward = rearrange(disc.intrinsic_reward(rollout_action_chunks, rollout_code_chunks), '(b c) -> b c', b = num_envs)
                    step_diayn_reward = repeat(chunk_diayn_reward / active_stride, 'b c -> b (c s)', s = active_stride)
                    actor_gae_per_level = calc_gae(
                        rewards + diayn_reward_weight * step_diayn_reward,
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
                    batch_base_advantages = actor_gae_per_level[0].advantages[batch_indices]

                    flat_states = rearrange(batch_states, 'b t d -> (b t) d')
                    flat_actions = rearrange(batch_actions, 'b t ... -> (b t) ...')
                    flat_old_log_probs = rearrange(batch_old_log_probs, 'b t ... -> (b t) ...')
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

                            if use_diayn:
                                state_chunks_l1 = rearrange(batch_states, 'b (c s) d -> (b c) s d', s = hierarchies[1])[:24]
                                with frozen((discriminators[1],)):
                                    cf_chunks_l1, cf_codes_l1 = actor.counterfactual_base_action_chunks(state_chunks_l1, 1)
                                    diayn_preserve = F.cross_entropy(discriminators[1](cf_chunks_l1), cf_codes_l1)
                                loss = loss + 0.25 * diayn_loss_weight * diayn_preserve

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
                        chunk_advantages = rearrange(actor_gae_per_level[level_index].advantages[batch_indices], 'b c -> (b c)')

                        internal_dist = Categorical(probs = chunk_probs)
                        level_policy_loss = hpo_actor_loss(internal_dist.log_prob(chunk_internal_actions), chunk_old_log_probs, chunk_advantages, eps_clip = eps_clip, normalize_advantages = True).mean()

                        sub_count = min(128, chunk_states.shape[0])
                        in_flight = (chunk_states[:sub_count, 6] + chunk_states[:sub_count, 7]) < 1.5 if state_dim >= 8 else torch.ones(sub_count, dtype = torch.bool, device = device)
                        active_slice = in_flight if in_flight.any() else slice(None)
                        binding_states = chunk_states[:sub_count][active_slice]

                        lower_is_continuous = continuous and lower_index == 0
                        lower_past_states = (
                            rearrange(past_states_stacked[lower_index][batch_indices][:, 0::stride], 'b c k d -> (b c) k d')
                            if (actor.use_state_history and lower_index >= 1) else None
                        )
                        binding_past_states = lower_past_states[:sub_count][active_slice] if exists(lower_past_states) else None

                        binding_means, binding_stds, binding_probs = actor.counterfactual_probe(
                            binding_states, level_index = lower_index,
                            time_indices = zeros(binding_states.shape[0], dtype = torch.long, device = device),
                            past_states = binding_past_states,
                            bind_receiver = False
                        )
                        binding_loss = continuous_binding_loss(binding_means, binding_stds) if lower_is_continuous else counterfactual_decoder_binding_loss(binding_probs)

                        with torch.no_grad():
                            cf_means, cf_stds, cf_probs = actor.counterfactual_probe(
                                chunk_states, level_index = lower_index,
                                time_indices = zeros(chunk_states.shape[0], dtype = torch.long, device = device),
                                past_states = lower_past_states,
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

                        diayn_aux_loss = zeros((), device = device)
                        if use_diayn:
                            disc = discriminators[level_index]
                            st_base_actions = base_level.straight_through_actions(level_out, flat_actions)
                            st_action_chunks = rearrange(st_base_actions, '(b c s) a -> (b c) s a', b = batch_size, s = stride)
                            state_chunks_mb = rearrange(batch_states, 'b (c s) d -> (b c) s d', s = stride)[:24]

                            with frozen((disc,)):
                                cf_action_chunks, cf_target_codes = actor.counterfactual_base_action_chunks(state_chunks_mb, level_index)
                                diayn_real_ce = F.cross_entropy(disc(st_action_chunks), chunk_internal_actions)
                                diayn_cf_ce = F.cross_entropy(disc(cf_action_chunks), cf_target_codes)

                            marginal_entropy_loss = disc.log_num_codes - categorical_entropy(chunk_probs.mean(dim = 0))
                            diayn_aux_loss = 0.5 * (diayn_real_ce + diayn_cf_ce + marginal_entropy_loss)

                        loss = (
                            base_policy_loss +
                            internal_policy_loss_weight * level_policy_loss -
                            entropy_coef * (base_entropy + internal_dist.entropy().mean()) +
                            binding_loss_weight * aux_scale * binding_loss +
                            coupling_loss_weight * aux_scale * coupling_loss +
                            diayn_loss_weight * diayn_aux_loss
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

    if return_actor:
        return actor

if __name__ == '__main__':
    fire.Fire(main)
