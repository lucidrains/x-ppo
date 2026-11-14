# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "einops>=0.8.0",
#   "einx>=0.4.0",
#   "ema-pytorch>=0.5.2",
#   "fire",
#   "gymnasium[box2d]>=1.0.0",
#   "imageio[ffmpeg]",
#   "mean-conc-beta>=0.2.7",
#   "numpy",
#   "pillow",
#   "torch>=2.5",
#   "torch-einops-utils>=0.2.5",
#   "x-transformers>=2.3.21",
# ]
# ///

# fql   - Seohong Park et al.      https://arxiv.org/abs/2502.02538
# aac   - Yongjae Shin et al.      https://arxiv.org/abs/2605.10044
# xm    - Alexi Gladstone et al.   https://arxiv.org/abs/2607.27372
# dsrl  - Andrew Wagenmaker et al. https://arxiv.org/abs/2506.15799
# t-sac - Dong Tian et al.         https://arxiv.org/abs/2503.03660
# floq  -                          https://arxiv.org/abs/2509.06863

from __future__ import annotations

import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import fire
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn, Tensor, arange, cat, is_tensor, stack
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import LayerNorm, Linear, Module, RMSNorm, Sequential, SiLU
from torch.nn.utils import clip_grad_norm_

import einx
from einops import einsum, rearrange, repeat

from ema_pytorch import EMA
from mean_conc_beta import Beta as MeanConcBeta
from torch_einops_utils import lens_to_mask, mask_after, masked_mean, pack_with_inverse
from x_transformers import Decoder

from x_ppo import calc_gae, clipped_value_loss_fn, hpo_actor_loss, ppo_actor_loss

# constants

torch.set_num_threads(2)

ppo_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

flow_device = torch.device(
    'cuda:0' if torch.cuda.is_available()
    else ('mps' if torch.backends.mps.is_available() else 'cpu')
)

BETA_ENTROPY_SCHEDULE = (0.0016, 0.0008, 0.0004, 0.0002, 0.0001, 0.00004)
LR_SCALE_SCHEDULE = (1., 0.90, 0.75, 0.55, 0.40, 0.25)

DT_BAND_RGB = ((72, 149, 239), (42, 157, 143), (244, 162, 97), (230, 57, 70))
OVERRIDE_GOLD = (255, 209, 102)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def append_dims(t, dims):
    return t.reshape(*t.shape, *((1,) * dims))

def module_device(module: Module):
    return next(module.parameters()).device

def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def schedule_at(schedule, progress):
    idx = min(int(progress * len(schedule)), len(schedule) - 1)
    return schedule[max(0, idx)]

def dim_rgb(rgb, scale = 0.35):
    return tuple(int(scale * v) for v in rgb)

def dt_color_rgb(dt, max_chunk_size = 16):
    band = min(3, max(0, (int(dt) - 1) * 4 // max(1, max_chunk_size)))
    return DT_BAND_RGB[band]

def update_network_(loss: Tensor, optimizer, max_grad_norm = None):
    optimizer.zero_grad()
    loss.backward()

    if exists(max_grad_norm):
        for group in optimizer.param_groups:
            clip_grad_norm_(group['params'], max_grad_norm)

    optimizer.step()

def make_orthonormal_dct(n: int):
    k = rearrange(arange(n, dtype = torch.float32), 'k -> k 1')
    t = rearrange(arange(n, dtype = torch.float32), 't -> 1 t')

    mat = torch.cos(math.pi / n * (t + 0.5) * k)
    mat[0] *= math.sqrt(1. / n)

    if n > 1:
        mat[1:] *= math.sqrt(2. / n)

    return mat

def sample_pyoco_noise(batch: int, seq_len: int, dim: int, corr_alpha: float, device):
    if corr_alpha <= 0.:
        return torch.randn((batch, seq_len, dim), device = device)

    shared = torch.randn((batch, 1, dim), device = device)
    indep = torch.randn((batch, seq_len, dim), device = device)
    return corr_alpha * shared + math.sqrt(max(0., 1. - corr_alpha ** 2)) * indep

def should_abort_chunk(est_total_q, planned_q, threshold = 0.25):
    return est_total_q < (planned_q - threshold)

# rectified flow + explorative modeling (Gladstone et al.)

class NanoFlow(Module):
    def __init__(
        self,
        model: Module,
        times_cond_kwarg = 'times',
        data_shape = None,
        max_timesteps = 100
    ):
        super().__init__()
        self.model = model
        self.times_cond_kwarg = times_cond_kwarg
        self.data_shape = data_shape
        self.max_timesteps = max_timesteps

    @torch.no_grad()
    def sample(self, *, steps = 4, batch_size = 1, data_shape = None, noise = None, **kwargs):
        data_shape = default(data_shape, self.data_shape)
        dev = module_device(self.model)

        state = noise.to(dev) if exists(noise) else torch.randn((batch_size, *data_shape), device = dev)
        times = torch.linspace(0., 1., steps + 1, device = dev)[:-1]
        delta = 1. / steps

        for t in times:
            time_kw = {self.times_cond_kwarg: t.expand(state.shape[0])} if exists(self.times_cond_kwarg) else {}
            state = state + delta * self.model(state, **time_kw, **kwargs)

        return state

    def forward(self, data, *, noise = None, times = None, loss_reduction = 'mean', **kwargs):
        b, dev = data.shape[0], data.device
        self.data_shape = default(self.data_shape, data.shape[1:])

        times = default(times, torch.rand(b, device = dev)) * (1. - self.max_timesteps ** -1)
        noise = default(noise, torch.randn_like(data))

        noised = noise.lerp(data, append_dims(times, data.ndim - 1))
        time_kw = {self.times_cond_kwarg: times} if exists(self.times_cond_kwarg) else {}
        pred_flow = self.model(noised, **time_kw, **kwargs)

        return F.mse_loss(data - noise, pred_flow, reduction = loss_reduction)

def xm_flow_loss(
    flow_model: NanoFlow,
    data: Tensor,
    *,
    mask: Tensor,
    candidates = 2,
    noise_fn = None,
    **cond_kwargs
):
    b, dev = data.shape[0], data.device
    times = torch.rand(b, device = dev)

    if candidates <= 1:
        noise = noise_fn(b) if exists(noise_fn) else None
        raw = flow_model(data, noise = noise, times = times, loss_reduction = 'none', **cond_kwargs)
        per_tok = raw.mean(dim = -1) if raw.ndim > mask.ndim else raw
        return masked_mean(per_tok, mask)

    def repeat_candidates(t):
        if is_tensor(t) and t.ndim > 0 and t.shape[0] == b:
            return repeat(t, 'b ... -> (b k) ...', k = candidates)
        return t

    data_k, times_k, mask_k = map(repeat_candidates, (data, times, mask))
    cond_k = {k: repeat_candidates(v) for k, v in cond_kwargs.items()}
    noise_k = noise_fn(b * candidates) if exists(noise_fn) else None

    raw = flow_model(data_k, noise = noise_k, times = times_k, loss_reduction = 'none', **cond_k)
    per_tok = raw.mean(dim = -1) if raw.ndim > mask_k.ndim else raw
    cand_losses = rearrange(masked_mean(per_tok, mask_k, dim = -1), '(b k) -> b k', b = b, k = candidates)

    return cand_losses.amin(dim = -1).mean()

# causal transformer flow actor & multi-chunk value critic

class TransformerFlowActor(Module):
    def __init__(
        self,
        *,
        dim_state: int,
        num_cont_actions: int,
        dim_hidden = 128,
        depth = 3,
        heads = 4,
        max_chunk_size = 16
    ):
        super().__init__()
        self.num_cont_actions = num_cont_actions
        self.max_chunk_size = max_chunk_size

        self.state_proj = Linear(dim_state, dim_hidden)
        self.noised_action_proj = Linear(num_cont_actions, dim_hidden)
        self.time_mlp = Sequential(Linear(1, dim_hidden), SiLU(), Linear(dim_hidden, dim_hidden))
        self.chunk_pos_emb = nn.Embedding(max_chunk_size, dim_hidden)

        self.decoder = Decoder(dim = dim_hidden, depth = depth, heads = heads, attn_dim_head = max(16, dim_hidden // heads))
        self.to_pred_flow = Linear(dim_hidden, num_cont_actions)

    def forward(self, noised_actions, *, times, cond_state):
        noised_actions, inverse_seq = pack_with_inverse(noised_actions, 'b * d')
        cond_state, _ = pack_with_inverse(cond_state, 'b * d')
        seq_len = noised_actions.shape[1]

        state_tokens = self.state_proj(cond_state)
        time_emb = self.time_mlp(rearrange(times, 'b -> b 1 1'))
        pos_emb = self.chunk_pos_emb(arange(seq_len, device = noised_actions.device))

        action_tokens = self.noised_action_proj(noised_actions) + time_emb + pos_emb
        out = self.decoder(cat((state_tokens, action_tokens), dim = 1))

        return inverse_seq(self.to_pred_flow(out[:, -seq_len:]))

class TransformerFlowCritic(Module):
    def __init__(
        self,
        *,
        dim_state: int,
        num_cont_actions: int,
        dim_hidden = 128,
        depth = 3,
        heads = 4,
        max_seq_len = 16
    ):
        super().__init__()
        self.state_proj = Linear(dim_state, dim_hidden)
        self.cont_action_proj = Linear(num_cont_actions, dim_hidden)
        self.value_proj = Linear(1, dim_hidden)
        self.time_proj = Sequential(Linear(1, dim_hidden), SiLU(), Linear(dim_hidden, dim_hidden))

        self.distill_token = nn.Parameter(torch.randn(dim_hidden) * 1e-2)
        self.pos_emb = nn.Embedding(max_seq_len, dim_hidden)

        self.decoder = Decoder(dim = dim_hidden, depth = depth, heads = heads, attn_dim_head = max(16, dim_hidden // heads))
        self.to_values = Sequential(LayerNorm(dim_hidden), Linear(dim_hidden, 1, bias = False))

    def forward(self, values = None, *, times = None, cond_state, cont_actions, is_distill = False):
        cont_actions, inverse_seq = pack_with_inverse(cont_actions, 'b * d')
        seq_len = cont_actions.shape[1]

        state_tokens = rearrange(self.state_proj(cond_state), 'b d -> b 1 d')
        action_tokens = self.cont_action_proj(cont_actions)

        if is_distill:
            action_tokens = einx.add('b n d, d', action_tokens, self.distill_token)
        else:
            values = rearrange(values, 'b s -> b s 1') if values.ndim == 2 else values
            time_emb = self.time_proj(rearrange(times, 'b -> b 1 1'))
            action_tokens = action_tokens + self.value_proj(values) + time_emb

        action_tokens = action_tokens + self.pos_emb(arange(seq_len, device = cont_actions.device))
        out = self.decoder(cat((state_tokens, action_tokens), dim = 1))

        return rearrange(inverse_seq(self.to_values(out[:, 1:])), '... 1 -> ...')

# smdp chunk replay buffer & flow trainer

class ChunkReplayBuffer:
    def __init__(self, capacity: int, max_chunk_size: int, state_dim: int, num_actions: int):
        self.capacity = capacity
        self.max_chunk_size = max_chunk_size

        self.states = np.zeros((capacity, state_dim), dtype = np.float32)
        self.actions = np.zeros((capacity, max_chunk_size, num_actions), dtype = np.float32)
        self.rewards = np.zeros((capacity, max_chunk_size), dtype = np.float32)
        self.durations = np.ones((capacity, max_chunk_size), dtype = np.float32)
        self.next_states = np.zeros((capacity, max_chunk_size, state_dim), dtype = np.float32)
        self.terminated = np.zeros((capacity, max_chunk_size), dtype = np.bool_)
        self.valid_lens = np.zeros((capacity,), dtype = np.int64)

        self.ptr = 0
        self.size = 0

    def add_rollout(
        self,
        states: Tensor,
        next_states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        terminated: Tensor,
        truncated: Tensor,
        *,
        durations: Tensor | None = None,
        stride = 2
    ):
        states_np, next_states_np = states.cpu().numpy(), next_states.cpu().numpy()
        actions_np, rewards_np = actions.cpu().numpy(), rewards.cpu().numpy()
        durations_np = durations.cpu().numpy() if exists(durations) else np.ones_like(rewards_np)
        term_np = terminated.cpu().numpy()
        done_np = (terminated | truncated).cpu().numpy()

        num_envs, rollout_len, _ = states_np.shape
        h = self.max_chunk_size

        for t in range(0, rollout_len - h + 1, stride):
            done_win = done_np[:, t:t + h]
            lens = np.where(done_win.any(axis = 1), done_win.argmax(axis = 1) + 1, h)
            idx = (np.arange(num_envs) + self.ptr) % self.capacity

            self.states[idx] = states_np[:, t]
            self.actions[idx] = actions_np[:, t:t + h]
            self.rewards[idx] = rewards_np[:, t:t + h]
            self.durations[idx] = durations_np[:, t:t + h]
            self.next_states[idx] = next_states_np[:, t:t + h]
            self.terminated[idx] = term_np[:, t:t + h]
            self.valid_lens[idx] = lens

            self.ptr = (self.ptr + num_envs) % self.capacity
            self.size = min(self.capacity, self.size + num_envs)

    def sample(self, batch_size: int, device, recent_fraction = 0.60):
        window = max(batch_size, int(self.size * recent_fraction))

        if self.size < self.capacity:
            idx = np.random.randint(max(0, self.size - window), self.size, size = batch_size)
        else:
            idx = (self.ptr - np.random.randint(1, window + 1, size = batch_size)) % self.capacity

        arrays = (self.states, self.actions, self.rewards, self.durations, self.next_states, self.terminated, self.valid_lens)
        return tuple(torch.from_numpy(arr[idx]).to(device) for arr in arrays)

class FlowChunkTrainer(Module):
    def __init__(
        self,
        *,
        state_dim: int,
        num_actions: int,
        max_chunk_size = 16,
        hidden_dim = 128,
        depth = 3,
        heads = 4,
        lr = 3e-4,
        ema_decay = 0.995,
        gamma = 0.99,
        corr_alpha = 0.7,
        xm_candidates = 2,
        critic_distill_steps = 4,
        max_grad_norm = 1.
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.max_chunk_size = max_chunk_size
        self.gamma = gamma
        self.corr_alpha = corr_alpha
        self.xm_candidates = xm_candidates
        self.critic_distill_steps = critic_distill_steps
        self.max_grad_norm = max_grad_norm

        self.actor = TransformerFlowActor(
            dim_state = state_dim,
            num_cont_actions = num_actions,
            dim_hidden = hidden_dim,
            depth = depth,
            heads = heads,
            max_chunk_size = max_chunk_size
        )

        self.critic = TransformerFlowCritic(
            dim_state = state_dim,
            num_cont_actions = num_actions,
            dim_hidden = hidden_dim,
            depth = depth,
            heads = heads,
            max_seq_len = max_chunk_size
        )

        ema_kw = dict(beta = ema_decay, include_online_model = False, update_every = 1, update_after_step = 0)
        self.ema_actor = EMA(self.actor, **ema_kw)
        self.ema_critic = EMA(self.critic, **ema_kw)

        self.actor_flow = NanoFlow(self.actor, data_shape = (max_chunk_size, num_actions))
        self.critic_flow = NanoFlow(self.critic, data_shape = (max_chunk_size,))
        self.ema_critic_flow = NanoFlow(self.ema_critic, data_shape = (max_chunk_size,))

        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr = lr, weight_decay = 1e-4)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr = lr, weight_decay = 1e-4)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

    def learn(self, buffer: ChunkReplayBuffer, value_net: Module, *, steps = 24, batch_size = 128, recent_frac = 0.60):
        dev, val_dev = module_device(self), module_device(value_net)
        self.train()

        actor_losses, critic_losses, q_means = [], [], []

        for _ in range(steps):
            states, actions, rewards, durations, next_states, terminated, lens = buffer.sample(
                batch_size, dev, recent_fraction = recent_frac
            )
            mask = mask_after(terminated, True) & lens_to_mask(lens, self.max_chunk_size)
            mask_f = mask.float()

            actor_loss = xm_flow_loss(
                self.actor_flow,
                actions,
                mask = mask,
                candidates = self.xm_candidates,
                noise_fn = lambda n: sample_pyoco_noise(n, self.max_chunk_size, self.num_actions, self.corr_alpha, dev),
                cond_state = states
            )
            update_network_(actor_loss, self.opt_actor, self.max_grad_norm)

            with torch.no_grad():
                flat_next = rearrange(next_states, 'b s d -> (b s) d').to(val_dev)
                next_values = rearrange(value_net(flat_next), '(b s) -> b s', b = batch_size).to(dev)
                alive_after = torch.cumprod((~terminated).float() * mask_f, dim = 1)

                # exact telescoping smdp multi-horizon bellman target

                step_durs = durations * mask_f
                cum_durs = torch.cumsum(step_durs, dim = 1)
                prev_gammas = self.gamma ** (cum_durs - step_durs)
                next_gammas = self.gamma ** cum_durs

                cum_rewards = torch.cumsum(rewards * mask_f * prev_gammas, dim = 1)
                target_q = (cum_rewards + next_gammas * alive_after * next_values).clamp(-5., 5.)

                flow_q = self.ema_critic_flow.sample(
                    steps = self.critic_distill_steps,
                    batch_size = batch_size,
                    data_shape = (self.max_chunk_size,),
                    cond_state = states,
                    cont_actions = actions
                ).clamp(-5., 5.)

                distill_target = 0.5 * flow_q + 0.5 * target_q

            critic_flow_loss = xm_flow_loss(
                self.critic_flow,
                target_q,
                mask = mask,
                candidates = self.xm_candidates,
                cond_state = states,
                cont_actions = actions
            )

            pred_q = self.critic(cond_state = states, cont_actions = actions, is_distill = True)
            distill_loss = masked_mean((pred_q - distill_target).square(), mask)

            update_network_(critic_flow_loss + distill_loss, self.opt_critic, self.max_grad_norm)

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_flow_loss.item())
            q_means.append(pred_q[mask].mean().item() * 100.)

        return dict(
            actor_flow = float(np.mean(actor_losses)),
            critic_flow = float(np.mean(critic_losses)),
            mean_q = float(np.mean(q_means))
        )

# composable flow chunk skill bank & recursive smdp wrapper

class FlowChunkBank(Module):
    def __init__(
        self,
        actor_model: TransformerFlowActor,
        critic_model: TransformerFlowCritic,
        *,
        dct_modes = 4,
        corr_alpha = 0.7,
        lerp_noise = False,
        override_chunk_len = False,
        actor_flow_steps = 4,
        dynamic_chunk = True,
        dynamic_chunk_eps = 0.05,
        min_chunk_len = 1,
        q_temp = 0.025,
        reassess_chunk = True,
        reassess_every = 4,
        reassess_threshold = 0.25,
        gamma = 0.99
    ):
        super().__init__()
        self.actor = deepcopy(actor_model).eval().requires_grad_(False)
        self.critic = deepcopy(critic_model).eval().requires_grad_(False)
        self.actor_flow = NanoFlow(self.actor, data_shape = (actor_model.max_chunk_size, actor_model.num_cont_actions))

        self.max_chunk_size = actor_model.max_chunk_size
        self.num_actions = actor_model.num_cont_actions
        self.dct_modes = min(dct_modes, self.max_chunk_size)
        self.dct_dim = self.dct_modes * self.num_actions

        self.lerp_noise = lerp_noise
        self.override_chunk_len = override_chunk_len
        self.aux_dim = int(lerp_noise) + (2 if override_chunk_len else 0)
        self.latent_dim = self.dct_dim + self.aux_dim

        self.corr_alpha = corr_alpha
        self.dct_scale = math.sqrt(self.max_chunk_size / self.dct_modes)
        self.actor_flow_steps = actor_flow_steps

        self.dynamic_chunk = dynamic_chunk
        self.dynamic_chunk_eps = dynamic_chunk_eps
        self.min_chunk_len = min_chunk_len
        self.q_temp = q_temp
        self.reassess_chunk = reassess_chunk
        self.reassess_every = max(1, reassess_every)
        self.reassess_threshold = reassess_threshold
        self.gamma = gamma

        self.register_buffer('dct_mat', make_orthonormal_dct(self.max_chunk_size), persistent = False)

    def dct_to_noise_seq(self, dct_latent: Tensor):
        if dct_latent.ndim == 1:
            dct_latent = rearrange(dct_latent, 'd -> 1 d')

        modes = rearrange(dct_latent, 'b (k a) -> b k a', k = self.dct_modes, a = self.num_actions)
        return self.dct_scale * einsum(self.dct_mat[:self.dct_modes], modes, 'k h, b k a -> b h a')

    def parse_latent_action(self, batch_size: int, latent_action: Tensor | None, device):
        ones_long = torch.ones(batch_size, device = device, dtype = torch.long)
        zeros = torch.zeros(batch_size, device = device)

        if not exists(latent_action):
            return None, zeros, ones_long, zeros

        latent = latent_action.to(device).clamp(-1., 1.)
        if latent.ndim == 1:
            latent = rearrange(latent, 'd -> 1 d')

        idx = self.dct_dim
        dct_latent = 3. * latent[:, :idx]
        alphas = 0.5 * (latent[:, idx] + 1.) if self.lerp_noise else torch.ones(batch_size, device = device)
        idx += int(self.lerp_noise)

        if not self.override_chunk_len:
            return dct_latent, alphas, ones_long, zeros

        len_code = 0.5 * (latent[:, idx] + 1.)
        override_gates = 0.5 * (latent[:, idx + 1] + 1.)
        span = self.max_chunk_size - self.min_chunk_len + 1
        desired_lens = (self.min_chunk_len + (len_code * span).long()).clamp(self.min_chunk_len, self.max_chunk_size)

        return dct_latent, alphas, desired_lens, override_gates

    def compose_noise(self, batch_size: int, dct_latent: Tensor | None, alphas: Tensor, *, zero_noise = False):
        dev = module_device(self)

        if zero_noise:
            return torch.zeros((batch_size, self.max_chunk_size, self.num_actions), device = dev)

        if not exists(dct_latent):
            return sample_pyoco_noise(batch_size, self.max_chunk_size, self.num_actions, self.corr_alpha, dev)

        constituted = self.dct_to_noise_seq(dct_latent)

        if not self.lerp_noise:
            return constituted

        init_noise = sample_pyoco_noise(batch_size, self.max_chunk_size, self.num_actions, self.corr_alpha, dev)
        return init_noise.lerp(constituted, append_dims(alphas, 2))

    @torch.no_grad()
    def critic_q(self, states: Tensor, action_chunks: Tensor):
        dev = module_device(self)
        return self.critic(cond_state = states.to(dev), cont_actions = action_chunks.to(dev), is_distill = True)

    @torch.no_grad()
    def plan_chunks(
        self,
        states: Tensor,
        latent_action: Tensor | None = None,
        *,
        eval_mode = False,
        zero_noise = False,
        return_probs = False
    ):
        dev = module_device(self)
        states_dev = states.to(dev)
        batch_size = states_dev.shape[0]

        dct_latent, alphas, desired_lens, override_gates = self.parse_latent_action(batch_size, latent_action, dev)
        noise_seq = self.compose_noise(batch_size, dct_latent, alphas, zero_noise = zero_noise)

        chunks = self.actor_flow.sample(
            steps = self.actor_flow_steps,
            batch_size = batch_size,
            noise = noise_seq,
            cond_state = states_dev
        ).clamp_(-1., 1.)

        q_vals = self.critic_q(states_dev, chunks)
        logits = q_vals / max(1e-4, self.q_temp)

        if self.min_chunk_len > 1:
            logits[:, :self.min_chunk_len - 1] = -1e9

        probs = logits.softmax(dim = -1)

        if not self.dynamic_chunk:
            critic_lens = torch.full((batch_size,), self.max_chunk_size, device = dev, dtype = torch.long)
        else:
            critic_lens = Categorical(probs = probs).sample() + 1

            if not eval_mode and self.dynamic_chunk_eps > 0.:
                rand_mask = torch.rand(batch_size, device = dev) < self.dynamic_chunk_eps
                rand_lens = torch.randint(self.min_chunk_len, self.max_chunk_size + 1, (batch_size,), device = dev)
                critic_lens = torch.where(rand_mask, rand_lens, critic_lens)

        can_override = self.override_chunk_len and exists(latent_action)
        overridden = (override_gates > 0.5) if can_override else torch.zeros(batch_size, device = dev, dtype = torch.bool)
        lens = torch.where(overridden, desired_lens, critic_lens)
        planned_q = rearrange(q_vals.gather(1, rearrange(lens - 1, 'b -> b 1')), 'b 1 -> b')

        meta = dict(
            alphas = alphas.cpu().numpy(),
            critic_lens = critic_lens.cpu().numpy(),
            desired_lens = desired_lens.cpu().numpy(),
            override_gates = override_gates.cpu().numpy(),
            overridden = overridden.cpu().numpy()
        )
        out = (chunks.cpu().numpy(), lens.cpu().numpy(), planned_q.cpu().numpy(), meta)
        return (*out, probs.cpu().numpy()) if return_probs else out

def step_envs(
    envs: list[gym.Env],
    obs: np.ndarray,
    actions: Tensor | np.ndarray | None = None,
    *,
    banks: tuple[FlowChunkBank, ...] = (),
    eval_mode = False,
    zero_noise = False
):
    num_envs = len(envs)

    if len(banks) == 0:
        act_np = actions.detach().cpu().numpy() if is_tensor(actions) else np.asarray(actions, dtype = np.float32)
        next_obs = np.empty_like(obs)
        rewards = np.empty(num_envs, dtype = np.float32)
        terms = np.zeros(num_envs, dtype = np.bool_)
        truncs = np.zeros(num_envs, dtype = np.bool_)

        for i, env in enumerate(envs):
            next_o, r, term_i, trunc_i, _ = env.step(act_np[i])
            next_obs[i] = next_o
            rewards[i] = float(r)
            terms[i] = term_i
            truncs[i] = trunc_i

        ones = np.ones(num_envs, dtype = np.float32)
        zeros = np.zeros(num_envs, dtype = np.float32)
        info = dict(
            raw_reward = rewards,
            last_reward = rewards,
            prim_steps = ones,
            exec_chunks = ones,
            planned_len = ones,
            overridden = zeros,
            inner_overridden = zeros,
            aborted = zeros,
            lerp_alpha = ones
        )
        return next_obs, rewards, terms, truncs, info

    bank, *rest_banks = banks
    latent = torch.as_tensor(actions, dtype = torch.float32) if exists(actions) else None
    chunks, plan_lens, planned_qs, meta = bank.plan_chunks(
        torch.from_numpy(obs).float(),
        latent,
        eval_mode = eval_mode,
        zero_noise = zero_noise
    )

    cur_obs = obs.copy()
    disc_rewards = np.zeros(num_envs, dtype = np.float32)
    raw_rewards = np.zeros(num_envs, dtype = np.float32)
    last_rewards = np.zeros(num_envs, dtype = np.float32)
    durations = np.zeros(num_envs, dtype = np.float32)
    exec_chunks = np.zeros(num_envs, dtype = np.float32)
    inner_ovrs = np.zeros(num_envs, dtype = np.float32)
    aborts = np.zeros(num_envs, dtype = np.float32)

    terms = np.zeros(num_envs, dtype = np.bool_)
    truncs = np.zeros(num_envs, dtype = np.bool_)
    active = np.ones(num_envs, dtype = np.bool_)
    discount = np.ones(num_envs, dtype = np.float32)

    for k in range(int(plan_lens.max())):
        active &= (k < plan_lens)
        if not active.any():
            break

        # closed-loop mid-chunk value reassessment (bypassed when higher policy overrides critic)

        if k > 0 and bank.reassess_chunk and divisible_by(k, bank.reassess_every):
            idx = np.flatnonzero(active & ~meta['overridden'])

            if len(idx) > 0:
                rem_lens = plan_lens[idx] - k
                rem_chunks = torch.from_numpy(chunks[idx, k:k + int(rem_lens.max())]).float()
                rem_q = bank.critic_q(torch.from_numpy(cur_obs[idx]).float(), rem_chunks).cpu().numpy()
                est_total = (disc_rewards[idx] / 100.) + discount[idx] * rem_q[np.arange(len(idx)), rem_lens - 1]

                abort_mask = should_abort_chunk(est_total, planned_qs[idx], bank.reassess_threshold)
                if abort_mask.any():
                    aborted_idx = idx[abort_mask]
                    aborts[aborted_idx] = 1.
                    active[aborted_idx] = False

        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break

        step_obs, step_disc_r, step_term, step_trunc, sub_info = step_envs(
            [envs[i] for i in idx],
            cur_obs[idx],
            chunks[idx, k],
            banks = tuple(rest_banks),
            eval_mode = eval_mode
        )

        cur_obs[idx] = step_obs
        raw_rewards[idx] += sub_info['raw_reward']
        disc_rewards[idx] += discount[idx] * step_disc_r
        last_rewards[idx] = sub_info['last_reward']
        discount[idx] *= bank.gamma ** sub_info['prim_steps']
        durations[idx] += sub_info['prim_steps']
        exec_chunks[idx] += 1.
        inner_ovrs[idx] += sub_info['overridden']
        aborts[idx] += sub_info['aborted']

        step_done = step_term | step_trunc
        if step_done.any():
            done_idx = idx[step_done]
            terms[done_idx] = step_term[step_done]
            truncs[done_idx] = step_trunc[step_done]
            active[done_idx] = False

    info = dict(
        raw_reward = raw_rewards,
        last_reward = last_rewards,
        prim_steps = durations,
        exec_chunks = exec_chunks,
        planned_len = plan_lens.astype(np.float32),
        critic_len = meta['critic_lens'].astype(np.float32),
        desired_len = meta['desired_lens'].astype(np.float32),
        override_gate = meta['override_gates'].astype(np.float32),
        overridden = meta['overridden'].astype(np.float32),
        inner_overridden = inner_ovrs / np.maximum(1., exec_chunks),
        aborted = aborts,
        lerp_alpha = meta['alphas'].astype(np.float32)
    )
    return cur_obs, disc_rewards, terms, truncs, info

class FlowChunkWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, bank: FlowChunkBank, eval_mode = True):
        super().__init__(env)
        if isinstance(env, FlowChunkWrapper):
            self.base_env = env.base_env
            self.banks = (bank, *env.banks)
        else:
            self.base_env = env
            self.banks = (bank,)

        self.bank = bank
        self.eval_mode = eval_mode
        self.action_space = gym.spaces.Box(low = -1., high = 1., shape = (bank.latent_dim,), dtype = np.float32)
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.base_env.reset(**kwargs)
        self._obs = np.asarray(obs, dtype = np.float32)
        return obs, info

    def step(self, latent_action = None, *, zero_noise = False):
        latent = rearrange(torch.as_tensor(latent_action, dtype = torch.float32), 'd -> 1 d') if exists(latent_action) else None
        next_obs, disc_r, terms, truncs, vec_info = step_envs(
            [self.base_env],
            rearrange(self._obs, 'd -> 1 d'),
            latent,
            banks = self.banks,
            eval_mode = self.eval_mode,
            zero_noise = zero_noise
        )
        self._obs = next_obs[0]
        info = {
            k: (bool(v[0] > 0.5) if k in ('aborted', 'overridden') else float(v[0]))
            for k, v in vec_info.items()
        }
        return self._obs.copy(), float(disc_r[0]), bool(terms[0]), bool(truncs[0]), info

# ppo networks (SimBa + MeanConcBeta) & smdp-hpo update

class SimBa(Module):
    def __init__(self, dim, expansion = 2):
        super().__init__()
        dim_inner = dim * expansion
        self.net = Sequential(RMSNorm(dim), Linear(dim, dim_inner), SiLU(), Linear(dim_inner, dim))
        self.norm = RMSNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class BetaActor(Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        *,
        bounds = (-1., 1.),
        init_conc = 5.,
        aux_dims = 0,
        aux_init_conc = 0.5
    ):
        super().__init__()
        dim = hidden_dim * 2
        self.net = Sequential(
            Linear(state_dim, dim),
            SimBa(dim),
            SimBa(dim),
            Linear(dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, 2 * action_dim)
        )

        nn.init.orthogonal_(self.net[-1].weight, gain = 0.01)
        nn.init.zeros_(self.net[-1].bias)

        if aux_dims > 0:
            inv_softplus = lambda y: math.log(math.expm1(y))
            with torch.no_grad():
                bias_pairs = rearrange(self.net[-1].bias, '(a two) -> a two', two = 2)
                bias_pairs[-aux_dims:, 1].fill_(inv_softplus(aux_init_conc) - inv_softplus(init_conc))

        clamp_exp = (-8., 4.) if aux_dims > 0 else (-4., 4.)
        self.beta = MeanConcBeta(bounds = bounds, init_conc = init_conc, unimodal = True, clamp_exp = clamp_exp, eps = 1e-5)

    def forward(self, state):
        return rearrange(self.net(state), '... (a two) -> ... a two', two = 2)

    def sample(self, state, sum_action_dim = True):
        dist = self.beta(self.forward(state))
        action = dist.sample()
        return action, self.beta.log_prob(dist, action, sum_action_dim = sum_action_dim)

    def sample_temp(self, state, temperature = 0.):
        dist = self.beta(self.forward(state), temperature = temperature)
        return dist.mean if temperature == 0. else dist.sample()

    def greedy(self, state):
        return self.sample_temp(state, temperature = 0.)

    def log_prob(self, params, action, sum_action_dim = True):
        return self.beta.log_prob(params, action, sum_action_dim = sum_action_dim)

    def entropy_per_dim(self, params):
        return self.beta.entropy(params, sum_action_dim = False)

class Critic(Module):
    def __init__(self, state_dim: int, hidden_dim = 192):
        super().__init__()
        self.net = Sequential(Linear(state_dim, hidden_dim), SimBa(hidden_dim), SimBa(hidden_dim), Linear(hidden_dim, 1))

    def forward(self, state):
        return rearrange(self.net(state), '... 1 -> ...')

def ppo_update(
    actor: BetaActor,
    critic: Critic,
    opt_actor,
    opt_critic,
    *,
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    durations: Tensor | None = None,
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.2,
    entropy_coef = 0.001,
    epochs = 6,
    minibatch_envs = 4,
    max_grad_norm = 0.5,
    use_hpo = True
):
    num_envs = states.shape[0]
    per_dim = old_log_probs.ndim == 3
    loss_fn = hpo_actor_loss if use_hpo else ppo_actor_loss

    with torch.no_grad():
        values = critic(states)
        gae = calc_gae(
            rewards = rewards,
            values = values,
            next_values = critic(next_states),
            terminated = terminated,
            truncated = truncated,
            durations = durations,
            gamma = gamma,
            lam = lam,
            use_accelerated = False,
            return_advantages = True
        )

    for _ in range(epochs):
        perm = torch.randperm(num_envs, device = states.device)

        for start in range(0, num_envs, minibatch_envs):
            mb = perm[start:start + minibatch_envs]

            mb_states = rearrange(states[mb], 'b t d -> (b t) d')
            mb_actions = rearrange(actions[mb], 'b t a -> (b t) a')
            mb_old_lp = rearrange(old_log_probs[mb], 'b t ... -> (b t) ...')
            mb_adv = rearrange(gae.advantages[mb], 'b t -> (b t)')
            mb_ret = rearrange(gae.returns[mb], 'b t -> (b t)')
            mb_val = rearrange(values[mb], 'b t -> (b t)')

            params = actor(mb_states)
            cur_lp = actor.log_prob(params, mb_actions, sum_action_dim = not per_dim)

            actor_loss = loss_fn(
                cur_lp,
                mb_old_lp,
                mb_adv,
                eps_clip = eps_clip,
                per_action_dim = per_dim,
                normalize_advantages = True
            ).mean()

            actor_loss = actor_loss - entropy_coef * actor.entropy_per_dim(params).mean() * 2.

            update_network_(actor_loss, opt_actor, max_grad_norm)
            update_network_(clipped_value_loss_fn(critic(mb_states), mb_val, mb_ret, clip = None), opt_critic)

# hud trajectory rendering & evaluation

def draw_badge(draw, font, box, text, *, fill = (20, 24, 33), outline = (255, 255, 255), text_fill = (255, 255, 255), width = 1):
    draw.rectangle(box, fill = fill, outline = outline, width = width)
    draw.text((box[0] + 6, box[1] + 6), text, fill = text_fill, font = font)

def draw_meter(draw, font, x, y, width, text, frac, bar_color, *, text_fill = (218, 226, 242), ticks = ()):
    fill_w = max(1, int(round((width - 2) * max(0., min(1., frac)))))
    draw.text((x, y + 1), text, fill = text_fill, font = font)
    draw.rectangle([x, y + 14, x + width, y + 26], fill = (32, 37, 48), outline = (82, 92, 114))
    draw.rectangle([x + 1, y + 15, x + 1 + fill_w, y + 25], fill = bar_color)

    for tick in ticks:
        tx = x + 1 + int(round((width - 2) * tick))
        draw.line([(tx, y + 12), (tx, y + 28)], fill = (255, 255, 255), width = 1)

def draw_step_boxes(draw, x0, y0, y1, *, total_slots, cur_step, exec_len, plan_len, color, step_dx, box_w, highlight_gold = False):
    for i in range(total_slots):
        step_num, bx = i + 1, x0 + step_dx * i

        if step_num == cur_step:
            fill, outline = color, (OVERRIDE_GOLD if highlight_gold else (255, 255, 255))
        elif step_num < cur_step:
            fill, outline = dim_rgb(color, 0.5), color
        elif step_num <= exec_len:
            fill, outline = (38, 44, 58), color
        elif step_num <= plan_len:
            fill, outline = (78, 28, 36), (200, 70, 80)
        else:
            fill, outline = (25, 29, 39), (46, 52, 66)

        draw.rectangle([bx, y0, bx + box_w, y1], fill = fill, outline = outline)

def draw_dt_histogram(
    draw,
    font,
    x0,
    y1,
    label,
    probs,
    plan_len,
    max_chunk_size,
    *,
    bar_step = 18,
    bar_w = 14,
    show_ovr = False,
    did_ovr = False,
    critic_len = 1,
    tick_vals = (1, 4, 8, 12, 16)
):
    draw.text((x0, y1 + 8), label, fill = (180, 188, 204), font = font)
    bx0 = x0 + 6 * len(label) + 4
    p_scale = max(0.16, float(probs.max()))

    for i in range(max_chunk_size):
        h_val, bx = i + 1, bx0 + bar_step * i
        is_sel = h_val == plan_len
        gold_sel = is_sel and show_ovr and did_ovr
        sel_out = OVERRIDE_GOLD if gold_sel else ((255, 255, 255) if is_sel else (56, 63, 79))

        draw.rectangle([bx, y1 + 1, bx + bar_w, y1 + 20], fill = (32, 37, 48), outline = sel_out, width = 2 if gold_sel else 1)
        bar_h = max(1, int(round(17 * min(1., float(probs[i]) / p_scale))))
        draw.rectangle([bx + 1, y1 + 20 - bar_h, bx + bar_w - 1, y1 + 19], fill = dt_color_rgb(h_val, max_chunk_size))

        if show_ovr and did_ovr and h_val == critic_len:
            draw.line([(bx + 2, y1 - 1), (bx + bar_w - 2, y1 - 1)], fill = (130, 210, 255), width = 2)

        if h_val in tick_vals:
            draw.text((bx + (2 if h_val < 10 else 0), y1 + 21), str(h_val), fill = sel_out if is_sel else (155, 165, 182), font = font)

def check_single_abort(bank: FlowChunkBank, obs: np.ndarray, rem_chunk: np.ndarray, disc_r: float, discount: float, planned_q: float):
    rem_q = bank.critic_q(
        rearrange(torch.from_numpy(obs).float(), 'd -> 1 d'),
        rearrange(torch.from_numpy(rem_chunk).float(), 's a -> 1 s a')
    )[0, len(rem_chunk) - 1].item()
    return should_abort_chunk((disc_r / 100.) + discount * rem_q, planned_q, bank.reassess_threshold)

def rollout_hud_chunk(
    env: gym.Env,
    obs: np.ndarray,
    banks: tuple[FlowChunkBank, ...],
    latent_action: Tensor | None,
    ep_return: float,
    *,
    zero_noise = False
):
    bank, *rest_banks = banks
    chunks, plan_lens, planned_qs, meta, probs = bank.plan_chunks(
        rearrange(torch.from_numpy(obs).float(), 'd -> 1 d'),
        latent_action,
        eval_mode = True,
        zero_noise = zero_noise,
        return_probs = True
    )

    chunk, plan_len, planned_q = chunks[0], int(plan_lens[0]), float(planned_qs[0])
    did_ovr = bool(meta['overridden'][0])
    critic_len, actor_len = int(meta['critic_lens'][0]), int(meta['desired_lens'][0])
    disc_reward, discount, was_aborted, done = 0., 1., False, False
    records = []

    for k in range(plan_len):
        if k > 0 and not did_ovr and bank.reassess_chunk and divisible_by(k, bank.reassess_every):
            if check_single_abort(bank, obs, chunk[k:plan_len], disc_reward, discount, planned_q):
                was_aborted = True
                break

        if len(rest_banks) > 0:
            obs, ep_return, sub_disc_r, done, sub_recs = rollout_hud_chunk(
                env, obs, tuple(rest_banks), torch.from_numpy(chunk[k:k + 1]).float(), ep_return
            )
            disc_reward += discount * sub_disc_r
            discount *= bank.gamma ** len(sub_recs)

            for r_i, rec in enumerate(sub_recs):
                rec.update(
                    plan_sup_k = plan_len,
                    sub_m = k + 1,
                    sup_probs = probs[0],
                    sup_ovr = did_ovr,
                    sup_critic_k = critic_len,
                    is_sup_start = (k == 0 and r_i == 0)
                )
            records.extend(sub_recs)
        else:
            obs_before = obs.copy()
            obs, reward, term, trunc, _ = env.step(chunk[k])
            reward_f = float(reward)
            ep_return += reward_f
            disc_reward += discount * reward_f
            discount *= bank.gamma
            done = bool(term or trunc)

            records.append(dict(
                rgb = env.render(),
                obs = obs_before,
                plan_h = plan_len,
                step_in_h = k + 1,
                h_probs = probs[0],
                alpha = float(meta['alphas'][0]),
                critic_h = critic_len,
                actor_h = actor_len,
                ovr_gate = float(meta['override_gates'][0]),
                did_ovr = did_ovr,
                cum_ret = ep_return,
                is_sub_start = (k == 0),
                is_sup_start = (k == 0)
            ))

        if done:
            break

    if len(rest_banks) > 0:
        exec_k = records[-1]['sub_m'] if records else 1
        for rec in records:
            rec.update(exec_sup_k = exec_k, sup_tot_dt = len(records), sup_aborted = was_aborted)
    else:
        for rec in records:
            rec.update(exec_h = len(records), aborted = was_aborted)

    return obs, ep_return, disc_reward, done, records

@torch.no_grad()
def render_hud_video(
    banks: tuple[FlowChunkBank, ...],
    seed: int,
    mp4_path: str | Path,
    gif_path: str | Path,
    *,
    student: BetaActor | None = None,
    zero_noise = False,
    mode_label = 'dct-8d',
    env_name = 'LunarLander-v3',
    max_episode_steps = 600
):
    if exists(student):
        student.eval()

    top_bank, prim_bank = banks[0], banks[-1]
    is_super = len(banks) > 1

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_episode_steps, continuous = True)
    obs, _ = env.reset(seed = seed)
    records, done, ep_return, dec_idx = [], False, 0., 0

    while not done:
        dec_idx += 1
        state_t = rearrange(torch.from_numpy(obs).float().to(ppo_device), 'd -> 1 d')
        latent = student.greedy(state_t) if exists(student) else None
        obs, ep_return, _, done, chunk_recs = rollout_hud_chunk(env, obs, banks, latent, ep_return, zero_noise = zero_noise)

        for rec in chunk_recs:
            rec['dec_idx'] = dec_idx
        records.extend(chunk_recs)

    env.close()

    total_frames = len(records)
    width, height, border = 608, (532 if is_super else 528), 4
    font = ImageFont.load_default()
    show_ovr = is_super or (top_bank.override_chunk_len and exists(student))
    show_lerp = (not is_super) and top_bank.lerp_noise and exists(student) and not show_ovr
    frames = []

    for frame_idx, rec in enumerate(records):
        exec_h, plan_h, step_in_h = rec['exec_h'], rec['plan_h'], rec['step_in_h']
        did_ovr, critic_h, actor_h = rec['did_ovr'], rec['critic_h'], rec['actor_h']

        in_col = dt_color_rgb(exec_h, prim_bank.max_chunk_size)
        top_col = dt_color_rgb(rec['exec_sup_k'], top_bank.max_chunk_size) if is_super else in_col
        ovr_col = OVERRIDE_GOLD if did_ovr else (98, 178, 228)
        ovr_txt_col = OVERRIDE_GOLD if did_ovr else (225, 236, 248)

        canvas = Image.new('RGB', (width, height), (20, 24, 33))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, width - 1, 400 + 2 * border - 1], fill = top_col)
        canvas.paste(Image.fromarray(rec['rgb']), (border, border))

        # lander tracking ring

        lx = int(border + 300. + 300. * float(rec['obs'][0]))
        ly = int(border + 282. - 200. * float(rec['obs'][1]))

        if border + 16 <= lx <= width - border - 16 and border + 16 <= ly <= 400 + border - 16:
            draw.ellipse([lx - 22, ly - 22, lx + 22, ly + 22], outline = in_col, width = 2)
            if rec['is_sup_start']:
                sup_gold = rec['sup_ovr'] if is_super else (show_ovr and did_ovr)
                draw.ellipse([lx - 28, ly - 28, lx + 28, ly + 28], outline = OVERRIDE_GOLD if sup_gold else (255, 255, 255), width = 2)
            elif is_super and rec['is_sub_start'] and did_ovr:
                draw.ellipse([lx - 26, ly - 26, lx + 26, ly + 26], outline = OVERRIDE_GOLD, width = 1)

        # top viewport badges

        if is_super:
            exec_k, plan_k, sub_m, sup_ovr = rec['exec_sup_k'], rec['plan_sup_k'], rec['sub_m'], rec['sup_ovr']
            abort_mark = '!' if rec['sup_aborted'] else ''
            left_text = f'FLOW2 K={exec_k}/{plan_k}{abort_mark} ({sub_m}/{exec_k}) | tot_dt={rec["sup_tot_dt"]:2d}'
            right_text = f'L2 {"OVR" if sup_ovr else "CRIT"} K={plan_k} | L1 {"OVR" if did_ovr else "CRIT"} dt={plan_h:2d} (c{critic_h:2d})'

            draw_badge(
                draw, font, [12, 12, 274, 36], left_text,
                fill = top_col, outline = OVERRIDE_GOLD if sup_ovr else (255, 255, 255),
                width = 2 if (rec['is_sup_start'] or sup_ovr) else 1
            )
            draw_badge(
                draw, font, [320, 12, 596, 36], right_text,
                outline = ovr_col, text_fill = ovr_txt_col,
                width = 2 if (sup_ovr or did_ovr) else 1
            )
        else:
            abort_tag = ' ABORT!' if rec['aborted'] else ''
            gold_b = show_ovr and did_ovr
            draw_badge(
                draw, font, [12, 12, 226, 36], f'ACTIVE dt={exec_h:2d} | step {step_in_h:2d}/{plan_h:2d}{abort_tag}',
                fill = in_col, outline = OVERRIDE_GOLD if gold_b else (255, 255, 255), width = 2 if gold_b else 1
            )

            if show_ovr:
                tag = f'OVERRIDE dt={actor_h:2d} (crit {critic_h:2d})' if did_ovr else f'CRITIC dt={critic_h:2d} (pol {actor_h:2d})'
                draw_badge(
                    draw, font, [346, 12, 596, 36], f'{tag} g={rec["ovr_gate"]:.2f}',
                    outline = ovr_col, text_fill = ovr_txt_col, width = 2 if did_ovr else 1
                )
            elif show_lerp:
                alpha_v = rec['alpha']
                gate_col = (int(72 + 165 * alpha_v), int(155 - 35 * alpha_v), int(239 - 145 * alpha_v))
                draw_badge(
                    draw, font, [404, 12, 596, 36], f'lerp={alpha_v:.2f} ({"REPLACE" if alpha_v >= 0.5 else "STEER"})',
                    outline = gate_col, text_fill = (245, 248, 255), width = 2
                )
            else:
                draw_badge(draw, font, [430, 12, 596, 36], mode_label, outline = (120, 130, 150), text_fill = (230, 236, 248))

        # chunk step progress row

        y0 = 413
        if is_super:
            draw.rectangle([12, y0, 56, y0 + 22], fill = top_col)
            draw.text((16, y0 + 6), f'K={sub_m}/{plan_k}', fill = (255, 255, 255), font = font)
            draw_step_boxes(
                draw, 60, y0 + 3, y0 + 19,
                total_slots = top_bank.max_chunk_size, cur_step = sub_m, exec_len = exec_k, plan_len = plan_k,
                color = top_col, step_dx = 14, box_w = 11, highlight_gold = sup_ovr
            )
            inner_x0, step_dx, box_w, stat_x, dec_tag = 64 + 14 * top_bank.max_chunk_size, 13, 10, 336, f'Sup#{rec["dec_idx"]:02d}'
        else:
            draw.rectangle([12, y0, 90, y0 + 22], fill = in_col)
            draw.text((17, y0 + 6), f'dt={exec_h:2d}/{plan_h:2d}', fill = (255, 255, 255), font = font)
            inner_x0, step_dx, box_w, stat_x, dec_tag = 96, 15, 12, 344, f'Dec#{rec["dec_idx"]:02d}'

        draw_step_boxes(
            draw, inner_x0, y0 + 4, y0 + 18,
            total_slots = prim_bank.max_chunk_size, cur_step = step_in_h, exec_len = exec_h, plan_len = plan_h,
            color = in_col, step_dx = step_dx, box_w = box_w, highlight_gold = is_super and did_ovr
        )
        stat_txt = f'Ret:{rec["cum_ret"]:+6.1f} | {dec_tag} | t={frame_idx + 1:03d}/{total_frames:03d}'
        draw.text((stat_x, y0 + 6), stat_txt, fill = (235, 240, 248), font = font)

        # live critic horizon distributions & override meters

        y1 = 442
        draw_dt_histogram(
            draw, font, 12, y1, 'P(dt1):', rec['h_probs'], plan_h, prim_bank.max_chunk_size,
            show_ovr = show_ovr, did_ovr = did_ovr, critic_len = critic_h
        )

        if is_super:
            draw_dt_histogram(
                draw, font, 362, y1, 'P(K2):', rec['sup_probs'], plan_k, top_bank.max_chunk_size,
                bar_step = 24, bar_w = 18, show_ovr = exists(student) and top_bank.override_chunk_len,
                did_ovr = sup_ovr, critic_len = rec['sup_critic_k'], tick_vals = tuple(range(1, top_bank.max_chunk_size + 1))
            )
            mode_x = 408 + 24 * top_bank.max_chunk_size
            if mode_x <= 520:
                l2_col = OVERRIDE_GOLD if sup_ovr else (195, 210, 232)
                draw.text((mode_x, y1 + 8), 'L2:OVR' if sup_ovr else 'L2:CRIT', fill = l2_col, font = font)
        elif show_ovr:
            ovr_g = rec['ovr_gate']
            bar_col = OVERRIDE_GOLD if did_ovr else (72, 165, 222)
            gate_lbl = f'gate={ovr_g:.2f} [{"OVERRIDE" if did_ovr else "CRITIC"}] pol={actor_h:2d}'
            draw_meter(draw, font, 366, y1, 230, gate_lbl, ovr_g, bar_col, text_fill = ovr_txt_col, ticks = (0.5,))
        elif show_lerp:
            alpha_v = rec['alpha']
            gate_col = (int(72 + 165 * alpha_v), int(155 - 35 * alpha_v), int(239 - 145 * alpha_v))
            draw_meter(draw, font, 366, y1, 230, f'lerp (steer->replace): {alpha_v:.2f}', alpha_v, gate_col)
        else:
            sel_pct = int(round(100 * rec['h_probs'][plan_h - 1]))
            draw.text((366, y1 + 8), f'{mode_label} | P(dt={plan_h})={sel_pct:2d}%', fill = (210, 218, 232), font = font)

        # episode chunk timeline

        y2, tx0, tw = (480 if is_super else 481), 12, width - 24
        draw.rectangle([tx0 - 1, y2 - 1, tx0 + tw + 1, y2 + (25 if is_super else 22)], outline = (95, 105, 125))

        for t_i, rec_t in enumerate(records):
            x0 = tx0 + int(t_i * tw / total_frames)
            x1 = max(x0 + 1, tx0 + int((t_i + 1) * tw / total_frames))
            past = t_i <= frame_idx
            inner_c = dt_color_rgb(rec_t['exec_h'], prim_bank.max_chunk_size)
            if not past:
                inner_c = dim_rgb(inner_c)

            if is_super:
                outer_c = dt_color_rgb(rec_t['exec_sup_k'], top_bank.max_chunk_size)
                gold_c = OVERRIDE_GOLD if past else dim_rgb(OVERRIDE_GOLD, 0.45)
                draw.rectangle([x0, y2, x1, y2 + 11], fill = outer_c if past else dim_rgb(outer_c))
                draw.rectangle([x0, y2 + 13, x1, y2 + 24], fill = inner_c)
                if rec_t['sup_ovr']:
                    draw.rectangle([x0, y2, x1, y2 + 2], fill = gold_c)
                if rec_t['did_ovr']:
                    draw.rectangle([x0, y2 + 13, x1, y2 + 15], fill = gold_c)
                if rec_t['is_sup_start'] and t_i > 0:
                    draw.line([(x0, y2), (x0, y2 + 11)], fill = (255, 255, 255))
            else:
                draw.rectangle([x0, y2, x1, y2 + 21], fill = inner_c)
                if show_ovr and rec_t['did_ovr']:
                    draw.rectangle([x0, y2, x1, y2 + 3], fill = OVERRIDE_GOLD if past else dim_rgb(OVERRIDE_GOLD, 0.45))
                if rec_t['is_sup_start'] and t_i > 0:
                    draw.line([(x0, y2), (x0, y2 + 7)], fill = OVERRIDE_GOLD if (show_ovr and rec_t['did_ovr']) else (240, 244, 252))

        if is_super:
            draw.line([(tx0, y2 + 12), (tx0 + tw, y2 + 12)], fill = (20, 24, 33))

        cur_x = tx0 + int((frame_idx + 0.5) * tw / total_frames)
        draw.line([(cur_x, y2 - 3), (cur_x, y2 + (27 if is_super else 24))], fill = (255, 255, 255), width = 2)

        # bottom color legend

        if is_super:
            legend = (
                (DT_BAND_RGB[0], 'Flow2 K=1', 14), (DT_BAND_RGB[1], 'K=2', 118),
                (DT_BAND_RGB[2], 'K=3', 186), (DT_BAND_RGB[3], 'K=4', 254), (OVERRIDE_GOLD, 'flow override', 344)
            )
            ly0 = 513
        elif show_ovr:
            legend = (
                (DT_BAND_RGB[0], 'dt 1-4', 14), (DT_BAND_RGB[1], 'dt 5-8', 108),
                (DT_BAND_RGB[2], 'dt 9-12', 202), (DT_BAND_RGB[3], 'dt 13-16', 304), (OVERRIDE_GOLD, 'policy override', 418)
            )
            ly0 = 511
        else:
            legend = (
                (DT_BAND_RGB[0], 'dt 1-4 fine', 14), (DT_BAND_RGB[1], 'dt 5-8 tact', 162),
                (DT_BAND_RGB[2], 'dt 9-12 glide', 310), (DT_BAND_RGB[3], 'dt 13-16 cruise', 458)
            )
            ly0 = 511

        for swatch_c, label_str, lx0 in legend:
            draw.rectangle([lx0, ly0, lx0 + 10, ly0 + 10], fill = swatch_c)
            draw.text((lx0 + 14, ly0), label_str, fill = OVERRIDE_GOLD if swatch_c == OVERRIDE_GOLD else (178, 186, 202), font = font)

        frames.append(np.asarray(canvas))

    mp4_path, gif_path = Path(mp4_path).resolve(), Path(gif_path).resolve()
    mp4_path.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(str(mp4_path), frames, fps = 50, macro_block_size = 1)
    imageio.mimsave(str(gif_path), frames[::2], fps = 25, loop = 0)

    return dict(
        seed = seed, return_val = ep_return, decisions = dec_idx,
        frames = total_frames, mp4 = str(mp4_path), gif = str(gif_path)
    )

@torch.no_grad()
def evaluate_policy(
    actor: BetaActor | None = None,
    *,
    banks: tuple[FlowChunkBank, ...] = (),
    zero_noise = False,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    num_episodes = 32,
    seed = 7000,
    video_folder = None,
    video_prefix = 'steered',
    mode_label = 'dct-8d',
    num_extra_videos = 0
):
    if exists(actor):
        actor.eval()

    is_base = len(banks) == 0
    is_super = len(banks) > 1
    env = gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True)

    for b in reversed(banks):
        env = FlowChunkWrapper(env, b, eval_mode = True)

    max_h = 1 if is_base else banks[0].max_chunk_size
    horizons = tuple(range(1, max_h + 1))

    returns, dts, plan_dts, ovrs, inner_ovrs, outer_ks = [], [], [], [], [], []
    dec_by_k = {k: 0 for k in horizons}
    step_by_k = {k: 0 for k in horizons}
    landings, aborts = 0, 0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        torch.manual_seed(ep_seed)
        np.random.seed(ep_seed)

        obs, _ = env.reset(seed = ep_seed)
        ep_ret, last_r, done = 0., 0., False

        while not done:
            state_t = torch.from_numpy(obs).float().to(ppo_device)
            action = actor.greedy(state_t).cpu().numpy() if exists(actor) else None

            if is_base:
                obs, reward, term, trunc, _ = env.step(action)
                last_r = float(reward)
                ep_ret += last_r
            else:
                obs, _, term, trunc, info = env.step(action, zero_noise = zero_noise)
                prim_steps = float(info['prim_steps'])
                exec_k = max(1, min(max_h, int(round(info['exec_chunks']))))
                last_r = float(info['last_reward'])
                ep_ret += float(info['raw_reward'])

                dts.append(prim_steps)
                plan_dts.append(float(info['planned_len']))
                ovrs.append(float(info['overridden']))
                inner_ovrs.append(float(info['inner_overridden']))
                outer_ks.append(float(exec_k))
                dec_by_k[exec_k] += 1
                step_by_k[exec_k] += int(round(prim_steps))
                aborts += int(round(float(info['aborted'])))

            done = term or trunc

        returns.append(ep_ret)
        landings += int(term and last_r >= 50.)

    env.close()
    arr = np.asarray(returns)

    if is_base:
        return dict(
            mean = float(arr.mean()),
            std = float(arr.std()),
            median = float(np.median(arr)),
            land_rate = 100. * landings / num_episodes,
            avg_dt = 1.
        )

    ranked_idx = np.argsort(arr)
    pick_seed = int(seed + (ranked_idx[-1] if exists(actor) else ranked_idx[len(ranked_idx) // 2]))

    video_meta, extra_videos = None, []
    if exists(video_folder):
        out_dir = Path(video_folder)
        out_dir.mkdir(parents = True, exist_ok = True)
        render_kw = dict(
            student = actor, zero_noise = zero_noise, mode_label = mode_label,
            env_name = env_name, max_episode_steps = max_episode_steps
        )
        video_meta = render_hud_video(banks, pick_seed, out_dir / f'{video_prefix}.mp4', out_dir / f'{video_prefix}.gif', **render_kw)

        for v_i in range(1, min(num_extra_videos, len(ranked_idx) - 1) + 1):
            extra_seed = int(seed + ranked_idx[-1 - v_i])
            stem = f'{video_prefix}-{v_i}'
            extra_videos.append(render_hud_video(banks, extra_seed, out_dir / f'{stem}.mp4', out_dir / f'{stem}.gif', **render_kw))

    total_decs, total_steps = max(1, sum(dec_by_k.values())), max(1, sum(step_by_k.values()))
    summary = dict(
        mean = float(arr.mean()),
        std = float(arr.std()),
        median = float(np.median(arr)),
        min = float(arr.min()),
        best = float(arr.max()),
        land_rate = 100. * landings / num_episodes,
        avg_dt = float(np.mean(dts)) if dts else 1.,
        avg_planned_dt = float(np.mean(plan_dts)) if plan_dts else 1.,
        override_rate = 100. * float(np.mean(ovrs)) if ovrs else 0.,
        aborts_per_ep = aborts / max(1, num_episodes),
        dec_frac = {str(k): 100. * dec_by_k[k] / total_decs for k in horizons},
        step_frac = {str(k): 100. * step_by_k[k] / total_steps for k in horizons},
        video = video_meta,
        extra_videos = extra_videos
    )

    if is_super:
        summary.update(
            avg_super_k = float(np.mean(outer_ks)) if outer_ks else 1.,
            inner_override_rate = 100. * float(np.mean(inner_ovrs)) if inner_ovrs else 0.
        )

    return summary

# recursive multi-stage policy + flow chunk bank training

@torch.no_grad()
def harvest_chunks(
    envs: list[gym.Env],
    banks: tuple[FlowChunkBank, ...],
    actor: BetaActor,
    buffer: ChunkReplayBuffer,
    *,
    rollout_steps = 64,
    seed = 5500
):
    actor.eval()

    with torch.random.fork_rng():
        seed_all(seed)
        obs = np.stack([env.reset(seed = seed + i)[0] for i, env in enumerate(envs)]).astype(np.float32)
        state = torch.from_numpy(obs).float().to(ppo_device)

        for temp in (0.15, 0.35, 0.60):
            states, next_states, actions, rewards, terms, truncs, durs = ([] for _ in range(7))

            for _ in range(rollout_steps):
                action = actor.sample_temp(state, temperature = temp)
                true_next, disc_r, term, trunc, info = step_envs(envs, obs, action, banks = banks, eval_mode = (temp <= 0.2))
                done = term | trunc
                next_obs = true_next.copy()

                for i in np.flatnonzero(done):
                    next_obs[i] = np.asarray(envs[i].reset()[0], dtype = np.float32)

                states.append(state.cpu())
                next_states.append(torch.from_numpy(true_next).float())
                actions.append(action.cpu())
                rewards.append(torch.from_numpy(disc_r / 100.).float())
                terms.append(torch.from_numpy(term).bool())
                truncs.append(torch.from_numpy(trunc).bool())
                durs.append(torch.from_numpy(info['prim_steps']).float())

                obs = next_obs
                state = torch.from_numpy(obs).float().to(ppo_device)

            buffer.add_rollout(
                stack(states, dim = 1),
                stack(next_states, dim = 1),
                stack(actions, dim = 1),
                stack(rewards, dim = 1),
                stack(terms, dim = 1),
                stack(truncs, dim = 1),
                durations = stack(durs, dim = 1),
                stride = 1
            )

def train_policy_stage(
    *,
    stage_label: str,
    banks: tuple[FlowChunkBank, ...] = (),
    next_flow_trainer: FlowChunkTrainer | None = None,
    next_buffer: ChunkReplayBuffer | None = None,
    buffer_stride = 1,
    flow_warmup_updates = 1,
    flow_train_every = 3,
    flow_steps_per_iter = 24,
    flow_consolidate_steps = 480,
    flow_batch_size = 128,
    harvest_final = False,
    stop_return: float | None = None,
    env_name = 'LunarLander-v3',
    num_envs = 16,
    updates = 165,
    rollout_steps = 64,
    max_episode_steps = 600,
    actor_hidden_dim = 108,
    init_conc = 6.,
    lr = 1e-3,
    critic_lr = 2e-3,
    val_every = 15,
    log_every = 15,
    eval_episodes = 32,
    seed = 42,
    video_folder = None,
    num_extra_videos = 0,
    smoke = False
):
    seed_all(seed)
    depth = len(banks)
    is_base = depth == 0
    is_super = depth > 1
    top_bank = banks[0] if not is_base else None

    envs = [gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True) for _ in range(num_envs)]
    obs = np.stack([env.reset(seed = seed + i)[0] for i, env in enumerate(envs)]).astype(np.float32)

    state_dim = obs.shape[-1]
    action_dim = envs[0].action_space.shape[0] if is_base else top_bank.latent_dim
    aux_dims = 0 if is_base else top_bank.aux_dim

    actor = BetaActor(
        state_dim,
        actor_hidden_dim,
        action_dim,
        bounds = (-1., 1.),
        init_conc = init_conc,
        aux_dims = aux_dims,
        aux_init_conc = 0.5
    ).to(ppo_device)

    critic = Critic(state_dim, 192).to(ppo_device)
    opt_actor = torch.optim.AdamW(actor.parameters(), lr = lr)
    opt_critic = torch.optim.AdamW(critic.parameters(), lr = critic_lr)

    state = torch.from_numpy(obs).float().to(ppo_device)
    running = np.zeros(num_envs, dtype = np.float32)
    recent_rets, recent_dts, recent_ovrs, recent_ks, history = [], [], [], [], []
    flow_stats = dict(actor_flow = 0., critic_flow = 0., mean_q = 0.)

    best_val = -float('inf')
    best_state = deepcopy(actor.state_dict())
    best_critic_state = deepcopy(critic.state_dict())
    stopped_early = False
    env_steps, decisions, start_time = 0, 0, time.time()

    for upd in range(1, updates + 1):
        progress = (upd - 1) / max(1, updates)
        ent_coef = schedule_at(BETA_ENTROPY_SCHEDULE, progress)

        for group in opt_actor.param_groups:
            group['lr'] = lr * schedule_at(LR_SCALE_SCHEDULE, progress)

        states, next_states, actions, log_probs, rewards, terms, truncs, durs = ([] for _ in range(8))
        actor.eval()

        for _ in range(rollout_steps):
            with torch.no_grad():
                action, lp = actor.sample(state, sum_action_dim = is_base)

            true_next, disc_r, term, trunc, info = step_envs(envs, obs, action, banks = banks, eval_mode = False)
            done = term | trunc
            next_obs = true_next.copy()

            running += info['raw_reward']
            env_steps += int(info['prim_steps'].sum())
            decisions += num_envs
            recent_dts.extend(info['prim_steps'].tolist())
            recent_ovrs.extend(info['overridden'].tolist())
            if is_super:
                recent_ks.extend(info['exec_chunks'].tolist())

            for i in np.flatnonzero(done):
                recent_rets.append(float(running[i]))
                running[i] = 0.
                next_obs[i] = np.asarray(envs[i].reset()[0], dtype = np.float32)

            states.append(state)
            next_states.append(torch.from_numpy(true_next).float().to(ppo_device))
            actions.append(action)
            log_probs.append(lp)
            rewards.append(torch.from_numpy(disc_r / 100.).float().to(ppo_device))
            terms.append(torch.from_numpy(term).bool().to(ppo_device))
            truncs.append(torch.from_numpy(trunc).bool().to(ppo_device))
            durs.append(torch.from_numpy(info['prim_steps']).float().to(ppo_device))

            obs = next_obs
            state = torch.from_numpy(obs).float().to(ppo_device)

        recent_dts, recent_ovrs, recent_ks = recent_dts[-2048:], recent_ovrs[-2048:], recent_ks[-2048:]
        states_t, next_states_t = stack(states, dim = 1), stack(next_states, dim = 1)
        actions_t, rewards_t = stack(actions, dim = 1), stack(rewards, dim = 1)
        terms_t, truncs_t, durs_t = stack(terms, dim = 1), stack(truncs, dim = 1), stack(durs, dim = 1)

        ret32 = float(np.mean(recent_rets[-32:])) if recent_rets else float('nan')
        stopped_early = exists(stop_return) and len(recent_rets) >= 16 and ret32 >= stop_return
        stopped_state = deepcopy(actor.state_dict()) if stopped_early else None

        actor.train()
        critic.train()

        ppo_update(
            actor,
            critic,
            opt_actor,
            opt_critic,
            states = states_t,
            next_states = next_states_t,
            actions = actions_t,
            old_log_probs = stack(log_probs, dim = 1),
            rewards = rewards_t,
            terminated = terms_t,
            truncated = truncs_t,
            durations = None if is_base else durs_t,
            entropy_coef = ent_coef
        )

        if stopped_early:
            actor.load_state_dict(stopped_state)

        if exists(next_buffer):
            with torch.random.fork_rng():
                next_buffer.add_rollout(
                    states_t, next_states_t, actions_t, rewards_t, terms_t, truncs_t,
                    durations = None if is_base else durs_t,
                    stride = buffer_stride
                )
                can_train_flow = (
                    exists(next_flow_trainer)
                    and upd >= flow_warmup_updates
                    and divisible_by(upd, flow_train_every)
                    and next_buffer.size >= flow_batch_size * 2
                )
                if can_train_flow:
                    flow_stats = next_flow_trainer.learn(
                        next_buffer, critic, steps = flow_steps_per_iter, batch_size = flow_batch_size
                    )

        avg_dt = float(np.mean(recent_dts[-512:])) if recent_dts else 1.
        avg_ovr = 100. * float(np.mean(recent_ovrs[-512:])) if recent_ovrs else 0.

        if divisible_by(upd, val_every) or upd == updates or stopped_early:
            val = evaluate_policy(
                actor,
                banks = banks,
                env_name = env_name,
                max_episode_steps = max_episode_steps,
                num_episodes = 12 if is_base else 16,
                seed = seed + (3000 if is_base else 4000)
            )
            if val['mean'] >= best_val:
                best_val = val['mean']
                best_state = deepcopy(actor.state_dict())
                best_critic_state = deepcopy(critic.state_dict())

        row = dict(
            update = upd, env_steps = env_steps, decisions = decisions,
            ret32 = ret32, avg_dt = avg_dt, override_rate = avg_ovr, best_val = best_val
        )
        if exists(next_flow_trainer):
            row.update(**flow_stats)
        if is_super:
            tail_ks = np.asarray(recent_ks[-512:])
            row['avg_super_k'] = float(tail_ks.mean()) if tail_ks.size > 0 else 1.
        history.append(row)

        if divisible_by(upd, log_every) or upd == updates or stopped_early:
            has_flow = exists(next_flow_trainer) and flow_stats['actor_flow'] > 0.
            k_str = f' | avg_K={row["avg_super_k"]:.2f}' if is_super else ''
            ovr_str = f' | ovr={avg_ovr:4.1f}%' if (not is_base and top_bank.override_chunk_len) else ''
            flow_str = f' | flow={flow_stats["actor_flow"]:.3f}/{flow_stats["critic_flow"]:.3f}' if has_flow else ''
            elapsed = time.time() - start_time
            print(
                f'  [{stage_label} upd {upd:3d}/{updates}] env={env_steps // 1000:4d}k (decs={decisions // 1000:3d}k) | '
                f'ret32={ret32:+6.1f} | val_best={best_val:+6.1f}{k_str} | dt={avg_dt:5.2f}{ovr_str}{flow_str} [{elapsed:.1f}s]',
                flush = True
            )

        if stopped_early:
            print(
                f'  [{stage_label} early stop] stopped unconverged policy at upd {upd} '
                f'(ret32={ret32:+.1f} >= {stop_return:+.1f})',
                flush = True
            )
            break

    if not stopped_early:
        actor.load_state_dict(best_state)
        critic.load_state_dict(best_critic_state)

    if exists(next_flow_trainer) and exists(next_buffer):
        if harvest_final:
            harvest_chunks(envs, banks, actor, next_buffer, rollout_steps = 12 if smoke else 64, seed = seed + 5500)
        with torch.random.fork_rng():
            flow_stats = next_flow_trainer.learn(
                next_buffer, critic, steps = flow_consolidate_steps,
                batch_size = flow_batch_size, recent_frac = 0.65 if harvest_final else 0.60
            )
        print(
            f'  [{stage_label} flow consolidated] chunks={next_buffer.size} | '
            f'a_flow={flow_stats["actor_flow"]:.3f} | c_flow={flow_stats["critic_flow"]:.3f} | mean_q={flow_stats["mean_q"]:+.1f}',
            flush = True
        )

    for env in envs:
        env.close()

    final_eval = evaluate_policy(
        actor,
        banks = banks,
        env_name = env_name,
        max_episode_steps = max_episode_steps,
        num_episodes = eval_episodes,
        seed = seed + (6000 if is_base else 9000),
        video_folder = None if is_base else video_folder,
        video_prefix = f'student-{stage_label}',
        mode_label = stage_label,
        num_extra_videos = 0 if is_base else num_extra_videos
    )

    if is_base:
        final_eval.update(stopped_upd = history[-1]['update'], stopped_ret32 = history[-1]['ret32'])
        print(
            f'  [{stage_label} eval] {final_eval["mean"]:+.1f} +/- {final_eval["std"]:.1f} '
            f'(median {final_eval["median"]:+.1f}, ret32 {final_eval["stopped_ret32"]:+.1f}) | land={final_eval["land_rate"]:.1f}%',
            flush = True
        )
    else:
        k_eval = f' | avg_K={final_eval["avg_super_k"]:.2f}' if is_super else ''
        print(
            f'  [{stage_label} eval] {final_eval["mean"]:+.1f} +/- {final_eval["std"]:.1f} (median {final_eval["median"]:+.1f}) | '
            f'land={final_eval["land_rate"]:.1f}%{k_eval} | avg_dt={final_eval["avg_dt"]:.2f} | ovr={final_eval["override_rate"]:.1f}%',
            flush = True
        )

    return actor, final_eval, history, flow_stats

# main

def main(
    env_name = 'LunarLander-v3',
    max_chunk_size = 16,
    dct_modes = 4,
    lerp_noise = False,
    override_chunk_len = False,
    super_chunk = False,
    super_chunk_size = 4,
    super_dct_modes = 2,
    super_updates = 150,
    super_rollout_steps = 48,
    num_envs = 16,
    max_episode_steps = 600,
    base_updates = 95,
    base_rollout_steps = 384,
    teacher_stop_return: float | None = 0.,
    buffer_capacity = 49152,
    super_buffer_capacity = 32768,
    flow_train_every = 3,
    flow_steps_per_iter = 24,
    flow_consolidate_steps = 480,
    super_consolidate_steps = 600,
    flow_batch_size = 128,
    flow_hidden_dim = 128,
    flow_depth = 3,
    flow_heads = 4,
    flow_lr = 3e-4,
    corr_alpha = 0.7,
    xm_candidates = 2,
    actor_flow_steps = 4,
    q_temp = 0.025,
    reassess_chunk = True,
    reassess_every = 4,
    reassess_threshold = 0.25,
    student_updates = 165,
    macro_rollout_steps = 64,
    eval_episodes = 32,
    seed = 42,
    render = True,
    num_extra_videos = 4,
    video_folder = './flow-chunk-steering-videos',
    smoke = False
):
    if super_chunk:
        override_chunk_len = True

    if smoke:
        num_envs, base_updates, base_rollout_steps = 4, 4, 64
        buffer_capacity, super_buffer_capacity = 2048, 1024
        flow_train_every, flow_steps_per_iter, flow_consolidate_steps, super_consolidate_steps, flow_batch_size = 2, 4, 8, 8, 32
        student_updates, macro_rollout_steps, eval_episodes, num_extra_videos = 4, 12, 4, 0
        super_updates, super_rollout_steps = 4, 8

    seed_all(seed)

    probe_env = gym.make(env_name, continuous = True)
    state_dim = probe_env.observation_space.shape[0]
    prim_action_dim = probe_env.action_space.shape[0]
    probe_env.close()

    flow_kw = dict(
        state_dim = state_dim,
        hidden_dim = flow_hidden_dim,
        depth = flow_depth,
        heads = flow_heads,
        lr = flow_lr,
        xm_candidates = xm_candidates
    )
    bank_kw = dict(
        lerp_noise = lerp_noise,
        override_chunk_len = override_chunk_len,
        actor_flow_steps = actor_flow_steps,
        dynamic_chunk = True,
        q_temp = q_temp,
        reassess_chunk = reassess_chunk,
        reassess_threshold = reassess_threshold
    )
    vid_dir = video_folder if render else None
    eval_kw = dict(env_name = env_name, max_episode_steps = max_episode_steps, num_episodes = eval_episodes, video_folder = vid_dir)
    stage_kw = dict(
        env_name = env_name,
        num_envs = num_envs,
        max_episode_steps = max_episode_steps,
        eval_episodes = eval_episodes,
        flow_train_every = flow_train_every,
        flow_steps_per_iter = flow_steps_per_iter,
        flow_batch_size = flow_batch_size,
        video_folder = vid_dir,
        num_extra_videos = num_extra_videos if render else 0,
        smoke = smoke
    )

    def eval_unsteered_baselines(banks, base_seed, specs, out_dict):
        for key, is_zero, prefix, mode_lbl in specs:
            ev = evaluate_policy(
                None, banks = banks, zero_noise = is_zero,
                seed = base_seed, video_prefix = prefix, mode_label = mode_lbl, **eval_kw
            )
            out_dict[key] = ev
            k_tag = f' | avg_K={ev["avg_super_k"]:.2f}' if len(banks) > 1 else ''
            print(
                f'  [{prefix} eval] {ev["mean"]:+.1f} +/- {ev["std"]:.1f} '
                f'(median {ev["median"]:+.1f}) | land={ev["land_rate"]:.1f}%{k_tag} | avg_dt={ev["avg_dt"]:.2f}',
                flush = True
            )

    # stage 1: dt = 1 teacher + layer-1 causal flow actor/critic (XM k = 2)

    buffer_1 = ChunkReplayBuffer(buffer_capacity, max_chunk_size, state_dim, prim_action_dim)
    flow_trainer_1 = FlowChunkTrainer(
        num_actions = prim_action_dim,
        max_chunk_size = max_chunk_size,
        corr_alpha = corr_alpha,
        **flow_kw
    ).to(flow_device)

    print(
        f'[stage 1] dt=1 PPO ({ppo_device}) + Layer-1 Causal Flow '
        f'(H1={max_chunk_size}, XM k={xm_candidates}, dev={flow_device})',
        flush = True
    )

    _, teacher_eval, stage1_history, _ = train_policy_stage(
        stage_label = 'dt1-teacher',
        next_flow_trainer = flow_trainer_1,
        next_buffer = buffer_1,
        buffer_stride = 2,
        flow_consolidate_steps = flow_consolidate_steps,
        stop_return = teacher_stop_return,
        updates = base_updates,
        rollout_steps = base_rollout_steps,
        actor_hidden_dim = 108,
        init_conc = 5.,
        critic_lr = 2.5e-3,
        seed = seed,
        **stage_kw
    )

    # stage 2: frozen layer-1 flow skill bank + reborn first student (+ online layer-2 flow trainer)

    bank_1 = FlowChunkBank(
        flow_trainer_1.ema_actor.ema_model,
        flow_trainer_1.ema_critic.ema_model,
        dct_modes = dct_modes,
        corr_alpha = corr_alpha,
        reassess_every = reassess_every,
        **bank_kw
    ).to(ppo_device)

    results = dict(
        config = dict(
            max_chunk_size = max_chunk_size,
            dct_modes = dct_modes,
            lerp_noise = lerp_noise,
            override_chunk_len = override_chunk_len,
            super_chunk = super_chunk,
            super_chunk_size = super_chunk_size,
            super_dct_modes = super_dct_modes,
            teacher_stop_return = teacher_stop_return,
            q_temp = q_temp,
            xm_candidates = xm_candidates,
            seed = seed
        ),
        teacher = teacher_eval,
        stage1_history = stage1_history,
        students = {},
        histories = {}
    )

    eval_unsteered_baselines(
        (bank_1,),
        seed + 8000,
        (
            ('zero_steer', True, 'zero-steer', 'zero-steer (z=0)'),
            ('unsteered', False, 'unsteered-prior', 'unsteered prior')
        ),
        results
    )

    label = '-'.join([f'dct-{bank_1.dct_dim}d', 'lerp' if lerp_noise else 'dsrl'] + (['ovr'] if override_chunk_len else []))
    flow_trainer_2, buffer_2 = None, None

    if super_chunk:
        with torch.random.fork_rng():
            buffer_2 = ChunkReplayBuffer(super_buffer_capacity, super_chunk_size, state_dim, bank_1.latent_dim)
            flow_trainer_2 = FlowChunkTrainer(
                num_actions = bank_1.latent_dim,
                max_chunk_size = super_chunk_size,
                corr_alpha = 0.5,
                **flow_kw
            ).to(flow_device)

    print(f'\n[stage 2 | {label}] reborn student dim={bank_1.latent_dim} steering layer-1 flow bank (H1={max_chunk_size})', flush = True)

    _, student1_eval, student1_hist, flow2_stats = train_policy_stage(
        stage_label = label,
        banks = (bank_1,),
        next_flow_trainer = flow_trainer_2,
        next_buffer = buffer_2,
        flow_warmup_updates = max(2, student_updates // 2),
        flow_consolidate_steps = super_consolidate_steps,
        harvest_final = super_chunk,
        updates = student_updates,
        rollout_steps = macro_rollout_steps,
        seed = seed + 400,
        **stage_kw
    )

    results['students']['layer1'] = student1_eval
    results['histories']['layer1'] = student1_hist

    # stage 3: frozen layer-2 flow skill bank (compressing layer-1 steering chunks) + reborn second student

    if super_chunk:
        bank_2 = FlowChunkBank(
            flow_trainer_2.ema_actor.ema_model,
            flow_trainer_2.ema_critic.ema_model,
            dct_modes = super_dct_modes,
            corr_alpha = 0.5,
            reassess_every = max(1, super_chunk_size // 2),
            **bank_kw
        ).to(ppo_device)

        eval_unsteered_baselines(
            (bank_2, bank_1),
            seed + 10000,
            (
                ('super_zero_steer', True, 'super-zero-steer', 'flow2 zero-steer'),
                ('super_unsteered', False, 'super-unsteered-prior', 'flow2 unsteered prior')
            ),
            results
        )

        super_label = f'flow2-h{super_chunk_size}-{bank_2.dct_dim}d' + ('-ovr' if override_chunk_len else '')
        print(
            f'\n[stage 3 | {super_label}] reborn student dim={bank_2.latent_dim} '
            f'steering 2-stack flow bank (H2={super_chunk_size} x H1={max_chunk_size})',
            flush = True
        )

        _, super_eval, super_hist, _ = train_policy_stage(
            stage_label = super_label,
            banks = (bank_2, bank_1),
            updates = super_updates,
            rollout_steps = super_rollout_steps,
            actor_hidden_dim = 128,
            seed = seed + 900,
            **stage_kw
        )

        results['layer2_flow'] = dict(buffer_size = buffer_2.size, **flow2_stats)
        results['students']['layer2'] = super_eval
        results['histories']['layer2'] = super_hist

    out_dir = Path(video_folder)
    out_dir.mkdir(parents = True, exist_ok = True)
    with open(out_dir / 'flow_steering_summary.json', 'w') as f:
        json.dump(results, f, indent = 2)

    def print_tier(title, ev, show_k = False):
        k_tag = f', avg_K = {ev["avg_super_k"]:.2f}' if show_k else ''
        print(
            f'  {title:33s}: {ev["mean"]:+6.1f} '
            f'(median {ev["median"]:+6.1f}, land {ev["land_rate"]:5.1f}%{k_tag}, avg_dt = {ev["avg_dt"]:.2f})',
            flush = True
        )

    h2_dim = super_chunk_size if super_chunk else 0
    print(f'\n=== Summary ({env_name} Continuous | H1 = {max_chunk_size} | H2 = {h2_dim} | XM k = {xm_candidates}) ===', flush = True)
    print_tier('Stage 1 Teacher (dt = 1)', teacher_eval)
    print_tier('Layer 1 Zero-Steer (z = 0)', results['zero_steer'])
    print_tier('Layer 1 Unsteered Prior', results['unsteered'])
    print_tier(f'Stage 2 Student 1 ({label:14s})', student1_eval)

    if super_chunk:
        print_tier('Layer 2 Flow Zero-Steer (z2 = 0)', results['super_zero_steer'], show_k = True)
        print_tier('Layer 2 Flow Unsteered Prior', results['super_unsteered'], show_k = True)
        print_tier(f'Stage 3 Student 2 ({super_label:14s})', results['students']['layer2'], show_k = True)

if __name__ == '__main__':
    fire.Fire(main)
