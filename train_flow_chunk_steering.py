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
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

import torch
from torch import nn, Tensor, arange, cat, stack
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import LayerNorm, Linear, Module, RMSNorm, Sequential, SiLU
from torch.nn.utils import clip_grad_norm_

import einx
from einops import rearrange, repeat

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
LR_SCALE_SCHEDULE = (1., 0.85, 0.55, 0.30, 0.15, 0.08)

DT_BAND_RGB = ((72, 149, 239), (42, 157, 143), (244, 162, 97), (230, 57, 70))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def append_dims(t, dims):
    return t.reshape(*t.shape, *((1,) * dims))

def schedule_at(schedule, progress):
    idx = min(int(progress * len(schedule)), len(schedule) - 1)
    return schedule[max(0, idx)]

def dt_color_rgb(dt, max_chunk_size = 16):
    band = min(3, max(0, int((dt - 1) * 4 // max(1, max_chunk_size))))
    return DT_BAND_RGB[band]

def update_network_(loss, optimizer, max_grad_norm = None):
    optimizer.zero_grad()
    loss.backward()

    if exists(max_grad_norm):
        params = [p for group in optimizer.param_groups for p in group['params']]
        clip_grad_norm_(params, max_grad_norm)

    optimizer.step()

def true_next_obs(next_obs, done, info):
    if not done.any() or 'final_obs' not in info:
        return next_obs

    out = next_obs.copy()

    for i in np.flatnonzero(done):
        if exists(info['final_obs'][i]):
            out[i] = info['final_obs'][i]

    return out

def make_orthonormal_dct(n):
    k = arange(n, dtype = torch.float32)[:, None]
    t = arange(n, dtype = torch.float32)[None, :]

    mat = torch.cos(math.pi / n * (t + 0.5) * k)
    mat[0] *= math.sqrt(1. / n)

    if n > 1:
        mat[1:] *= math.sqrt(2. / n)

    return mat

def sample_pyoco_noise(batch, seq_len, dim, corr_alpha, device):
    if corr_alpha <= 0.:
        return torch.randn((batch, seq_len, dim), device = device)

    z_shared = torch.randn((batch, 1, dim), device = device)
    z_indep = torch.randn((batch, seq_len, dim), device = device)

    return corr_alpha * z_shared + math.sqrt(max(0., 1. - corr_alpha ** 2)) * z_indep

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
        dev = next(self.model.parameters()).device

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

    rep = lambda t: repeat(t, 'b ... -> (b k) ...', k = candidates) if torch.is_tensor(t) and t.ndim > 0 and t.shape[0] == b else t
    data_k, times_k, mask_k = rep(data), rep(times), rep(mask)
    cond_k = {k: rep(v) for k, v in cond_kwargs.items()}
    noise_k = noise_fn(b * candidates) if exists(noise_fn) else None

    raw = flow_model(data_k, noise = noise_k, times = times_k, loss_reduction = 'none', **cond_k)
    per_tok = raw.mean(dim = -1) if raw.ndim > mask_k.ndim else raw
    cand_losses = rearrange(masked_mean(per_tok, mask_k, dim = -1), '(b k) -> b k', b = b, k = candidates)

    return cand_losses.amin(dim = -1).mean()

# causal transformer flow actor & multi-chunk value critic

class TransformerFlowActor(Module):
    """ causal transformer velocity field over action chunks """

    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        dim_hidden = 128,
        depth = 3,
        heads = 4,
        max_chunk_size = 16
    ):
        super().__init__()
        self.num_cont_actions = num_cont_actions
        self.max_chunk_size = max_chunk_size

        self.state_proj = Linear(dim_state, dim_hidden)
        self.action_proj = Linear(num_cont_actions, dim_hidden)
        self.time_proj = Sequential(Linear(1, dim_hidden), nn.GELU(), Linear(dim_hidden, dim_hidden))
        self.pos_emb = nn.Embedding(max_chunk_size, dim_hidden)

        self.decoder = Decoder(dim = dim_hidden, depth = depth, heads = heads)
        self.to_flow = Sequential(LayerNorm(dim_hidden), Linear(dim_hidden, num_cont_actions))

    def forward(self, noised_actions, *, cond_state, times):
        noised_actions, inverse_seq = pack_with_inverse(noised_actions, 'b * d')
        seq = noised_actions.shape[1]

        state_tokens = rearrange(self.state_proj(cond_state), 'b d -> b 1 d')
        action_tokens = (
            self.action_proj(noised_actions) +
            self.time_proj(rearrange(times, 'b -> b 1 1')) +
            self.pos_emb(arange(seq, device = noised_actions.device))
        )

        out = self.decoder(cat((state_tokens, action_tokens), dim = 1))
        return inverse_seq(self.to_flow(out[:, 1:]))

class TransformerFlowCritic(Module):
    """ causal multi-chunk value critic with flow matching & single-pass distillation """

    def __init__(
        self,
        *,
        dim_state,
        num_cont_actions,
        dim_hidden = 128,
        depth = 3,
        heads = 4,
        max_seq_len = 16
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.state_proj = Linear(dim_state, dim_hidden)
        self.action_proj = Linear(num_cont_actions, dim_hidden)
        self.value_proj = Linear(1, dim_hidden)
        self.time_proj = Sequential(Linear(1, dim_hidden), nn.GELU(), Linear(dim_hidden, dim_hidden))
        self.distill_token = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
        self.pos_emb = nn.Embedding(max_seq_len, dim_hidden)

        self.decoder = Decoder(dim = dim_hidden, depth = depth, heads = heads)
        self.to_values = Sequential(LayerNorm(dim_hidden), Linear(dim_hidden, 1))

    def forward(self, values = None, *, cond_state, cont_actions, times = None, is_distill = False):
        cont_actions, inverse_seq = pack_with_inverse(cont_actions, 'b * d')
        seq = cont_actions.shape[1]

        state_tokens = rearrange(self.state_proj(cond_state), 'b d -> b 1 d')
        action_tokens = self.action_proj(cont_actions)

        if is_distill:
            action_tokens = einx.add('b s d, 1 1 d', action_tokens, self.distill_token)
        else:
            val_in = rearrange(values, 'b s -> b s 1') if values.ndim == 2 else values
            action_tokens = action_tokens + self.value_proj(val_in) + self.time_proj(rearrange(times, 'b -> b 1 1'))

        action_tokens = action_tokens + self.pos_emb(arange(seq, device = cont_actions.device))
        out = self.decoder(cat((state_tokens, action_tokens), dim = 1))

        return rearrange(inverse_seq(self.to_values(out[:, 1:])), '... 1 -> ...')

# chunk replay buffer & flow trainer

class ChunkReplayBuffer:
    def __init__(self, capacity, max_chunk_size, state_dim, num_actions):
        self.capacity = capacity
        self.max_chunk_size = max_chunk_size

        self.states = np.zeros((capacity, state_dim), dtype = np.float32)
        self.actions = np.zeros((capacity, max_chunk_size, num_actions), dtype = np.float32)
        self.rewards = np.zeros((capacity, max_chunk_size), dtype = np.float32)
        self.next_states = np.zeros((capacity, max_chunk_size, state_dim), dtype = np.float32)
        self.terminated = np.zeros((capacity, max_chunk_size), dtype = np.bool_)
        self.valid_lens = np.zeros((capacity,), dtype = np.int64)

        self.ptr = 0
        self.size = 0

    def add_rollout(self, states, next_states, actions, rewards, terminated, truncated, stride = 2):
        s_np, ns_np = states.cpu().numpy(), next_states.cpu().numpy()
        a_np, r_np = actions.cpu().numpy(), rewards.cpu().numpy()
        term_np = terminated.cpu().numpy()
        done_np = (terminated | truncated).cpu().numpy()

        n, rollout_len, _ = s_np.shape
        h = self.max_chunk_size

        for t in range(0, rollout_len - h + 1, stride):
            d_win = done_np[:, t:t + h]
            lens = np.where(d_win.any(axis = 1), d_win.argmax(axis = 1) + 1, h)
            idx = (np.arange(n) + self.ptr) % self.capacity

            self.states[idx] = s_np[:, t]
            self.actions[idx] = a_np[:, t:t + h]
            self.rewards[idx] = r_np[:, t:t + h]
            self.next_states[idx] = ns_np[:, t:t + h]
            self.terminated[idx] = term_np[:, t:t + h]
            self.valid_lens[idx] = lens

            self.ptr = (self.ptr + n) % self.capacity
            self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size, dev, recent_fraction = 0.55):
        win = max(batch_size, int(self.size * recent_fraction))

        if self.size < self.capacity:
            idx = np.random.randint(max(0, self.size - win), self.size, size = batch_size)
        else:
            idx = (self.ptr - np.random.randint(1, win + 1, size = batch_size)) % self.capacity

        return tuple(
            torch.from_numpy(arr[idx]).to(dev)
            for arr in (self.states, self.actions, self.rewards, self.next_states, self.terminated, self.valid_lens)
        )

class FlowChunkTrainer(Module):
    def __init__(
        self,
        *,
        state_dim,
        num_actions,
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

        self.ema_actor = EMA(self.actor, beta = ema_decay, include_online_model = False, update_every = 1, update_after_step = 0)
        self.ema_critic = EMA(self.critic, beta = ema_decay, include_online_model = False, update_every = 1, update_after_step = 0)

        self.actor_flow = NanoFlow(self.actor, data_shape = (max_chunk_size, num_actions))
        self.critic_flow = NanoFlow(self.critic, data_shape = (max_chunk_size,))
        self.ema_critic_flow = NanoFlow(self.ema_critic, data_shape = (max_chunk_size,))

        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr = lr, weight_decay = 1e-4)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr = lr, weight_decay = 1e-4)

        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

    def learn(self, buffer: ChunkReplayBuffer, value_net: Module, *, steps = 24, batch_size = 128, recent_frac = 0.55):
        dev = next(self.parameters()).device
        v_dev = next(value_net.parameters()).device
        self.train()

        gammas = self.gamma ** arange(self.max_chunk_size, device = dev, dtype = torch.float32)
        next_gammas = gammas * self.gamma
        a_losses, c_losses, q_means = [], [], []

        for _ in range(steps):
            s, act, rew, ns, term, lens = buffer.sample(batch_size, dev, recent_fraction = recent_frac)
            mask = mask_after(term, True) & lens_to_mask(lens, self.max_chunk_size)

            # causal flow actor with explorative modeling (k = xm_candidates)

            actor_loss = xm_flow_loss(
                self.actor_flow,
                act,
                mask = mask,
                candidates = self.xm_candidates,
                noise_fn = lambda n: sample_pyoco_noise(n, self.max_chunk_size, self.num_actions, self.corr_alpha, dev),
                cond_state = s
            )
            update_network_(actor_loss, self.opt_actor, self.max_grad_norm)

            # causal flow critic over SMDP prefix returns bootstrapped from Stage-1 value net

            with torch.no_grad():
                flat_ns = rearrange(ns, 'b s d -> (b s) d').to(v_dev)
                v_next = rearrange(value_net(flat_ns), '(b s) -> b s', b = batch_size).to(dev)

                alive_after = torch.cumprod((~term).float() * mask.float(), dim = 1)
                cum_rew = torch.cumsum(rew * mask.float() * gammas[None, :], dim = 1)
                target_q = (cum_rew + next_gammas[None, :] * alive_after * v_next).clamp(-5., 5.)

                flow_q = self.ema_critic_flow.sample(
                    steps = self.critic_distill_steps,
                    batch_size = batch_size,
                    data_shape = (self.max_chunk_size,),
                    cond_state = s,
                    cont_actions = act
                ).clamp(-5., 5.)

                distill_target = 0.5 * flow_q + 0.5 * target_q

            c_flow_loss = xm_flow_loss(
                self.critic_flow,
                target_q,
                mask = mask,
                candidates = self.xm_candidates,
                cond_state = s,
                cont_actions = act
            )

            pred_q = self.critic(cond_state = s, cont_actions = act, is_distill = True)
            c_dist_loss = masked_mean((pred_q - distill_target).square(), mask)

            update_network_(c_flow_loss + c_dist_loss, self.opt_critic, self.max_grad_norm)

            a_losses.append(actor_loss.item())
            c_losses.append(c_flow_loss.item())
            q_means.append(pred_q[mask].mean().item() * 100.)

        return dict(
            actor_flow = float(np.mean(a_losses)),
            critic_flow = float(np.mean(c_losses)),
            mean_q = float(np.mean(q_means))
        )

# frozen DCT-steered flow skill bank + adaptive chunking / value-reassessment wrapper

class FlowChunkBank(Module):
    """ frozen flow actor/critic skill bank steered in low-frequency DCT-II noise space """

    def __init__(
        self,
        actor_model: TransformerFlowActor,
        critic_model: TransformerFlowCritic,
        *,
        dct_modes = 4,
        corr_alpha = 0.7,
        lerp_noise = False,
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
        self.latent_dim = self.dct_dim + int(lerp_noise)

        self.corr_alpha = corr_alpha
        self.dct_scale = math.sqrt(self.max_chunk_size / self.dct_modes)
        self.actor_flow_steps = actor_flow_steps

        self.dynamic_chunk = dynamic_chunk
        self.dynamic_chunk_eps = dynamic_chunk_eps
        self.min_chunk_len = min_chunk_len
        self.q_temp = q_temp
        self.reassess_chunk = reassess_chunk
        self.reassess_every = reassess_every
        self.reassess_threshold = reassess_threshold
        self.gamma = gamma

        self.register_buffer('dct_mat', make_orthonormal_dct(self.max_chunk_size), persistent = False)

    def dct_to_noise_seq(self, dct_latent: Tensor):
        if dct_latent.ndim == 1:
            dct_latent = rearrange(dct_latent, 'd -> 1 d')

        z = rearrange(dct_latent, 'b (k a) -> b k a', k = self.dct_modes, a = self.num_actions)
        return self.dct_scale * torch.einsum('k h, b k a -> b h a', self.dct_mat[:self.dct_modes], z)

    def compose_noise(self, batch_size: int, latent_action: Tensor | None = None, *, zero_noise = False):
        dev = next(self.parameters()).device

        if zero_noise:
            return torch.zeros((batch_size, self.max_chunk_size, self.num_actions), device = dev), torch.zeros(batch_size, device = dev)

        init_noise = sample_pyoco_noise(batch_size, self.max_chunk_size, self.num_actions, self.corr_alpha, dev)

        if not exists(latent_action):
            return init_noise, torch.zeros(batch_size, device = dev)

        latent = latent_action.to(dev)
        if latent.ndim == 1:
            latent = rearrange(latent, 'd -> 1 d')

        if self.lerp_noise:
            dct_latent, alpha = latent[:, :-1], latent[:, -1].clamp(0., 1.)
            constituted = self.dct_to_noise_seq(dct_latent)
            return init_noise.lerp(constituted, append_dims(alpha, 2)), alpha

        constituted = self.dct_to_noise_seq(latent)
        return init_noise + constituted, torch.ones(batch_size, device = dev)

    @torch.no_grad()
    def critic_q(self, states: Tensor, action_chunks: Tensor):
        dev = next(self.parameters()).device
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
        dev = next(self.parameters()).device
        states_d = states.to(dev)
        b = states_d.shape[0]

        noise_seq, alphas = self.compose_noise(b, latent_action, zero_noise = zero_noise)

        chunks = self.actor_flow.sample(
            steps = self.actor_flow_steps,
            batch_size = b,
            noise = noise_seq,
            cond_state = states_d
        ).clamp_(-1., 1.)

        q_vals = self.critic_q(states_d, chunks)
        logits = q_vals / max(1e-4, self.q_temp)

        if self.min_chunk_len > 1:
            logits[:, :self.min_chunk_len - 1] = -1e9

        probs = logits.softmax(dim = -1)

        # adaptive action chunking via multi-chunk value estimation (Shin et al.)

        if not self.dynamic_chunk:
            lens = torch.full((b,), self.max_chunk_size, device = dev, dtype = torch.long)
        else:
            lens = Categorical(probs = probs).sample() + 1

            if not eval_mode and self.dynamic_chunk_eps > 0.:
                rand_mask = torch.rand(b, device = dev) < self.dynamic_chunk_eps
                rand_lens = torch.randint(self.min_chunk_len, self.max_chunk_size + 1, (b,), device = dev)
                lens = torch.where(rand_mask, rand_lens, lens)

        planned_q = q_vals.gather(1, (lens - 1)[:, None]).squeeze(1)
        out = (chunks.cpu().numpy(), lens.cpu().numpy(), planned_q.cpu().numpy(), alphas.cpu().numpy())

        return (*out, probs.cpu().numpy()) if return_probs else out

def step_chunk_envs(
    envs: list[gym.Env],
    obs: np.ndarray,
    bank: FlowChunkBank,
    latent_actions: Tensor | None = None,
    *,
    eval_mode = False,
    zero_noise = False
):
    n = len(envs)
    chunks, plan_lens, planned_qs, alphas = bank.plan_chunks(
        torch.from_numpy(obs).float(),
        latent_actions,
        eval_mode = eval_mode,
        zero_noise = zero_noise
    )

    cur_obs = obs.copy()
    disc_r = np.zeros(n, dtype = np.float32)
    raw_r = np.zeros(n, dtype = np.float32)
    last_r = np.zeros(n, dtype = np.float32)
    durs = np.zeros(n, dtype = np.float32)
    aborts = np.zeros(n, dtype = np.float32)
    terms = np.zeros(n, dtype = np.bool_)
    truncs = np.zeros(n, dtype = np.bool_)
    active = np.ones(n, dtype = np.bool_)
    g = np.ones(n, dtype = np.float32)

    for k in range(int(plan_lens.max())):
        active &= (k < plan_lens)
        if not active.any():
            break

        # closed-loop mid-chunk value reassessment

        if k > 0 and bank.reassess_chunk and divisible_by(k, bank.reassess_every):
            idx = np.flatnonzero(active)
            rem_lens = plan_lens[idx] - k
            max_rem = int(rem_lens.max())

            rem_chunks = torch.from_numpy(chunks[idx, k:k + max_rem]).float()
            rem_s = torch.from_numpy(cur_obs[idx]).float()

            rem_q = bank.critic_q(rem_s, rem_chunks).cpu().numpy()
            rem_q_end = rem_q[np.arange(len(idx)), rem_lens - 1]
            est_total = (disc_r[idx] / 100.) + g[idx] * rem_q_end

            abort_mask = should_abort_chunk(est_total, planned_qs[idx], bank.reassess_threshold)
            if abort_mask.any():
                aborted_idx = idx[abort_mask]
                aborts[aborted_idx] = 1.
                active[aborted_idx] = False

        for i in np.flatnonzero(active):
            next_o, r, term_i, trunc_i, _ = envs[i].step(chunks[i, k])
            rf = float(r)

            cur_obs[i] = np.asarray(next_o, dtype = np.float32)
            raw_r[i] += rf
            disc_r[i] += g[i] * rf
            last_r[i] = rf
            g[i] *= bank.gamma
            durs[i] += 1.

            if term_i or trunc_i:
                terms[i], truncs[i] = term_i, trunc_i
                active[i] = False

    info = dict(
        raw_reward = raw_r,
        last_reward = last_r,
        prim_steps = durs,
        planned_len = plan_lens.astype(np.float32),
        aborted = aborts,
        lerp_alpha = alphas.astype(np.float32)
    )
    return cur_obs, disc_r, terms, truncs, info

class FlowChunkWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, bank: FlowChunkBank, eval_mode = True):
        super().__init__(env)
        self.bank = bank
        self.eval_mode = eval_mode

        low = np.array([-3.] * bank.dct_dim + ([0.] if bank.lerp_noise else []), dtype = np.float32)
        high = np.array([3.] * bank.dct_dim + ([1.] if bank.lerp_noise else []), dtype = np.float32)
        self.action_space = gym.spaces.Box(low = low, high = high, dtype = np.float32)
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs = np.asarray(obs, dtype = np.float32)
        return obs, info

    def step(self, latent_action = None, *, zero_noise = False):
        z = rearrange(torch.as_tensor(latent_action, dtype = torch.float32), 'd -> 1 d') if exists(latent_action) else None
        next_obs, disc_r, terms, truncs, vinfo = step_chunk_envs([self.env], self._obs[None], self.bank, z, eval_mode = self.eval_mode, zero_noise = zero_noise)

        self._obs = next_obs[0]
        info = {k: float(v[0]) if k != 'aborted' else bool(v[0] > 0.5) for k, v in vinfo.items()}
        return self._obs.copy(), float(disc_r[0]), bool(terms[0]), bool(truncs[0]), info

# PPO networks (SimBa + MeanConcBeta)

class SimBa(Module):
    def __init__(self, dim, expansion = 2):
        super().__init__()
        inner = dim * expansion
        self.net = Sequential(RMSNorm(dim), Linear(dim, inner), SiLU(), Linear(inner, dim))
        self.norm = RMSNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class BetaActor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        *,
        bounds = (-1., 1.),
        init_conc = 5.
    ):
        super().__init__()
        self.action_dim = action_dim
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

        self.beta = MeanConcBeta(bounds = bounds, init_conc = init_conc, unimodal = True, eps = 1e-5)

    def forward(self, state):
        return rearrange(self.net(state), '... (a two) -> ... a two', two = 2)

    def sample(self, state, sum_action_dim = True):
        dist = self.beta(self.forward(state))
        action = dist.sample()
        return action, self.beta.log_prob(dist, action, sum_action_dim = sum_action_dim)

    def greedy(self, state):
        return self.beta(self.forward(state), temperature = 0.).mean

    def log_prob(self, params, action, sum_action_dim = True):
        return self.beta.log_prob(params, action, sum_action_dim = sum_action_dim)

    def entropy_per_dim(self, params):
        return self.beta.entropy(params, sum_action_dim = False)

class Critic(Module):
    def __init__(self, state_dim, hidden_dim = 192):
        super().__init__()
        self.net = Sequential(
            Linear(state_dim, hidden_dim),
            SimBa(hidden_dim),
            SimBa(hidden_dim),
            Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

# PPO / SMDP-HPO update

def ppo_update(
    actor: BetaActor,
    critic: Critic,
    opt_actor,
    opt_critic,
    *,
    states,
    next_states,
    actions,
    old_log_probs,
    rewards,
    terminated,
    truncated,
    durations = None,
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

            mb_s = rearrange(states[mb], 'b t d -> (b t) d')
            mb_a = rearrange(actions[mb], 'b t a -> (b t) a')
            mb_adv = rearrange(gae.advantages[mb], 'b t -> (b t)')
            mb_ret = rearrange(gae.returns[mb], 'b t -> (b t)')
            mb_val = rearrange(values[mb], 'b t -> (b t)')

            params = actor(mb_s)

            if old_log_probs.ndim == 3:
                mb_old_lp = rearrange(old_log_probs[mb], 'b t a -> (b t) a')
                cur_lp = actor.log_prob(params, mb_a, sum_action_dim = False)
                actor_loss = loss_fn(cur_lp, mb_old_lp, mb_adv, eps_clip = eps_clip, per_action_dim = True, normalize_advantages = True).mean()
            else:
                mb_old_lp = rearrange(old_log_probs[mb], 'b t -> (b t)')
                actor_loss = loss_fn(actor.log_prob(params, mb_a, sum_action_dim = True), mb_old_lp, mb_adv, eps_clip = eps_clip, normalize_advantages = True).mean()

            actor_loss = actor_loss - entropy_coef * actor.entropy_per_dim(params).mean() * 2.

            update_network_(actor_loss, opt_actor, max_grad_norm)
            update_network_(clipped_value_loss_fn(critic(mb_s), mb_val, mb_ret, clip = None), opt_critic)

# evaluation & HUD trajectory rendering

@torch.no_grad()
def render_steered_video(
    bank: FlowChunkBank,
    student: BetaActor | None,
    seed: int,
    mp4_path: str | Path,
    gif_path: str | Path,
    *,
    zero_noise = False,
    mode_label = 'dct-8d',
    env_name = 'LunarLander-v3',
    max_episode_steps = 600
):
    if exists(student):
        student.eval()

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_episode_steps, continuous = True)
    obs, _ = env.reset(seed = seed)
    records, done, ep_ret, dec_idx = [], False, 0., 0

    while not done:
        z_t = student.greedy(rearrange(torch.from_numpy(obs).float().to(ppo_device), 'd -> 1 d')) if exists(student) else None
        chunks, plan_lens, planned_qs, alphas, probs = bank.plan_chunks(
            rearrange(torch.from_numpy(obs).float(), 'd -> 1 d'),
            z_t,
            eval_mode = True,
            zero_noise = zero_noise,
            return_probs = True
        )

        chunk, plan_h, planned_q, alpha, h_probs = chunks[0], int(plan_lens[0]), float(planned_qs[0]), float(alphas[0]), probs[0]
        dec_idx += 1
        disc_r, g, chunk_recs, was_aborted = 0., 1., [], False

        for k in range(plan_h):
            if k > 0 and bank.reassess_chunk and divisible_by(k, bank.reassess_every):
                rem_q = bank.critic_q(
                    rearrange(torch.from_numpy(obs).float(), 'd -> 1 d'),
                    rearrange(torch.from_numpy(chunk[k:plan_h]).float(), 's a -> 1 s a')
                )[0, plan_h - k - 1].item()

                if should_abort_chunk((disc_r / 100.) + g * rem_q, planned_q, bank.reassess_threshold):
                    was_aborted = True
                    break

            obs_before = obs.copy()
            obs, reward, term, trunc, _ = env.step(chunk[k])
            rf = float(reward)
            ep_ret += rf
            disc_r += g * rf
            g *= bank.gamma
            chunk_recs.append([env.render(), obs_before, plan_h, k + 1, plan_h, h_probs, alpha, ep_ret, dec_idx, k == 0, False])

            if term or trunc:
                done = True
                break

        exec_h = len(chunk_recs)
        for rec in chunk_recs:
            rec[4], rec[10] = exec_h, was_aborted

        records.extend(chunk_recs)

    env.close()

    total_frames = len(records)
    W, H, B = 608, 528, 4
    font = ImageFont.load_default()
    show_lerp = bank.lerp_noise and exists(student)
    frames = []

    for f_idx, (rgb, obs_b, plan_h, step_in_h, exec_h, h_probs, alpha, cum_ret, d_idx, is_start, was_aborted) in enumerate(records):
        col = dt_color_rgb(exec_h, bank.max_chunk_size)
        canvas = Image.new('RGB', (W, H), (20, 24, 33))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, W - 1, 400 + 2 * B - 1], fill = col)
        canvas.paste(Image.fromarray(rgb), (B, B))

        lx, ly = int(B + 300. + 300. * float(obs_b[0])), int(B + 282. - 200. * float(obs_b[1]))
        if B + 16 <= lx <= W - B - 16 and B + 16 <= ly <= 400 + B - 16:
            draw.ellipse([lx - 23, ly - 23, lx + 23, ly + 23], outline = col, width = 2)
            if is_start:
                draw.ellipse([lx - 27, ly - 27, lx + 27, ly + 27], outline = (255, 255, 255), width = 1)

        abort_tag = ' ABORT!' if was_aborted else ''
        draw.rectangle([12, 12, 224, 36], fill = col, outline = (255, 255, 255), width = 1)
        draw.text((18, 18), f'ACTIVE dt={exec_h:2d} | step {step_in_h:2d}/{plan_h:2d}{abort_tag}', fill = (255, 255, 255), font = font)

        if show_lerp:
            gate_col = (int(72 + 165 * alpha), int(155 - 35 * alpha), int(239 - 145 * alpha))
            draw.rectangle([404, 12, 596, 36], fill = (20, 24, 33), outline = gate_col, width = 2)
            draw.text((412, 18), f'lerp={alpha:.2f} ({"REPLACE" if alpha >= 0.5 else "STEER"})', fill = (245, 248, 255), font = font)
        else:
            draw.rectangle([430, 12, 596, 36], fill = (20, 24, 33), outline = (120, 130, 150), width = 1)
            draw.text((438, 18), mode_label, fill = (230, 236, 248), font = font)

        y0 = 413
        draw.rectangle([12, y0, 90, y0 + 22], fill = col)
        draw.text((17, y0 + 6), f'dt={exec_h:2d}/{plan_h:2d}', fill = (255, 255, 255), font = font)

        for s_i in range(bank.max_chunk_size):
            cx, step_num = 96 + 15 * s_i, s_i + 1
            if step_num == step_in_h:
                c_fill, c_out = col, (255, 255, 255)
            elif step_num < step_in_h:
                c_fill, c_out = tuple(int(0.50 * v) for v in col), col
            elif step_num <= exec_h:
                c_fill, c_out = (38, 44, 58), col
            elif step_num <= plan_h:
                c_fill, c_out = (78, 28, 36), (200, 70, 80)
            else:
                c_fill, c_out = (27, 31, 42), (48, 54, 68)
            draw.rectangle([cx, y0 + 3, cx + 12, y0 + 19], fill = c_fill, outline = c_out)

        draw.text((344, y0 + 6), f'Ret:{cum_ret:+6.1f} | Dec#{d_idx:02d} | t={f_idx + 1:03d}/{total_frames:03d}', fill = (235, 240, 248), font = font)

        y1 = 442
        draw.text((12, y1 + 8), 'P(dt|s):', fill = (180, 188, 204), font = font)
        p_scale = max(0.16, float(h_probs.max()))

        for h_i in range(bank.max_chunk_size):
            h_val, bx = h_i + 1, 64 + 18 * h_i
            c_h = dt_color_rgb(h_val, bank.max_chunk_size)
            is_sel = (h_val == plan_h)
            draw.rectangle([bx, y1 + 1, bx + 14, y1 + 20], fill = (32, 37, 48), outline = (255, 255, 255) if is_sel else (56, 63, 79))
            bh = max(1, int(round(17 * min(1., float(h_probs[h_i]) / p_scale))))
            draw.rectangle([bx + 1, y1 + 20 - bh, bx + 13, y1 + 19], fill = c_h)
            if h_val in (1, 4, 8, 12, 16):
                draw.text((bx + (1 if h_val < 10 else -1), y1 + 21), str(h_val), fill = (255, 255, 255) if is_sel else (155, 165, 182), font = font)

        if show_lerp:
            gate_col = (int(72 + 165 * alpha), int(155 - 35 * alpha), int(239 - 145 * alpha))
            draw.text((366, y1 + 1), f'lerp (steer->replace): {alpha:.2f}', fill = (218, 226, 242), font = font)
            draw.rectangle([366, y1 + 14, 596, y1 + 26], fill = (32, 37, 48), outline = (82, 92, 114))
            draw.rectangle([367, y1 + 15, 367 + max(1, int(round(228 * alpha))), y1 + 25], fill = gate_col)
        else:
            draw.text((366, y1 + 8), f'{mode_label} | P(dt={plan_h})={int(round(100 * h_probs[plan_h - 1])):2d}%', fill = (210, 218, 232), font = font)

        y2, tx0, tw = 481, 12, W - 24
        draw.rectangle([tx0 - 1, y2 - 1, tx0 + tw + 1, y2 + 22], outline = (95, 105, 125))

        for t_i, rec_i in enumerate(records):
            x0 = tx0 + int(t_i * tw / total_frames)
            x1 = max(x0 + 1, tx0 + int((t_i + 1) * tw / total_frames))
            c_bar = dt_color_rgb(rec_i[4], bank.max_chunk_size)
            if t_i > f_idx:
                c_bar = tuple(int(0.35 * v) for v in c_bar)
            draw.rectangle([x0, y2, x1, y2 + 21], fill = c_bar)
            if rec_i[9] and t_i > 0:
                draw.line([(x0, y2), (x0, y2 + 7)], fill = (240, 244, 252))

        cur_x = tx0 + int((f_idx + 0.5) * tw / total_frames)
        draw.line([(cur_x, y2 - 3), (cur_x, y2 + 24)], fill = (255, 255, 255), width = 2)

        for b_i, lbl in enumerate(('dt 1-4 fine', 'dt 5-8 tact', 'dt 9-12 glide', 'dt 13-16 cruise')):
            lx0 = 14 + 148 * b_i
            draw.rectangle([lx0, 511, lx0 + 10, 521], fill = DT_BAND_RGB[b_i])
            draw.text((lx0 + 15, 511), lbl, fill = (178, 186, 202), font = font)

        frames.append(np.asarray(canvas))

    mp4_path, gif_path = Path(mp4_path).resolve(), Path(gif_path).resolve()
    mp4_path.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(str(mp4_path), frames, fps = 50, macro_block_size = 1)
    imageio.mimsave(str(gif_path), frames[::2], fps = 25, loop = 0)

    return dict(seed = seed, return_val = ep_ret, decisions = dec_idx, frames = total_frames, mp4 = str(mp4_path), gif = str(gif_path))

@torch.no_grad()
def evaluate_base_policy(
    actor: BetaActor,
    *,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    num_episodes = 32,
    seed = 5000
):
    actor.eval()
    env = gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True)
    rets, lands = [], 0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed = seed + ep)
        ep_r, last_r, term = 0., 0., False

        while True:
            a = actor.greedy(rearrange(torch.from_numpy(obs).float().to(ppo_device), 'd -> 1 d')).squeeze(0).cpu().numpy()
            obs, r, term, trunc, _ = env.step(a)
            ep_r += float(r)
            last_r = float(r)
            if term or trunc:
                break

        rets.append(ep_r)
        lands += int(term and last_r >= 50.)

    env.close()
    arr = np.asarray(rets)

    return dict(
        mean = float(arr.mean()),
        std = float(arr.std()),
        median = float(np.median(arr)),
        land_rate = 100. * lands / num_episodes
    )

@torch.no_grad()
def evaluate_wrapped_student(
    bank: FlowChunkBank,
    student: BetaActor | None,
    *,
    zero_noise = False,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    num_episodes = 32,
    seed = 7000,
    video_folder = None,
    video_prefix = 'flow-steered',
    mode_label = 'dct-8d'
):
    if exists(student):
        student.eval()

    base_env = gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True)
    env = FlowChunkWrapper(base_env, bank, eval_mode = True)

    rets, dts, planned_dts, alphas_all, alt_dts = [], [], [], [], []
    aborts, lands = 0, 0
    best_ep_r, best_traj = -float('inf'), []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        torch.manual_seed(ep_seed)
        np.random.seed(ep_seed)

        obs, _ = env.reset(seed = ep_seed)
        ep_r, last_r, term, done = 0., 0., False, False
        ep_traj = []

        while not done:
            x0, y0 = float(obs[0]), float(obs[1])
            z = student.greedy(rearrange(torch.from_numpy(obs).float().to(ppo_device), 'd -> 1 d')).squeeze(0).cpu().numpy() if exists(student) else None

            obs, _, term, trunc, info = env.step(z, zero_noise = zero_noise)
            dt_step, dt_plan, alpha_val = float(info['prim_steps']), float(info['planned_len']), float(info['lerp_alpha'])

            ep_r += float(info['raw_reward'])
            last_r = float(info['last_reward'])
            dts.append(dt_step)
            planned_dts.append(dt_plan)
            alphas_all.append(alpha_val)
            alt_dts.append([y0, dt_step, dt_plan, float(info['aborted']), alpha_val])
            ep_traj.append([x0, y0, float(obs[0]), float(obs[1]), dt_step, alpha_val])

            aborts += int(info['aborted'])
            done = term or trunc

        rets.append(ep_r)
        lands += int(term and last_r >= 50.)
        if ep_r > best_ep_r:
            best_ep_r, best_traj = ep_r, ep_traj

    env.close()
    arr = np.asarray(rets)
    ranked_idx = np.argsort(arr)
    median_seed = int(seed + ranked_idx[len(ranked_idx) // 2])
    best_seed = int(seed + ranked_idx[-1])

    video_meta = None
    if exists(video_folder):
        out_dir = Path(video_folder)
        out_dir.mkdir(parents = True, exist_ok = True)
        pick_seed = best_seed if exists(student) else median_seed

        video_meta = render_steered_video(
            bank, student, pick_seed,
            out_dir / f'{video_prefix}.mp4',
            out_dir / f'{video_prefix}.gif',
            zero_noise = zero_noise,
            mode_label = mode_label,
            env_name = env_name,
            max_episode_steps = max_episode_steps
        )

    return dict(
        mean = float(arr.mean()),
        std = float(arr.std()),
        median = float(np.median(arr)),
        best = float(arr.max()),
        land_rate = 100. * lands / num_episodes,
        avg_dt = float(np.mean(dts)) if dts else 1.,
        avg_planned_dt = float(np.mean(planned_dts)) if planned_dts else 1.,
        mean_lerp = float(np.mean(alphas_all)) if alphas_all else 0.,
        std_lerp = float(np.std(alphas_all)) if alphas_all else 0.,
        aborts_per_ep = aborts / max(1, num_episodes),
        video = video_meta,
        alt_dts = alt_dts[:1600],
        best_traj = best_traj
    )

# stage 2: reborn DCT latent-noise steering student (Wagenmaker et al. + MeanConcBeta)

def train_reborn_student(
    bank: FlowChunkBank,
    *,
    label = 'dct-8d',
    env_name = 'LunarLander-v3',
    num_envs = 16,
    student_updates = 105,
    macro_rollout_steps = 56,
    max_episode_steps = 600,
    actor_hidden_dim = 108,
    critic_hidden_dim = 192,
    student_init_conc = 6.,
    lr = 1e-3,
    critic_lr = 2e-3,
    val_every = 15,
    eval_episodes = 32,
    log_every = 15,
    seed = 300,
    video_folder = None
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    envs = [gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True) for _ in range(num_envs)]
    obs = np.stack([e.reset(seed = seed + i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    state_dim = obs.shape[-1]

    # DCT coords in [-3, 3] (Beta(4, 4) => mean 0, var 1); optional lerp gate in [0, 1] (mean 0.5)

    bounds = [(-3., 3.)] * bank.dct_dim + [(0., 1.)] if bank.lerp_noise else (-3., 3.)
    student = BetaActor(state_dim, actor_hidden_dim, bank.latent_dim, bounds = bounds, init_conc = student_init_conc).to(ppo_device)
    critic = Critic(state_dim, critic_hidden_dim).to(ppo_device)

    opt_actor = torch.optim.AdamW(student.parameters(), lr = lr)
    opt_critic = torch.optim.AdamW(critic.parameters(), lr = critic_lr)

    state = torch.from_numpy(obs).float().to(ppo_device)
    running = np.zeros(num_envs, dtype = np.float32)
    recent_rets, recent_dts, recent_aborts, recent_lerps, history = [], [], [], [], []
    best_val, best_state = -float('inf'), deepcopy(student.state_dict())
    env_steps, macro_decs, t0 = 0, 0, time.time()

    mode_desc = 'lerp(init, dct, alpha in [0, 1])' if bank.lerp_noise else 'init + dct'
    print(f'\n[stage 2 | {label}] reborn student dim={bank.latent_dim} ({mode_desc})', flush = True)

    for upd in range(1, student_updates + 1):
        progress = (upd - 1) / max(1, student_updates)
        ent_coef = schedule_at(BETA_ENTROPY_SCHEDULE, progress)

        for pg in opt_actor.param_groups:
            pg['lr'] = lr * schedule_at(LR_SCALE_SCHEDULE, progress)

        states, next_states, latents, log_probs, rewards, terms, truncs, durs = [], [], [], [], [], [], [], []
        student.eval()

        for _ in range(macro_rollout_steps):
            with torch.no_grad():
                z, lp = student.sample(state, sum_action_dim = False)

            true_next, disc_r, term, trunc, info = step_chunk_envs(envs, obs, bank, z.cpu(), eval_mode = False)
            done = term | trunc
            next_obs = true_next.copy()

            running += info['raw_reward']
            env_steps += int(info['prim_steps'].sum())
            macro_decs += num_envs
            recent_dts.extend(info['prim_steps'].tolist())
            recent_aborts.extend(info['aborted'].tolist())
            recent_lerps.extend(info['lerp_alpha'].tolist())

            for i in np.flatnonzero(done):
                recent_rets.append(float(running[i]))
                running[i] = 0.
                next_obs[i] = np.asarray(envs[i].reset()[0], dtype = np.float32)

            states.append(state)
            next_states.append(torch.from_numpy(true_next).float().to(ppo_device))
            latents.append(z)
            log_probs.append(lp)
            rewards.append(torch.from_numpy(disc_r / 100.).float().to(ppo_device))
            terms.append(torch.from_numpy(term).bool().to(ppo_device))
            truncs.append(torch.from_numpy(trunc).bool().to(ppo_device))
            durs.append(torch.from_numpy(info['prim_steps']).float().to(ppo_device))

            obs = next_obs
            state = torch.from_numpy(obs).float().to(ppo_device)

        recent_dts = recent_dts[-2048:]
        recent_aborts = recent_aborts[-2048:]
        recent_lerps = recent_lerps[-2048:]

        student.train()
        critic.train()

        ppo_update(
            student, critic, opt_actor, opt_critic,
            states = stack(states, dim = 1),
            next_states = stack(next_states, dim = 1),
            actions = stack(latents, dim = 1),
            old_log_probs = stack(log_probs, dim = 1),
            rewards = stack(rewards, dim = 1),
            terminated = stack(terms, dim = 1),
            truncated = stack(truncs, dim = 1),
            durations = stack(durs, dim = 1),
            entropy_coef = ent_coef
        )

        rec32 = float(np.mean(recent_rets[-32:])) if recent_rets else float('nan')
        avg_dt = float(np.mean(recent_dts[-512:])) if recent_dts else 1.
        abort_rate = 100. * float(np.mean(recent_aborts[-512:])) if recent_aborts else 0.
        avg_lerp = float(np.mean(recent_lerps[-512:])) if recent_lerps else 1.

        if divisible_by(upd, val_every) or upd == student_updates:
            v = evaluate_wrapped_student(bank, student, env_name = env_name, max_episode_steps = max_episode_steps, num_episodes = 16, seed = seed + 4000)
            if v['mean'] >= best_val:
                best_val, best_state = v['mean'], deepcopy(student.state_dict())

        history.append(dict(
            update = upd,
            env_steps = env_steps,
            decisions = macro_decs,
            ret32 = rec32,
            avg_dt = avg_dt,
            abort_rate = abort_rate,
            mean_lerp = avg_lerp,
            best_val = best_val
        ))

        if divisible_by(upd, log_every) or upd == student_updates:
            lerp_str = f' | lerp={avg_lerp:.2f}' if bank.lerp_noise else ''
            print(
                f'  [{label} upd {upd:3d}/{student_updates}] env={env_steps // 1000:4d}k (decs={macro_decs // 1000:3d}k) | '
                f'ret32={rec32:+6.1f} | val_best={best_val:+6.1f} | avg_dt={avg_dt:4.2f}{lerp_str} | abort={abort_rate:4.1f}% [{time.time() - t0:.1f}s]',
                flush = True
            )

    for e in envs:
        e.close()

    student.load_state_dict(best_state)

    final_eval = evaluate_wrapped_student(
        bank, student,
        env_name = env_name,
        max_episode_steps = max_episode_steps,
        num_episodes = eval_episodes,
        seed = seed + 9000,
        video_folder = video_folder,
        video_prefix = f'student-{label}',
        mode_label = label
    )

    lerp_eval_str = f' | lerp={final_eval["mean_lerp"]:.2f}+/-{final_eval["std_lerp"]:.2f}' if bank.lerp_noise else ''
    print(
        f'  [{label} eval] {final_eval["mean"]:+.1f} +/- {final_eval["std"]:.1f} '
        f'(median {final_eval["median"]:+.1f}) | land={final_eval["land_rate"]:.1f}% | '
        f'avg_dt={final_eval["avg_dt"]:.2f} (plan {final_eval["avg_planned_dt"]:.2f}){lerp_eval_str} | aborts/ep={final_eval["aborts_per_ep"]:.2f}',
        flush = True
    )
    return student, final_eval, history

# main

def main(
    env_name = 'LunarLander-v3',
    max_chunk_size = 16,
    dct_modes = 4,
    lerp_noise = False,
    num_envs = 16,
    max_episode_steps = 600,
    base_updates = 95,
    base_rollout_steps = 384,
    buffer_capacity = 49152,
    flow_train_every = 3,
    flow_steps_per_iter = 24,
    flow_consolidate_steps = 360,
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
    student_updates = 105,
    macro_rollout_steps = 56,
    eval_episodes = 32,
    seed = 42,
    render = True,
    video_folder = './flow-chunk-steering-videos',
    smoke = False
):
    if smoke:
        num_envs, base_updates, base_rollout_steps = 4, 4, 64
        buffer_capacity, flow_train_every, flow_steps_per_iter, flow_consolidate_steps, flow_batch_size = 2048, 2, 4, 8, 32
        student_updates, macro_rollout_steps, eval_episodes, render = 4, 12, 4, False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # stage 1: dt = 1 continuous PPO + causal TransformerFlowActor & TransformerFlowCritic (XM k = 2)

    base_env = SyncVectorEnv(
        [lambda: gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True) for _ in range(num_envs)],
        autoreset_mode = AutoresetMode.SAME_STEP
    )

    state_dim = base_env.single_observation_space.shape[0]
    num_actions = base_env.single_action_space.shape[0]

    base_actor = BetaActor(state_dim, 108, num_actions, bounds = (-1., 1.), init_conc = 5.).to(ppo_device)
    base_critic = Critic(state_dim, 192).to(ppo_device)

    opt_actor = torch.optim.AdamW(base_actor.parameters(), lr = 1e-3)
    opt_critic = torch.optim.AdamW(base_critic.parameters(), lr = 2.5e-3)

    buffer = ChunkReplayBuffer(buffer_capacity, max_chunk_size, state_dim, num_actions)

    flow_trainer = FlowChunkTrainer(
        state_dim = state_dim,
        num_actions = num_actions,
        max_chunk_size = max_chunk_size,
        hidden_dim = flow_hidden_dim,
        depth = flow_depth,
        heads = flow_heads,
        lr = flow_lr,
        corr_alpha = corr_alpha,
        xm_candidates = xm_candidates
    ).to(flow_device)

    print(f'[stage 1] dt=1 PPO ({ppo_device}) + Causal Flow Actor/Critic (H={max_chunk_size}, XM k={xm_candidates}, dev={flow_device})', flush = True)

    obs, _ = base_env.reset(seed = seed)
    state = torch.from_numpy(obs).float().to(ppo_device)
    running = np.zeros(num_envs, dtype = np.float32)
    recent_rets, stage1_history = [], []
    flow_stats = dict(actor_flow = 0., critic_flow = 0., mean_q = 0.)
    best_teacher_val, best_teacher_state = -float('inf'), deepcopy(base_actor.state_dict())
    env_steps, t0 = 0, time.time()

    for upd in range(1, base_updates + 1):
        progress = (upd - 1) / max(1, base_updates)
        ent_coef = schedule_at(BETA_ENTROPY_SCHEDULE, progress)

        for pg in opt_actor.param_groups:
            pg['lr'] = 1e-3 * schedule_at(LR_SCALE_SCHEDULE, progress)

        states, next_states, actions, log_probs, rewards, terms, truncs = [], [], [], [], [], [], []
        base_actor.eval()

        for _ in range(base_rollout_steps):
            with torch.no_grad():
                a, lp = base_actor.sample(state, sum_action_dim = True)

            next_obs, reward, term, trunc, info = base_env.step(a.cpu().numpy())
            done = term | trunc
            running += reward

            for i in np.flatnonzero(done):
                recent_rets.append(float(running[i]))
                running[i] = 0.

            states.append(state)
            next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(ppo_device))
            actions.append(a)
            log_probs.append(lp)
            rewards.append(torch.from_numpy(reward / 100.).float().to(ppo_device))
            terms.append(torch.from_numpy(term).bool().to(ppo_device))
            truncs.append(torch.from_numpy(trunc).bool().to(ppo_device))

            state = torch.from_numpy(next_obs).float().to(ppo_device)

        env_steps += num_envs * base_rollout_steps
        s_t, ns_t = stack(states, dim = 1), stack(next_states, dim = 1)
        a_t, r_t = stack(actions, dim = 1), stack(rewards, dim = 1)
        term_t, trunc_t = stack(terms, dim = 1), stack(truncs, dim = 1)

        buffer.add_rollout(s_t, ns_t, a_t, r_t, term_t, trunc_t, stride = 2)

        base_actor.train()
        base_critic.train()

        ppo_update(
            base_actor, base_critic, opt_actor, opt_critic,
            states = s_t, next_states = ns_t, actions = a_t,
            old_log_probs = stack(log_probs, dim = 1),
            rewards = r_t, terminated = term_t, truncated = trunc_t,
            entropy_coef = ent_coef
        )

        if divisible_by(upd, flow_train_every) and buffer.size >= flow_batch_size * 2:
            flow_stats = flow_trainer.learn(buffer, base_critic, steps = flow_steps_per_iter, batch_size = flow_batch_size)

        rec32 = float(np.mean(recent_rets[-32:])) if recent_rets else float('nan')
        stage1_history.append(dict(update = upd, env_steps = env_steps, ret32 = rec32, **flow_stats))

        if divisible_by(upd, 15) or upd == base_updates:
            v = evaluate_base_policy(base_actor, env_name = env_name, max_episode_steps = max_episode_steps, num_episodes = 12, seed = seed + 3000)
            if v['mean'] >= best_teacher_val:
                best_teacher_val, best_teacher_state = v['mean'], deepcopy(base_actor.state_dict())

            print(
                f'  [dt=1 upd {upd:3d}/{base_updates}] env={env_steps // 1000:4d}k | ret32={rec32:+6.1f} | val={v["mean"]:+6.1f} | '
                f'a_flow={flow_stats["actor_flow"]:.3f} c_flow={flow_stats["critic_flow"]:.3f} q={flow_stats["mean_q"]:+5.1f} [{time.time() - t0:.1f}s]',
                flush = True
            )

    base_env.close()

    flow_stats = flow_trainer.learn(buffer, base_critic, steps = flow_consolidate_steps, batch_size = flow_batch_size)
    print(f'  [flow consolidated] chunks={buffer.size} | a_flow={flow_stats["actor_flow"]:.3f} | c_flow={flow_stats["critic_flow"]:.3f} | mean_q={flow_stats["mean_q"]:+.1f}', flush = True)

    base_actor.load_state_dict(best_teacher_state)
    teacher_eval = evaluate_base_policy(base_actor, env_name = env_name, max_episode_steps = max_episode_steps, num_episodes = eval_episodes, seed = seed + 6000)
    print(f'  [dt=1 teacher eval] {teacher_eval["mean"]:+.1f} +/- {teacher_eval["std"]:.1f} (median {teacher_eval["median"]:+.1f}) | land={teacher_eval["land_rate"]:.1f}%', flush = True)

    # stage 2: ablations (z = 0 & unsteered prior) + reborn DCT-8D latent-steering student

    bank = FlowChunkBank(
        flow_trainer.ema_actor.ema_model,
        flow_trainer.ema_critic.ema_model,
        dct_modes = dct_modes,
        corr_alpha = corr_alpha,
        lerp_noise = lerp_noise,
        actor_flow_steps = actor_flow_steps,
        dynamic_chunk = True,
        q_temp = q_temp,
        reassess_chunk = reassess_chunk,
        reassess_every = reassess_every,
        reassess_threshold = reassess_threshold
    ).to(ppo_device)

    results = dict(
        config = dict(max_chunk_size = max_chunk_size, dct_modes = dct_modes, lerp_noise = lerp_noise, q_temp = q_temp, xm_candidates = xm_candidates, seed = seed),
        teacher = teacher_eval,
        stage1_history = stage1_history,
        zero_steer = None,
        unsteered = None,
        students = {},
        histories = {}
    )

    for key, is_zero, prefix, m_lbl in (
        ('zero_steer', True, 'zero-steer', 'zero-steer (z=0)'),
        ('unsteered', False, 'unsteered-prior', 'unsteered prior')
    ):
        ev = evaluate_wrapped_student(
            bank, None,
            zero_noise = is_zero,
            env_name = env_name,
            max_episode_steps = max_episode_steps,
            num_episodes = eval_episodes,
            seed = seed + 8000,
            video_folder = video_folder if render else None,
            video_prefix = prefix,
            mode_label = m_lbl
        )
        results[key] = ev
        print(
            f'  [{prefix} eval] {ev["mean"]:+.1f} +/- {ev["std"]:.1f} (median {ev["median"]:+.1f}) | '
            f'land={ev["land_rate"]:.1f}% | avg_dt={ev["avg_dt"]:.2f} (plan {ev["avg_planned_dt"]:.2f}) | aborts/ep={ev["aborts_per_ep"]:.2f}',
            flush = True
        )

    suffix = 'lerp' if lerp_noise else 'add'
    label = f'dct-{bank.dct_dim}d-{suffix}'

    _, student_eval, student_hist = train_reborn_student(
        bank,
        label = label,
        env_name = env_name,
        num_envs = num_envs,
        student_updates = student_updates,
        macro_rollout_steps = macro_rollout_steps,
        max_episode_steps = max_episode_steps,
        eval_episodes = eval_episodes,
        seed = seed + 400,
        video_folder = video_folder if render else None
    )

    results['students']['dct'] = student_eval
    results['histories']['dct'] = student_hist

    out_dir = Path(video_folder)
    out_dir.mkdir(parents = True, exist_ok = True)
    with open(out_dir / 'flow_steering_summary.json', 'w') as f:
        json.dump(results, f, indent = 2)

    z_ev, u_ev = results['zero_steer'], results['unsteered']
    print(f'\n=== Summary (LunarLander-v3 Continuous | H_max = {max_chunk_size} | XM k = {xm_candidates} | {label}) ===', flush = True)
    print(f'  Stage 1 Teacher (dt = 1)       : {teacher_eval["mean"]:+6.1f} (median {teacher_eval["median"]:+6.1f}, land {teacher_eval["land_rate"]:4.1f}%, avg_dt = 1.00)', flush = True)
    print(f'  Zero-Steer Flow Bank (z = 0)   : {z_ev["mean"]:+6.1f} (median {z_ev["median"]:+6.1f}, land {z_ev["land_rate"]:4.1f}%, avg_dt = {z_ev["avg_dt"]:.2f}, aborts/ep = {z_ev["aborts_per_ep"]:.2f})', flush = True)
    print(f'  Unsteered Prior Bank           : {u_ev["mean"]:+6.1f} (median {u_ev["median"]:+6.1f}, land {u_ev["land_rate"]:4.1f}%, avg_dt = {u_ev["avg_dt"]:.2f}, aborts/ep = {u_ev["aborts_per_ep"]:.2f})', flush = True)
    print(f'  Reborn Student ({label:12s})  : {student_eval["mean"]:+6.1f} (median {student_eval["median"]:+6.1f}, land {student_eval["land_rate"]:4.1f}%, avg_dt = {student_eval["avg_dt"]:.2f}, aborts/ep = {student_eval["aborts_per_ep"]:.2f})', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
