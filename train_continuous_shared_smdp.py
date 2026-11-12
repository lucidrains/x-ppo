# /// script
# dependencies = [
#   "assoc-scan",
#   "einops",
#   "fire",
#   "gymnasium[box2d,other]",
#   "imageio[ffmpeg]",
#   "matplotlib",
#   "mean-conc-beta>=0.2.7",
#   "numpy",
#   "pillow",
#   "torch",
#   "torch-einops-utils>=0.2.5",
# ]
# ///

from __future__ import annotations

import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import NamedTuple

import fire
import numpy as np

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn, Tensor, arange, cat, full, randint, randn_like, randperm, stack, tensor
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Module, RMSNorm, Sequential, SiLU
from torch.nn.utils import clip_grad_norm_

from einops import pack, rearrange, reduce, repeat

from mean_conc_beta import Beta as MeanConcBeta

from torch_einops_utils import (
    lens_to_mask,
    masked_mean,
    pad_right_at_dim_to
)

from x_ppo import (
    NextLatentQCritic,
    calc_gae,
    clipped_value_loss_fn,
    hpo_actor_loss,
    soft_exit_weights
)

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

BETA_ENTROPY_SCHEDULE = (0.0016, 0.0008, 0.0004, 0.0002, 0.0001, 0.00004)
LR_SCALE_SCHEDULE = (1., 0.85, 0.55, 0.30, 0.15, 0.08)

HIER_COLORS_RGB = {1: (72, 149, 239), 2: (42, 157, 143), 4: (244, 162, 97), 8: (230, 57, 70)}
HIER_COLORS_HEX = {1: '#4895ef', 2: '#2a9d8f', 4: '#f4a261', 8: '#e63946'}

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def normalized_entropy(prob):
    ent = -(prob * log(prob)).sum(dim = -1)
    return ent / math.log(max(2, prob.shape[-1]))

def schedule_at(schedule, progress):
    idx = min(int(progress * len(schedule)), len(schedule) - 1)
    return schedule[max(0, idx)]

def hadamard_directions(dim):
    mat = tensor([[1.]])

    while mat.shape[0] < dim:
        top = cat((mat, mat), dim = 1)
        bot = cat((mat, -mat), dim = 1)
        mat = cat((top, bot), dim = 0)

    dirs = mat[:dim, :dim]
    return F.normalize(cat((dirs, -dirs), dim = 0), dim = -1)

def true_next_obs(next_obs, done, info):
    if not done.any() or 'final_obs' not in info:
        return next_obs

    out = next_obs.copy()

    for i in np.flatnonzero(done):
        if exists(info['final_obs'][i]):
            out[i] = info['final_obs'][i]

    return out

def step_info_array(info, key, done, default_val):
    out = np.full(done.shape[0], default_val, dtype = np.float32)

    if key in info:
        mask = info.get(f'_{key}', ~done)
        out[mask] = info[key][mask]

    if done.any() and 'final_info' in info and key in info['final_info']:
        final_info = info['final_info']
        final_mask = final_info.get(f'_{key}', done)
        out[final_mask] = final_info[key][final_mask]

    return out

def is_strict_landing(obs, terminated, last_reward):
    x, y, vx, vy, angle, ang_vel, leg_l, leg_r = (float(v) for v in obs[:8])

    return (
        terminated and
        last_reward >= 50. and
        abs(x) < 0.20 and
        abs(y) < 0.15 and
        abs(vx) < 0.50 and
        abs(vy) < 0.50 and
        abs(angle) < 0.20 and
        abs(ang_vel) < 0.50 and
        leg_l > 0.5 and
        leg_r > 0.5
    )

# networks

class SimBa(Module):
    def __init__(
        self,
        dim,
        expansion = 2,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * expansion)

        self.net = Sequential(
            RMSNorm(dim),
            Linear(dim, dim_inner),
            SiLU(),
            Dropout(dropout),
            Linear(dim_inner, dim)
        )

        self.norm = RMSNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class BetaActor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        beta_init_conc = 5.
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

        self.beta = MeanConcBeta(
            bounds = (-1., 1.),
            init_conc = beta_init_conc,
            unimodal = True,
            eps = 1e-5
        )

        self.register_buffer('probe_dirs', hadamard_directions(action_dim), persistent = False)

    def forward(self, state):
        out = self.net(state)
        return rearrange(out, '... (a two) -> ... a two', two = 2)

    def sample(self, state):
        dist = self.beta(self.forward(state))
        action = dist.sample()
        return action, self.beta.log_prob(dist, action, sum_action_dim = True)

    def sample_candidates(self, state, k):
        params = self.forward(state)
        dist = self.beta(params)
        cands = rearrange(dist.sample((k,)), 'k b a -> b k a')
        cand_lp = self.beta.log_prob(rearrange(params, 'b a p -> b 1 a p'), cands, sum_action_dim = True)
        return cands, cand_lp

    def greedy(self, state):
        return self.beta.mean(self.forward(state))

    def eval_candidates(self, state, k = 9, step_scale = 0.35):
        params = self.forward(state)
        mu = self.beta.mean(params)

        if k <= 1:
            return rearrange(mu, 'b a -> b 1 a')

        std = self.beta(params).variance.clamp_min(1e-8).sqrt()
        dirs = self.probe_dirs[:k - 1]

        offset = step_scale * math.sqrt(self.action_dim) * rearrange(std, 'b a -> b 1 a') * rearrange(dirs, 'k a -> 1 k a')
        probes = (rearrange(mu, 'b a -> b 1 a') + offset).clamp(-1. + 1e-4, 1. - 1e-4)

        packed, _ = pack((mu, probes), 'b * a')
        return packed

    def log_prob(self, params, action):
        return self.beta.log_prob(params, action, sum_action_dim = True)

    def entropy(self, params):
        # mean per-dim entropy scaled to 2d primitive space so schedules stay invariant to latent_dim

        return self.beta.entropy(params, sum_action_dim = False).mean(dim = -1) * 2.

class Critic(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim = 192
    ):
        super().__init__()
        self.net = Sequential(
            Linear(state_dim, hidden_dim),
            RMSNorm(hidden_dim),
            SiLU(),
            Linear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            SiLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return rearrange(self.net(state), '... 1 -> ...')

# shared padded hierarchical tanh autoencoder

class SharedActionChunkTanh(Module):
    def __init__(
        self,
        hierarchies = (1, 2, 4, 8),
        num_actions = 2,
        latent_dim = 8,
        dim = 192,
        pad_value = -2.,
        latent_noise = 0.05,
        len_loss_weight = 0.5,
        pre_tanh_reg = 1e-3,
        prior_quantile_weight = 0.05
    ):
        super().__init__()
        self.hierarchies = tuple(sorted(int(h) for h in hierarchies))
        self.num_hierarchies = len(self.hierarchies)
        self.max_hierarchy = max(self.hierarchies)
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.dim = dim

        self.pad_value = float(pad_value)
        self.latent_noise = latent_noise
        self.len_loss_weight = len_loss_weight
        self.pre_tanh_reg = pre_tanh_reg
        self.prior_quantile_weight = prior_quantile_weight

        self.packed_chunk_dim = 1 + self.max_hierarchy * num_actions
        self.max_log2 = math.log2(max(2, self.max_hierarchy))

        self.horizons_np = np.asarray(self.hierarchies, dtype = np.float32)
        self.register_buffer('horizons_t', tensor(self.hierarchies, dtype = torch.long), persistent = False)
        self.register_buffer('horizons_f', tensor(self.hierarchies, dtype = torch.float32), persistent = False)
        self.register_buffer('horizons_log2', tensor(self.hierarchies, dtype = torch.float32).log2(), persistent = False)

        # encoder and decoder with linear skips so length and action variations form global manifolds across (-1, 1)^d

        self.encoder = Sequential(
            Linear(self.packed_chunk_dim, dim),
            SimBa(dim),
            SimBa(dim),
            RMSNorm(dim),
            Linear(dim, latent_dim)
        )
        self.enc_skip = Linear(self.packed_chunk_dim, latent_dim)

        self.decoder = Sequential(
            Linear(latent_dim, dim),
            SimBa(dim),
            SimBa(dim),
            RMSNorm(dim),
            Linear(dim, self.packed_chunk_dim)
        )
        self.dec_skip = Linear(latent_dim, self.packed_chunk_dim, bias = False)

    def pad_and_prepend_length(self, chunk, length = None):
        chunk = chunk.float()
        h = chunk.shape[-2]
        assert h <= self.max_hierarchy

        padded = pad_right_at_dim_to(chunk, self.max_hierarchy, dim = -2, value = self.pad_value)
        lengths = length.float() if torch.is_tensor(length) else full(chunk.shape[:-2], float(h), device = chunk.device)
        return self.pack_padded(padded, lengths)

    def pack_padded(self, padded_actions, lengths):
        flat_actions = rearrange(padded_actions.float(), '... h a -> ... (h a)')
        return cat((rearrange(lengths.float(), '... -> ... 1'), flat_actions), dim = -1)

    def round_to_legit(self, pred_lengths):
        h_idx = (rearrange(pred_lengths, '... -> ... 1') - self.horizons_f).abs().argmin(dim = -1)
        return self.horizons_t[h_idx], h_idx

    def encode(self, packed_chunks):
        x = packed_chunks.float()
        pre_tanh = self.encoder(x) + self.enc_skip(x)
        return pre_tanh.tanh(), pre_tanh

    def decode_raw(self, latents):
        raw = self.decoder(latents) + self.dec_skip(latents)
        pred_log2 = 0.5 * self.max_log2 * (1. + 1.35 * raw[..., 0].tanh())
        pred_lengths = 2. ** pred_log2
        pred_actions = rearrange(raw[..., 1:], '... (h a) -> ... h a', h = self.max_hierarchy).tanh()
        return pred_lengths, pred_actions, pred_log2

    def decode(self, latents):
        pred_lengths, pred_actions, _ = self.decode_raw(latents.clamp(-1., 1.))
        legit_h, h_idx = self.round_to_legit(pred_lengths)
        return legit_h, h_idx, pred_lengths, pred_actions

    def soft_hierarchy_probs(self, latents, temp = 0.35):
        _, _, pred_log2 = self.decode_raw(latents.clamp(-1., 1.))
        logits = -(rearrange(pred_log2, '... -> ... 1') - self.horizons_log2).square() / temp
        return logits.softmax(dim = -1)

    @torch.no_grad()
    def to_env_commands(self, latents):
        pred_lengths, pred_actions, _ = self.decode_raw(latents.clamp(-1., 1.))
        cmd = cat((rearrange(pred_lengths, '... -> ... 1'), rearrange(pred_actions, '... h a -> ... (h a)')), dim = -1)
        return cmd.cpu().numpy().astype(np.float32)

    def forward(self, padded_actions, lengths, prior_samples = None):
        packed = self.pack_padded(padded_actions, lengths)
        latents, pre_tanh = self.encode(packed)

        should_noise = self.training and self.latent_noise > 0.
        noisy = (pre_tanh + self.latent_noise * randn_like(pre_tanh)).tanh() if should_noise else latents

        pred_lengths, pred_actions, pred_log2 = self.decode_raw(noisy)

        # average valid-step mse within each chunk first so h = 1 is weighted equally to h = 8

        valid_mask = lens_to_mask(lengths, self.max_hierarchy)
        per_step_mse = reduce((pred_actions - padded_actions).square(), 'b h a -> b h', 'mean')
        act_loss = masked_mean(per_step_mse, mask = valid_mask, dim = -1).mean()

        len_loss = F.mse_loss(pred_log2, lengths.float().log2())
        reg_loss = self.pre_tanh_reg * pre_tanh.square().mean()

        # 1d quantile alignment keeps all four horizons well-represented under the initial policy prior

        if exists(prior_samples) and self.prior_quantile_weight > 0.:
            _, _, prior_log2 = self.decode_raw(prior_samples)
            sorted_prior = rearrange(prior_log2, '... -> (...)').sort().values
            target_quantiles = torch.linspace(-0.2, self.max_log2 + 0.2, sorted_prior.numel(), device = sorted_prior.device)
            reg_loss = reg_loss + self.prior_quantile_weight * F.mse_loss(sorted_prior, target_quantiles)

        total_loss = act_loss + self.len_loss_weight * len_loss + reg_loss

        with torch.no_grad():
            legit_h, _, _, clean_act = self.decode(latents)
            clean_step_mse = reduce((clean_act - padded_actions).square(), 'b h a -> b h', 'mean')
            clean_mse = masked_mean(clean_step_mse, mask = valid_mask, dim = -1).mean().item()
            round_acc = (legit_h == lengths).float().mean().item()

        return total_loss, dict(act_mse = clean_mse, len_mse = len_loss.item(), round_acc = round_acc)

class SharedChunkBuffer:
    def __init__(
        self,
        capacity_per_h,
        hierarchies,
        num_actions,
        pad_value = -2.,
        stride = 2
    ):
        self.capacity = capacity_per_h
        self.hierarchies = tuple(sorted(int(h) for h in hierarchies))
        self.max_h = max(self.hierarchies)
        self.num_actions = num_actions
        self.pad_value = float(pad_value)
        self.stride = max(1, stride)

        self.padded_actions = {
            h: full((capacity_per_h, self.max_h, num_actions), self.pad_value, dtype = torch.float32)
            for h in self.hierarchies
        }
        self.ptrs = {h: 0 for h in self.hierarchies}
        self.sizes = {h: 0 for h in self.hierarchies}

    def add_rollout(self, actions, dones):
        acts_cpu, dones_cpu = actions.detach().cpu(), dones.detach().cpu()

        for h in self.hierarchies:
            if acts_cpu.shape[1] < h:
                continue

            step_stride = 1 if h == 1 else self.stride
            d_win = dones_cpu.unfold(1, h, step_stride)
            a_win = acts_cpu.unfold(1, h, step_stride)

            valid = ~d_win[..., :-1].any(dim = -1) if h > 1 else torch.ones_like(d_win[..., 0], dtype = torch.bool)
            if not valid.any():
                continue

            act_chunks = rearrange(a_win[valid], 'n a h -> n h a').float()[-self.capacity:]
            n = act_chunks.shape[0]
            idx = (self.ptrs[h] + arange(n)) % self.capacity

            buf = self.padded_actions[h]
            buf[idx] = self.pad_value
            buf[idx, :h] = act_chunks

            self.ptrs[h] = (self.ptrs[h] + n) % self.capacity
            self.sizes[h] = min(self.capacity, self.sizes[h] + n)

    def ready(self, min_per_h = 64):
        return all(sz >= min_per_h for sz in self.sizes.values())

    def sample(self, batch_size, dev):
        per_h = max(1, batch_size // len(self.hierarchies))
        xs, ls = [], []

        for h in self.hierarchies:
            idx = randint(0, self.sizes[h], (per_h,))
            xs.append(self.padded_actions[h][idx])
            ls.append(full((per_h,), h, dtype = torch.long))

        return cat(xs, dim = 0).to(dev), cat(ls, dim = 0).to(dev)

    def train_steps(
        self,
        model,
        opt,
        steps = 100,
        batch_size = 256,
        prior_init_conc = 3.
    ):
        if not self.ready(16):
            return dict(act_mse = 0., len_mse = 0., round_acc = 0.)

        dev = next(model.parameters()).device
        model.train()

        prior_beta = MeanConcBeta(bounds = (-1., 1.), init_conc = prior_init_conc, unimodal = True, eps = 1e-5).to(dev)
        prior_dist = prior_beta(torch.zeros(batch_size, model.latent_dim, 2, device = dev))

        stats_hist = []

        with torch.random.fork_rng():
            for _ in range(steps):
                bx, bl = self.sample(batch_size, dev)
                loss, stats = model(bx, bl, prior_samples = prior_dist.sample())

                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.)
                opt.step()

                stats_hist.append(stats)

        tail = stats_hist[-16:]
        return {k: float(np.mean([s[k] for s in tail])) for k in tail[-1]}

    @torch.no_grad()
    def evaluate_per_hierarchy(self, model, max_samples = 2048):
        dev = next(model.parameters()).device
        model.eval()
        report = {}

        for h in self.hierarchies:
            n = min(max_samples, self.sizes[h])
            x = self.padded_actions[h][:n].to(dev)
            l = full((n,), h, dtype = torch.long, device = dev)

            z, _ = model.encode(model.pack_padded(x, l))
            legit_h, _, pred_l, pred_a = model.decode(z)

            diff = pred_a[:, :h] - x[:, :h]
            report[h] = dict(
                mse = diff.square().mean().item(),
                mae = diff.abs().mean().item(),
                round_acc = 100. * (legit_h == h).float().mean().item(),
                mean_pred_len = pred_l.mean().item(),
                std_pred_len = pred_l.std().item(),
                z_std = z.std().item()
            )

        return report

def load_shared_autoencoder(ckpt_path):
    ck = torch.load(Path(ckpt_path), map_location = device, weights_only = False)

    ae = SharedActionChunkTanh(
        hierarchies = tuple(ck['hierarchies']),
        num_actions = int(ck['num_actions']),
        latent_dim = int(ck['latent_dim']),
        dim = int(ck['dim']),
        pad_value = float(ck['pad_value'])
    ).to(device)

    ae.load_state_dict(ck['autoencoder'])
    ae.eval()
    ae.requires_grad_(False)

    print(f'[loaded shared AE] hierarchies={ae.hierarchies} | packed={ae.packed_chunk_dim}->{ae.latent_dim}D | pad={ae.pad_value}', flush = True)
    return ae, ck.get('ae_report', {})

def discover_shared_autoencoder(
    env_name = 'LunarLander-v3',
    hierarchies = (1, 2, 4, 8),
    latent_dim = 8,
    compressor_dim = 192,
    pad_value = -2.,
    prior_init_conc = 3.,
    base_updates = 135,
    rollout_steps = 384,
    num_envs = 16,
    max_episode_steps = 600,
    seed = 42
):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = SyncVectorEnv(
            [lambda: gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True) for _ in range(num_envs)],
            autoreset_mode = AutoresetMode.SAME_STEP
        )

        state_dim = int(env.single_observation_space.shape[0])
        num_actions = int(env.single_action_space.shape[0])

        actor = BetaActor(state_dim, 108, num_actions).to(device)
        critic = Critic(state_dim, 192).to(device)
        opt_actor = torch.optim.AdamW(actor.parameters(), lr = 1e-3)
        opt_critic = torch.optim.AdamW(critic.parameters(), lr = 2.5e-3)

        autoencoder = SharedActionChunkTanh(
            hierarchies = hierarchies,
            num_actions = num_actions,
            latent_dim = latent_dim,
            dim = compressor_dim,
            pad_value = pad_value
        ).to(device)

        opt_ae = torch.optim.AdamW(autoencoder.parameters(), lr = 2e-3)
        buffer = SharedChunkBuffer(49152, hierarchies, num_actions, pad_value = pad_value)

        obs, _ = env.reset(seed = seed)
        state = torch.from_numpy(obs).float().to(device)
        running, recent, t0 = np.zeros(num_envs), [], time.time()

        print(f'[stage 1] discovering shared hierarchical autoencoder for h = {hierarchies} (pad = {pad_value}, D = {latent_dim})', flush = True)

        for upd in range(1, base_updates + 1):
            progress = (upd - 1) / max(1, base_updates)
            ent_coef = schedule_at(BETA_ENTROPY_SCHEDULE, progress)

            for pg in opt_actor.param_groups:
                pg['lr'] = 1e-3 * schedule_at(LR_SCALE_SCHEDULE, progress)

            states, next_states, actions, log_probs, rewards, terms, truncs = [], [], [], [], [], [], []
            actor.eval()

            for _ in range(rollout_steps):
                with torch.no_grad():
                    act, lp = actor.sample(state)

                next_obs, reward, terminated, truncated, info = env.step(act.cpu().numpy())
                done = terminated | truncated
                running += reward

                for idx in np.flatnonzero(done):
                    recent.append(float(running[idx]))
                    running[idx] = 0.

                states.append(state)
                next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                actions.append(act)
                log_probs.append(lp)
                rewards.append(torch.from_numpy(reward / 100.).float().to(device))
                terms.append(torch.from_numpy(terminated).bool().to(device))
                truncs.append(torch.from_numpy(truncated).bool().to(device))
                state = torch.from_numpy(next_obs).float().to(device)

            states_t, next_states_t = stack(states, dim = 1), stack(next_states, dim = 1)
            actions_t, old_lp_t = stack(actions, dim = 1), stack(log_probs, dim = 1)
            rewards_t, term_t, trunc_t = stack(rewards, dim = 1), stack(terms, dim = 1), stack(truncs, dim = 1)

            buffer.add_rollout(actions_t, term_t | trunc_t)

            with torch.no_grad():
                values = critic(states_t)
                gae = calc_gae(
                    rewards = rewards_t,
                    values = values,
                    next_values = critic(next_states_t),
                    terminated = term_t,
                    truncated = trunc_t,
                    gamma = 0.99,
                    lam = 0.95,
                    use_accelerated = False,
                    return_advantages = True
                )

            actor.train()
            critic.train()
            eff_eps = 0.2 * math.sqrt(num_actions / 2.)

            for _ in range(6):
                perm = randperm(num_envs, device = device)

                for start in range(0, num_envs, 4):
                    mb = perm[start:start + 4]
                    mb_s = rearrange(states_t[mb], 'b t d -> (b t) d')
                    mb_a = rearrange(actions_t[mb], 'b t a -> (b t) a')
                    params = actor(mb_s)

                    a_loss = hpo_actor_loss(
                        actor.log_prob(params, mb_a),
                        rearrange(old_lp_t[mb], 'b t -> (b t)'),
                        rearrange(gae.advantages[mb], 'b t -> (b t)'),
                        eps_clip = eff_eps
                    ).mean() - ent_coef * actor.entropy(params).mean()

                    opt_actor.zero_grad()
                    a_loss.backward()
                    clip_grad_norm_(actor.parameters(), 0.5)
                    opt_actor.step()

                    c_loss = clipped_value_loss_fn(
                        critic(mb_s),
                        rearrange(values[mb], 'b t -> (b t)'),
                        rearrange(gae.returns[mb], 'b t -> (b t)'),
                        clip = None
                    )

                    opt_critic.zero_grad()
                    c_loss.backward()
                    opt_critic.step()

            if divisible_by(upd, 5) and buffer.ready(64):
                buffer.train_steps(autoencoder, opt_ae, steps = 120, batch_size = 256, prior_init_conc = prior_init_conc)

            if divisible_by(upd, 15) or upd == base_updates:
                rec32 = float(np.mean(recent[-32:])) if recent else float('nan')
                print(f'  [stage 1 upd {upd:3d}/{base_updates}] ret32 = {rec32:+6.1f} [{time.time() - t0:.1f}s]', flush = True)

        env.close()

        buffer.train_steps(autoencoder, opt_ae, steps = 500, batch_size = 512, prior_init_conc = prior_init_conc)
        autoencoder.eval()
        autoencoder.requires_grad_(False)

        ae_report = buffer.evaluate_per_hierarchy(autoencoder)

        with torch.no_grad():
            probe_actor = BetaActor(state_dim, 108, latent_dim, beta_init_conc = prior_init_conc).to(device)
            z_prior, _ = probe_actor.sample(torch.randn(8192, state_dim, device = device))
            leg_prior, *_ = autoencoder.decode(z_prior)
            prior_mix = {h: 100. * float((leg_prior == h).float().mean().item()) for h in autoencoder.hierarchies}

        for h, st in ae_report.items():
            print(
                f'  [frozen shared AE h={h}] mse={st["mse"]:.5f} | mae={st["mae"]:.4f} | '
                f'round_acc={st["round_acc"]:.2f}% | pred_len={st["mean_pred_len"]:.2f}+/-{st["std_pred_len"]:.2f} | '
                f'prior_share={prior_mix[h]:.1f}%',
                flush = True
            )

        return autoencoder, dict(per_hierarchy = ae_report, prior_mix = prior_mix), buffer

# macro environment

class SharedContinuousHierarchyEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        autoencoder: SharedActionChunkTanh,
        gamma = 0.99
    ):
        super().__init__(env)
        self.autoencoder = autoencoder
        self.hierarchies = np.asarray(autoencoder.hierarchies, dtype = np.int64)
        self.horizons_f = np.asarray(autoencoder.hierarchies, dtype = np.float32)
        self.max_hierarchy = autoencoder.max_hierarchy
        self.num_actions = autoencoder.num_actions
        self.latent_dim = autoencoder.latent_dim
        self.gamma = gamma

        self.action_space = gym.spaces.Box(
            low = -1.,
            high = 1.,
            shape = (self.latent_dim,),
            dtype = np.float32
        )

    def nearest_legit_hierarchy(self, pred_len):
        idx = int(np.argmin(np.abs(self.horizons_f - pred_len)))
        return int(self.hierarchies[idx])

    def step(self, action):
        arr = np.asarray(action, dtype = np.float32)

        # accepts either raw latent z of shape (latent_dim,) or pre-decoded [pred_len_float, flat_actions]

        if arr.shape == (self.latent_dim,):
            dev = next(self.autoencoder.parameters()).device

            with torch.no_grad():
                z = rearrange(torch.from_numpy(arr).float().to(dev), 'd -> 1 d')
                pred_len_t, pred_act_t, _ = self.autoencoder.decode_raw(z.clamp(-1., 1.))
                pred_len = float(pred_len_t[0].item())
                prim_seq = pred_act_t[0].cpu().numpy()
        else:
            pred_len = float(arr[0])
            prim_seq = arr[1:].reshape(self.max_hierarchy, self.num_actions)

        h = self.nearest_legit_hierarchy(pred_len)

        disc_reward, raw_reward, duration = 0., 0., 0

        for step_idx in range(h):
            obs, reward, terminated, truncated, info = self.env.step(np.clip(prim_seq[step_idx], -1., 1.))
            last_reward = float(reward)
            disc_reward += (self.gamma ** step_idx) * last_reward
            raw_reward += last_reward
            duration += 1

            if terminated or truncated:
                break

        return obs, disc_reward, terminated, truncated, {
            **info,
            'raw_reward': raw_reward,
            'duration': duration,
            'horizon': float(h),
            'pred_length': pred_len
        }

# option embedding & semi-markov ppo

class SharedContinuousOptionEmbed(Module):
    def __init__(
        self,
        autoencoder: SharedActionChunkTanh,
        embed_dim = 64,
        gamma = 0.99
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.num_hierarchies = autoencoder.num_hierarchies
        self.num_actions = autoencoder.num_actions
        self.max_hierarchy = autoencoder.max_hierarchy

        gamma_pows = rearrange(tensor([gamma ** j for j in range(self.max_hierarchy)], dtype = torch.float32), 'h -> h 1')
        disc_sums = tensor([sum(gamma ** j for j in range(h)) for h in autoencoder.hierarchies], dtype = torch.float32)
        hor_stats = stack([
            disc_sums / disc_sums.max().clamp_min(1e-6),
            tensor([gamma ** h for h in autoencoder.hierarchies], dtype = torch.float32)
        ], dim = -1)

        self.register_buffer('gamma_pows', gamma_pows, persistent = False)
        self.register_buffer('disc_sums', disc_sums, persistent = False)
        self.register_buffer('hor_stats', hor_stats, persistent = False)

        phys_dim = self.max_hierarchy * self.num_actions + 2 * self.num_actions + self.num_hierarchies + 3

        self.phys_proj = Sequential(
            Linear(phys_dim, embed_dim),
            SiLU(),
            Linear(embed_dim, embed_dim),
            SiLU()
        )

        self.latent_proj = Sequential(
            Linear(autoencoder.latent_dim, embed_dim),
            SiLU(),
            Linear(embed_dim, embed_dim)
        )

    def forward(self, packed_q_actions):
        # packed_q_actions carries [h_idx, z_1..z_D] so NextLatentQCritic reads nominal horizon h_idx

        h_idx, z = packed_q_actions[..., 0].long(), packed_q_actions[..., 1:]

        with torch.no_grad():
            pred_len, pred_act, _ = self.autoencoder.decode_raw(z.clamp(-1., 1.))
            horizons = self.autoencoder.horizons_t[h_idx]
            valid_steps = rearrange(lens_to_mask(horizons, self.max_hierarchy).float(), '... h -> ... h 1')

            disc_seq = pred_act * valid_steps * self.gamma_pows
            duty = disc_seq.sum(dim = -2) / rearrange(self.disc_sums[h_idx], '... -> ... 1')

            last_idx = repeat(horizons - 1, '... -> ... 1 a', a = self.num_actions)
            last_a = rearrange(pred_act.gather(-2, last_idx), '... 1 a -> ... a')

            phys_feats = cat((
                rearrange(disc_seq, '... h a -> ... (h a)'),
                duty,
                last_a,
                F.one_hot(h_idx, self.num_hierarchies).float(),
                self.hor_stats[h_idx],
                rearrange(pred_len / float(self.max_hierarchy), '... -> ... 1')
            ), dim = -1)

        return self.phys_proj(phys_feats) + self.latent_proj(z)

class SharedSMDPRollout(NamedTuple):
    states: Tensor
    next_states: Tensor
    latents: Tensor
    old_log_probs: Tensor
    rewards: Tensor
    durations: Tensor
    terminated: Tensor
    truncated: Tensor
    cand_latents: Tensor | None = None
    cand_old_lp: Tensor | None = None

class SemiMarkovSharedContinuousPPO(Module):
    def __init__(
        self,
        state_dim,
        autoencoder: SharedActionChunkTanh,
        *,
        hidden_dim = 108,
        critic_hidden_dim = 192,
        beta_init_conc = 3.,
        trace_power = 1.,
        marginal_ent_coef = 0.015,
        use_q_filter = True,
        num_candidates = 8,
        q_action_embed_dim = 64,
        q_lam = 0.8,
        q_rate_advantage = False,
        conservative_weight = 0.25,
        soft_exit_coef = 0.12,
        soft_exit_kl_budget = 0.25,
        lr = 1e-3,
        critic_lr = 1.8e-3,
        gamma = 0.99,
        lam = 0.95,
        eps_clip = 0.2,
        epochs = 6,
        minibatch_envs = 4
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.actor = BetaActor(state_dim, hidden_dim, autoencoder.latent_dim, beta_init_conc = beta_init_conc)
        self.critic = Critic(state_dim, hidden_dim = critic_hidden_dim)

        self.trace_power = float(trace_power)
        self.marginal_ent_coef = float(marginal_ent_coef)
        self.num_candidates = max(1, int(num_candidates))
        self.q_lam = float(q_lam)
        self.q_rate_advantage = q_rate_advantage
        self.conservative_weight = float(conservative_weight)
        self.soft_exit_coef = float(soft_exit_coef)
        self.soft_exit_kl_budget = float(soft_exit_kl_budget)

        self.base_lr = lr
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.eff_eps_clip = eps_clip * math.sqrt(autoencoder.latent_dim / 2.)
        self.epochs = epochs
        self.minibatch_envs = minibatch_envs

        if use_q_filter:
            self.q_critic = NextLatentQCritic(
                self.critic,
                dim_embed = state_dim,
                action_embed = SharedContinuousOptionEmbed(autoencoder, embed_dim = q_action_embed_dim, gamma = gamma),
                action_embed_dim = q_action_embed_dim,
                action_horizons = autoencoder.horizons_np,
                discrete_actions = False,
                hidden_dim = critic_hidden_dim,
                depth = 2,
                gamma = gamma,
                next_latent_coef = 0.25,
                reward_coef = 0.5,
                terminal_coef = 0.25,
                duration_coef = 0.25
            )
            val_ids = {id(p) for p in self.critic.parameters()}
            self._q_trunk_params = [p for p in self.q_critic.parameters() if id(p) not in val_ids]
            self.opt_critic = torch.optim.AdamW(self.q_critic.parameters(), lr = critic_lr)
        else:
            self.q_critic = None
            self._q_trunk_params = []
            self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr = critic_lr)

        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr = lr)

    def set_lr_scale(self, scale):
        for pg in self.opt_actor.param_groups:
            pg['lr'] = self.base_lr * scale

    @torch.no_grad()
    def pack_for_q(self, latents):
        _, h_idx, _, _ = self.autoencoder.decode(latents)
        return cat((rearrange(h_idx.float(), '... -> ... 1'), latents), dim = -1)

    @torch.no_grad()
    def sample_rollout_action(self, state, filter_prob = 0.):
        if not exists(self.q_critic):
            z, lp = self.actor.sample(state)
            return z, lp, None, None

        cand_z, cand_lp = self.actor.sample_candidates(state, self.num_candidates)
        base_z, base_lp = cand_z[:, 0], cand_lp[:, 0]

        if filter_prob <= 0. or self.num_candidates <= 1:
            return base_z, base_lp, cand_z, cand_lp

        filtered = self.q_critic.filter_actions(
            state,
            self.pack_for_q(cand_z),
            rate_advantage = self.q_rate_advantage,
            conservative_weight = self.conservative_weight
        )

        b_idx = arange(state.shape[0], device = state.device)
        filt_z = cand_z[b_idx, filtered.indices]
        filt_lp = cand_lp[b_idx, filtered.indices]

        if filter_prob >= 1.:
            return filt_z, filt_lp, cand_z, cand_lp

        use_f = torch.rand(state.shape[0], device = state.device) < filter_prob

        return (
            torch.where(rearrange(use_f, 'b -> b 1'), filt_z, base_z),
            torch.where(use_f, filt_lp, base_lp),
            cand_z,
            cand_lp
        )

    @torch.no_grad()
    def eval_action(self, state, use_q = True):
        if not (use_q and exists(self.q_critic) and self.num_candidates > 1):
            return self.actor.greedy(state)

        cand_z = self.actor.eval_candidates(state, k = self.num_candidates + 1)

        filtered = self.q_critic.filter_actions(
            state,
            self.pack_for_q(cand_z),
            rate_advantage = self.q_rate_advantage,
            conservative_weight = self.conservative_weight
        )

        return cand_z[arange(state.shape[0], device = state.device), filtered.indices]

    def update(
        self,
        rollout: SharedSMDPRollout,
        *,
        beta_entropy_coef = 0.0004,
        distill_active = True
    ):
        num_envs = rollout.states.shape[0]

        with torch.no_grad():
            values, next_values = self.critic(rollout.states), self.critic(rollout.next_states)

            gae = calc_gae(
                rewards = rollout.rewards,
                values = values,
                next_values = next_values,
                terminated = rollout.terminated,
                truncated = rollout.truncated,
                durations = rollout.durations,
                trace_power = self.trace_power,
                gamma = self.gamma,
                lam = self.lam,
                use_accelerated = False,
                return_advantages = True
            )

            q_gae, cand_adv = None, None

            if exists(self.q_critic):
                q_gae = calc_gae(
                    rewards = rollout.rewards,
                    values = values,
                    next_values = next_values,
                    terminated = rollout.terminated,
                    truncated = rollout.truncated,
                    durations = rollout.durations,
                    trace_power = self.trace_power,
                    gamma = self.gamma,
                    lam = self.q_lam,
                    use_accelerated = False,
                    return_advantages = True
                )

                if distill_active and self.soft_exit_coef > 0. and exists(rollout.cand_latents) and rollout.cand_latents.shape[-2] > 1:
                    scores = self.q_critic.smdp_advantage_scores(
                        rollout.states,
                        self.pack_for_q(rollout.cand_latents),
                        rate_advantage = self.q_rate_advantage,
                        conservative_weight = self.conservative_weight
                    )
                    weights = soft_exit_weights(scores, kl_budget = self.soft_exit_kl_budget)
                    k_cands = rollout.cand_latents.shape[-2]
                    cand_adv = (weights - (1. / k_cands)) * k_cands

        self.train()
        self.autoencoder.eval()

        for _ in range(self.epochs):
            perm = randperm(num_envs, device = rollout.states.device)

            for start in range(0, num_envs, self.minibatch_envs):
                mb = perm[start:start + self.minibatch_envs]
                mb_s = rearrange(rollout.states[mb], 'b t d -> (b t) d')
                mb_z = rearrange(rollout.latents[mb], 'b t a -> (b t) a')
                mb_old_lp = rearrange(rollout.old_log_probs[mb], 'b t -> (b t)')
                mb_adv = rearrange(gae.advantages[mb], 'b t -> (b t)')
                mb_ret = rearrange(gae.returns[mb], 'b t -> (b t)')

                params = self.actor(mb_s)
                new_lp = self.actor.log_prob(params, mb_z)

                surr_loss = hpo_actor_loss(
                    new_lp,
                    mb_old_lp,
                    mb_adv,
                    eps_clip = self.eff_eps_clip,
                    normalize_advantages = True
                ).mean()

                if exists(cand_adv):
                    mb_cand_z = rearrange(rollout.cand_latents[mb], 'b t k a -> (b t) k a')
                    mb_cand_old_lp = rearrange(rollout.cand_old_lp[mb], 'b t k -> (b t) k')
                    mb_cand_adv = rearrange(cand_adv[mb], 'b t k -> (b t) k')

                    new_cand_lp = self.actor.beta.log_prob(rearrange(params, 'b a p -> b 1 a p'), mb_cand_z, sum_action_dim = True)

                    exit_loss = hpo_actor_loss(
                        new_cand_lp,
                        mb_cand_old_lp,
                        mb_cand_adv,
                        eps_clip = self.eff_eps_clip,
                        normalize_advantages = False
                    ).mean()

                    surr_loss = surr_loss + self.soft_exit_coef * exit_loss

                ent_bonus = beta_entropy_coef * self.actor.entropy(params).mean()

                if self.marginal_ent_coef > 0.:
                    soft_h = self.autoencoder.soft_hierarchy_probs(self.actor.beta.mean(params))
                    ent_bonus = ent_bonus + self.marginal_ent_coef * normalized_entropy(soft_h.mean(dim = 0))

                self.opt_actor.zero_grad()
                (surr_loss - ent_bonus).backward()
                clip_grad_norm_(self.actor.parameters(), 0.5)
                self.opt_actor.step()

                if exists(self.q_critic):
                    critic_loss = self.q_critic(
                        mb_s,
                        self.pack_for_q(mb_z),
                        rewards = rearrange(rollout.rewards[mb], 'b t -> (b t)'),
                        next_embeddings = rearrange(rollout.next_states[mb], 'b t d -> (b t) d'),
                        returns = mb_ret,
                        dones = rearrange(rollout.terminated[mb], 'b t -> (b t)'),
                        durations = rearrange(rollout.durations[mb], 'b t -> (b t)'),
                        q_targets = rearrange(q_gae.returns[mb], 'b t -> (b t)')
                    )
                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    clip_grad_norm_(self._q_trunk_params, 1.)
                    self.opt_critic.step()
                else:
                    critic_loss = clipped_value_loss_fn(
                        self.critic(mb_s),
                        rearrange(values[mb], 'b t -> (b t)'),
                        mb_ret,
                        clip = None
                    )
                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    self.opt_critic.step()

# evaluation, video & plotting

@torch.no_grad()
def evaluate_policy(
    ppo: SemiMarkovSharedContinuousPPO,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    num_episodes = 48,
    use_q = True,
    seed = 1000
):
    ppo.eval()
    ae = ppo.autoencoder
    base_env = gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True)
    env = SharedContinuousHierarchyEnv(base_env, ae)

    episodes = []
    dec_by_h = {h: 0 for h in ae.hierarchies}
    step_by_h = {h: 0 for h in ae.hierarchies}
    q_overrides = 0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed = ep_seed)
        done, ep_ret, prim_steps, macro_steps, last_r, term_flag = False, 0., 0, 0, 0., False
        segments, dec_hs, pred_lens, latents_seq = [], [], [], []
        ep_dec_by_h = {h: 0 for h in ae.hierarchies}

        while not done:
            obs_t = rearrange(torch.from_numpy(obs).float().to(device), 'd -> 1 d')
            z_t = ppo.eval_action(obs_t, use_q = use_q)

            if use_q and exists(ppo.q_critic):
                greedy_z = ppo.actor.greedy(obs_t)
                q_overrides += int(not torch.allclose(z_t, greedy_z, atol = 1e-4))

            legit_h_t, _, pred_len_t, pred_act_t = ae.decode(z_t)
            h = int(legit_h_t[0].item())
            pred_len = float(pred_len_t[0].item())
            prim_seq = pred_act_t[0, :h].cpu().numpy()

            dec_by_h[h] += 1
            ep_dec_by_h[h] += 1
            dec_hs.append(h)
            pred_lens.append(pred_len)
            latents_seq.append(z_t[0].cpu().numpy().tolist())
            macro_steps += 1

            seg_pts = [(float(obs[0]), float(obs[1]))]

            for step_i in range(h):
                obs, reward, terminated, truncated, _ = base_env.step(np.clip(prim_seq[step_i], -1., 1.))
                seg_pts.append((float(obs[0]), float(obs[1])))
                last_r = float(reward)
                ep_ret += last_r
                prim_steps += 1
                step_by_h[h] += 1

                if terminated or truncated:
                    done = True
                    term_flag = bool(terminated)
                    break

            segments.append((h, seg_pts))

        episodes.append(dict(
            seed = ep_seed,
            return_val = ep_ret,
            prim_steps = prim_steps,
            macro_steps = macro_steps,
            landed = 1. if last_r >= 50. else 0.,
            strict_landed = 1. if is_strict_landing(obs, term_flag, last_r) else 0.,
            final_xy = [float(obs[0]), float(obs[1])],
            final_angle = float(obs[4]),
            dec_counts = ep_dec_by_h,
            used_hier_count = sum(1 for c in ep_dec_by_h.values() if c > 0),
            segments = segments,
            dec_hs = dec_hs,
            pred_lens = pred_lens,
            latents = latents_seq
        ))

    env.close()

    ranked_ret = sorted(episodes, key = lambda e: e['return_val'], reverse = True)
    ranked_mixed = sorted(episodes, key = lambda e: e['return_val'] + 6. * e['used_hier_count'], reverse = True)
    tot_dec, tot_step = max(1, sum(dec_by_h.values())), max(1, sum(step_by_h.values()))
    returns = [e['return_val'] for e in episodes]

    top_keys = ('seed', 'return_val', 'prim_steps', 'macro_steps', 'landed', 'strict_landed', 'final_xy', 'final_angle', 'dec_counts')

    return dict(
        mean_return = float(np.mean(returns)),
        std_return = float(np.std(returns)),
        median_return = float(np.median(returns)),
        best_return = float(ranked_ret[0]['return_val']),
        land_rate = 100. * float(np.mean([e['landed'] for e in episodes])),
        strict_land_rate = 100. * float(np.mean([e['strict_landed'] for e in episodes])),
        mean_prim_steps = float(np.mean([e['prim_steps'] for e in episodes])),
        mean_macro_steps = float(np.mean([e['macro_steps'] for e in episodes])),
        q_override_pct = 100. * q_overrides / tot_dec if (use_q and exists(ppo.q_critic)) else 0.,
        dec_frac = {h: 100. * dec_by_h[h] / tot_dec for h in ae.hierarchies},
        step_frac = {h: 100. * step_by_h[h] / tot_step for h in ae.hierarchies},
        top_seeds = [e['seed'] for e in ranked_mixed[:5]],
        top5 = [{k: e[k] for k in top_keys} for e in ranked_ret[:5]],
        best_episode = ranked_mixed[0],
        episodes = episodes
    )

@torch.no_grad()
def render_mixed_video(
    ppo: SemiMarkovSharedContinuousPPO,
    seed,
    mp4_path,
    gif_path,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600
):
    ppo.eval()
    ae = ppo.autoencoder
    env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_episode_steps, continuous = True)
    obs, _ = env.reset(seed = seed)
    font = ImageFont.load_default()

    frames, done, ep_ret, dec_idx = [], False, 0., 0
    dec_counts = {h: 0 for h in ae.hierarchies}

    while not done:
        obs_t = rearrange(torch.from_numpy(obs).float().to(device), 'd -> 1 d')
        z_t = ppo.eval_action(obs_t, use_q = True)
        legit_h_t, _, pred_len_t, pred_act_t = ae.decode(z_t)

        h = int(legit_h_t[0].item())
        pred_len = float(pred_len_t[0].item())
        prim_seq = pred_act_t[0, :h].cpu().numpy()

        z_str = ','.join(f'{v:+.2f}' for v in z_t[0, :4].cpu().tolist()) + '..'
        col = HIER_COLORS_RGB.get(h, (120, 130, 150))
        dec_counts[h] += 1
        dec_idx += 1

        for step_i in range(h):
            obs, reward, terminated, truncated, _ = env.step(np.clip(prim_seq[step_i], -1., 1.))
            ep_ret += float(reward)
            rgb = env.render()

            canvas = Image.new('RGB', (rgb.shape[1] + 8, rgb.shape[0] + 44), (20, 24, 33))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([0, 0, canvas.width - 1, rgb.shape[0] + 7], fill = col)
            canvas.paste(Image.fromarray(rgb), (4, 4))
            draw.text(
                (10, rgb.shape[0] + 12),
                f'len={pred_len:4.2f}->dt={h} ({step_i + 1}/{h}) | z=[{z_str}] | ret={ep_ret:+6.1f}',
                fill = (240, 244, 252),
                font = font
            )
            frames.append(np.asarray(canvas))

            if terminated or truncated:
                done = True
                break

    env.close()

    mp4_path, gif_path = Path(mp4_path).resolve(), Path(gif_path).resolve()
    mp4_path.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(str(mp4_path), frames, fps = 50, macro_block_size = 1)
    imageio.mimsave(str(gif_path), frames[::2], fps = 25)

    print(f'[video] seed={seed} | return={ep_ret:+.1f} | decisions={dec_idx}/{len(frames)}f | {dec_counts} -> {mp4_path}', flush = True)
    return dict(seed = seed, episode_return = ep_ret, decisions = dec_idx, frames = len(frames), dec_counts = dec_counts, mp4_path = str(mp4_path))

def plot_results(
    histories,
    evals,
    ae: SharedActionChunkTanh,
    ae_report,
    out_path: Path,
    diag_path: Path | None = None
):
    fig, axes = plt.subplots(2, 2, figsize = (14.8, 9.8), dpi = 150)
    colors = {'shared_smdp_pure': '#457b9d', 'shared_smdp': '#d62828', 'shared_smdp_q_filter': '#0f766e'}
    ax_curve, ax_mix, ax_traj, ax_len = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for k, rows in histories.items():
        ax_curve.plot(
            [r['env_steps'] / 1e3 for r in rows],
            [r['recent_return_32'] for r in rows],
            color = colors.get(k, '#2b6cb0'),
            linewidth = 2.2,
            label = f'{k} (eval {evals[k]["mean_return"]:+.1f}, land {evals[k]["land_rate"]:.1f}%)'
        )

    ax_curve.axhline(200., color = 'green', linestyle = '--', linewidth = 1.)
    ax_curve.set(xlabel = 'primitive env steps (k)', ylabel = 'rolling 32-ep return', title = f'Shared Hierarchical Autoencoder ({ae.latent_dim}D, pad={ae.pad_value}) Learning Curve')
    ax_curve.legend(loc = 'lower right', fontsize = 8.5)
    ax_curve.grid(alpha = 0.25)

    primary_key = 'shared_smdp_q_filter' if 'shared_smdp_q_filter' in histories else next(iter(histories))
    div_rows = histories[primary_key]

    ax_mix.stackplot(
        [r['env_steps'] / 1e3 for r in div_rows],
        *[[r['dec_pct'][h] for r in div_rows] for h in ae.hierarchies],
        labels = [f'dt = {h}' for h in ae.hierarchies],
        colors = [HIER_COLORS_HEX.get(h, '#555555') for h in ae.hierarchies],
        alpha = 0.85
    )
    ax_mix.set(ylim = (0, 100), xlabel = 'primitive env steps (k)', ylabel = 'decision share (%)', title = f'Emergent Rounded Hierarchy Share ({primary_key})')
    ax_mix.legend(loc = 'upper right', fontsize = 8.5)

    ev_d = evals[primary_key]
    ep = ev_d['best_episode']
    segs = ep['segments']
    lines = [[pts[i], pts[i + 1]] for _, pts in segs for i in range(len(pts) - 1)]
    cols = [HIER_COLORS_HEX.get(h, '#555555') for h, pts in segs for _ in range(len(pts) - 1)]

    if lines:
        ax_traj.add_collection(LineCollection(lines, colors = cols, linewidths = 2.4))
        ax_traj.scatter([pts[0][0] for _, pts in segs], [pts[0][1] for _, pts in segs], c = [HIER_COLORS_HEX.get(h, '#555555') for h, _ in segs], s = 18, zorder = 4)

    ax_traj.axvline(-0.2, color = '#2ecc71', linestyle = '--', linewidth = 1.)
    ax_traj.axvline(0.2, color = '#2ecc71', linestyle = '--', linewidth = 1.)
    ax_traj.axhline(0., color = '#555555', linewidth = 1.)
    ax_traj.legend(
        handles = [Patch(color = HIER_COLORS_HEX.get(h, '#555555'), label = f'dt={h}: {ev_d["dec_frac"][h]:.1f}% dec ({ev_d["step_frac"][h]:.1f}% steps)') for h in ae.hierarchies],
        loc = 'upper right',
        fontsize = 8
    )
    ax_traj.set(xlim = (-0.75, 0.75), ylim = (-0.08, 1.5), xlabel = 'x', ylabel = 'y', title = f'Trajectory Colored by Decoded & Rounded dt (seed {ep["seed"]}, ret {ep["return_val"]:+.1f})')
    ax_traj.grid(alpha = 0.25)

    pred_lens = np.asarray(ep['pred_lens'], dtype = np.float32)
    dec_hs = np.asarray(ep['dec_hs'], dtype = np.int64)
    t_steps = np.arange(pred_lens.shape[0])

    for t_i, h_chosen in enumerate(dec_hs):
        ax_len.axvspan(t_i - 0.5, t_i + 0.5, color = HIER_COLORS_HEX.get(int(h_chosen), '#cccccc'), alpha = 0.18, lw = 0)

    ax_len.plot(t_steps, pred_lens, color = '#1d3557', linewidth = 2., marker = 'o', markersize = 3.5, label = 'decoded float length $\\hat{h}(z_t)$')
    ax_len.step(t_steps, dec_hs, where = 'mid', color = '#e63946', linewidth = 1.4, linestyle = '--', label = 'rounded legit $h \\in \\{1,2,4,8\\}$')

    for mid in 0.5 * (ae.horizons_np[:-1] + ae.horizons_np[1:]):
        ax_len.axhline(float(mid), color = 'gray', linestyle = ':', linewidth = 0.9)

    ax_len.set(
        xlim = (-0.5, max(1., pred_lens.shape[0] - 0.5)),
        ylim = (0.5, float(ae.max_hierarchy) + 0.8),
        xlabel = 'decision index (dotted lines = rounding midpoints 1.5, 3.0, 6.0)',
        ylabel = 'chunk length (steps)',
        title = f'Continuous Decoded Length $\\hat{{h}}(z_t)$ & Rounding ({pred_lens.shape[0]} decisions)'
    )
    ax_len.legend(loc = 'upper right', fontsize = 8)
    ax_len.grid(alpha = 0.25)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches = 'tight')
    plt.close(fig)

    if not exists(diag_path):
        return

    fig2, axes2 = plt.subplots(1, 3, figsize = (15.5, 4.6), dpi = 150)
    per_h = ae_report.get('per_hierarchy', {})

    if per_h:
        hs = sorted(int(k) for k in per_h.keys())
        mses = [per_h[h]['mse'] if h in per_h else per_h[str(h)]['mse'] for h in hs]
        bars = axes2[0].bar([f'h={h}' for h in hs], mses, color = [HIER_COLORS_HEX.get(h, '#555555') for h in hs], alpha = 0.88)
        for b_rect, m_val in zip(bars, mses):
            axes2[0].text(b_rect.get_x() + b_rect.get_width() / 2., m_val + 0.001, f'{m_val:.4f}', ha = 'center', va = 'bottom', fontsize = 8)
        axes2[0].set(ylabel = 'valid-step action MSE', title = 'Stage 1 Shared AE Reconstruction MSE (100% Round Acc)')
        axes2[0].grid(axis = 'y', alpha = 0.25)

    all_pred_lens = [pl for e in ev_d['episodes'] for pl in e['pred_lens']]
    axes2[1].hist(all_pred_lens, bins = 44, color = '#2a9d8f', alpha = 0.82, edgecolor = 'white')

    for mid in 0.5 * (ae.horizons_np[:-1] + ae.horizons_np[1:]):
        axes2[1].axvline(float(mid), color = '#d62828', linestyle = '--', linewidth = 1.2)
    for h in ae.hierarchies:
        axes2[1].axvline(float(h), color = '#1d3557', linestyle = ':', linewidth = 1.)

    axes2[1].set(xlabel = 'decoded float length $\\hat{h}(z)$', ylabel = 'count across 48 eval episodes', title = f'Eval Decoded Length Distribution ({primary_key})')
    axes2[1].grid(alpha = 0.25)

    lat_arr = np.asarray(ep['latents'], dtype = np.float32)
    for t_i, h_chosen in enumerate(dec_hs):
        axes2[2].axvspan(t_i - 0.5, t_i + 0.5, color = HIER_COLORS_HEX.get(int(h_chosen), '#cccccc'), alpha = 0.16, lw = 0)
    for d_i in range(lat_arr.shape[1]):
        axes2[2].plot(t_steps, lat_arr[:, d_i], linewidth = 1.2, alpha = 0.85, label = f'z[{d_i}]')

    axes2[2].set(
        xlim = (-0.5, max(1., lat_arr.shape[0] - 0.5)),
        ylim = (-1.05, 1.05),
        xlabel = 'decision index',
        ylabel = 'shared tanh latent $z \\in (-1, 1)^D$',
        title = f'Shared {ae.latent_dim}D Latent Trajectory (seed {ep["seed"]})'
    )
    axes2[2].legend(loc = 'lower right', fontsize = 7, ncol = 2)
    axes2[2].grid(alpha = 0.25)

    plt.tight_layout()
    fig2.savefig(diag_path, bbox_inches = 'tight')
    plt.close(fig2)

# training

def train_shared_variant(
    *,
    name,
    autoencoder: SharedActionChunkTanh,
    beta_init_conc = 3.,
    marginal_ent_coef = 0.015,
    trace_power = 1.,
    use_q_filter = True,
    num_candidates = 8,
    q_lam = 0.8,
    q_filter_prob = 0.5,
    q_warmup_frac = 0.08,
    q_rate_advantage = False,
    conservative_weight = 0.25,
    soft_exit_coef = 0.12,
    soft_exit_kl_budget = 0.25,
    env_step_budget = 1_100_000,
    env_name = 'LunarLander-v3',
    num_envs = 16,
    rollout_decisions = 128,
    max_episode_steps = 600,
    val_every = 15,
    seed = 42
):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = SyncVectorEnv(
            [lambda: SharedContinuousHierarchyEnv(gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True), autoencoder) for _ in range(num_envs)],
            autoreset_mode = AutoresetMode.SAME_STEP
        )

        state_dim = int(env.single_observation_space.shape[0])

        ppo = SemiMarkovSharedContinuousPPO(
            state_dim,
            autoencoder,
            beta_init_conc = beta_init_conc,
            trace_power = trace_power,
            marginal_ent_coef = marginal_ent_coef,
            use_q_filter = use_q_filter,
            num_candidates = num_candidates,
            q_lam = q_lam,
            q_rate_advantage = q_rate_advantage,
            conservative_weight = conservative_weight,
            soft_exit_coef = soft_exit_coef,
            soft_exit_kl_budget = soft_exit_kl_budget
        ).to(device)

        obs, _ = env.reset(seed = seed + 100)
        state = torch.from_numpy(obs).float().to(device)
        running, recent, history = np.zeros(num_envs), [], []

        best_val = -float('inf')
        best_actor_state = deepcopy(ppo.actor.state_dict())
        best_q_state = deepcopy(ppo.q_critic.state_dict()) if exists(ppo.q_critic) else None
        env_steps, upd, t0 = 0, 0, time.time()

        while env_steps < env_step_budget:
            upd += 1
            progress = min(1., env_steps / max(1, env_step_budget))
            ppo.set_lr_scale(schedule_at(LR_SCALE_SCHEDULE, progress))

            warmed_up = progress >= q_warmup_frac
            active_filter_prob = q_filter_prob if (use_q_filter and warmed_up) else 0.

            states, next_states, latents_list, old_lps = [], [], [], []
            cand_zs, cand_lps = [], []
            rewards, durations, terms, truncs, ep_rets = [], [], [], [], []
            h_counts = {h: 0 for h in autoencoder.hierarchies}

            ppo.eval()

            for _ in range(rollout_decisions):
                with torch.no_grad():
                    z_t, lp_t, cand_z_t, cand_lp_t = ppo.sample_rollout_action(state, filter_prob = active_filter_prob)
                    cmds_np = autoencoder.to_env_commands(z_t)

                next_obs, disc_r, terminated, truncated, info = env.step(cmds_np)
                done = terminated | truncated
                raw_r = step_info_array(info, 'raw_reward', done, 0.)
                durs = step_info_array(info, 'duration', done, 1.)
                chosen_hs = step_info_array(info, 'horizon', done, 1.)

                for h_val in chosen_hs.astype(np.int64).tolist():
                    if h_val in h_counts:
                        h_counts[h_val] += 1

                running += raw_r
                env_steps += int(durs.sum())

                for idx in np.flatnonzero(done):
                    ep_rets.append(float(running[idx]))
                    recent.append(float(running[idx]))
                    running[idx] = 0.

                states.append(state)
                next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                latents_list.append(z_t)
                old_lps.append(lp_t)

                if exists(cand_z_t):
                    cand_zs.append(cand_z_t)
                    cand_lps.append(cand_lp_t)

                rewards.append(torch.from_numpy(disc_r / 100.).float().to(device))
                durations.append(torch.from_numpy(durs).float().to(device))
                terms.append(torch.from_numpy(terminated).bool().to(device))
                truncs.append(torch.from_numpy(truncated).bool().to(device))
                state = torch.from_numpy(next_obs).float().to(device)

            ppo.update(
                SharedSMDPRollout(
                    states = stack(states, dim = 1),
                    next_states = stack(next_states, dim = 1),
                    latents = stack(latents_list, dim = 1),
                    old_log_probs = stack(old_lps, dim = 1),
                    rewards = stack(rewards, dim = 1),
                    durations = stack(durations, dim = 1),
                    terminated = stack(terms, dim = 1),
                    truncated = stack(truncs, dim = 1),
                    cand_latents = stack(cand_zs, dim = 1) if cand_zs else None,
                    cand_old_lp = stack(cand_lps, dim = 1) if cand_lps else None
                ),
                beta_entropy_coef = schedule_at(BETA_ENTROPY_SCHEDULE, progress),
                distill_active = warmed_up
            )

            rec32 = float(np.mean(recent[-32:])) if recent else float('nan')
            upd_ret = float(np.mean(ep_rets)) if ep_rets else rec32

            if divisible_by(upd, val_every) or env_steps >= env_step_budget:
                val_ret = evaluate_policy(
                    ppo,
                    env_name = env_name,
                    max_episode_steps = max_episode_steps,
                    num_episodes = 12,
                    use_q = warmed_up,
                    seed = seed + 5000
                )['mean_return']

                if val_ret >= best_val:
                    best_val = val_ret
                    best_actor_state = deepcopy(ppo.actor.state_dict())
                    if exists(ppo.q_critic):
                        best_q_state = deepcopy(ppo.q_critic.state_dict())

            tot_dec = max(1, sum(h_counts.values()))
            dec_pct = {h: 100. * h_counts[h] / tot_dec for h in autoencoder.hierarchies}

            history.append(dict(
                variant = name,
                seed = seed,
                update = upd,
                env_steps = env_steps,
                recent_return_32 = rec32,
                update_return = upd_ret,
                dec_pct = dec_pct
            ))

            if divisible_by(upd, 15) or env_steps >= env_step_budget:
                mix_str = ' '.join(f'dt{h}:{dec_pct[h]:4.1f}%' for h in autoencoder.hierarchies)
                print(f'  [{name:20s} upd {upd:3d}] steps = {env_steps // 1000:4d}k | ret32 = {rec32:+6.1f} (val = {best_val:+6.1f}) | {mix_str} [{time.time() - t0:.1f}s]', flush = True)

        env.close()

    ppo.actor.load_state_dict(best_actor_state)
    if exists(ppo.q_critic) and exists(best_q_state):
        ppo.q_critic.load_state_dict(best_q_state)

    eval_stats = evaluate_policy(ppo, env_name = env_name, max_episode_steps = max_episode_steps, use_q = True)

    if exists(ppo.q_critic):
        actor_ev = evaluate_policy(ppo, env_name = env_name, max_episode_steps = max_episode_steps, use_q = False)
        eval_stats['actor_only_mean_return'] = actor_ev['mean_return']
        eval_stats['actor_only_land_rate'] = actor_ev['land_rate']
        eval_stats['actor_only_strict_land_rate'] = actor_ev['strict_land_rate']
    else:
        eval_stats['actor_only_mean_return'] = None
        eval_stats['actor_only_land_rate'] = None
        eval_stats['actor_only_strict_land_rate'] = None

    return history, ppo, eval_stats

def main(
    env_name = 'LunarLander-v3',
    out_folder = './continuous-shared-smdp',
    shared_ae_ckpt = None,
    hierarchies = (1, 2, 4, 8),
    latent_dim = 8,
    compressor_dim = 192,
    pad_value = -2.,
    beta_init_conc = 3.,
    base_updates = 135,
    base_rollout_steps = 384,
    env_step_budget = 1_100_000,
    marginal_ent_coef = 0.015,
    trace_power = 1.,
    use_q_filter = True,
    compare_q_filter = False,
    include_pure_unreg = False,
    num_candidates = 8,
    q_lam = 0.8,
    q_filter_prob = 0.5,
    q_warmup_frac = 0.08,
    q_rate_advantage = False,
    conservative_weight = 0.25,
    soft_exit_coef = 0.12,
    soft_exit_kl_budget = 0.25,
    num_videos = 2,
    seed = 42
):
    out_dir = Path(out_folder)
    out_dir.mkdir(parents = True, exist_ok = True)

    hs = tuple(sorted(int(h) for h in hierarchies))

    if exists(shared_ae_ckpt):
        autoencoder, ae_report = load_shared_autoencoder(shared_ae_ckpt)
    else:
        autoencoder, ae_report, _ = discover_shared_autoencoder(
            env_name = env_name,
            hierarchies = hs,
            latent_dim = latent_dim,
            compressor_dim = compressor_dim,
            pad_value = pad_value,
            prior_init_conc = beta_init_conc,
            base_updates = base_updates,
            rollout_steps = base_rollout_steps,
            seed = seed
        )

        torch.save(dict(
            hierarchies = autoencoder.hierarchies,
            num_actions = autoencoder.num_actions,
            latent_dim = autoencoder.latent_dim,
            dim = autoencoder.dim,
            pad_value = autoencoder.pad_value,
            autoencoder = autoencoder.state_dict(),
            ae_report = ae_report
        ), out_dir / 'shared_tanh_ae.pt')

    print(
        f'\n[stage 2] unified continuous SMDP training over single {autoencoder.latent_dim}D shared AE '
        f'(hierarchies={autoencoder.hierarchies}, pad={autoencoder.pad_value})',
        flush = True
    )

    if compare_q_filter:
        variants = ([('shared_smdp_pure', False, 0.)] if include_pure_unreg else []) + [
            ('shared_smdp', False, marginal_ent_coef),
            ('shared_smdp_q_filter', True, marginal_ent_coef)
        ]
    else:
        variants = [('shared_smdp_q_filter' if use_q_filter else 'shared_smdp', use_q_filter, marginal_ent_coef)]

    histories, agents, evals = {}, {}, {}

    for name, q_filt, marg_coef in variants:
        hist, agent, ev = train_shared_variant(
            name = name,
            autoencoder = autoencoder,
            beta_init_conc = beta_init_conc,
            marginal_ent_coef = marg_coef,
            trace_power = trace_power,
            use_q_filter = q_filt,
            num_candidates = num_candidates,
            q_lam = q_lam,
            q_filter_prob = q_filter_prob,
            q_warmup_frac = q_warmup_frac,
            q_rate_advantage = q_rate_advantage,
            conservative_weight = conservative_weight,
            soft_exit_coef = soft_exit_coef,
            soft_exit_kl_budget = soft_exit_kl_budget,
            env_step_budget = env_step_budget,
            env_name = env_name,
            seed = seed
        )

        histories[name], agents[name], evals[name] = hist, agent, ev
        mix = ' '.join(f'dt{h}:{ev["dec_frac"][h]:.1f}%' for h in autoencoder.hierarchies)
        extra = f' | actor_only {ev["actor_only_mean_return"]:+.1f} | q_override {ev["q_override_pct"]:.1f}%' if q_filt else ''

        print(
            f'  [{name:20s} eval] {ev["mean_return"]:+.1f} +/- {ev["std_return"]:.1f} '
            f'(median {ev["median_return"]:+.1f}, land {ev["land_rate"]:.1f}%, strict {ev["strict_land_rate"]:.1f}%, '
            f'best {ev["best_return"]:+.1f}) | {mix}{extra}',
            flush = True
        )

    with open(out_dir / 'continuous_shared_smdp_curves.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'seed', 'update', 'env_steps', 'recent_return_32', 'update_return', *[f'dec_pct_h{h}' for h in autoencoder.hierarchies]])
        for rows in histories.values():
            for r in rows:
                writer.writerow([r['variant'], r['seed'], r['update'], r['env_steps'], r['recent_return_32'], r['update_return'], *[r['dec_pct'][h] for h in autoencoder.hierarchies]])

    primary_key = 'shared_smdp_q_filter' if 'shared_smdp_q_filter' in agents else next(reversed(agents))
    primary_agent = agents[primary_key]
    videos = []

    if num_videos > 0:
        for i, sd in enumerate(evals[primary_key]['top_seeds'][:num_videos]):
            prefix = 'continuous-shared-smdp-q-filter' if primary_key == 'shared_smdp_q_filter' else 'continuous-shared-smdp-mixed'
            stem = f'{prefix}-best' if i == 0 else f'{prefix}-rank{i + 1}-s{sd}'
            videos.append(render_mixed_video(
                primary_agent, sd, out_dir / f'{stem}.mp4', out_dir / f'{stem}.gif', env_name = env_name
            ))

    plot_results(
        histories,
        evals,
        autoencoder,
        ae_report,
        out_dir / 'continuous_shared_smdp_results.png',
        diag_path = out_dir / 'shared_autoencoder_diagnostics.png'
    )

    summary_payload = dict(
        env_name = env_name,
        continuous = True,
        hierarchies = list(autoencoder.hierarchies),
        latent_dim = autoencoder.latent_dim,
        pad_value = autoencoder.pad_value,
        seed = seed,
        marginal_ent_coef = marginal_ent_coef,
        ae_report = ae_report,
        variants = {
            k: {sk: sv for sk, sv in ev.items() if sk not in ('best_episode', 'episodes', 'top_seeds')}
            for k, ev in evals.items()
        },
        videos = videos
    )

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary_payload, f, indent = 2)

    torch.save(dict(
        hierarchies = autoencoder.hierarchies,
        latent_dim = autoencoder.latent_dim,
        pad_value = autoencoder.pad_value,
        autoencoder = autoencoder.state_dict(),
        ae_report = ae_report,
        evals = {k: {sk: sv for sk, sv in ev.items() if sk != 'episodes'} for k, ev in evals.items()},
        histories = histories,
        videos = videos,
        actor = primary_agent.actor.state_dict(),
        critic = primary_agent.critic.state_dict(),
        q_critic = primary_agent.q_critic.state_dict() if exists(primary_agent.q_critic) else None
    ), out_dir / 'ppo_continuous_shared_smdp.pt')

if __name__ == '__main__':
    fire.Fire(main)
