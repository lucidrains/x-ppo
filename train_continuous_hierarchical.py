# /// script
# dependencies = [
#   "assoc-scan",
#   "einops",
#   "einx",
#   "fire",
#   "gymnasium[box2d,other]",
#   "imageio[ffmpeg]",
#   "matplotlib",
#   "mean-conc-beta>=0.2.7",
#   "numpy",
#   "torch",
#   "torch-einops-utils>=0.2.5",
# ]
# ///

"""
Hierarchical Continuous Action Compression via Tanh Bottlenecks.

Continuous counterpart to `train_vq_hierarchical.py` on `LunarLander-v3` (continuous = True):
Stage 1 trains a dt = 1 policy with a `mean-conc-beta` actor while intermittently compressing
contiguous h-step action chunks `(h, 2)` into lower-dimensional `tanh`-squashed continuous
latents `z in (-1, 1)^d`.
Stage 2 freezes each tanh decoder inside an h-times-slower semi-Markov macro-environment and
trains a fresh `mean-conc-beta` macro-actor over `z in (-1, 1)^d`.

uv run train_continuous_hierarchical.py --sweep
"""

from __future__ import annotations

import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import fire
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import torch
from torch import Tensor, arange, cat, randint, randn_like, randperm, stack, zeros
import torch.nn.functional as F
from torch.nn import Dropout, Linear, Module, RMSNorm, Sequential, SiLU, init
from torch.nn.utils import clip_grad_norm_

from einops import rearrange
from einops.layers.torch import Rearrange

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from mean_conc_beta import Beta as MeanConcBeta

from x_ppo import (
    calc_gae,
    clipped_value_loss_fn,
    hpo_actor_loss,
    ppo_actor_loss
)

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

ENTROPY_SCHEDULE = (0.0016, 0.0008, 0.0004, 0.0002, 0.0001, 0.00004)
LR_SCALE_SCHEDULE = (1., 0.85, 0.50, 0.20, 0.10, 0.05)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def schedule_at(schedule, progress):
    idx = min(int(progress * len(schedule)), len(schedule) - 1)
    return schedule[max(0, idx)]

def resolve_latent_dim(hierarchy: int, num_actions: int, latent_dim: int | None = None) -> int:
    if exists(latent_dim):
        return int(latent_dim)

    chunk_dim = hierarchy * num_actions
    if hierarchy <= 1:
        return num_actions

    # logarithmic latent growth matching VQ bit scaling: (4 -> 3, 8 -> 4, 16 -> 6) for h = (2, 4, 8)

    target = max(num_actions + 1, num_actions * int(round(math.log2(hierarchy))))
    return min(chunk_dim - 1, target)

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
        f_info = info['final_info']
        f_mask = f_info.get(f'_{key}', done)
        out[f_mask] = f_info[key][f_mask]

    return out

def is_strict_landing(obs: np.ndarray, terminated: bool, last_reward: float) -> bool:
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
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        *,
        beta_init_conc = 5.,
        beta_min_conc = 0.,
        beta_eps = 1e-5,
        dropout = 0.
    ):
        super().__init__()
        self.action_dim = action_dim
        dim = hidden_dim * 2

        self.net = Sequential(
            Linear(state_dim, dim),
            SimBa(dim, dropout = dropout),
            SimBa(dim, dropout = dropout),
            Linear(dim, hidden_dim),
            SiLU(),
            Linear(hidden_dim, 2 * action_dim)
        )

        init.orthogonal_(self.net[-1].weight, gain = 0.01)
        init.zeros_(self.net[-1].bias)

        self.beta = MeanConcBeta(
            bounds = (-1., 1.),
            init_conc = beta_init_conc,
            min_conc = beta_min_conc,
            unimodal = True,
            eps = beta_eps
        )

    def forward(self, state: Tensor) -> Tensor:
        out = self.net(state)
        return rearrange(out, '... (a two) -> ... a two', two = 2)

    def dist(self, params: Tensor):
        return self.beta(params)

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        params = self.forward(state)
        dist = self.beta(params)
        action = dist.sample()
        return action, self.beta.log_prob(dist, action, sum_action_dim = True)

    def greedy(self, state: Tensor) -> Tensor:
        return self.beta.mean(self.forward(state))

    def log_prob(self, params: Tensor, action: Tensor) -> Tensor:
        return self.beta.log_prob(params, action, sum_action_dim = True)

    def entropy(self, params: Tensor) -> Tensor:
        # mean per-dim entropy anchored to 2D primitive space so schedule stays invariant across latent_dim

        return self.beta.entropy(params, sum_action_dim = False).mean(dim = -1) * 2.

class Critic(Module):
    def __init__(
        self,
        state_dim: int,
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

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).squeeze(-1)

# tanh action chunk compressor

class ActionChunkTanh(Module):
    def __init__(
        self,
        *,
        hierarchy: int,
        num_actions: int,
        state_dim: int,
        latent_dim: int | None = None,
        dim = 128,
        latent_noise = 0.02,
        prepend_first_state = False,
        state_recon_weight = 0.1
    ):
        super().__init__()
        resolved_latent_dim = resolve_latent_dim(hierarchy, num_actions, latent_dim)

        self.hierarchy = hierarchy
        self.num_actions = num_actions
        self.chunk_dim = hierarchy * num_actions
        self.latent_dim = resolved_latent_dim
        self.latent_noise = latent_noise
        self.prepend_first_state = prepend_first_state
        self.state_recon_weight = state_recon_weight

        enc_in_dim = self.chunk_dim + (state_dim if prepend_first_state else 0)
        self.state_norm = RMSNorm(state_dim) if prepend_first_state else None

        # continuous bottleneck: RMSNorm -> Linear -> tanh

        self.encoder = Sequential(
            Linear(enc_in_dim, dim),
            SimBa(dim),
            SimBa(dim),
            RMSNorm(dim),
            Linear(dim, resolved_latent_dim)
        )

        self.decoder = Sequential(
            Linear(resolved_latent_dim, dim),
            SimBa(dim),
            SimBa(dim),
            RMSNorm(dim),
            Linear(dim, self.chunk_dim),
            Rearrange('... (h a) -> ... h a', h = hierarchy)
        )

        self.state_decoder = Linear(resolved_latent_dim, state_dim) if prepend_first_state else None

    def encode(
        self,
        chunks: Tensor,
        first_states: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        flat = rearrange(chunks.float(), 'b h a -> b (h a)')

        if self.prepend_first_state:
            assert exists(first_states)
            flat = cat((self.state_norm(first_states), flat), dim = -1)

        pre_tanh = self.encoder(flat)
        return pre_tanh.tanh(), pre_tanh

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents).tanh()

    def forward(
        self,
        chunks: Tensor,
        first_states: Tensor | None = None
    ):
        chunks = chunks.float()
        latents, pre_tanh = self.encode(chunks, first_states)

        should_noise = self.training and self.latent_noise > 0.
        decoder_latents = (pre_tanh + self.latent_noise * randn_like(pre_tanh)).tanh() if should_noise else latents

        pred = self.decode(decoder_latents)
        recon_loss = F.mse_loss(pred, chunks)
        loss = recon_loss

        if self.prepend_first_state and exists(first_states):
            loss = loss + self.state_recon_weight * F.mse_loss(self.state_decoder(latents), self.state_norm(first_states).detach())

        with torch.no_grad():
            clean_pred = self.decode(latents)
            abs_err = (clean_pred - chunks).abs()
            mae = abs_err.mean().item()
            close_acc = (abs_err.mean(dim = -1) < 0.18).float().mean().item()

            # engine regime agreement: main > 0 and lateral in <-0.5 / [-0.5, 0.5] / >0.5

            if self.num_actions >= 2:
                main_ok = (clean_pred[..., 0] > 0.) == (chunks[..., 0] > 0.)
                side_bucket = lambda x: (x > 0.5).long() - (x < -0.5).long()
                side_ok = side_bucket(clean_pred[..., 1]) == side_bucket(chunks[..., 1])
                mode_acc = (main_ok & side_ok).float().mean().item()
            else:
                mode_acc = close_acc

        stats = dict(
            recon_mse = recon_loss.item(),
            mae = mae,
            close_acc = close_acc,
            mode_acc = mode_acc,
            z_std = latents.std().item()
        )

        return loss, stats

class ChunkBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        hierarchy: int,
        state_dim: int,
        num_actions: int,
        stride = 2
    ):
        self.capacity = capacity
        self.hierarchy = hierarchy
        self.stride = max(1, stride)

        self.actions = zeros((capacity, hierarchy, num_actions), dtype = torch.float32)
        self.first_states = zeros((capacity, state_dim), dtype = torch.float32)
        self.ptr = 0
        self.size = 0

    def add_rollout(
        self,
        states: Tensor,
        actions: Tensor,
        dones: Tensor
    ):
        h = self.hierarchy
        if states.shape[1] < h:
            return

        s_win = states.detach().cpu().unfold(1, h, self.stride)
        d_win = dones.detach().cpu().unfold(1, h, self.stride)
        a_win = actions.detach().cpu().unfold(1, h, self.stride)

        valid = ~d_win[..., :-1].any(dim = -1) if h > 1 else torch.ones_like(d_win[..., 0], dtype = torch.bool)
        if not valid.any():
            return

        first_s = s_win[..., 0][valid][-self.capacity:]
        act_chunks = rearrange(a_win[valid], 'n a h -> n h a').float()[-self.capacity:]
        n = act_chunks.shape[0]

        idx = (self.ptr + arange(n)) % self.capacity
        self.actions[idx] = act_chunks
        self.first_states[idx] = first_s
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def train_steps(
        self,
        model: ActionChunkTanh,
        opt: torch.optim.Optimizer,
        steps = 80,
        batch_size = 256
    ) -> dict:
        if self.size < 1:
            return dict(recon_mse = 0., mae = 0., close_acc = 0., mode_acc = 0., z_std = 0.)

        dev = next(model.parameters()).device
        bs = min(batch_size, self.size)
        model.train()

        history = []
        with torch.random.fork_rng():
            for _ in range(steps):
                idx = randint(0, self.size, (bs,))
                loss, stats = model(self.actions[idx].to(dev), self.first_states[idx].to(dev))

                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.)
                opt.step()

                history.append(stats)

        tail = history[-16:]
        return {k: float(np.mean([s[k] for s in tail])) for k in tail[-1]}

# slower macro environment committing to decoded tanh chunks

class TanhMacroEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        compressor: ActionChunkTanh | None,
        hierarchy: int,
        num_actions: int,
        gamma = 0.99
    ):
        super().__init__(env)
        self.compressor = compressor
        self.hierarchy = hierarchy
        self.num_actions = num_actions
        self.gamma = gamma

        latent_dim = compressor.latent_dim if exists(compressor) else num_actions
        self.action_space = gym.spaces.Box(low = -1., high = 1., shape = (latent_dim,), dtype = np.float32)

    def step(self, action):
        act_arr = np.asarray(action, dtype = np.float32)

        # accepts either a latent vector (latent_dim,) or a pre-decoded chunk (hierarchy, num_actions)

        if act_arr.ndim == 2 and act_arr.shape == (self.hierarchy, self.num_actions):
            prim_seq = act_arr
        else:
            assert exists(self.compressor), 'compressor required when stepping with latent vector z'
            dev = next(self.compressor.parameters()).device
            with torch.no_grad():
                z = torch.from_numpy(act_arr).float().unsqueeze(0).to(dev)
                prim_seq = self.compressor.decode(z).squeeze(0).cpu().numpy()

        disc_reward, raw_reward, prim_steps = 0., 0., 0

        for h in range(self.hierarchy):
            obs, reward, terminated, truncated, info = self.env.step(prim_seq[h])

            r = float(reward)
            disc_reward += (self.gamma ** h) * r
            raw_reward += r
            prim_steps += 1

            if terminated or truncated:
                break

        return obs, disc_reward, terminated, truncated, {
            **info,
            'raw_reward': raw_reward,
            'prim_steps': prim_steps
        }

# ppo update

def ppo_update(
    actor: BetaActor,
    critic: Critic,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    *,
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    durations: Tensor | None = None,
    trace_power = 1.,
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

    # joint log-ratio std grows as sqrt(action_dim) for factorized policies

    eff_eps_clip = eps_clip * math.sqrt(actor.action_dim / 2.)

    with torch.no_grad():
        values = critic(states)
        gae = calc_gae(
            rewards = rewards,
            values = values,
            next_values = critic(next_states),
            terminated = terminated,
            truncated = truncated,
            durations = durations,
            trace_power = trace_power,
            gamma = gamma,
            lam = lam,
            use_accelerated = False,
            return_advantages = True
        )

    for _ in range(epochs):
        perm = randperm(num_envs, device = states.device)

        for start in range(0, num_envs, minibatch_envs):
            mb = perm[start:start + minibatch_envs]

            mb_s = rearrange(states[mb], 'b t d -> (b t) d')
            mb_a = rearrange(actions[mb], 'b t a -> (b t) a')
            mb_old_lp = rearrange(old_log_probs[mb], 'b t -> (b t)')
            mb_adv = rearrange(gae.advantages[mb], 'b t -> (b t)')
            mb_ret = rearrange(gae.returns[mb], 'b t -> (b t)')
            mb_val = rearrange(values[mb], 'b t -> (b t)')

            params = actor(mb_s)
            actor_loss = loss_fn(
                actor.log_prob(params, mb_a),
                mb_old_lp,
                mb_adv,
                eps_clip = eff_eps_clip,
                normalize_advantages = True
            ).mean()
            actor_loss = actor_loss - entropy_coef * actor.entropy(params).mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt_actor.step()

            critic_loss = clipped_value_loss_fn(critic(mb_s), mb_val, mb_ret, clip = None)
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

# evaluation & video recording

@torch.no_grad()
def quick_validate(
    actor: BetaActor,
    compressor: ActionChunkTanh | None,
    hierarchy: int,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    num_episodes = 12,
    seed = 5000
) -> float:
    actor.eval()
    env = gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True)
    rets = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed = seed + ep)
        ep_ret, done = 0., False

        while not done:
            act = actor.greedy(torch.from_numpy(obs).float().unsqueeze(0).to(device))

            if not exists(compressor):
                obs, reward, terminated, truncated, _ = env.step(act.squeeze(0).cpu().numpy())
                ep_ret += float(reward)
                done = terminated or truncated
            else:
                prim_seq = compressor.decode(act).squeeze(0).cpu().numpy()
                for h in range(hierarchy):
                    obs, reward, terminated, truncated, _ = env.step(prim_seq[h])
                    ep_ret += float(reward)
                    if terminated or truncated:
                        done = True
                        break

        rets.append(ep_ret)

    env.close()
    return float(np.mean(rets))

@torch.no_grad()
def evaluate_macro_policy(
    actor: BetaActor,
    compressor: ActionChunkTanh | None,
    hierarchy: int,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    num_episodes = 48,
    top_k = 5,
    seed = 1000,
    video_folder: str | None = None,
    video_prefix: str | None = None
) -> dict:
    actor.eval()
    env = gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True)

    episodes = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed = ep_seed)
        ep_ret, prim_t, macro_t, last_r, done, term_flag = 0., 0, 0, 0., False, False

        xs, ys = [float(obs[0])], [float(obs[1])]
        dec_xs, dec_ys = [], []
        latents_log, prim_acts_log = [], []

        while not done:
            dec_xs.append(float(obs[0]))
            dec_ys.append(float(obs[1]))

            act = actor.greedy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
            act_np = act.squeeze(0).cpu().numpy()
            latents_log.append(act_np.tolist())
            macro_t += 1

            if not exists(compressor):
                prim_acts_log.append(act_np.tolist())
                obs, reward, terminated, truncated, _ = env.step(act_np)
                xs.append(float(obs[0]))
                ys.append(float(obs[1]))
                last_r = float(reward)
                ep_ret += last_r
                prim_t += 1
                term_flag = bool(terminated)
                done = terminated or truncated
            else:
                prim_seq = compressor.decode(act).squeeze(0).cpu().numpy()
                for h in range(hierarchy):
                    prim_a = prim_seq[h]
                    prim_acts_log.append(prim_a.tolist())
                    obs, reward, terminated, truncated, _ = env.step(prim_a)
                    xs.append(float(obs[0]))
                    ys.append(float(obs[1]))
                    last_r = float(reward)
                    ep_ret += last_r
                    prim_t += 1
                    if terminated or truncated:
                        term_flag = bool(terminated)
                        done = True
                        break

        episodes.append(dict(
            seed = int(ep_seed),
            return_val = float(ep_ret),
            prim_steps = int(prim_t),
            macro_steps = int(macro_t),
            landed = 1. if last_r >= 50. else 0.,
            strict_landed = 1. if is_strict_landing(obs, term_flag, last_r) else 0.,
            final_xy = [float(obs[0]), float(obs[1])],
            final_angle = float(obs[4]),
            xs = xs,
            ys = ys,
            dec_xs = dec_xs,
            dec_ys = dec_ys,
            latents = latents_log,
            prim_actions = prim_acts_log
        ))

    env.close()

    ranked = sorted(episodes, key = lambda e: e['return_val'], reverse = True)
    best_top_k = ranked[:top_k]
    best_ep = best_top_k[0]
    best_seed, best_return = best_ep['seed'], best_ep['return_val']

    video_path = None
    if exists(video_folder):
        render_env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_episode_steps, continuous = True)
        obs, _ = render_env.reset(seed = best_seed)
        frames, done = [render_env.render()], False

        while not done:
            act = actor.greedy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
            if not exists(compressor):
                obs, _, terminated, truncated, _ = render_env.step(act.squeeze(0).cpu().numpy())
                frames.append(render_env.render())
                done = terminated or truncated
            else:
                prim_seq = compressor.decode(act).squeeze(0).cpu().numpy()
                for h in range(hierarchy):
                    obs, _, terminated, truncated, _ = render_env.step(prim_seq[h])
                    frames.append(render_env.render())
                    if terminated or truncated:
                        done = True
                        break

        render_env.close()

        out_dir = Path(video_folder).resolve()
        out_dir.mkdir(parents = True, exist_ok = True)
        prefix = default(video_prefix, f'tanh-macro-h{hierarchy}-best')
        mp4_path = out_dir / f'{prefix}.mp4'
        imageio.mimsave(str(mp4_path), frames, fps = 50, macro_block_size = 1)
        imageio.mimsave(str(out_dir / f'{prefix}.gif'), frames[::2], fps = 25)
        video_path = str(mp4_path)
        print(f'[video] h={hierarchy} best eval = {best_return:+.1f} (seed={best_seed}) -> {mp4_path}', flush = True)

    returns = [e['return_val'] for e in episodes]
    return dict(
        mean_return = float(np.mean(returns)),
        std_return = float(np.std(returns)),
        median_return = float(np.median(returns)),
        best_return = float(best_return),
        best_seed = int(best_seed),
        land_rate = 100. * float(np.mean([e['landed'] for e in episodes])),
        strict_land_rate = 100. * float(np.mean([e['strict_landed'] for e in episodes])),
        mean_prim_steps = float(np.mean([e['prim_steps'] for e in episodes])),
        mean_macro_steps = float(np.mean([e['macro_steps'] for e in episodes])),
        top_trajectories = best_top_k,
        video_path = video_path
    )

# plotting

def draw_helipad(ax):
    ax.axhline(0., color = '#333333', linewidth = 1.)
    ax.fill_between([-0.2, 0.2], -0.04, 0., color = '#f4d35e', alpha = 0.55)
    ax.plot([-0.2, -0.2], [0., 0.08], color = '#d99b26', linewidth = 2.)
    ax.plot([0.2, 0.2], [0., 0.08], color = '#d99b26', linewidth = 2.)
    ax.set_xlim(-0.68, 0.68)
    ax.set_ylim(-0.08, 1.52)
    ax.grid(alpha = 0.22)

def plot_sweep(
    base_hist: list[dict],
    base_eval: dict,
    macro_runs: list[dict],
    out_dir: Path,
    env_name: str
):
    palette = ('#3b6fb6', '#e07a2b', '#2a9d6f', '#d13b40', '#7a5195')

    series = [(
        f'dt=1 flat (2D Beta | mean {base_eval["mean_return"]:+.1f}, best {base_eval["best_return"]:+.1f}, land {base_eval["land_rate"]:.0f}%)',
        base_hist,
        palette[0]
    )]

    for idx, run in enumerate(macro_runs):
        ev = run['eval']
        lbl = (
            f'dt={run["h"]} tanh ({run["chunk_dim"]}->{run["latent_dim"]}D | '
            f'mean {ev["mean_return"]:+.1f}, best {ev["best_return"]:+.1f}, land {ev["land_rate"]:.0f}%)'
        )
        series.append((lbl, run['history'], palette[(idx + 1) % len(palette)]))

    fig, axes = plt.subplots(1, 2, figsize = (15.2, 5.6), dpi = 160)

    for lbl, hist, col in series:
        steps_k = [r['env_steps'] / 1e3 for r in hist]
        decs_k = [r['decisions'] / 1e3 for r in hist]
        rets = [r['recent_return_32'] for r in hist]
        axes[0].plot(steps_k, rets, color = col, linewidth = 2.2, label = lbl)
        axes[1].plot(decs_k, rets, color = col, linewidth = 2.2, label = lbl)

    for ax, xlabel, title in (
        (axes[0], 'primitive environment steps (k)', f'{env_name} Continuous: Return vs Environment Steps'),
        (axes[1], 'policy decision steps (k)', f'{env_name} Continuous: Return vs Decision Steps')
    ):
        ax.axhline(200., color = 'forestgreen', linestyle = '--', linewidth = 1.1, label = 'solved (+200)')
        ax.axhline(0., color = 'gray', linestyle = ':', linewidth = 0.8)
        ax.set(xlabel = xlabel, ylabel = 'rolling 32-episode return', title = title)
        ax.legend(loc = 'lower right', fontsize = 8.2, framealpha = 0.92)
        ax.grid(alpha = 0.25)

    plt.tight_layout()
    sweep_path = out_dir / 'continuous_hierarchy_sweep.png'
    fig.savefig(sweep_path, bbox_inches = 'tight')
    plt.close(fig)
    print(f'[plot] saved -> {sweep_path}', flush = True)

    for key, xlabel, fname, title in (
        ('env_steps', 'primitive environment steps (k)', 'returns_over_env_steps.png', f'{env_name} Continuous: All Hierarchies over Environment Steps'),
        ('decisions', 'policy decision steps (k)', 'returns_over_decision_steps.png', f'{env_name} Continuous: All Hierarchies over Decision Steps')
    ):
        fig_s, ax_s = plt.subplots(figsize = (8.4, 5.2), dpi = 160)
        for lbl, hist, col in series:
            ax_s.plot([r[key] / 1e3 for r in hist], [r['recent_return_32'] for r in hist], color = col, linewidth = 2.2, label = lbl)
        ax_s.axhline(200., color = 'forestgreen', linestyle = '--', linewidth = 1.1, label = 'solved (+200)')
        ax_s.axhline(0., color = 'gray', linestyle = ':', linewidth = 0.8)
        ax_s.set(xlabel = xlabel, ylabel = 'rolling 32-episode return', title = title)
        ax_s.legend(loc = 'lower right', fontsize = 8.4, framealpha = 0.92)
        ax_s.grid(alpha = 0.25)
        fig_s.tight_layout()
        fig_s.savefig(out_dir / fname, bbox_inches = 'tight')
        plt.close(fig_s)

def plot_best_trajectories(
    base_eval: dict,
    macro_runs: list[dict],
    out_dir: Path
):
    all_rows = [('dt=1 Flat (2D Beta)', 1, 2, base_eval)] + [
        (f'dt={run["h"]} Tanh ({run["chunk_dim"]}->{run["latent_dim"]}D)', run['h'], run['latent_dim'], run['eval'])
        for run in macro_runs
    ]

    num_rows = len(all_rows)
    top_k = min(5, min(len(r[3]['top_trajectories']) for r in all_rows))

    # top-k grid per hierarchy colored by main-engine thrust + macro decision markers

    fig, axes = plt.subplots(num_rows, top_k, figsize = (3.4 * top_k, 3.15 * num_rows), dpi = 160, sharex = True, sharey = True)
    if num_rows == 1:
        axes = axes[None, :]

    norm = plt.Normalize(0., 1.)

    for r_idx, (row_lbl, h_val, _, ev) in enumerate(all_rows):
        for c_idx in range(top_k):
            ax = axes[r_idx, c_idx]
            ep = ev['top_trajectories'][c_idx]
            xs, ys = np.asarray(ep['xs']), np.asarray(ep['ys'])
            main_a = np.asarray(ep['prim_actions'])[:, 0]
            thrust = np.where(main_a > 0., 0.5 * (np.clip(main_a, 0., 1.) + 1.), 0.)

            pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis = 1), cmap = plt.cm.plasma, norm = norm, linewidths = 2.2, zorder = 3)
            lc.set_array(thrust)
            ax.add_collection(lc)

            ax.scatter(ep['dec_xs'], ep['dec_ys'], s = 13 if h_val > 1 else 5, color = '#1d3557', alpha = 0.72, zorder = 4)
            ax.scatter([xs[0]], [ys[0]], s = 38, color = '#2b6cb0', zorder = 5)
            ax.scatter([xs[-1]], [ys[-1]], s = 72, marker = '*', color = '#d62828', zorder = 6)
            draw_helipad(ax)

            ax.set_title(
                f'#{c_idx + 1} (s{ep["seed"]}): {ep["return_val"]:+.1f}\n'
                f'{ep["macro_steps"]} decs / {ep["prim_steps"]}f | x={ep["final_xy"][0]:+.2f}',
                fontsize = 8.3
            )
            if c_idx == 0:
                ax.set_ylabel(f'{row_lbl}\naltitude y', fontsize = 8.8, fontweight = 'bold')
            if r_idx == num_rows - 1:
                ax.set_xlabel('x position', fontsize = 8.5)

    fig.suptitle(
        'LunarLander Continuous — Best 5 Eval Trajectories per Hierarchy (Line Color = Main Thrust, Dots = Policy Decisions)',
        fontsize = 11.5, fontweight = 'bold', y = 1.01
    )
    plt.tight_layout()
    grid_path = out_dir / 'continuous_best5_trajectories.png'
    fig.savefig(grid_path, bbox_inches = 'tight')
    plt.close(fig)
    print(f'[plot] saved -> {grid_path}', flush = True)

    # overlaid top-k trajectories + best episode tanh latent trace

    fig2, axes2 = plt.subplots(2, num_rows, figsize = (4.1 * num_rows, 6.8), dpi = 160)
    if num_rows == 1:
        axes2 = axes2[:, None]

    rank_cols = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd')

    for idx, (row_lbl, h_val, lat_dim, ev) in enumerate(all_rows):
        ax_top, ax_bot = axes2[0, idx], axes2[1, idx]

        for r_i in range(top_k):
            ep = ev['top_trajectories'][r_i]
            col = rank_cols[r_i % len(rank_cols)]
            ax_top.plot(
                ep['xs'], ep['ys'],
                color = col,
                linewidth = 2.0 if r_i == 0 else 1.4,
                alpha = 0.95 if r_i == 0 else 0.72,
                label = f'#{r_i + 1} s{ep["seed"]}: {ep["return_val"]:+.1f} ({ep["macro_steps"]}d)'
            )
            ax_top.scatter([ep['xs'][-1]], [ep['ys'][-1]], s = 45, marker = '*', color = col, zorder = 5)

        draw_helipad(ax_top)
        ax_top.set_title(f'{row_lbl}\nMean: {ev["mean_return"]:+.1f} | Land: {ev["land_rate"]:.0f}%', fontsize = 9.2, fontweight = 'bold')
        ax_top.set_xlabel('x position', fontsize = 8.5)
        if idx == 0:
            ax_top.set_ylabel('altitude y', fontsize = 9.)
        ax_top.legend(loc = 'upper right', fontsize = 7.2, framealpha = 0.9)

        lat_arr = np.asarray(ev['top_trajectories'][0]['latents'])
        t_steps = np.arange(lat_arr.shape[0])
        for d_i in range(lat_arr.shape[1]):
            lbl = f'z[{d_i}]' if h_val > 1 else ('main' if d_i == 0 else 'side')
            ax_bot.plot(t_steps, lat_arr[:, d_i], linewidth = 1.4, alpha = 0.85, label = lbl)

        ax_bot.axhline(0., color = 'gray', linestyle = ':', linewidth = 0.8)
        ax_bot.set_ylim(-1.05, 1.05)
        ax_bot.set_title(f'Best Episode Latents ({lat_arr.shape[0]} decisions, dim={lat_dim})', fontsize = 8.8)
        ax_bot.set_xlabel(f'decision index (1 step = {h_val} env frame{"s" if h_val > 1 else ""})', fontsize = 8.3)
        if idx == 0:
            ax_bot.set_ylabel('action / tanh latent in (-1, 1)', fontsize = 8.8)
        ax_bot.legend(loc = 'lower right', fontsize = 7., ncol = 2, framealpha = 0.88)
        ax_bot.grid(alpha = 0.22)

    plt.tight_layout()
    overlay_path = out_dir / 'continuous_winning_latents.png'
    fig2.savefig(overlay_path, bbox_inches = 'tight')
    plt.close(fig2)
    print(f'[plot] saved -> {overlay_path}', flush = True)

# main training pipeline

def main(
    env_name = 'LunarLander-v3',
    hierarchy = 4,
    sweep = False,
    hierarchies: tuple[int, ...] = (2, 4, 8),
    latent_dim: int | None = None,
    compressor_dim = 128,
    compressor_lr = 2e-3,
    latent_noise = 0.02,
    prepend_first_state = False,
    compress_every = 5,
    compress_steps = 80,
    compress_final_steps = 300,
    buffer_capacity = 49152,
    base_updates = 135,
    macro_updates = 160,
    num_envs = 16,
    base_rollout_steps = 384,
    macro_rollout_steps: int | None = None,
    max_episode_steps = 600,
    actor_hidden_dim = 108,
    critic_hidden_dim = 192,
    beta_init_conc = 5.,
    epochs = 6,
    minibatch_envs = 4,
    lr = 1e-3,
    critic_lr = 2.5e-3,
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.2,
    max_grad_norm = 0.5,
    use_hpo = True,
    val_every = 15,
    eval_episodes = 48,
    seed = 42,
    render = True,
    out_folder = './continuous-hierarchical-recording',
    log_every = 15
):
    active_hierarchies = tuple(int(h) for h in hierarchies) if sweep else (int(hierarchy),)

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_folder)
    out_dir.mkdir(parents = True, exist_ok = True)

    # stage 1: dt = 1 ppo + intermittent tanh action compression

    base_env = SyncVectorEnv(
        [lambda: gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True) for _ in range(num_envs)],
        autoreset_mode = AutoresetMode.SAME_STEP
    )

    state_dim = base_env.single_observation_space.shape[0]
    num_actions = base_env.single_action_space.shape[0]

    base_actor = BetaActor(state_dim, actor_hidden_dim, num_actions, beta_init_conc = beta_init_conc).to(device)
    base_critic = Critic(state_dim, critic_hidden_dim).to(device)
    opt_actor = torch.optim.AdamW(base_actor.parameters(), lr = lr)
    opt_critic = torch.optim.AdamW(base_critic.parameters(), lr = critic_lr)

    with torch.random.fork_rng():
        compressors = {
            h: ActionChunkTanh(
                hierarchy = h,
                num_actions = num_actions,
                state_dim = state_dim,
                latent_dim = latent_dim,
                dim = compressor_dim,
                latent_noise = latent_noise,
                prepend_first_state = prepend_first_state
            ).to(device)
            for h in active_hierarchies
        }
        comp_opts = {h: torch.optim.AdamW(compressors[h].parameters(), lr = compressor_lr) for h in active_hierarchies}
        buffers = {
            h: ChunkBuffer(capacity = buffer_capacity, hierarchy = h, state_dim = state_dim, num_actions = num_actions)
            for h in active_hierarchies
        }
        comp_stats = {h: dict(recon_mse = 0., mae = 0., close_acc = 0., mode_acc = 0., z_std = 0.) for h in active_hierarchies}

    dims_desc = ', '.join(f'h={h}: {compressors[h].chunk_dim}->{compressors[h].latent_dim}D' for h in active_hierarchies)
    print(f'[stage 1] dt = 1 MeanConcBeta PPO + Tanh compression ({dims_desc})', flush = True)

    obs, _ = base_env.reset(seed = seed)
    state = torch.from_numpy(obs).float().to(device)
    running = np.zeros(num_envs)
    recent, base_hist = [], []
    best_base_val = -float('inf')
    best_base_state = deepcopy(base_actor.state_dict())
    env_steps = 0
    t0 = time.time()

    for upd in range(1, base_updates + 1):
        progress = (upd - 1) / max(1, base_updates)
        ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)
        for pg in opt_actor.param_groups:
            pg['lr'] = lr * schedule_at(LR_SCALE_SCHEDULE, progress)

        states, next_states, actions, log_probs, rewards, terms, truncs = [], [], [], [], [], [], []

        base_actor.eval()
        for _ in range(base_rollout_steps):
            with torch.no_grad():
                action, lp = base_actor.sample(state)

            next_obs, reward, terminated, truncated, info = base_env.step(action.cpu().numpy())
            done = terminated | truncated
            running += reward

            for idx in np.flatnonzero(done):
                recent.append(float(running[idx]))
                running[idx] = 0.

            states.append(state)
            next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
            actions.append(action)
            log_probs.append(lp)
            rewards.append(torch.from_numpy(reward / 100.).float().to(device))
            terms.append(torch.from_numpy(terminated).bool().to(device))
            truncs.append(torch.from_numpy(truncated).bool().to(device))
            state = torch.from_numpy(next_obs).float().to(device)

        env_steps += num_envs * base_rollout_steps
        states_t, actions_t = stack(states, dim = 1), stack(actions, dim = 1)
        term_t, trunc_t = stack(terms, dim = 1), stack(truncs, dim = 1)

        for h in active_hierarchies:
            buffers[h].add_rollout(states_t, actions_t, term_t | trunc_t)

        base_actor.train()
        base_critic.train()
        ppo_update(
            base_actor, base_critic, opt_actor, opt_critic,
            states = states_t, next_states = stack(next_states, dim = 1),
            actions = actions_t, old_log_probs = stack(log_probs, dim = 1),
            rewards = stack(rewards, dim = 1), terminated = term_t, truncated = trunc_t,
            gamma = gamma, lam = lam, eps_clip = eps_clip,
            entropy_coef = ent_coef, epochs = epochs, minibatch_envs = minibatch_envs,
            max_grad_norm = max_grad_norm, use_hpo = use_hpo
        )

        if divisible_by(upd, compress_every):
            for h in active_hierarchies:
                if buffers[h].size >= 256:
                    comp_stats[h] = buffers[h].train_steps(compressors[h], comp_opts[h], steps = compress_steps)

        rec32 = float(np.mean(recent[-32:])) if recent else float('nan')

        if divisible_by(upd, val_every) or upd == base_updates:
            val_ret = quick_validate(base_actor, None, 1, env_name = env_name, max_episode_steps = max_episode_steps, seed = seed + 5000)
            if val_ret >= best_base_val:
                best_base_val = val_ret
                best_base_state = deepcopy(base_actor.state_dict())

        base_hist.append(dict(update = upd, env_steps = env_steps, decisions = env_steps, recent_return_32 = rec32))

        if divisible_by(upd, log_every) or upd == base_updates:
            c_msg = ' | '.join(
                f'h={h}({compressors[h].latent_dim}D): mse={comp_stats[h]["recon_mse"]:.3f} mode={100 * comp_stats[h]["mode_acc"]:4.1f}%'
                for h in active_hierarchies
            )
            print(
                f'  [dt=1 upd {upd:3d}/{base_updates}] steps = {env_steps // 1000:4d}k | '
                f'ret32 = {rec32:+6.1f} (val_best = {best_base_val:+6.1f}) | {c_msg} [{time.time() - t0:.1f}s]',
                flush = True
            )

    base_env.close()

    base_actor.load_state_dict(best_base_state)
    base_eval = evaluate_macro_policy(
        base_actor, None, 1,
        env_name = env_name, max_episode_steps = max_episode_steps,
        num_episodes = eval_episodes,
        video_folder = out_folder if render else None,
        video_prefix = 'tanh-flat-h1-best'
    )
    print(
        f'  [dt=1 eval] {base_eval["mean_return"]:+.1f} +/- {base_eval["std_return"]:.1f} '
        f'(best {base_eval["best_return"]:+.1f}) | land = {base_eval["land_rate"]:.1f}% '
        f'(strict = {base_eval["strict_land_rate"]:.1f}%)',
        flush = True
    )

    torch.save(dict(hierarchy = 1, actor = base_actor.state_dict(), eval = base_eval), out_dir / 'ppo_tanh_h1.pt')

    with open(out_dir / 'tanh_flat_h1_returns.csv', 'w', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['update', 'env_steps', 'decisions', 'recent_return_32'])
        writer.writeheader()
        writer.writerows(base_hist)

    for h in active_hierarchies:
        comp_stats[h] = buffers[h].train_steps(compressors[h], comp_opts[h], steps = compress_final_steps, batch_size = 512)
        compressors[h].eval()
        compressors[h].requires_grad_(False)
        st = comp_stats[h]
        print(
            f'  [tanh frozen h={h}] {compressors[h].chunk_dim}->{compressors[h].latent_dim}D | '
            f'mse = {st["recon_mse"]:.4f} | mae = {st["mae"]:.3f} | '
            f'mode_acc = {100 * st["mode_acc"]:.1f}% | z_std = {st["z_std"]:.2f}',
            flush = True
        )

    # stage 2: slower macro-environment ppo for each hierarchy h > 1

    macro_runs = []

    for h in active_hierarchies:
        comp = compressors[h]
        lat_dim = comp.latent_dim
        rollout_macro = default(macro_rollout_steps, max(128, base_rollout_steps // h))

        with torch.random.fork_rng():
            torch.manual_seed(seed + 1000 * h)
            np.random.seed(seed + 1000 * h)

            macro_env = SyncVectorEnv(
                [
                    lambda h_val = h, c_mod = comp: TanhMacroEnv(
                        gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True),
                        c_mod, h_val, num_actions = num_actions, gamma = gamma
                    )
                    for _ in range(num_envs)
                ],
                autoreset_mode = AutoresetMode.SAME_STEP
            )

            macro_actor = BetaActor(state_dim, actor_hidden_dim, lat_dim, beta_init_conc = beta_init_conc).to(device)
            macro_critic = Critic(state_dim, critic_hidden_dim).to(device)
            opt_m_actor = torch.optim.AdamW(macro_actor.parameters(), lr = lr)
            opt_m_critic = torch.optim.AdamW(macro_critic.parameters(), lr = critic_lr)

            obs, _ = macro_env.reset(seed = seed + 100)
            state = torch.from_numpy(obs).float().to(device)
            running = np.zeros(num_envs)
            m_recent, m_hist = [], []
            best_m_val = -float('inf')
            best_m_state = deepcopy(macro_actor.state_dict())
            m_steps, m_decs = 0, 0
            t_h = time.time()

            print(
                f'\n[stage 2] dt = {h} MeanConcBeta macro-actor '
                f'({comp.chunk_dim}->{lat_dim}D tanh bottleneck, rollout = {rollout_macro})',
                flush = True
            )

            for upd in range(1, macro_updates + 1):
                progress = (upd - 1) / max(1, macro_updates)
                ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)
                for pg in opt_m_actor.param_groups:
                    pg['lr'] = lr * schedule_at(LR_SCALE_SCHEDULE, progress)

                states, next_states, latents, log_probs, rewards, terms, truncs, psteps = [], [], [], [], [], [], [], []

                macro_actor.eval()
                for _ in range(rollout_macro):
                    with torch.no_grad():
                        z, lp = macro_actor.sample(state)
                        prim_seqs = comp.decode(z).cpu().numpy()

                    next_obs, disc_r, terminated, truncated, info = macro_env.step(prim_seqs)
                    done = terminated | truncated

                    raw_r = step_info_array(info, 'raw_reward', done, 0.)
                    durs = step_info_array(info, 'prim_steps', done, float(h))

                    running += raw_r
                    m_steps += int(durs.sum())
                    m_decs += num_envs

                    for idx in np.flatnonzero(done):
                        m_recent.append(float(running[idx]))
                        running[idx] = 0.

                    states.append(state)
                    next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                    latents.append(z)
                    log_probs.append(lp)
                    rewards.append(torch.from_numpy(disc_r / 100.).float().to(device))
                    terms.append(torch.from_numpy(terminated).bool().to(device))
                    truncs.append(torch.from_numpy(truncated).bool().to(device))
                    psteps.append(torch.from_numpy(durs).float().to(device))
                    state = torch.from_numpy(next_obs).float().to(device)

                macro_actor.train()
                macro_critic.train()
                ppo_update(
                    macro_actor, macro_critic, opt_m_actor, opt_m_critic,
                    states = stack(states, dim = 1), next_states = stack(next_states, dim = 1),
                    actions = stack(latents, dim = 1), old_log_probs = stack(log_probs, dim = 1),
                    rewards = stack(rewards, dim = 1), terminated = stack(terms, dim = 1),
                    truncated = stack(truncs, dim = 1), durations = stack(psteps, dim = 1),
                    gamma = gamma, lam = lam, eps_clip = eps_clip, entropy_coef = ent_coef,
                    epochs = epochs, minibatch_envs = minibatch_envs, max_grad_norm = max_grad_norm, use_hpo = use_hpo
                )

                rec32 = float(np.mean(m_recent[-32:])) if m_recent else float('nan')

                if divisible_by(upd, val_every) or upd == macro_updates:
                    val_ret = quick_validate(
                        macro_actor, comp, h,
                        env_name = env_name, max_episode_steps = max_episode_steps, seed = seed + 5000 + 100 * h
                    )
                    if val_ret >= best_m_val:
                        best_m_val = val_ret
                        best_m_state = deepcopy(macro_actor.state_dict())

                m_hist.append(dict(update = upd, env_steps = m_steps, decisions = m_decs, recent_return_32 = rec32))

                if divisible_by(upd, log_every) or upd == macro_updates:
                    print(
                        f'  [dt={h} upd {upd:3d}/{macro_updates}] decs = {m_decs // 1000:3d}k '
                        f'(env = {m_steps // 1000:4d}k) | ret32 = {rec32:+6.1f} '
                        f'(val_best = {best_m_val:+6.1f}) [{time.time() - t_h:.1f}s]',
                        flush = True
                    )

            macro_env.close()

        macro_actor.load_state_dict(best_m_state)
        m_eval = evaluate_macro_policy(
            macro_actor, comp, h,
            env_name = env_name, max_episode_steps = max_episode_steps,
            num_episodes = eval_episodes,
            video_folder = out_folder if render else None,
            video_prefix = f'tanh-macro-h{h}-best'
        )

        torch.save(dict(
            hierarchy = h,
            chunk_dim = comp.chunk_dim,
            latent_dim = lat_dim,
            compressor_stats = comp_stats[h],
            tanh_compressor = comp.state_dict(),
            macro_actor = macro_actor.state_dict(),
            macro_eval = m_eval
        ), out_dir / f'ppo_tanh_h{h}.pt')

        with open(out_dir / f'tanh_h{h}_returns.csv', 'w', newline = '') as f:
            writer = csv.DictWriter(f, fieldnames = ['update', 'env_steps', 'decisions', 'recent_return_32'])
            writer.writeheader()
            writer.writerows(m_hist)

        print(
            f'  [dt={h} eval] {m_eval["mean_return"]:+.1f} +/- {m_eval["std_return"]:.1f} '
            f'(best {m_eval["best_return"]:+.1f}) | land = {m_eval["land_rate"]:.1f}% '
            f'(strict = {m_eval["strict_land_rate"]:.1f}%)',
            flush = True
        )

        macro_runs.append(dict(
            h = h,
            chunk_dim = comp.chunk_dim,
            latent_dim = lat_dim,
            compressor_stats = comp_stats[h],
            history = m_hist,
            eval = m_eval
        ))

    # plots & summary

    plot_sweep(base_hist, base_eval, macro_runs, out_dir, env_name)
    plot_best_trajectories(base_eval, macro_runs, out_dir)

    top_keys = ('seed', 'return_val', 'prim_steps', 'macro_steps', 'landed', 'strict_landed', 'final_xy', 'final_angle')
    summary = dict(
        env_name = env_name,
        continuous = True,
        seed = seed,
        base = dict(
            hierarchy = 1,
            action_dim = num_actions,
            mean_return = base_eval['mean_return'],
            std_return = base_eval['std_return'],
            best_return = base_eval['best_return'],
            land_rate = base_eval['land_rate'],
            strict_land_rate = base_eval['strict_land_rate'],
            mean_prim_steps = base_eval['mean_prim_steps'],
            mean_macro_steps = base_eval['mean_macro_steps'],
            top5 = [{k: ep[k] for k in top_keys} for ep in base_eval['top_trajectories']]
        ),
        macros = {
            str(r['h']): dict(
                hierarchy = r['h'],
                chunk_dim = r['chunk_dim'],
                latent_dim = r['latent_dim'],
                compression_ratio = round(r['chunk_dim'] / r['latent_dim'], 2),
                compressor_stats = r['compressor_stats'],
                mean_return = r['eval']['mean_return'],
                std_return = r['eval']['std_return'],
                best_return = r['eval']['best_return'],
                land_rate = r['eval']['land_rate'],
                strict_land_rate = r['eval']['strict_land_rate'],
                mean_prim_steps = r['eval']['mean_prim_steps'],
                mean_macro_steps = r['eval']['mean_macro_steps'],
                top5 = [{k: ep[k] for k in top_keys} for ep in r['eval']['top_trajectories']]
            )
            for r in macro_runs
        }
    )

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent = 2))
    print(f'[summary] saved -> {out_dir / "summary.json"}', flush = True)

if __name__ == '__main__':
    fire.Fire(main)
