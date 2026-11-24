# /// script
# dependencies = [
#   "torch>=2.5",
#   "torch-population>=0.0.4",
#   "x-ppo>=0.1.1",
#   "einops>=0.8.0",
#   "gymnasium[box2d]>=1.0.0",
#   "imageio[ffmpeg]",
#   "matplotlib",
#   "numpy",
#   "pillow",
#   "fire",
# ]
# ///

"""Evolutionary Policy Optimization (EPO) with `torch-population` and `x-ppo`.

Mustafaoglu, Pingali & Miikkulainen (2025) - https://arxiv.org/abs/2504.12568
With optional CEM meta-evolution of the Delightful Policy Gradient gate temperature
(Osband, 2026 - https://arxiv.org/abs/2603.14608).
"""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
import time
from functools import lru_cache, partial
from pathlib import Path

import fire
import gymnasium as gym
import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib
import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import Module
from torch.optim import Adam
from torch_population import Population

from x_ppo import calc_gae, clipped_value_loss_fn, ppo_actor_loss

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# constants

LINEAGES = ('BASE', 'ELITE', 'PPO-FT', 'MUT')
BASE, ELITE, PPO_FT, MUT = range(len(LINEAGES))

PALETTE = {
    'BEST':   '#f5c542',
    'ELITE':  '#34d399',
    'PPO-FT': '#38bdf8',
    'MUT':    '#fb923c',
    'BASE':   '#94a3b8',
}

PLOT_STYLE = {
    'figure.facecolor': '#0f121a',
    'axes.facecolor':   '#161b26',
    'axes.edgecolor':   '#2d3748',
    'axes.labelcolor':  '#e2e8f0',
    'text.color':       '#e2e8f0',
    'xtick.color':      '#cbd5e1',
    'ytick.color':      '#cbd5e1',
    'grid.color':       '#263043',
}

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return den > 0 and (num % den) == 0

def resolve_device(device: str | None = None) -> torch.device:
    if exists(device):
        return torch.device(device)
    # small MLP RL policies run faster on CPU than MPS due to kernel launch overhead
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_env(env_id: str, max_episode_steps: int | None = None, render_mode: str | None = None):
    env = gym.make(env_id, render_mode = render_mode)
    return gym.wrappers.TimeLimit(env, max_episode_steps = max_episode_steps) if exists(max_episode_steps) else env

def init_orthogonal_(module: Module, gain: float = math.sqrt(2.)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain = gain)
        nn.init.zeros_(module.bias)

def lineage_labels(pop: Population) -> list[str]:
    return [LINEAGES[int(i)] for i in pop.tracked('lineage').tolist()]

# synchronous env pool preserving exact final_obs on truncation

class EnvPool:
    def __init__(self, env_id: str, num_envs: int, max_episode_steps: int | None = None):
        self.envs = [make_env(env_id, max_episode_steps = max_episode_steps) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self, seeds: list[int] | int | None = None) -> np.ndarray:
        obs = []
        for i, env in enumerate(self.envs):
            s = seeds[i] if isinstance(seeds, list) else (seeds + i if exists(seeds) else None)
            obs.append(env.reset(seed = s)[0])
        return np.asarray(obs, dtype = np.float32)

    def step(self, actions: np.ndarray):
        next_obs, final_obs = [], []
        rewards = np.empty(self.num_envs, dtype = np.float32)
        terminated = np.empty(self.num_envs, dtype = bool)
        truncated = np.empty(self.num_envs, dtype = bool)

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, r, term, trunc, _ = env.step(int(action))
            rewards[i], terminated[i], truncated[i] = r, term, trunc
            final_obs.append(obs)
            next_obs.append(env.reset()[0] if (term or trunc) else obs)

        return (
            np.asarray(next_obs, dtype = np.float32),
            np.asarray(final_obs, dtype = np.float32),
            rewards,
            terminated,
            truncated,
        )

    def close(self):
        for env in self.envs:
            env.close()

# networks

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, out_dim),
    )

class ActorCritic(Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = build_mlp(state_dim, hidden_dim, action_dim)
        self.critic = build_mlp(state_dim, hidden_dim, 1)

        self.actor.apply(init_orthogonal_)
        self.critic.apply(init_orthogonal_)
        init_orthogonal_(self.actor[-1], gain = 0.01)
        init_orthogonal_(self.critic[-1], gain = 1.0)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.actor(state), self.critic(state).squeeze(-1)

# CEM meta-evolution of the delightful gate temperature via torch-population

class MetaTemperature(Module):
    def __init__(self, min_bound: float = -1., max_bound: float = 3.):
        super().__init__()
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self) -> Tensor:
        return (self.param.sigmoid() * (self.max_bound - self.min_bound) + self.min_bound).exp()

class MetaCEM:
    def __init__(
        self,
        pop_size: int,
        num_elite: int,
        device: torch.device,
        tau: float = 0.5,
        sigma_init: float = 0.25,
        sigma_end: float = 0.01,
        min_bound: float = -1.,
        max_bound: float = 3.,
    ):
        self.pop_size = pop_size
        self.num_elite = min(num_elite, pop_size)
        self.tau = tau
        self.sigma_end = sigma_end
        self.device = device

        self.pop = Population(
            MetaTemperature(min_bound = min_bound, max_bound = max_bound).to(device),
            pop_size = pop_size,
            init = 'clone',
        )
        self.mu = self.pop.flat_params[0].detach().clone()
        self.epsilon = torch.full_like(self.mu, sigma_init)
        self.sigma = self.epsilon.sqrt()
        self.cem_updates = 0
        self.resample_()

    @torch.no_grad()
    def resample_(self):
        self.pop.flat_params.copy_(self.mu[None] + self.sigma[None] * torch.randn_like(self.pop.flat_params))
        self.fitness_sum = torch.zeros(self.pop_size, device = self.device)
        self.usage_count = torch.zeros(self.pop_size, device = self.device)
        self.cursor = 0

    @torch.no_grad()
    def assign(self, count: int) -> tuple[list[int], Tensor]:
        slots = [(self.cursor + i) % self.pop_size for i in range(count)]
        return slots, self.pop(members = slots).squeeze(-1)

    @torch.no_grad()
    def record(self, slots: list[int], fitnesses: Tensor):
        for slot, fit in zip(slots, fitnesses.tolist()):
            self.fitness_sum[slot] += fit
            self.usage_count[slot] += 1.
            self.cursor += 1
            if self.cursor >= self.pop_size:
                avg_fit = self.fitness_sum / self.usage_count.clamp(min = 1.)
                elites = self.pop.flat_params[avg_fit.topk(self.num_elite).indices]
                self.mu = elites.mean(dim = 0)
                self.epsilon = self.tau * self.epsilon + (1. - self.tau) * self.sigma_end
                self.sigma = (elites.var(dim = 0, unbiased = False) + self.epsilon).sqrt()
                self.cem_updates += 1
                self.resample_()

    @torch.no_grad()
    def stats(self) -> dict[str, float]:
        temps = self.pop().squeeze(-1)
        return dict(
            mean = float(temps.mean().item()),
            min = float(temps.min().item()),
            max = float(temps.max().item()),
            std = float(temps.std(unbiased = False).item()),
            updates = self.cem_updates,
        )

# unified PPO (single noop pre-training AND batched subset Lamarckian fine-tuning)

def forward_policy(
    pop: Population,
    obs: Tensor,
    members: list[int] | None,
    num_agents: int,
    envs_per_agent: int | None = None,
) -> tuple[Tensor, Tensor]:
    if not pop.is_population:
        return pop(obs)
    if exists(envs_per_agent):
        logits, values = pop(rearrange(obs, '(m e) d -> m e d', m = num_agents, e = envs_per_agent), members = members, per_individual = True)
        return rearrange(logits, 'm e a -> (m e) a'), rearrange(values, 'm e -> (m e)')
    return pop(obs, members = members, per_individual = True)

def train_ppo(
    pop: Population,
    opt,
    env_id: str,
    total_steps_per_agent: int,
    *,
    device: torch.device,
    members: list[int] | None = None,
    envs_per_agent: int = 4,
    rollout_steps: int = 256,
    minibatch_size: int = 64,
    k_epochs: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    eps_clip: float = 0.2,
    value_clip: float | None = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    use_delightful_gating: bool = False,
    delight_temp: float | Tensor = 1.0,
    max_episode_steps: int | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, float]:
    is_pop = pop.is_population
    active = members if is_pop else None
    num_agents = len(members) if exists(members) else (pop.pop_size if is_pop else 1)
    if num_agents == 0:
        return dict(policy_loss = 0., value_loss = 0., entropy = 0., steps = 0)

    total_envs = num_agents * envs_per_agent
    num_updates = max(1, math.ceil(total_steps_per_agent / (envs_per_agent * rollout_steps)))
    actual_steps = min(rollout_steps, max(16, math.ceil(total_steps_per_agent / envs_per_agent)))
    n_per_agent = envs_per_agent * actual_steps
    mb_size = min(minibatch_size, n_per_agent)

    if torch.is_tensor(delight_temp):
        delight_temp_t = delight_temp.to(device = device, dtype = torch.float32)
        if is_pop and delight_temp_t.ndim == 1:
            delight_temp_t = delight_temp_t[:, None]
    else:
        delight_temp_t = float(delight_temp)

    pool = EnvPool(env_id, total_envs, max_episode_steps = max_episode_steps)
    obs = pool.reset(seeds = seed)
    pop.train()

    last_pi = last_v = last_ent = 0.
    total_env_steps = 0

    for update_idx in range(num_updates):
        rollout: list[tuple[Tensor, ...]] = []

        for _ in range(actual_steps):
            obs_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                logits, values = forward_policy(pop, obs_t, active, num_agents, envs_per_agent)
                dist = Categorical(logits = logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            next_obs, final_obs, rewards, terminated, truncated = pool.step(actions.cpu().numpy())

            trunc_boot = torch.zeros(total_envs, device = device)
            if truncated.any():
                with torch.no_grad():
                    _, final_vals = forward_policy(pop, torch.from_numpy(final_obs).to(device), active, num_agents, envs_per_agent)
                trunc_boot = torch.where(torch.from_numpy(truncated).to(device), final_vals, trunc_boot)

            rollout.append((
                obs_t, actions, logprobs, values,
                torch.from_numpy(rewards).to(device),
                torch.from_numpy(terminated).to(device),
                torch.from_numpy(truncated).to(device),
                trunc_boot,
            ))
            obs = next_obs
            total_env_steps += total_envs

        with torch.no_grad():
            _, next_values = forward_policy(pop, torch.from_numpy(obs).to(device), active, num_agents, envs_per_agent)

        b_obs, b_act, b_lp, b_val, b_rew, b_term, b_trunc, b_boot = (torch.stack(col, dim = 1) for col in zip(*rollout))

        gae = calc_gae(
            rewards = b_rew + gamma * b_boot,
            values = b_val,
            terminated = b_term | b_trunc,
            next_value = next_values,
            gamma = gamma,
            lam = gae_lambda,
            return_advantages = True,
        )

        # flatten per individual: (num_agents, N, ...) if populated else (N, ...)
        def flatten_batch(t: Tensor) -> Tensor:
            return t.reshape(num_agents, n_per_agent, *t.shape[2:]) if is_pop else t.reshape(n_per_agent, *t.shape[2:])

        flat_batch = [flatten_batch(t) for t in (b_obs, b_act, b_lp, b_val, gae.returns, gae.advantages)]
        # normalize advantages per individual (never across population members)
        adv = flat_batch[-1]
        flat_batch[-1] = (adv - adv.mean(dim = -1, keepdim = True)) / (adv.std(dim = -1, keepdim = True) + 1e-8)

        for _ in range(k_epochs):
            perm = torch.randperm(n_per_agent, device = device)
            for start in range(0, n_per_agent, mb_size):
                idx = perm[start : start + mb_size]
                if idx.numel() < 2:
                    continue

                sl = (slice(None), idx) if is_pop else idx
                mb_obs, mb_act, mb_old_lp, mb_old_val, mb_ret, mb_adv = (t[sl] for t in flat_batch)

                logits, values = forward_policy(pop, mb_obs, active, num_agents)
                dist = Categorical(logits = logits)

                pi_loss = ppo_actor_loss(
                    dist.log_prob(mb_act), mb_old_lp, mb_adv,
                    eps_clip = eps_clip,
                    normalize_advantages = False,
                    delightful = use_delightful_gating,
                    delight_temp = delight_temp_t,
                ).mean(dim = -1)

                v_loss = value_coef * clipped_value_loss_fn(
                    values, mb_old_val, mb_ret,
                    clip = value_clip,
                    reduction = 'none',
                ).mean(dim = -1)

                ent = dist.entropy()
                # sum across active members so gradient magnitude per member is independent of K
                loss = (pi_loss + v_loss - entropy_coef * ent.mean(dim = -1)).sum()

                opt.zero_grad()
                loss.backward()
                if exists(max_grad_norm):
                    # clip actor and critic independently per member so value-MSE grads cannot crush policy grads
                    pop.clip_grad_norm_(max_grad_norm, members = active, names = 'actor')
                    pop.clip_grad_norm_(max_grad_norm, members = active, names = 'critic')
                opt.step(members = active)

                last_pi = float(pi_loss.mean().item())
                last_v = float(v_loss.mean().item())
                last_ent = float(ent.mean().item())

        if verbose and (divisible_by(update_idx + 1, max(1, num_updates // 4)) or update_idx + 1 == num_updates):
            print(f'  pretrain [{update_idx + 1:2d}/{num_updates:2d}] steps={total_env_steps:6d} | pi_loss={last_pi:+.4f} | v_loss={last_v:.3f} | ent={last_ent:.3f}')

    pool.close()
    return dict(policy_loss = last_pi, value_loss = last_v, entropy = last_ent, steps = total_env_steps)

# vectorized population evaluation on paired seeds

@torch.no_grad()
def evaluate_population(
    pop: Population,
    env_id: str,
    eval_seeds: list[int],
    *,
    device: torch.device,
    members: list[int] | None = None,
    max_episode_steps: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    pop.eval()
    is_pop = pop.is_population
    num_agents = len(members) if exists(members) else (pop.pop_size if is_pop else 1)

    envs = [make_env(env_id, max_episode_steps = max_episode_steps) for _ in range(num_agents)]
    returns = np.zeros((num_agents, len(eval_seeds)), dtype = np.float32)
    ent_sum = np.zeros(num_agents, dtype = np.float64)
    steps = np.zeros(num_agents, dtype = np.int64)

    for ep, ep_seed in enumerate(eval_seeds):
        obs = np.asarray([env.reset(seed = ep_seed)[0] for env in envs], dtype = np.float32)
        active = np.ones(num_agents, dtype = bool)
        ep_ret = np.zeros(num_agents, dtype = np.float32)

        while active.any():
            obs_t = torch.from_numpy(obs).to(device)
            logits = pop(obs_t, members = members, per_individual = True)[0] if is_pop else pop(obs_t[0])[0].unsqueeze(0)
            actions = logits.argmax(dim = -1).cpu().numpy()
            ents = Categorical(logits = logits).entropy().cpu().numpy()

            for i in np.flatnonzero(active):
                next_o, r, term, trunc, _ = envs[i].step(int(actions[i]))
                obs[i] = next_o
                ep_ret[i] += r
                ent_sum[i] += float(ents[i])
                steps[i] += 1
                if term or trunc:
                    active[i] = False

        returns[:, ep] = ep_ret

    for env in envs:
        env.close()

    return (
        torch.from_numpy(returns.mean(axis = 1)).to(device),
        torch.from_numpy((ent_sum / np.maximum(steps, 1)).astype(np.float32)).to(device),
        torch.from_numpy(returns.std(axis = 1)).to(device),
    )

# evolutionary operators via torch-population

def select_elite_parents(elite_fitness: Tensor, num_offspring: int, frac_tournament: float) -> tuple[Tensor, Tensor]:
    n, device = elite_fitness.numel(), elite_fitness.device
    if n == 1:
        zeros = torch.zeros(num_offspring, dtype = torch.long, device = device)
        return zeros, zeros
    k = max(2, min(n, int(n * frac_tournament)))
    contenders = torch.rand(num_offspring, n, device = device).argsort(dim = -1)[:, :k]
    top2 = contenders.gather(-1, elite_fitness[contenders].topk(2, dim = -1).indices)
    return top2[:, 0], top2[:, 1]

def crossover_and_branch_(
    pop: Population,
    opt,
    fitness: Tensor,
    *,
    elite_count: int,
    mutation_prob: float,
    frac_tournament: float,
    mutation_scale_bounds: tuple[float, float] = (0.005, 0.05),
    eps: float = 1e-5,
) -> dict:
    device = fitness.device
    order = fitness.argsort(descending = True)
    pop.keep(order)
    sorted_fit = fitness[order]
    pop.ema_('fitness_ema', sorted_fit, decay = 0.7)

    elite_slots = list(range(elite_count))
    offspring_slots = list(range(elite_count, pop.pop_size))
    pop.track_('lineage', ELITE, members = elite_slots)
    pop.track_('age', pop.tracked('age')[:elite_count] + 1, members = elite_slots)

    if not offspring_slots:
        return dict(order = order, sorted_fitness = sorted_fit, mut_slots = [], ft_slots = [], expected_ft_fitness = torch.empty(0, device = device))

    elite_fit = sorted_fit[:elite_count]
    p1, p2 = select_elite_parents(elite_fit, len(offspring_slots), frac_tournament)

    f1, f2 = elite_fit[p1], elite_fit[p2]
    min_fit = sorted_fit.min()
    s1, s2 = f1 - min_fit + eps, f2 - min_fit + eps
    alpha = s1 / (s1 + s2 + eps)

    # offspring_slots [E..P-1] is disjoint from elites [0..E-1], so view copy + lerp_ is aliasing-safe
    pop[offspring_slots] = pop.view(p1)
    pop.view(offspring_slots).lerp_(pop.view(p2), 1. - alpha)
    opt.reset(members = offspring_slots)
    pop.track_('age', 0, members = offspring_slots)

    mut_mask = torch.rand(len(offspring_slots), device = device) < mutation_prob
    mut_local = mut_mask.nonzero(as_tuple = False).flatten()
    ft_local = (~mut_mask).nonzero(as_tuple = False).flatten()
    mut_slots = [offspring_slots[i] for i in mut_local.tolist()]
    ft_slots = [offspring_slots[i] for i in ft_local.tolist()]

    if mut_slots:
        lo, hi = mutation_scale_bounds
        scale = ((f1[mut_local] - f2[mut_local]).abs() / (s1[mut_local] + s2[mut_local]).clamp(min = eps)).clamp(lo, hi)
        with torch.no_grad():
            pop.flat_params[mut_slots] += torch.randn_like(pop.flat_params[mut_slots]) * scale[:, None]
        pop.track_('lineage', MUT, members = mut_slots)

    if ft_slots:
        pop.track_('lineage', PPO_FT, members = ft_slots)

    expected_ft = (alpha * f1 + (1. - alpha) * f2)[ft_local] if ft_slots else torch.empty(0, device = device)
    return dict(order = order, sorted_fitness = sorted_fit, mut_slots = mut_slots, ft_slots = ft_slots, expected_ft_fitness = expected_ft)

@torch.no_grad()
def population_weight_diversity(pop: Population) -> float:
    return float(torch.pdist(pop.flat_params.detach().float()).mean().item()) if (pop.is_population and pop.pop_size >= 2) else 0.

# video recording & dashboards

@lru_cache(maxsize = 8)
def load_font(size: int):
    try:
        return ImageFont.load_default(size = size)
    except TypeError:
        return ImageFont.load_default()

def save_video_pair(frames: list[np.ndarray], video_dir: Path, stem: str, *, fps: int = 30, gif_fps: int = 12, gif_width: int = 760) -> tuple[Path, Path]:
    video_dir.mkdir(parents = True, exist_ok = True)
    mp4, gif = video_dir / f'{stem}.mp4', video_dir / f'{stem}.gif'
    imageio.mimsave(str(mp4), frames, fps = fps, codec = 'libx264', macro_block_size = 1)
    vf = f'fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(mp4), '-vf', vf, str(gif)], check = True)
    return mp4, gif

def make_grid_frame(
    tiles: list[np.ndarray],
    tile_returns: np.ndarray,
    tile_done: np.ndarray,
    eval_fitness: np.ndarray,
    lineages: list[str],
    gen: int,
    step_idx: int,
    diversity: float,
    tile_w: int = 300,
    tile_h: int = 200,
) -> np.ndarray:
    n = len(tiles)
    cols = 4 if n in (4, 8, 12, 16) else min(4, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    header_h, banner_h, pad = 26, 34, 4
    cell_w, cell_h = tile_w + 2 * pad, tile_h + header_h + 2 * pad
    canvas_w = cols * cell_w + (cols * cell_w) % 2
    canvas_h = banner_h + rows * cell_h + (banner_h + rows * cell_h) % 2

    canvas = Image.new('RGB', (canvas_w, canvas_h), '#0f121a')
    draw = ImageDraw.Draw(canvas)
    font_banner, font_tile, font_badge = load_font(15), load_font(13), load_font(14)

    banner = (
        f'EPO (Miikkulainen) + torch-population  |  Gen {gen:02d}  |  '
        f'Best {eval_fitness.max():+6.1f}  Mean {eval_fitness.mean():+6.1f}  '
        f'Min {eval_fitness.min():+6.1f}  |  Div {diversity:.2f}  |  t={step_idx:03d}'
    )
    draw.rectangle([0, 0, canvas_w, banner_h], fill = '#161b26')
    draw.text((10, 8), banner, fill = '#eef2fa', font = font_banner)

    for idx in range(n):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, banner_h + r * cell_h
        is_best = idx == 0
        lineage = lineages[idx]
        accent = PALETTE['BEST'] if is_best else PALETTE.get(lineage, PALETTE['BASE'])

        draw.rectangle([x0 + 1, y0 + 1, x0 + cell_w - 2, y0 + cell_h - 2], outline = accent, width = 2)
        draw.rectangle([x0 + pad, y0 + pad, x0 + cell_w - pad, y0 + pad + header_h], fill = '#1a202c')
        prefix = f'#{idx + 1}{"*" if is_best else " "} {lineage:6s}'
        draw.text((x0 + pad + 6, y0 + pad + 5), f'{prefix} | eval {eval_fitness[idx]:+6.1f} | live {tile_returns[idx]:+6.1f}', fill = accent, font = font_tile)

        canvas.paste(Image.fromarray(tiles[idx]).resize((tile_w, tile_h), Image.BILINEAR), (x0 + pad, y0 + pad + header_h))

        if tile_done[idx]:
            ret = float(tile_returns[idx])
            status, bg = (f'LANDED {ret:+.1f}', '#16a34a') if ret >= 200. else ((f'DONE {ret:+.1f}', '#2563eb') if ret >= 0. else (f'CRASH {ret:+.1f}', '#dc2626'))
            px0, py0 = x0 + pad + 8, y0 + pad + header_h + 8
            draw.rectangle([px0, py0, px0 + 132, py0 + 24], fill = bg)
            draw.text((px0 + 8, py0 + 4), status, fill = '#ffffff', font = font_badge)

    return np.asarray(canvas, dtype = np.uint8)

@torch.no_grad()
def record_population_grid(
    pop: Population,
    fitness: Tensor,
    lineages: list[str],
    *,
    env_id: str,
    record_seed: int,
    gen: int,
    diversity: float,
    video_dir: Path,
    device: torch.device,
    max_episode_steps: int = 500,
    fps: int = 30,
) -> dict:
    pop.eval()
    n = pop.pop_size
    order = fitness.argsort(descending = True).cpu().tolist()
    sorted_fit = fitness.detach().cpu().numpy()[order]
    sorted_lin = [lineages[i] for i in order]

    envs = [make_env(env_id, max_episode_steps = max_episode_steps, render_mode = 'rgb_array') for _ in range(n)]
    obs = np.asarray([env.reset(seed = record_seed)[0] for env in envs], dtype = np.float32)

    active = np.ones(n, dtype = bool)
    ep_ret = np.zeros(n, dtype = np.float32)
    tiles = [env.render() for env in envs]
    trajectories = [[(float(obs[i, 0]), float(obs[i, 1]))] for i in range(n)]
    frames, step_idx = [], 0

    while active.any() and step_idx < max_episode_steps:
        frames.append(make_grid_frame(tiles, ep_ret, ~active, sorted_fit, sorted_lin, gen, step_idx, diversity))
        logits, _ = pop(torch.from_numpy(obs).to(device), members = order, per_individual = True)
        actions = logits.argmax(dim = -1).cpu().numpy()

        for i in np.flatnonzero(active):
            next_o, r, term, trunc, _ = envs[i].step(int(actions[i]))
            obs[i] = next_o
            ep_ret[i] += r
            trajectories[i].append((float(next_o[0]), float(next_o[1])))
            tiles[i] = envs[i].render()
            if term or trunc:
                active[i] = False
        step_idx += 1

    final_grid = make_grid_frame(tiles, ep_ret, np.ones(n, dtype = bool), sorted_fit, sorted_lin, gen, step_idx, diversity)
    frames.extend([final_grid] * (fps // 2))
    for env in envs:
        env.close()

    mp4, gif = save_video_pair(frames, video_dir, f'gen_{gen:03d}_population', fps = fps)
    png = video_dir / f'gen_{gen:03d}_final_frame.png'
    Image.fromarray(final_grid).save(png)

    return dict(
        gen = gen,
        mp4 = str(mp4),
        gif = str(gif),
        final_png = str(png),
        showcase_returns = ep_ret.tolist(),
        eval_returns = sorted_fit.tolist(),
        lineages = sorted_lin,
        trajectories = trajectories,
        grid_frames_subsampled = frames[::2],
    )

def plot_returns_over_time(history: list[dict], out_path: Path, eval_episodes: int = 5, solve_return: float = 200.):
    plt.rcParams.update(PLOT_STYLE)
    def col(k: str) -> np.ndarray:
        return np.array([h[k] for h in history])
    gens = col('generation')
    has_meta = 'meta_temp_mean' in history[0]

    ncols = 3 if has_meta else 2
    fig, axes = plt.subplots(1, ncols, figsize = (5.8 * ncols, 4.8))
    legend_kw = dict(frameon = True, facecolor = '#0f121a', edgecolor = '#2d3748')

    # panel 1: returns
    ax = axes[0]
    ax.fill_between(gens, col('min_return'), col('best_return'), color = '#38bdf8', alpha = 0.14, label = 'Pop [min, max]')
    seen = set()
    for h in history:
        for ret, lin in zip(h['member_returns'], h['lineages']):
            ax.scatter(h['generation'], ret, color = PALETTE.get(lin, PALETTE['BASE']), s = 18, alpha = 0.65, zorder = 3, label = None if lin in seen else lin)
            seen.add(lin)

    ax.plot(gens, col('mean_return'), color = '#38bdf8', lw = 2.0, label = 'Pop mean', zorder = 4)
    ax.plot(gens, col('elite_mean_return'), color = '#34d399', lw = 2.0, ls = '--', label = 'Elite mean', zorder = 4)
    ax.plot(gens, col('best_return'), color = '#f5c542', lw = 2.6, label = 'Best individual', zorder = 5)
    ax.axhline(solve_return, color = '#22c55e', lw = 1.2, ls = ':', alpha = 0.85, label = f'Solved ({solve_return:+.0f})')
    ax.axhline(0., color = '#64748b', lw = 0.8, alpha = 0.5)
    ax.set(title = 'Population Returns Over Generations', xlabel = 'Generation', ylabel = f'Mean Eval Return ({eval_episodes} paired episodes)')
    ax.title.set(fontsize = 12, fontweight = 'bold')
    ax.grid(alpha = 0.4)
    ax.legend(fontsize = 8, loc = 'lower right', ncol = 2, **legend_kw)

    # panel 2: diversity & entropy
    ax2 = axes[1]
    l1 = ax2.plot(gens, col('weight_diversity'), color = '#fb923c', lw = 2.2, label = 'Weight diversity (pdist)')
    ax2.set(title = 'Exploration vs Exploitation Dynamics', xlabel = 'Generation')
    ax2.title.set(fontsize = 12, fontweight = 'bold')
    ax2.set_ylabel('Mean Pairwise L2 Weight Distance', color = '#fb923c')
    ax2.tick_params(axis = 'y', colors = '#fb923c')
    ax2.grid(alpha = 0.4)

    ax2_r = ax2.twinx()
    l2 = ax2_r.plot(gens, col('mean_entropy'), color = '#a78bfa', lw = 2.0, ls = '--', label = 'Policy entropy')
    ax2_r.set_ylabel('Mean Policy Entropy (nats)', color = '#a78bfa')
    ax2_r.tick_params(axis = 'y', colors = '#a78bfa')
    ax2_r.spines['right'].set_color('#a78bfa')
    ax2.legend(l1 + l2, [l.get_label() for l in l1 + l2], fontsize = 9, loc = 'upper right', **legend_kw)

    # panel 3: meta-evolved temperature
    if has_meta:
        ax3 = axes[2]
        ax3.fill_between(gens, col('meta_temp_min'), col('meta_temp_max'), color = '#f472b6', alpha = 0.2, label = 'CEM pop [min, max]')
        ax3.plot(gens, col('meta_temp_mean'), color = '#f472b6', lw = 2.2, label = 'CEM mean temp')
        ax3.set(title = 'Meta-Evolved Delightful Gate Temp (CEM)', xlabel = 'Generation', ylabel = 'Temperature')
        ax3.title.set(fontsize = 12, fontweight = 'bold')
        ax3.grid(alpha = 0.4)
        ax3.legend(fontsize = 9, **legend_kw)

    fig.tight_layout()
    fig.savefig(out_path, dpi = 180, bbox_inches = 'tight')
    plt.close(fig)

def plot_population_trajectories(snapshots: list[dict], out_path: Path):
    if not snapshots:
        return
    plt.rcParams.update(PLOT_STYLE)

    idxs = dict.fromkeys(np.linspace(0, len(snapshots) - 1, min(6, len(snapshots))).round().astype(int).tolist())
    chosen = [snapshots[i] for i in idxs]
    cols = min(3, len(chosen))
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize = (5.0 * cols, 4.2 * rows), sharex = True, sharey = True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, snap in zip(axes_flat, chosen):
        ax.plot([-0.2, 0.2], [0., 0.], color = '#f5c542', lw = 3.5, solid_capstyle = 'round', zorder = 2)
        for x_pad in (-0.2, 0.2):
            ax.axvline(x_pad, color = '#f5c542', lw = 0.8, ls = ':', alpha = 0.4)
        ax.axhline(0., color = '#475569', lw = 0.8, ls = '--', alpha = 0.5)

        for idx in reversed(range(len(snap['trajectories']))):
            pts = np.asarray(snap['trajectories'][idx])
            color = PALETTE['BEST'] if idx == 0 else PALETTE.get(snap['lineages'][idx], PALETTE['BASE'])
            ax.plot(pts[:, 0], pts[:, 1], color = color, lw = 2.4 if idx == 0 else 1.3, alpha = 0.95 if idx == 0 else 0.72, zorder = 4 if idx == 0 else 3)
            ax.scatter(pts[-1, 0], pts[-1, 1], color = color, s = 34 if idx == 0 else 18, edgecolor = '#0f121a', lw = 0.6, zorder = 5)

        rets = np.asarray(snap['showcase_returns'])
        ax.set(title = f"Gen {snap['gen']:02d}  |  Best {rets.max():+.0f}  Mean {rets.mean():+.0f}", xlim = (-1.05, 1.05), ylim = (-0.15, 1.55), xlabel = 'x position', ylabel = 'y altitude')
        ax.title.set(fontsize = 11, fontweight = 'bold')
        ax.grid(alpha = 0.35)

    for ax in axes_flat[len(chosen):]:
        ax.axis('off')

    legend_items = [('BEST', '#1 Best', 2.4), ('ELITE', 'Elite', 1.6), ('PPO-FT', 'PPO-FT', 1.6), ('MUT', 'Mutated', 1.6), ('BASE', 'Base (Gen 0)', 1.6)]
    handles = [plt.Line2D([0], [0], color = PALETTE[k], lw = lw, label = lbl) for k, lbl, lw in legend_items]
    fig.legend(handles = handles, loc = 'lower center', ncol = 5, frameon = True, facecolor = '#161b26', edgecolor = '#2d3748', fontsize = 9.5)
    fig.suptitle('Population 2D Flight Trajectories Across Evolution (Same Showcase Seed)', fontsize = 13, fontweight = 'bold', y = 1.01)
    fig.tight_layout(rect = [0, 0.05, 1, 1])
    fig.savefig(out_path, dpi = 180, bbox_inches = 'tight')
    plt.close(fig)

# main

def main(
    env_id: str = 'LunarLander-v3',
    population_size: int = 8,
    elite_count: int = 3,
    mutation_prob: float = 0.3,
    mutation_scale_min: float = 0.005,
    mutation_scale_max: float = 0.05,
    frac_tournament: float = 0.67,
    init_noise_std: float = 0.01,
    total_generations: int = 30,
    record_every: int = 5,
    pretrain_steps: int = 32768,
    pretrain_envs: int = 8,
    finetune_steps: int = 4096,
    finetune_envs_per_child: int = 4,
    eval_episodes: int = 5,
    final_eval_episodes: int = 20,
    hidden_dim: int = 64,
    ppo_lr: float = 1e-3,
    ppo_gamma: float = 0.99,
    ppo_gae_lambda: float = 0.95,
    ppo_eps_clip: float = 0.2,
    ppo_value_clip: float | None = None,
    ppo_k_epochs: int = 6,
    ppo_minibatch_size: int = 64,
    ppo_rollout_steps: int = 256,
    ppo_entropy_coef: float = 0.01,
    ppo_max_grad_norm: float = 0.5,
    max_episode_steps: int = 500,
    use_delightful_gating: bool = True,
    delight_temp: float = 2.7,
    use_meta_evolution: bool = True,
    meta_pop_size: int = 8,
    meta_num_elite: int = 3,
    meta_kappa: float = 1.0,
    out_dir: str = './epo-miikkulainen-run',
    seed: int = 42,
    record_seed: int = 4242,
    device: str | None = None,
    use_wandb: bool = False,
    wandb_project: str = 'epo-lunarlander',
    clear_out_dir: bool = True,
):
    cli_args = dict(locals())
    seed_everything(seed)
    torch.set_num_threads(1)
    dev = resolve_device(device)

    try:
        probe_env = make_env(env_id)
    except gym.error.NameNotFound:
        print(f'environment {env_id} not found, falling back to LunarLander-v2')
        env_id = 'LunarLander-v2'
        probe_env = make_env(env_id)

    state_dim, action_dim = int(probe_env.observation_space.shape[0]), int(probe_env.action_space.n)
    probe_env.close()

    use_delightful_gating = use_delightful_gating or use_meta_evolution
    out_path = Path(out_dir)
    if clear_out_dir and out_path.exists():
        shutil.rmtree(out_path)
    video_dir = out_path / 'videos'
    video_dir.mkdir(parents = True, exist_ok = True)

    config = {**cli_args, 'env_id': env_id, 'state_dim': state_dim, 'action_dim': action_dim, 'use_delightful_gating': use_delightful_gating, 'device': str(dev)}
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(project = wandb_project, config = config)

    t_start = time.perf_counter()
    print(f'EPO (Miikkulainen) + torch-population | env={env_id} | pop={population_size} (elites={elite_count}) | device={dev}')
    print(f'outputs -> {out_path.resolve()}')

    # 1. pre-train base policy inside a zero-overhead noop Population wrapper
    pop = Population(ActorCritic(state_dim, action_dim, hidden_dim = hidden_dim).to(dev))
    opt = pop.optimizer(Adam, lr = ppo_lr, eps = 1e-5)

    run_ppo = partial(
        train_ppo, pop, opt, env_id,
        device = dev,
        minibatch_size = ppo_minibatch_size,
        k_epochs = ppo_k_epochs,
        gamma = ppo_gamma,
        gae_lambda = ppo_gae_lambda,
        eps_clip = ppo_eps_clip,
        value_clip = ppo_value_clip,
        entropy_coef = ppo_entropy_coef,
        max_grad_norm = ppo_max_grad_norm,
        use_delightful_gating = use_delightful_gating,
        max_episode_steps = max_episode_steps,
    )

    print(f'\n[Phase 1] Pre-training base PPO policy ({pretrain_steps:,} steps across {pretrain_envs} envs)...')
    pretrain_stats = run_ppo(
        pretrain_steps,
        envs_per_agent = pretrain_envs,
        rollout_steps = ppo_rollout_steps,
        delight_temp = delight_temp,
        seed = seed,
        verbose = True,
    )

    # 2. fan out base policy -> vectorized population (slot 0 stays untouched base clone)
    pop.populate(population_size, init = 'clone', noise_std = init_noise_std, keep_first = True)
    pop.track_('lineage', torch.full((population_size,), BASE, dtype = torch.long, device = dev))
    pop.track_('age', torch.zeros(population_size, dtype = torch.long, device = dev))
    print(f'\n[Phase 2] Populated P={pop.pop_size} agents via pop.populate(init="clone", noise_std={init_noise_std}).')

    meta_cem = MetaCEM(pop_size = meta_pop_size, num_elite = meta_num_elite, device = dev) if use_meta_evolution else None

    history: list[dict] = []
    snapshots: list[dict] = []
    total_steps = pretrain_stats['steps']
    best_ever = -float('inf')

    # 3. evolutionary generations
    for gen in range(total_generations + 1):
        eval_seeds = [seed * 1000 + gen * 100 + e for e in range(eval_episodes)]
        fitness, entropies, _ = evaluate_population(pop, env_id, eval_seeds, device = dev, max_episode_steps = max_episode_steps)

        lineages = lineage_labels(pop)
        order = fitness.argsort(descending = True)
        sorted_fit = fitness[order]
        sorted_lin = [lineages[i] for i in order.tolist()]
        diversity = population_weight_diversity(pop)

        best_ret = float(sorted_fit[0].item())
        if best_ret > best_ever:
            best_ever = best_ret
            torch.save(pop[int(order[0].item())].state_dict(), out_path / 'best_policy.pt')

        row = dict(
            generation = gen,
            env_steps = total_steps,
            elapsed_sec = round(time.perf_counter() - t_start, 2),
            best_return = best_ret,
            elite_mean_return = float(sorted_fit[:elite_count].mean().item()),
            mean_return = float(sorted_fit.mean().item()),
            min_return = float(sorted_fit[-1].item()),
            mean_entropy = float(entropies.mean().item()),
            weight_diversity = diversity,
            member_returns = [float(x) for x in sorted_fit.tolist()],
            lineages = sorted_lin,
        )
        if exists(meta_cem):
            ms = meta_cem.stats()
            row.update(meta_temp_mean = ms['mean'], meta_temp_min = ms['min'], meta_temp_max = ms['max'], meta_cem_updates = ms['updates'])
        history.append(row)

        meta_str = f' | temp={row["meta_temp_mean"]:.2f}' if exists(meta_cem) else ''
        print(
            f'gen {gen:2d}/{total_generations:2d} | best={best_ret:+7.1f} | elite={row["elite_mean_return"]:+7.1f} | '
            f'mean={row["mean_return"]:+7.1f} | min={row["min_return"]:+7.1f} | ent={row["mean_entropy"]:.3f} | '
            f'div={diversity:.2f}{meta_str} | steps={total_steps:,} | {row["elapsed_sec"]:5.1f}s'
        )
        if exists(wandb_run):
            wandb_run.log({k: v for k, v in row.items() if isinstance(v, (int, float))})

        if gen in (0, total_generations) or divisible_by(gen, record_every):
            snap = record_population_grid(
                pop, fitness, lineages,
                env_id = env_id, record_seed = record_seed, gen = gen, diversity = diversity,
                video_dir = video_dir, device = dev, max_episode_steps = max_episode_steps,
            )
            snapshots.append(snap)
            print(f'  -> recorded {snap["mp4"]} (showcase: {[round(r, 1) for r in snap["showcase_returns"]]})')
            if exists(wandb_run):
                import wandb
                wandb_run.log(dict(generation = gen, population_video = wandb.Video(snap['mp4'], fps = 30, format = 'mp4')))

        if gen == total_generations:
            break

        # 4. evolve: rank -> keep elites -> crossover -> mutate or Lamarckian PPO fine-tune
        evo = crossover_and_branch_(
            pop, opt, fitness,
            elite_count = elite_count,
            mutation_prob = mutation_prob,
            frac_tournament = frac_tournament,
            mutation_scale_bounds = (mutation_scale_min, mutation_scale_max),
        )
        ft_slots = evo['ft_slots']

        if ft_slots:
            meta_ids, ft_temps = meta_cem.assign(len(ft_slots)) if exists(meta_cem) else (None, delight_temp)
            ft_stats = run_ppo(
                finetune_steps,
                members = ft_slots,
                envs_per_agent = finetune_envs_per_child,
                rollout_steps = min(ppo_rollout_steps, max(32, finetune_steps // finetune_envs_per_child)),
                delight_temp = ft_temps,
                seed = seed + (gen + 1) * 10000,
            )
            total_steps += ft_stats['steps']

            if exists(meta_cem):
                ft_fit, ft_ent, _ = evaluate_population(pop, env_id, eval_seeds, device = dev, members = ft_slots, max_episode_steps = max_episode_steps)
                meta_cem.record(meta_ids, (ft_fit - evo['expected_ft_fitness']) + meta_kappa * ft_ent)

    # 5. held-out evaluation on unseen seeds
    heldout_seeds = [900_000 + i for i in range(final_eval_episodes)]
    ho_mean, ho_ent, ho_std = evaluate_population(pop, env_id, heldout_seeds, device = dev, max_episode_steps = max_episode_steps)
    ho_order = ho_mean.argsort(descending = True)
    final_lin, final_age = lineage_labels(pop), pop.tracked('age').tolist()

    print(f'\n[Final Held-Out Eval ({final_eval_episodes} unseen seeds)]')
    for rank, idx in enumerate(ho_order.tolist(), start = 1):
        print(f'  #{rank} (slot {idx}, {final_lin[idx]:6s}, age={final_age[idx]:2d}): {ho_mean[idx].item():+7.1f} +/- {ho_std[idx].item():5.1f} | ent={ho_ent[idx].item():.3f}')

    # 6. plots, timelapse video, CSV & JSON metrics
    returns_plot, traj_plot = out_path / 'returns_over_time.png', out_path / 'population_trajectories.png'
    plot_returns_over_time(history, returns_plot, eval_episodes = eval_episodes)
    plot_population_trajectories(snapshots, traj_plot)

    timelapse_frames = [f for snap in snapshots for f in snap['grid_frames_subsampled']]
    tl_mp4, tl_gif = save_video_pair(timelapse_frames, video_dir, 'evolution_timelapse', fps = 24)

    csv_cols = ['generation', 'env_steps', 'elapsed_sec', 'best_return', 'elite_mean_return', 'mean_return', 'min_return', 'mean_entropy', 'weight_diversity']
    if exists(meta_cem):
        csv_cols += ['meta_temp_mean', 'meta_temp_min', 'meta_temp_max']
    csv_path = out_path / 'returns.csv'
    with open(csv_path, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(csv_cols + [f'rank_{i + 1}_return' for i in range(population_size)])
        for h in history:
            writer.writerow([h[k] for k in csv_cols] + h['member_returns'])

    summary = dict(
        config = config,
        history = history,
        heldout_eval = dict(
            episodes = final_eval_episodes,
            best_return = float(ho_mean.max().item()),
            mean_return = float(ho_mean.mean().item()),
            member_means = [float(x) for x in ho_mean[ho_order].tolist()],
            member_stds = [float(x) for x in ho_std[ho_order].tolist()],
        ),
        artifacts = dict(
            returns_plot = str(returns_plot),
            trajectories_plot = str(traj_plot),
            timelapse_mp4 = str(tl_mp4),
            timelapse_gif = str(tl_gif),
            periodic_videos = [
                {k: s[k] for k in ('gen', 'mp4', 'gif', 'final_png', 'showcase_returns')}
                for s in snapshots
            ],
            returns_csv = str(csv_path),
            best_checkpoint = str(out_path / 'best_policy.pt'),
        ),
    )
    with open(out_path / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent = 2)

    print(f'\nSaved summary -> {returns_plot}, {traj_plot}, {tl_mp4}, {out_path / "metrics.json"}')
    if exists(wandb_run):
        wandb_run.finish()

if __name__ == '__main__':
    fire.Fire(main)
