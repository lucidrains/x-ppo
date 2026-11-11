# /// script
# dependencies = [
#   "assoc-scan",
#   "einops",
#   "einx",
#   "fire",
#   "gymnasium[box2d,other]",
#   "imageio[ffmpeg]",
#   "matplotlib",
#   "numpy",
#   "pillow",
#   "torch",
#   "torch-einops-utils>=0.2.5",
#   "vector-quantize-pytorch>=1.34.0",
# ]
# ///
"""unified multi-horizon semi-markov PPO across VQ action hierarchies.

stage 1 trains a dt = 1 policy while intermittently quantizing h-step action chunks
into discrete macro-codebooks (num_actions * h codes for h in 2, 4, 8).
stage 2 exposes the disjoint union of all hierarchies (dt = 1, 2, 4, 8 -> 60 actions)
to a factorized actor trained with exact semi-markov GAE:

    delta_k = R_k^(gamma) + gamma^(tau_k) * (1 - term_k) * V(S_{k+1}) - V(S_k)
    A_k     = delta_k + gamma^(tau_k) * lam^(tau_k ^ trace_power) * (1 - done_k) * A_{k+1}

where trace_power = 1. (default clock trace, equal physical lookahead & rich dt mix),
      trace_power = 0. (decision trace, h-times longer horizon on sparse rewards),
   or trace_power in (0., 1.) (smooth interpolation, e.g. 0.5 -> sqrt(tau_k) trace).

    uv run train_unified_smdp.py
"""

from __future__ import annotations

import csv
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import fire
import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import Tensor, cat, nn, stack, tensor, zeros
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import VectorQuantize

from x_ppo import (
    calc_gae,
    clipped_value_loss_fn,
    hpo_actor_loss
)

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

ENTROPY_SCHEDULE = (0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002)
LR_SCALE_SCHEDULE = (1., 0.85, 0.50, 0.20, 0.10, 0.05)

PRIM_NAMES = ('N', 'L', 'M', 'R')
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

def entropy(prob):
    return -(prob * log(prob)).sum(dim = -1)

def normalized_entropy(prob):
    return entropy(prob) / math.log(max(2, prob.shape[-1]))

def schedule_at(schedule, progress):
    idx = min(int(progress * len(schedule)), len(schedule) - 1)
    return schedule[max(0, idx)]

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

# networks

class SimBa(Module):
    def __init__(self, dim, expansion = 2, dropout = 0.):
        super().__init__()
        dim_inner = int(dim * expansion)

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

class Critic(Module):
    def __init__(self, state_dim, hidden_dim = 192):
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

    def forward(self, state):
        return self.net(state).squeeze(-1)

class UnifiedHierarchyActor(Module):
    def __init__(
        self,
        state_dim: int,
        action_counts: tuple[int, ...],
        hidden_dim = 64,
        factorized = True,
        dropout = 0.
    ):
        super().__init__()
        self.action_counts = action_counts
        self.num_hierarchies = len(action_counts)
        self.total_actions = sum(action_counts)
        self.factorized = factorized and self.num_hierarchies > 1

        offsets = tensor([0, *action_counts[:-1]]).cumsum(0)
        self.register_buffer('offsets', offsets, persistent = False)
        self.register_buffer('log_counts', tensor(action_counts, dtype = torch.float32).log(), persistent = False)

        dim = hidden_dim * 2

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, dim),
            SimBa(dim, dropout = dropout),
            SimBa(dim, dropout = dropout),
            nn.Linear(dim, hidden_dim),
            nn.SiLU()
        )

        if self.factorized:
            self.to_hier_logits = nn.Linear(hidden_dim, self.num_hierarchies)
            nn.init.zeros_(self.to_hier_logits.weight)
            nn.init.zeros_(self.to_hier_logits.bias)

            self.to_code_logits = ModuleList([
                nn.Linear(hidden_dim, count) for count in action_counts
            ])
        else:
            self.to_flat_logits = nn.Linear(hidden_dim, self.total_actions)

    def forward(self, state: Tensor):
        feat = self.backbone(state)

        if not self.factorized:
            log_probs = self.to_flat_logits(feat).log_softmax(dim = -1)
            probs = log_probs.exp()
            hier_probs = stack([c.sum(dim = -1) for c in probs.split(self.action_counts, dim = -1)], dim = -1)
            return log_probs, hier_probs, None

        hier_log_probs = self.to_hier_logits(feat).log_softmax(dim = -1)
        hier_probs = hier_log_probs.exp()
        code_log_probs = [head(feat).log_softmax(dim = -1) for head in self.to_code_logits]

        joint_log_probs = cat([
            hier_lp + code_lp
            for hier_lp, code_lp in zip(hier_log_probs.split(1, dim = -1), code_log_probs)
        ], dim = -1)

        within_entropies = stack([-(lp.exp() * lp).sum(dim = -1) for lp in code_log_probs], dim = -1) / self.log_counts
        within_ent = (hier_probs * within_entropies).sum(dim = -1)
        return joint_log_probs, hier_probs, within_ent

    def sample(self, state: Tensor):
        joint_log_probs, *_ = self.forward(state)
        dist = Categorical(logits = joint_log_probs, validate_args = False)
        action = dist.sample()
        return action, dist.log_prob(action)

    def greedy(self, state: Tensor):
        joint_log_probs, hier_probs, _ = self.forward(state)

        if not self.factorized:
            return joint_log_probs.argmax(dim = -1), hier_probs

        best_level = hier_probs.argmax(dim = -1)
        best_local = stack([c.argmax(dim = -1) for c in joint_log_probs.split(self.action_counts, dim = -1)], dim = -1)
        local_code = best_local.gather(-1, best_level[..., None]).squeeze(-1)
        return self.offsets[best_level] + local_code, hier_probs

# vector quantized action chunks

class ActionChunkVQ(Module):
    def __init__(
        self,
        hierarchy: int,
        num_actions: int,
        codebook_size: int | None = None,
        dim = 128,
        codebook_dim = 16,
        decay = 0.9
    ):
        super().__init__()
        num_codes = default(codebook_size, num_actions * hierarchy)
        chunk_dim = hierarchy * num_actions

        self.hierarchy = hierarchy
        self.num_actions = num_actions
        self.num_codes = num_codes

        self.encoder = nn.Sequential(
            nn.Linear(chunk_dim, dim),
            SimBa(dim),
            nn.Linear(dim, dim)
        )

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = num_codes,
            codebook_dim = codebook_dim,
            layernorm_after_project_in = True,
            decay = decay,
            commitment_weight = 1.,
            kmeans_init = True,
            kmeans_iters = 10,
            dead_code_threshold_ratio = 0.05,
            rotation_trick = True
        )

        self.decoder = nn.Sequential(
            SimBa(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, chunk_dim),
            Rearrange('... (h a) -> ... h a', h = hierarchy)
        )

    def forward(self, chunks: Tensor):
        one_hot = F.one_hot(chunks.long(), self.num_actions).float()
        quantized, _, vq_loss = self.vq(self.encoder(rearrange(one_hot, 'b h a -> b (h a)')))
        logits = self.decoder(quantized)

        recon_loss = F.cross_entropy(
            rearrange(logits, 'b h a -> (b h) a'),
            rearrange(chunks.long(), 'b h -> (b h)')
        )
        seq_acc = (logits.argmax(dim = -1) == chunks).all(dim = -1).float().mean()
        return recon_loss + vq_loss, seq_acc

    @torch.no_grad()
    def codebook_table(self) -> np.ndarray:
        dev = next(self.parameters()).device
        quantized = self.vq.get_output_from_indices(torch.arange(self.num_codes, device = dev))
        return self.decoder(quantized).argmax(dim = -1).cpu().numpy()

class ChunkBuffer:
    def __init__(self, capacity: int, hierarchy: int, stride = 2):
        self.capacity = capacity
        self.hierarchy = hierarchy
        self.stride = stride
        self.chunks = zeros((capacity, hierarchy), dtype = torch.long)
        self.ptr, self.size = 0, 0

    def add_rollout(self, actions: Tensor, dones: Tensor):
        h = self.hierarchy
        if actions.shape[1] < h:
            return

        act_win = actions.detach().cpu().unfold(1, h, self.stride)
        done_win = dones.detach().cpu().unfold(1, h, self.stride)
        valid = ~done_win[..., :-1].any(dim = -1) if h > 1 else torch.ones_like(done_win[..., 0], dtype = torch.bool)
        chunks = act_win[valid].long()[-self.capacity:]
        n = chunks.shape[0]
        if n == 0:
            return

        idx = (self.ptr + torch.arange(n)) % self.capacity
        self.chunks[idx] = chunks
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def train_steps(self, model: ActionChunkVQ, opt: torch.optim.Optimizer, steps = 80, batch_size = 256):
        if self.size < batch_size:
            return 0., 0

        dev = next(model.parameters()).device
        model.train()

        with torch.random.fork_rng():
            if not bool(model.vq._codebook.initted.item()):
                init_n = min(self.size, max(1024, model.num_codes * 16))
                with torch.no_grad():
                    model(self.chunks[torch.randint(0, self.size, (init_n,))].to(dev))

            accs = []
            for _ in range(steps):
                loss, seq_acc = model(self.chunks[torch.randint(0, self.size, (batch_size,))].to(dev))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                accs.append(float(seq_acc))

        unique_codes = int(np.unique(model.codebook_table(), axis = 0).shape[0])
        return float(np.mean(accs[-16:])), unique_codes

# unified environment & semi-markov ppo

class ActionCatalog:
    def __init__(self, codebooks: dict[int, np.ndarray]):
        self.hierarchies = tuple(sorted(codebooks))
        self.counts = tuple(int(codebooks[h].shape[0]) for h in self.hierarchies)
        self.total_actions = sum(self.counts)

        self.horizons, self.local_codes, self.sequences = [], [], []
        for h in self.hierarchies:
            for code_idx, seq in enumerate(codebooks[h]):
                self.horizons.append(h)
                self.local_codes.append(code_idx)
                self.sequences.append(tuple(int(a) for a in seq))

        self.horizons_np = np.asarray(self.horizons, dtype = np.int64)

class UnifiedHierarchyEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, catalog: ActionCatalog, gamma = 0.99):
        super().__init__(env)
        self.catalog = catalog
        self.gamma = gamma
        self.action_space = gym.spaces.Discrete(catalog.total_actions)

    def step(self, action):
        seq = self.catalog.sequences[int(action)]
        disc_reward, raw_reward, duration, last_reward = 0., 0., 0, 0.

        for step_idx, prim_action in enumerate(seq):
            obs, reward, terminated, truncated, info = self.env.step(prim_action)
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
            'last_reward': last_reward
        }

@dataclass
class SMDPRollout:
    states: Tensor
    next_states: Tensor
    actions: Tensor
    old_log_probs: Tensor
    rewards: Tensor
    durations: Tensor
    terminated: Tensor
    truncated: Tensor

class SemiMarkovPPO(Module):
    def __init__(
        self,
        state_dim: int,
        action_counts: tuple[int, ...],
        *,
        hidden_dim = 64,
        critic_hidden_dim = 192,
        factorized = True,
        use_smdp_gae = True,
        trace_power = 1.,
        marginal_ent_coef = 0.03,
        lr = 1e-3,
        critic_lr = 1e-3,
        gamma = 0.99,
        lam = 0.95,
        eps_clip = 0.2,
        max_grad_norm = 0.5,
        epochs = 6,
        minibatch_envs = 4
    ):
        super().__init__()
        self.actor = UnifiedHierarchyActor(state_dim, action_counts, hidden_dim = hidden_dim, factorized = factorized)
        self.critic = Critic(state_dim, hidden_dim = critic_hidden_dim)

        self.use_smdp_gae = use_smdp_gae
        self.trace_power = float(trace_power)
        self.marginal_ent_coef = marginal_ent_coef
        self.base_lr = lr
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_envs = minibatch_envs

        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr = lr)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr = critic_lr)

    def set_lr_scale(self, scale: float):
        for pg in self.opt_actor.param_groups:
            pg['lr'] = self.base_lr * scale

    def update(self, rollout: SMDPRollout, entropy_coef = 0.005):
        num_envs = rollout.states.shape[0]

        with torch.no_grad():
            values = self.critic(rollout.states)
            next_values = self.critic(rollout.next_states)
            gae = calc_gae(
                rewards = rollout.rewards,
                values = values,
                next_values = next_values,
                terminated = rollout.terminated,
                truncated = rollout.truncated,
                durations = rollout.durations if self.use_smdp_gae else None,
                trace_power = self.trace_power,
                gamma = self.gamma,
                lam = self.lam,
                use_accelerated = False,
                return_advantages = True
            )

        self.train()
        for _ in range(self.epochs):
            perm = torch.randperm(num_envs, device = rollout.states.device)

            for start in range(0, num_envs, self.minibatch_envs):
                mb = perm[start:start + self.minibatch_envs]

                mb_s = rearrange(rollout.states[mb], 'b k d -> (b k) d')
                mb_a = rearrange(rollout.actions[mb], 'b k -> (b k)')
                mb_old_lp = rearrange(rollout.old_log_probs[mb], 'b k -> (b k)')
                mb_adv = rearrange(gae.advantages[mb], 'b k -> (b k)')
                mb_ret = rearrange(gae.returns[mb], 'b k -> (b k)')
                mb_val = rearrange(values[mb], 'b k -> (b k)')

                joint_lp, hier_probs, within_ent = self.actor(mb_s)
                log_probs = joint_lp.gather(-1, mb_a[..., None]).squeeze(-1)
                surr_loss = hpo_actor_loss(log_probs, mb_old_lp, mb_adv, eps_clip = self.eps_clip, normalize_advantages = True).mean()

                if self.actor.factorized:
                    state_ent = 0.5 * (normalized_entropy(hier_probs) + within_ent).mean()
                    marginal_ent = normalized_entropy(hier_probs.mean(dim = 0))
                    ent_bonus = entropy_coef * state_ent + self.marginal_ent_coef * marginal_ent
                else:
                    ent_bonus = entropy_coef * entropy(joint_lp.exp()).mean()

                actor_loss = surr_loss - ent_bonus

                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                critic_loss = clipped_value_loss_fn(self.critic(mb_s), mb_val, mb_ret, clip = None)
                self.opt_critic.zero_grad()
                critic_loss.backward()
                self.opt_critic.step()

# stage 1: dt = 1 ppo + intermittent vq action-chunk discovery

def discover_vq_codebooks(
    env_name = 'LunarLander-v3',
    macro_hierarchies: tuple[int, ...] = (2, 4, 8),
    base_updates = 90,
    num_envs = 16,
    rollout_steps = 1024,
    max_episode_steps = 600,
    compress_every = 5,
    vq_steps = 80,
    vq_final_steps = 250,
    buffer_capacity = 65536,
    seed = 42
) -> dict[int, np.ndarray]:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = SyncVectorEnv(
            [lambda: gym.make(env_name, max_episode_steps = max_episode_steps) for _ in range(num_envs)],
            autoreset_mode = AutoresetMode.SAME_STEP
        )
        state_dim = env.single_observation_space.shape[0]
        num_actions = int(env.single_action_space.n)

        ppo = SemiMarkovPPO(
            state_dim, (num_actions,),
            hidden_dim = 108, factorized = False, use_smdp_gae = False,
            lr = 1e-3, critic_lr = 2.5e-3
        ).to(device)

        vqs = {h: ActionChunkVQ(h, num_actions).to(device) for h in macro_hierarchies}
        vq_opts = {h: torch.optim.AdamW(vqs[h].parameters(), lr = 2e-3) for h in macro_hierarchies}
        buffers = {h: ChunkBuffer(buffer_capacity, h) for h in macro_hierarchies}
        vq_info = {h: (0., 0) for h in macro_hierarchies}

        obs, _ = env.reset(seed = seed)
        state = torch.from_numpy(obs).float().to(device)
        ones_dur = torch.ones((num_envs, rollout_steps), device = device)
        running, recent, t0 = np.zeros(num_envs), [], time.time()

        print(f'[stage 1] dt = 1 PPO + VQ discovery for hierarchies {macro_hierarchies}', flush = True)

        for upd in range(1, base_updates + 1):
            progress = (upd - 1) / max(1, base_updates)
            ppo.set_lr_scale(schedule_at(LR_SCALE_SCHEDULE, progress))
            ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)

            states, next_states, actions, log_probs, rewards, terms, truncs = [], [], [], [], [], [], []

            ppo.eval()
            for _ in range(rollout_steps):
                with torch.no_grad():
                    action, lp = ppo.actor.sample(state)

                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
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

            actions_t = stack(actions, dim = 1)
            term_t, trunc_t = stack(terms, dim = 1), stack(truncs, dim = 1)

            for h in macro_hierarchies:
                buffers[h].add_rollout(actions_t, term_t | trunc_t)

            ppo.update(SMDPRollout(
                states = stack(states, dim = 1),
                next_states = stack(next_states, dim = 1),
                actions = actions_t,
                old_log_probs = stack(log_probs, dim = 1),
                rewards = stack(rewards, dim = 1),
                durations = ones_dur,
                terminated = term_t,
                truncated = trunc_t
            ), entropy_coef = ent_coef)

            if divisible_by(upd, compress_every):
                for h in macro_hierarchies:
                    vq_info[h] = buffers[h].train_steps(vqs[h], vq_opts[h], steps = vq_steps)

            if divisible_by(upd, 10) or upd == base_updates:
                ret32 = float(np.mean(recent[-32:])) if recent else float('nan')
                vq_str = ' | '.join(f'h={h}: {100 * vq_info[h][0]:.0f}% ({vq_info[h][1]}/{vqs[h].num_codes})' for h in macro_hierarchies)
                print(f'  [dt=1 upd {upd:2d}/{base_updates}] ret32 = {ret32:+6.1f} | {vq_str} [{time.time() - t0:.1f}s]', flush = True)

        env.close()

        codebooks = {1: np.arange(num_actions, dtype = np.int64)[:, None]}
        for h in macro_hierarchies:
            acc, uniq = buffers[h].train_steps(vqs[h], vq_opts[h], steps = vq_final_steps, batch_size = 512)
            codebooks[h] = vqs[h].codebook_table()
            macros = [''.join(PRIM_NAMES[int(a)] for a in row) for row in codebooks[h]]
            print(f'  [vq h={h}] seq_acc = {100 * acc:.1f}% | unique = {uniq}/{vqs[h].num_codes} | {macros}', flush = True)

    return codebooks

# stage 2: unified multi-hierarchy smdp training

def train_unified_variant(
    *,
    name: str,
    catalog: ActionCatalog,
    factorized: bool,
    use_smdp_gae: bool,
    marginal_ent_coef: float,
    trace_power = 1.,
    env_step_budget = 800_000,
    env_name = 'LunarLander-v3',
    num_envs = 16,
    rollout_decisions = 192,
    max_episode_steps = 600,
    lr = 1e-3,
    critic_lr = 1e-3,
    gamma = 0.99,
    lam = 0.95,
    seed = 42
):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = SyncVectorEnv(
            [lambda: UnifiedHierarchyEnv(gym.make(env_name, max_episode_steps = max_episode_steps), catalog, gamma = gamma) for _ in range(num_envs)],
            autoreset_mode = AutoresetMode.SAME_STEP
        )

        state_dim = env.single_observation_space.shape[0]
        ppo = SemiMarkovPPO(
            state_dim, catalog.counts,
            factorized = factorized,
            use_smdp_gae = use_smdp_gae,
            trace_power = trace_power,
            marginal_ent_coef = marginal_ent_coef,
            lr = lr, critic_lr = critic_lr,
            gamma = gamma, lam = lam
        ).to(device)

        obs, _ = env.reset(seed = seed + 100)
        state = torch.from_numpy(obs).float().to(device)
        running, recent, history = np.zeros(num_envs), [], []
        best_rolling = -float('inf')
        best_actor_state = deepcopy(ppo.actor.state_dict())
        env_steps, upd, t0 = 0, 0, time.time()

        while env_steps < env_step_budget:
            upd += 1
            progress = min(1., env_steps / max(1, env_step_budget))
            ppo.set_lr_scale(schedule_at(LR_SCALE_SCHEDULE, progress))
            ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)

            states, next_states, actions, log_probs = [], [], [], []
            rewards, durations, terms, truncs = [], [], [], []
            ep_rets = []
            h_counts = {h: 0 for h in catalog.hierarchies}

            ppo.eval()
            for _ in range(rollout_decisions):
                with torch.no_grad():
                    act_t, lp_t = ppo.actor.sample(state)

                act_np = act_t.cpu().numpy()
                next_obs, disc_r, terminated, truncated, info = env.step(act_np)
                done = terminated | truncated

                raw_r = step_info_array(info, 'raw_reward', done, 0.)
                durs = step_info_array(info, 'duration', done, 1.)

                for h in catalog.horizons_np[act_np]:
                    h_counts[int(h)] += 1

                running += raw_r
                env_steps += int(durs.sum())

                for idx in np.flatnonzero(done):
                    ep_rets.append(float(running[idx]))
                    recent.append(float(running[idx]))
                    running[idx] = 0.

                states.append(state)
                next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                actions.append(act_t)
                log_probs.append(lp_t)
                rewards.append(torch.from_numpy(disc_r / 100.).float().to(device))
                durations.append(torch.from_numpy(durs).float().to(device))
                terms.append(torch.from_numpy(terminated).bool().to(device))
                truncs.append(torch.from_numpy(truncated).bool().to(device))
                state = torch.from_numpy(next_obs).float().to(device)

            ppo.update(SMDPRollout(
                states = stack(states, dim = 1),
                next_states = stack(next_states, dim = 1),
                actions = stack(actions, dim = 1),
                old_log_probs = stack(log_probs, dim = 1),
                rewards = stack(rewards, dim = 1),
                durations = stack(durations, dim = 1),
                terminated = stack(terms, dim = 1),
                truncated = stack(truncs, dim = 1)
            ), entropy_coef = ent_coef)

            rec32 = float(np.mean(recent[-32:])) if recent else float('nan')
            upd_ret = float(np.mean(ep_rets)) if ep_rets else rec32

            if not np.isnan(rec32) and rec32 > best_rolling:
                best_rolling = rec32
                best_actor_state = deepcopy(ppo.actor.state_dict())

            tot_dec = max(1, sum(h_counts.values()))
            dec_pct = {h: 100. * h_counts[h] / tot_dec for h in catalog.hierarchies}
            history.append(dict(variant = name, update = upd, env_steps = env_steps, recent_return_32 = rec32, update_return = upd_ret, dec_pct = dec_pct))

            if divisible_by(upd, 10) or env_steps >= env_step_budget:
                mix_str = ' '.join(f'dt{h}:{dec_pct[h]:4.1f}%' for h in catalog.hierarchies)
                print(f'  [{name:12s} upd {upd:2d}] steps = {env_steps // 1000:3d}k | ret32 = {rec32:+6.1f} (best = {best_rolling:+6.1f}) | {mix_str} [{time.time() - t0:.1f}s]', flush = True)

        env.close()

    ppo.actor.load_state_dict(best_actor_state)
    eval_stats = evaluate_policy(ppo.actor, catalog, env_name = env_name, max_episode_steps = max_episode_steps)
    eval_stats['peak_rolling_32'] = best_rolling
    return history, ppo.actor, eval_stats

# evaluation, hud video rendering & summary plot

@torch.no_grad()
def evaluate_policy(
    actor: UnifiedHierarchyActor,
    catalog: ActionCatalog,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    num_episodes = 48,
    seed = 1000
) -> dict:
    actor.eval()
    env = gym.make(env_name, max_episode_steps = max_episode_steps)

    returns, landed, ranked_seeds = [], [], []
    dec_by_h = {h: 0 for h in catalog.hierarchies}
    step_by_h = {h: 0 for h in catalog.hierarchies}
    best_segments, best_score = [], -float('inf')

    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed = ep_seed)
        done, ep_ret, last_r = False, 0., 0.
        used_h, segments = set(), []

        while not done:
            u_t, _ = actor.greedy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
            u = int(u_t.item())
            h, seq = catalog.horizons[u], catalog.sequences[u]

            used_h.add(h)
            dec_by_h[h] += 1
            pts = [(float(obs[0]), float(obs[1]))]

            for prim_a in seq:
                obs, reward, terminated, truncated, _ = env.step(prim_a)
                pts.append((float(obs[0]), float(obs[1])))
                last_r = float(reward)
                ep_ret += last_r
                step_by_h[h] += 1
                if terminated or truncated:
                    done = True
                    break

            segments.append((h, pts))

        returns.append(ep_ret)
        landed.append(1. if last_r >= 50. else 0.)

        score = ep_ret + 8. * len(used_h)
        ranked_seeds.append((score, ep_ret, ep_seed))
        if score > best_score:
            best_score, best_segments = score, segments

    env.close()
    ranked_seeds.sort(reverse = True)
    tot_dec, tot_step = max(1, sum(dec_by_h.values())), max(1, sum(step_by_h.values()))

    return dict(
        mean_return = float(np.mean(returns)),
        std_return = float(np.std(returns)),
        best_return = float(max(returns)),
        land_rate = 100. * float(np.mean(landed)),
        top_seeds = [s for _, _, s in ranked_seeds],
        dec_frac = {h: 100. * dec_by_h[h] / tot_dec for h in catalog.hierarchies},
        step_frac = {h: 100. * step_by_h[h] / tot_step for h in catalog.hierarchies},
        best_segments = best_segments
    )

@torch.no_grad()
def render_mixed_video(
    actor: UnifiedHierarchyActor,
    catalog: ActionCatalog,
    seed: int,
    mp4_path: str | Path,
    gif_path: str | Path,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600
) -> dict:
    actor.eval()
    env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_episode_steps)
    obs, _ = env.reset(seed = seed)

    records, done, ep_ret, dec_idx = [], False, 0., 0

    while not done:
        u_t, hier_probs_t = actor.greedy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
        u = int(u_t.item())
        h, code, seq = catalog.horizons[u], catalog.local_codes[u], catalog.sequences[u]
        h_probs = hier_probs_t.squeeze(0).cpu().numpy()
        dec_idx += 1

        for step_i, prim_a in enumerate(seq):
            obs_before = obs.copy()
            obs, reward, terminated, truncated, _ = env.step(prim_a)
            ep_ret += float(reward)
            records.append((env.render(), obs_before, h, code, seq, step_i + 1, h_probs, ep_ret, dec_idx, step_i == 0))
            if terminated or truncated:
                done = True
                break

    env.close()

    total_frames = len(records)
    W, H, B = 608, 512, 4
    font = ImageFont.load_default()
    frames = []

    for f_idx, (rgb, obs_b, h, code, seq, step_in_h, h_probs, cum_ret, d_idx, is_start) in enumerate(records):
        col = HIER_COLORS_RGB.get(h, (120, 130, 150))
        canvas = Image.new('RGB', (W, H), (20, 24, 33))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([0, 0, W - 1, 400 + 2 * B - 1], fill = col)
        canvas.paste(Image.fromarray(rgb), (B, B))

        lx, ly = int(B + 300. + 300. * float(obs_b[0])), int(B + 282. - 200. * float(obs_b[1]))
        if B + 16 <= lx <= W - B - 16 and B + 16 <= ly <= 400 + B - 16:
            draw.ellipse([lx - 23, ly - 23, lx + 23, ly + 23], outline = col, width = 2)
            if is_start:
                draw.ellipse([lx - 27, ly - 27, lx + 27, ly + 27], outline = (255, 255, 255), width = 1)

        draw.rectangle([12, 12, 208, 36], fill = col, outline = (255, 255, 255), width = 1)
        draw.text((19, 18), f'ACTIVE dt={h} | commit {step_in_h}/{h}', fill = (255, 255, 255), font = font)

        y0 = 412
        draw.rectangle([12, y0, 124, y0 + 24], fill = col)
        draw.text((18, y0 + 6), f'dt = {h} (code #{code:02d})', fill = (255, 255, 255), font = font)
        draw.text((134, y0 + 6), 'seq:', fill = (190, 198, 212), font = font)

        for s_i, prim_a in enumerate(seq):
            cx = 164 + 21 * s_i
            active = (s_i + 1 == step_in_h)
            draw.rectangle([cx, y0 + 2, cx + 18, y0 + 22], fill = col if active else (42, 48, 62), outline = (255, 255, 255) if active else (88, 96, 115))
            draw.text((cx + 5, y0 + 6), PRIM_NAMES[prim_a], fill = (255, 255, 255), font = font)

        draw.text((342, y0 + 6), f'Return: {cum_ret:+6.1f}  |  Dec #{d_idx:02d}  |  t={f_idx + 1:03d}/{total_frames}', fill = (235, 240, 248), font = font)

        y1 = 444
        draw.text((12, y1 + 3), 'pi(dt|s):', fill = (180, 188, 204), font = font)
        for idx_h, h_opt in enumerate(catalog.hierarchies):
            bx = 74 + 132 * idx_h
            p_val = float(h_probs[idx_h])
            c_rgb = HIER_COLORS_RGB.get(h_opt, (120, 130, 150))
            draw.text((bx, y1 + 3), f'dt={h_opt}', fill = c_rgb, font = font)
            draw.rectangle([bx + 30, y1 + 4, bx + 102, y1 + 14], fill = (38, 43, 56), outline = (70, 78, 96))
            draw.rectangle([bx + 30, y1 + 4, bx + 30 + max(1, int(round(72 * p_val))), y1 + 14], fill = c_rgb)
            draw.text((bx + 106, y1 + 3), f'{int(round(100 * p_val)):2d}%', fill = (210, 216, 228), font = font)

        y2, tx0, tw = 472, 12, W - 24
        draw.rectangle([tx0 - 1, y2 - 1, tx0 + tw + 1, y2 + 24], outline = (95, 105, 125))
        for t_i, rec_i in enumerate(records):
            x0, x1 = tx0 + int(t_i * tw / total_frames), max(tx0 + int(t_i * tw / total_frames) + 1, tx0 + int((t_i + 1) * tw / total_frames))
            c_bar = HIER_COLORS_RGB.get(rec_i[2], (120, 130, 150))
            if t_i > f_idx:
                c_bar = tuple(int(0.35 * v) for v in c_bar)
            draw.rectangle([x0, y2, x1, y2 + 23], fill = c_bar)
            if rec_i[9] and t_i > 0:
                draw.line([(x0, y2), (x0, y2 + 7)], fill = (240, 244, 252))

        cur_x = tx0 + int((f_idx + 0.5) * tw / total_frames)
        draw.line([(cur_x, y2 - 3), (cur_x, y2 + 26)], fill = (255, 255, 255), width = 2)
        frames.append(np.asarray(canvas))

    mp4_path, gif_path = Path(mp4_path).resolve(), Path(gif_path).resolve()
    mp4_path.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(str(mp4_path), frames, fps = 50, macro_block_size = 1)
    imageio.mimsave(str(gif_path), frames[::2], fps = 25)

    dec_counts = {h: sum(1 for r in records if r[9] and r[2] == h) for h in catalog.hierarchies}
    print(f'[video] seed={seed} | return={ep_ret:+.1f} | decisions={dec_idx}/{total_frames} frames | {dec_counts} -> {mp4_path}', flush = True)
    return dict(seed = seed, episode_return = ep_ret, decisions = dec_idx, frames = total_frames, dec_counts = dec_counts, mp4_path = str(mp4_path))

def plot_results(histories: dict[str, list[dict]], evals: dict[str, dict], hierarchies: tuple[int, ...], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize = (17.5, 5.2), dpi = 150)
    colors = {'naive_flat': '#7f8c8d', 'smdp_pure': '#2b6cb0', 'smdp_diverse': '#d62828'}
    titles = {
        'naive_flat': 'Naive Flat 60-Way (gamma^1, lam^1)',
        'smdp_pure': 'SMDP-GAE (gamma^tau, lam_k)',
        'smdp_diverse': 'SMDP-GAE + Marginal Diversity'
    }

    for k, rows in histories.items():
        xs = [r['env_steps'] / 1e3 for r in rows]
        ys = [r['recent_return_32'] for r in rows]
        axes[0].plot(xs, ys, color = colors.get(k, '#333333'), linewidth = 2.2, label = f'{titles.get(k, k)} (eval {evals[k]["mean_return"]:+.1f})')
    axes[0].axhline(200., color = 'green', linestyle = '--', linewidth = 1.)
    axes[0].set(xlabel = 'primitive env steps (k)', ylabel = 'rolling 32-ep return', title = 'Unified Multi-Horizon Learning Curve')
    axes[0].legend(loc = 'lower right', fontsize = 8)
    axes[0].grid(alpha = 0.25)

    div_rows = histories['smdp_diverse']
    axes[1].stackplot(
        [r['env_steps'] / 1e3 for r in div_rows],
        *[[r['dec_pct'][h] for r in div_rows] for h in hierarchies],
        labels = [f'dt = {h}' for h in hierarchies],
        colors = [HIER_COLORS_HEX.get(h, '#555555') for h in hierarchies],
        alpha = 0.85
    )
    axes[1].set(ylim = (0, 100), xlabel = 'primitive env steps (k)', ylabel = 'decision share (%)', title = 'Hierarchy Mix Over Training')
    axes[1].legend(loc = 'upper right', fontsize = 8)

    segs = evals['smdp_diverse']['best_segments']
    lines = [[pts[i], pts[i + 1]] for _, pts in segs for i in range(len(pts) - 1)]
    cols = [HIER_COLORS_HEX.get(h, '#555555') for h, pts in segs for _ in range(len(pts) - 1)]
    if lines:
        axes[2].add_collection(LineCollection(lines, colors = cols, linewidths = 2.4))
        axes[2].scatter([pts[0][0] for _, pts in segs], [pts[0][1] for _, pts in segs], c = [HIER_COLORS_HEX.get(h, '#555555') for h, _ in segs], s = 18, zorder = 4)
    axes[2].axvline(-0.2, color = '#2ecc71', linestyle = '--', linewidth = 1.)
    axes[2].axvline(0.2, color = '#2ecc71', linestyle = '--', linewidth = 1.)
    axes[2].axhline(0., color = '#555555', linewidth = 1.)
    ev_d = evals['smdp_diverse']
    axes[2].legend(handles = [Patch(color = HIER_COLORS_HEX.get(h, '#555555'), label = f'dt={h}: {ev_d["dec_frac"][h]:.1f}% dec') for h in hierarchies], loc = 'upper right', fontsize = 8)
    axes[2].set(xlim = (-0.75, 0.75), ylim = (-0.08, 1.5), xlabel = 'x', ylabel = 'y', title = 'Mixed-Hierarchy Trajectory')
    axes[2].grid(alpha = 0.25)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches = 'tight')
    plt.close(fig)

# main

def main(
    env_name = 'LunarLander-v3',
    out_folder = './vq-hierarchical-recording',
    vq_ckpt_dir: str | None = None,
    macro_hierarchies: tuple[int, ...] = (2, 4, 8),
    base_updates = 90,
    base_rollout_steps = 1024,
    env_step_budget = 800_000,
    marginal_ent_coef = 0.03,
    trace_power = 1.,
    compare_ablations = False,
    num_videos = 3,
    seed = 42
):
    out_dir = Path(out_folder)
    out_dir.mkdir(parents = True, exist_ok = True)

    if exists(vq_ckpt_dir):
        ckpt_dir = Path(vq_ckpt_dir)
        codebooks = {1: np.arange(4, dtype = np.int64)[:, None]}
        for h in macro_hierarchies:
            tbl = torch.load(ckpt_dir / f'ppo_vq_h{h}.pt', map_location = 'cpu', weights_only = False)['decoded_table']
            codebooks[h] = (tbl.cpu().numpy() if torch.is_tensor(tbl) else np.asarray(tbl)).astype(np.int64)
    else:
        codebooks = discover_vq_codebooks(
            env_name = env_name,
            macro_hierarchies = macro_hierarchies,
            base_updates = base_updates,
            rollout_steps = base_rollout_steps,
            seed = seed
        )

    catalog = ActionCatalog(codebooks)
    print(f'\n[stage 2] unified SMDP training ({catalog.total_actions} actions across dt = {catalog.hierarchies})', flush = True)

    variants = []
    if compare_ablations:
        variants.extend([
            ('naive_flat', False, False, 0.),
            ('smdp_pure', True, True, 0.)
        ])
    variants.append(('smdp_diverse', True, True, marginal_ent_coef))

    histories, actors, evals = {}, {}, {}

    for name, factorized, use_smdp, marg_coef in variants:
        hist, actor, ev = train_unified_variant(
            name = name,
            catalog = catalog,
            factorized = factorized,
            use_smdp_gae = use_smdp,
            marginal_ent_coef = marg_coef,
            trace_power = trace_power,
            env_step_budget = env_step_budget,
            env_name = env_name,
            seed = seed
        )
        histories[name], actors[name], evals[name] = hist, actor, ev
        mix = ' '.join(f'dt{h}:{ev["dec_frac"][h]:.1f}%' for h in catalog.hierarchies)
        print(f'  [{name:12s} eval] {ev["mean_return"]:+.1f} +/- {ev["std_return"]:.1f} (best {ev["best_return"]:+.1f}) | {mix}', flush = True)

    with open(out_dir / 'unified_smdp_curves.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'update', 'env_steps', 'recent_return_32', 'update_return', *[f'dec_pct_h{h}' for h in catalog.hierarchies]])
        for rows in histories.values():
            for r in rows:
                writer.writerow([r['variant'], r['update'], r['env_steps'], r['recent_return_32'], r['update_return'], *[r['dec_pct'][h] for h in catalog.hierarchies]])

    videos = []
    for i, sd in enumerate(evals['smdp_diverse']['top_seeds'][:max(1, num_videos)]):
        stem = 'unified-smdp-mixed-best' if i == 0 else f'unified-smdp-mixed-rank{i + 1}-s{sd}'
        videos.append(render_mixed_video(
            actors['smdp_diverse'], catalog, sd,
            out_dir / f'{stem}.mp4', out_dir / f'{stem}.gif',
            env_name = env_name
        ))

    plot_results(histories, evals, catalog.hierarchies, out_dir / 'unified_smdp_results.png')

    torch.save(dict(
        codebooks = codebooks,
        evals = evals,
        videos = videos,
        actor = actors['smdp_diverse'].state_dict()
    ), out_dir / 'ppo_unified_smdp.pt')

if __name__ == '__main__':
    fire.Fire(main)
