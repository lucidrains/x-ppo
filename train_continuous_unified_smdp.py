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
#   "pillow",
#   "torch",
#   "torch-einops-utils>=0.2.5",
# ]
# ///
"""unified multi-horizon continuous semi-markov PPO across heterogeneous tanh hierarchies.

continuous counterpart to `train_unified_smdp.py` on continuous `LunarLander-v3`:
stage 1 trains a dt = 1 Beta policy while compressing contiguous h-step action chunks `(h, 2)`
into `tanh`-squashed continuous latents `z_h in (-1, 1)^{d_h}` with `d_h = (3, 4, 6)` for `h = (2, 4, 8)`.
stage 2 exposes all hierarchies `dt = (1, 2, 4, 8)` with differing dimensions `d_h = (2, 3, 4, 6)`
to a trans-dimensional factorized actor (discrete router `pi(h | s)` + masked `MeanConcBeta` heads),
trained with exact semi-markov GAE, decoupled router + `sqrt(d_h / 2)` latent HPO trust regions,
and `NextLatentQCritic` candidate action filtering:

    uv run train_continuous_unified_smdp.py
    uv run train_continuous_unified_smdp.py --compare_q_filter --tanh_ckpt_dir ./continuous-hierarchical-recording
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
from torch import Tensor, arange, cat, randint, randn_like, randperm, stack, tensor, zeros
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import Dropout, Linear, Module, ModuleDict, Parameter, RMSNorm, Sequential, SiLU, init
from torch.nn.utils import clip_grad_norm_

from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from mean_conc_beta import Beta as MeanConcBeta

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

HIER_ENTROPY_SCHEDULE = (0.005, 0.003, 0.0018, 0.001, 0.0005, 0.0002)
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

def resolve_latent_dim(hierarchy: int, num_actions: int, latent_dim: int | None = None) -> int:
    if exists(latent_dim):
        return int(latent_dim)
    if hierarchy <= 1:
        return num_actions
    target = max(num_actions + 1, num_actions * int(round(math.log2(hierarchy))))
    return min(hierarchy * num_actions - 1, target)

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
        self.net = Sequential(RMSNorm(dim), Linear(dim, dim_inner), SiLU(), Dropout(dropout), Linear(dim_inner, dim))
        self.norm = RMSNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class BetaActor(Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, beta_init_conc = 5.):
        super().__init__()
        dim = hidden_dim * 2
        self.net = Sequential(Linear(state_dim, dim), SimBa(dim), SimBa(dim), Linear(dim, hidden_dim), SiLU(), Linear(hidden_dim, 2 * action_dim))
        init.orthogonal_(self.net[-1].weight, gain = 0.01)
        init.zeros_(self.net[-1].bias)
        self.beta = MeanConcBeta(bounds = (-1., 1.), init_conc = beta_init_conc, unimodal = True, eps = 1e-5)

    def forward(self, state: Tensor) -> Tensor:
        return rearrange(self.net(state), '... (a two) -> ... a two', two = 2)

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        dist = self.beta(self.forward(state))
        action = dist.sample()
        return action, self.beta.log_prob(dist, action, sum_action_dim = True)

    def log_prob(self, params: Tensor, action: Tensor) -> Tensor:
        return self.beta.log_prob(params, action, sum_action_dim = True)

    def entropy(self, params: Tensor) -> Tensor:
        return self.beta.entropy(params, sum_action_dim = False).mean(dim = -1) * 2.

class Critic(Module):
    def __init__(self, state_dim: int, hidden_dim = 192):
        super().__init__()
        self.net = Sequential(
            Linear(state_dim, hidden_dim), RMSNorm(hidden_dim), SiLU(),
            Linear(hidden_dim, hidden_dim), RMSNorm(hidden_dim), SiLU(),
            Linear(hidden_dim, 1)
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).squeeze(-1)

# stage 1: tanh action chunk compressor & discovery

class ActionChunkTanh(Module):
    def __init__(self, hierarchy: int, num_actions: int, latent_dim: int | None = None, dim = 128, latent_noise = 0.02):
        super().__init__()
        resolved_dim = resolve_latent_dim(hierarchy, num_actions, latent_dim)
        self.hierarchy = hierarchy
        self.num_actions = num_actions
        self.chunk_dim = hierarchy * num_actions
        self.latent_dim = resolved_dim
        self.latent_noise = latent_noise

        self.encoder = Sequential(Linear(self.chunk_dim, dim), SimBa(dim), SimBa(dim), RMSNorm(dim), Linear(dim, resolved_dim))
        self.decoder = Sequential(
            Linear(resolved_dim, dim), SimBa(dim), SimBa(dim), RMSNorm(dim), Linear(dim, self.chunk_dim),
            Rearrange('... (h a) -> ... h a', h = hierarchy)
        )

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents).tanh()

    def forward(self, chunks: Tensor) -> Tensor:
        chunks = chunks.float()
        pre_tanh = self.encoder(rearrange(chunks, 'b h a -> b (h a)'))
        noisy = (pre_tanh + self.latent_noise * randn_like(pre_tanh)).tanh() if (self.training and self.latent_noise > 0.) else pre_tanh.tanh()
        return F.mse_loss(self.decode(noisy), chunks)

class ChunkBuffer:
    def __init__(self, capacity: int, hierarchy: int, num_actions: int, stride = 2):
        self.capacity = capacity
        self.hierarchy = hierarchy
        self.stride = max(1, stride)
        self.actions = zeros((capacity, hierarchy, num_actions), dtype = torch.float32)
        self.ptr = 0
        self.size = 0

    def add_rollout(self, actions: Tensor, dones: Tensor):
        h = self.hierarchy
        if actions.shape[1] < h:
            return
        d_win = dones.detach().cpu().unfold(1, h, self.stride)
        a_win = actions.detach().cpu().unfold(1, h, self.stride)
        valid = ~d_win[..., :-1].any(dim = -1) if h > 1 else torch.ones_like(d_win[..., 0], dtype = torch.bool)
        if not valid.any():
            return

        act_chunks = rearrange(a_win[valid], 'n a h -> n h a').float()[-self.capacity:]
        n = act_chunks.shape[0]
        idx = (self.ptr + arange(n)) % self.capacity
        self.actions[idx] = act_chunks
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def train_steps(self, model: ActionChunkTanh, opt: torch.optim.Optimizer, steps = 80, batch_size = 256) -> float:
        if self.size < 1:
            return 0.
        dev = next(model.parameters()).device
        bs = min(batch_size, self.size)
        model.train()
        mses = []
        with torch.random.fork_rng():
            for _ in range(steps):
                loss = model(self.actions[randint(0, self.size, (bs,))].to(dev))
                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                mses.append(loss.item())
        return float(np.mean(mses[-16:]))

def load_tanh_compressors(ckpt_dir: str | Path, macro_hierarchies: tuple[int, ...], num_actions: int):
    ckpt_path = Path(ckpt_dir)
    compressors = {}
    for h in macro_hierarchies:
        ck = torch.load(ckpt_path / f'ppo_tanh_h{h}.pt', map_location = device, weights_only = False)
        lat_dim = int(ck.get('latent_dim', resolve_latent_dim(h, num_actions)))
        dim = int(ck['tanh_compressor']['encoder.0.weight'].shape[0])
        comp = ActionChunkTanh(h, num_actions, latent_dim = lat_dim, dim = dim).to(device)
        comp.load_state_dict(ck['tanh_compressor'], strict = False)
        comp.eval()
        comp.requires_grad_(False)
        compressors[h] = comp
        print(f'[loaded tanh h={h}] {comp.chunk_dim}->{comp.latent_dim}D', flush = True)
    return compressors

def discover_tanh_compressors(
    env_name = 'LunarLander-v3',
    macro_hierarchies: tuple[int, ...] = (2, 4, 8),
    compressor_dim = 128,
    base_updates = 135,
    rollout_steps = 384,
    num_envs = 16,
    max_episode_steps = 600,
    seed = 42
) -> dict[int, ActionChunkTanh]:
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

        compressors = {h: ActionChunkTanh(h, num_actions, dim = compressor_dim).to(device) for h in macro_hierarchies}
        comp_opts = {h: torch.optim.AdamW(compressors[h].parameters(), lr = 2e-3) for h in macro_hierarchies}
        buffers = {h: ChunkBuffer(49152, h, num_actions) for h in macro_hierarchies}

        obs, _ = env.reset(seed = seed)
        state = torch.from_numpy(obs).float().to(device)
        running, recent, t0 = np.zeros(num_envs), [], time.time()
        print(f'[stage 1] discovering tanh bottlenecks for h = {macro_hierarchies}', flush = True)

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

            for h in macro_hierarchies:
                buffers[h].add_rollout(actions_t, term_t | trunc_t)

            with torch.no_grad():
                values = critic(states_t)
                gae = calc_gae(
                    rewards = rewards_t, values = values, next_values = critic(next_states_t),
                    terminated = term_t, truncated = trunc_t, gamma = 0.99, lam = 0.95,
                    use_accelerated = False, return_advantages = True
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
                        actor.log_prob(params, mb_a), rearrange(old_lp_t[mb], 'b t -> (b t)'),
                        rearrange(gae.advantages[mb], 'b t -> (b t)'), eps_clip = eff_eps
                    ).mean() - ent_coef * actor.entropy(params).mean()

                    opt_actor.zero_grad()
                    a_loss.backward()
                    clip_grad_norm_(actor.parameters(), 0.5)
                    opt_actor.step()

                    c_loss = clipped_value_loss_fn(
                        critic(mb_s), rearrange(values[mb], 'b t -> (b t)'),
                        rearrange(gae.returns[mb], 'b t -> (b t)'), clip = None
                    )
                    opt_critic.zero_grad()
                    c_loss.backward()
                    opt_critic.step()

            if divisible_by(upd, 5):
                for h in macro_hierarchies:
                    if buffers[h].size >= 256:
                        buffers[h].train_steps(compressors[h], comp_opts[h], steps = 80)

            if divisible_by(upd, 15) or upd == base_updates:
                rec32 = float(np.mean(recent[-32:])) if recent else float('nan')
                print(f'  [stage 1 upd {upd:3d}/{base_updates}] ret32 = {rec32:+6.1f} [{time.time() - t0:.1f}s]', flush = True)

        env.close()

        for h in macro_hierarchies:
            mse = buffers[h].train_steps(compressors[h], comp_opts[h], steps = 300, batch_size = 512)
            compressors[h].eval()
            compressors[h].requires_grad_(False)
            print(f'  [frozen tanh h={h}] {compressors[h].chunk_dim}->{compressors[h].latent_dim}D | mse={mse:.4f}', flush = True)

        return compressors

# trans-dimensional option catalog & macro environment

class TanhOptionCatalog(Module):
    def __init__(self, compressors: dict[int, ActionChunkTanh], num_actions = 2, include_primitive = True):
        super().__init__()
        macro_hs = sorted(int(h) for h in compressors.keys())
        self.hierarchies = ((1,) if include_primitive else ()) + tuple(macro_hs)
        self.num_hierarchies = len(self.hierarchies)
        self.num_actions = num_actions
        self.max_hierarchy = max(self.hierarchies)

        self.latent_dims = tuple(num_actions if h == 1 else int(compressors[h].latent_dim) for h in self.hierarchies)
        self.max_latent_dim = max(self.latent_dims)
        self.packed_dim = 1 + self.max_latent_dim

        self.compressors = ModuleDict({str(h): compressors[h] for h in macro_hs})
        for comp in self.compressors.values():
            comp.eval()
            comp.requires_grad_(False)

        self.horizons_np = np.asarray(self.hierarchies, dtype = np.float32)
        self.register_buffer('horizons_t', tensor(self.hierarchies, dtype = torch.long), persistent = False)
        self.register_buffer('valid_mask', arange(self.max_latent_dim)[None, :] < tensor(self.latent_dims)[:, None], persistent = False)

    @torch.no_grad()
    def decode_padded(self, h_idx: Tensor, z_pad: Tensor) -> Tensor:
        lead_shape = h_idx.shape
        flat_h, flat_z = h_idx.reshape(-1).long(), z_pad.reshape(-1, self.max_latent_dim)
        prim_padded = flat_z.new_zeros((flat_h.shape[0], self.max_hierarchy, self.num_actions))

        for idx, (h, d_h) in enumerate(zip(self.hierarchies, self.latent_dims)):
            mask = flat_h == idx
            if bool(mask.any()):
                z_sub = flat_z[mask, :d_h].clamp(-1., 1.)
                prim_padded[mask, :h] = z_sub[:, None, :] if h == 1 else self.compressors[str(h)].decode(z_sub)

        return prim_padded.reshape(*lead_shape, self.max_hierarchy, self.num_actions)

    @torch.no_grad()
    def to_env_commands(self, packed_actions: Tensor) -> np.ndarray:
        h_idx = packed_actions[..., 0].long()
        prim_padded = self.decode_padded(h_idx, packed_actions[..., 1:])
        cmd = cat((self.horizons_t[h_idx].float()[..., None], rearrange(prim_padded, '... h a -> ... (h a)')), dim = -1)
        return cmd.cpu().numpy().astype(np.float32)

class UnifiedContinuousHierarchyEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, max_hierarchy: int, num_actions: int, gamma = 0.99):
        super().__init__(env)
        self.max_hierarchy = max_hierarchy
        self.num_actions = num_actions
        self.gamma = gamma
        self.action_space = gym.spaces.Box(low = -1., high = float(max_hierarchy), shape = (1 + max_hierarchy * num_actions,), dtype = np.float32)

    def step(self, action):
        cmd = np.asarray(action, dtype = np.float32)
        h = max(1, min(self.max_hierarchy, int(round(float(cmd[0])))))
        prim_seq = cmd[1:].reshape(self.max_hierarchy, self.num_actions)

        disc_reward, raw_reward, duration, last_reward = 0., 0., 0, 0.
        for step_idx in range(h):
            obs, reward, terminated, truncated, info = self.env.step(np.clip(prim_seq[step_idx], -1., 1.))
            last_reward = float(reward)
            disc_reward += (self.gamma ** step_idx) * last_reward
            raw_reward += last_reward
            duration += 1
            if terminated or truncated:
                break

        return obs, disc_reward, terminated, truncated, {**info, 'raw_reward': raw_reward, 'duration': duration}

# factorized continuous hierarchy actor (discrete router + variable-dim Beta heads)

class FactorizedContinuousHierarchyActor(Module):
    def __init__(self, state_dim: int, latent_dims: tuple[int, ...], hidden_dim = 108, head_dim = 80, beta_init_conc = 5.):
        super().__init__()
        self.latent_dims = tuple(int(d) for d in latent_dims)
        self.num_hierarchies = len(self.latent_dims)
        self.max_latent_dim = max(self.latent_dims)
        self.packed_dim = 1 + self.max_latent_dim

        self.register_buffer('valid_mask', arange(self.max_latent_dim)[None, :] < tensor(self.latent_dims)[:, None], persistent = False)
        self.register_buffer('latent_dims_f', tensor(self.latent_dims, dtype = torch.float32), persistent = False)
        self.register_buffer('eps_scale', (tensor(self.latent_dims, dtype = torch.float32) / 2.).sqrt(), persistent = False)

        dim = hidden_dim * 2
        self.backbone = Sequential(Linear(state_dim, dim), SimBa(dim), SimBa(dim), Linear(dim, hidden_dim), SiLU())

        self.to_hier_logits = Linear(hidden_dim, self.num_hierarchies)
        init.zeros_(self.to_hier_logits.weight)
        init.zeros_(self.to_hier_logits.bias)

        self.head_w1 = Parameter(torch.empty(self.num_hierarchies, hidden_dim, head_dim))
        self.head_b1 = Parameter(zeros(self.num_hierarchies, head_dim))
        self.head_norm = RMSNorm(head_dim)
        self.head_w2 = Parameter(zeros(self.num_hierarchies, head_dim, 2 * self.max_latent_dim))
        self.head_b2 = Parameter(zeros(self.num_hierarchies, 2 * self.max_latent_dim))

        for i, d_i in enumerate(self.latent_dims):
            init.orthogonal_(self.head_w1[i], gain = math.sqrt(2.))
            init.orthogonal_(self.head_w2[i, :, :2 * d_i], gain = 0.01)

        self.beta = MeanConcBeta(bounds = (-1., 1.), init_conc = beta_init_conc, unimodal = True, eps = 1e-5)

    def forward(self, state: Tensor):
        feat = self.backbone(state)
        hier_log_probs = self.to_hier_logits(feat).log_softmax(dim = -1)
        hier_probs = hier_log_probs.exp()

        h_feat = F.silu(self.head_norm(einsum(feat, self.head_w1, '... d, h d m -> ... h m') + self.head_b1))
        all_params = rearrange(einsum(h_feat, self.head_w2, '... h m, h m p -> ... h p') + self.head_b2, '... h (a two) -> ... h a two', two = 2)

        per_dim_ent = self.beta.entropy(all_params, sum_action_dim = False)
        beta_ent_by_h = ((per_dim_ent * self.valid_mask).sum(dim = -1) / self.latent_dims_f) * 2.
        return hier_log_probs, hier_probs, all_params, beta_ent_by_h

    def sample_candidates(self, state: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
        b = state.shape[0]
        hier_log_probs, _, all_params, _ = self.forward(state)
        cand_h = rearrange(Categorical(logits = hier_log_probs, validate_args = False).sample((k,)), 'k b -> b k')
        cand_hier_lp = hier_log_probs.gather(-1, cand_h)

        cand_params = all_params[arange(b, device = state.device)[:, None], cand_h]
        cand_z_raw = self.beta(cand_params).sample()
        mask = self.valid_mask[cand_h]
        cand_lat_lp = (self.beta.log_prob(cand_params, cand_z_raw, sum_action_dim = False) * mask).sum(dim = -1)
        return cat((cand_h.float()[..., None], cand_z_raw * mask.float()), dim = -1), cand_hier_lp, cand_lat_lp

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        packed, hier_lp, lat_lp = self.sample_candidates(state, 1)
        return packed.squeeze(1), hier_lp.squeeze(1), lat_lp.squeeze(1)

    def evaluate_packed(self, state: Tensor, packed_actions: Tensor):
        hier_log_probs, hier_probs, all_params, beta_ent_by_h = self.forward(state)
        is_single = packed_actions.ndim == 2
        if is_single:
            packed_actions = packed_actions.unsqueeze(1)

        h_idx, z_pad = packed_actions[..., 0].long(), packed_actions[..., 1:]
        hier_lp = hier_log_probs.gather(-1, h_idx)
        chosen_params = all_params[arange(state.shape[0], device = state.device)[:, None], h_idx]
        mask = self.valid_mask[h_idx]
        lat_lp = (self.beta.log_prob(chosen_params, z_pad.masked_fill(~mask, 0.), sum_action_dim = False) * mask).sum(dim = -1)

        if is_single:
            hier_lp, lat_lp = hier_lp.squeeze(1), lat_lp.squeeze(1)

        # detach hier_probs so continuous differential entropy never biases router logits
        within_beta_ent = (hier_probs.detach() * beta_ent_by_h).sum(dim = -1)
        return hier_lp, lat_lp, hier_probs, within_beta_ent

    def greedy(self, state: Tensor) -> tuple[Tensor, Tensor]:
        _, hier_probs, all_params, _ = self.forward(state)
        best_h = hier_probs.argmax(dim = -1)
        chosen_params = all_params[arange(state.shape[0], device = state.device), best_h]
        z_mean = self.beta.mean(chosen_params) * self.valid_mask[best_h].float()
        return cat((best_h.float()[:, None], z_mean), dim = -1), hier_probs

    def eval_candidates(self, state: Tensor, rel_prob_min = 0.2):
        b, H = state.shape[0], self.num_hierarchies
        _, hier_probs, all_params, _ = self.forward(state)
        plausible = hier_probs >= (rel_prob_min * hier_probs.amax(dim = -1, keepdim = True))
        det_h = arange(H, device = state.device)[None, :].expand(b, H)
        det_z = self.beta.mean(all_params) * self.valid_mask[None, :, :].float()
        return cat((det_h.float()[..., None], det_z), dim = -1), plausible, hier_probs

# option embedding & semi-markov continuous ppo

class ContinuousOptionEmbed(Module):
    def __init__(self, catalog: TanhOptionCatalog, embed_dim = 64, gamma = 0.99):
        super().__init__()
        self.catalog = catalog
        self.num_hierarchies = catalog.num_hierarchies
        self.num_actions = catalog.num_actions

        gamma_pows = tensor([gamma ** j for j in range(catalog.max_hierarchy)], dtype = torch.float32)[:, None]
        disc_sums = tensor([sum(gamma ** j for j in range(h)) for h in catalog.hierarchies], dtype = torch.float32)
        hor_stats = stack([disc_sums / disc_sums.max().clamp_min(1e-6), tensor([gamma ** h for h in catalog.hierarchies], dtype = torch.float32)], dim = -1)

        self.register_buffer('gamma_pows', gamma_pows, persistent = False)
        self.register_buffer('disc_sums', disc_sums, persistent = False)
        self.register_buffer('hor_stats', hor_stats, persistent = False)

        phys_dim = catalog.max_hierarchy * self.num_actions + 2 * self.num_actions + self.num_hierarchies + 2
        self.phys_proj = Sequential(Linear(phys_dim, embed_dim), SiLU(), Linear(embed_dim, embed_dim), SiLU())
        self.latent_weight = Parameter(zeros(self.num_hierarchies, catalog.max_latent_dim, embed_dim))
        self.latent_bias = Parameter(zeros(self.num_hierarchies, embed_dim))

    def forward(self, packed_actions: Tensor) -> Tensor:
        h_idx, z_pad = packed_actions[..., 0].long(), packed_actions[..., 1:]
        with torch.no_grad():
            prim_seq = self.catalog.decode_padded(h_idx, z_pad)
            disc_seq = prim_seq * self.gamma_pows
            duty = disc_seq.sum(dim = -2) / self.disc_sums[h_idx][..., None]
            last_idx = (self.catalog.horizons_t[h_idx] - 1)[..., None, None].expand(*h_idx.shape, 1, self.num_actions)
            last_a = prim_seq.gather(-2, last_idx).squeeze(-2)
            phys_feats = cat((rearrange(disc_seq, '... h a -> ... (h a)'), duty, last_a, F.one_hot(h_idx, self.num_hierarchies).float(), self.hor_stats[h_idx]), dim = -1)

        z_masked = z_pad * self.catalog.valid_mask[h_idx].float()
        return self.phys_proj(phys_feats) + einsum(z_masked, self.latent_weight[h_idx], '... d, ... d e -> ... e') + self.latent_bias[h_idx]

@dataclass
class ContinuousSMDPRollout:
    states: Tensor
    next_states: Tensor
    packed_actions: Tensor
    old_hier_log_probs: Tensor
    old_lat_log_probs: Tensor
    rewards: Tensor
    durations: Tensor
    terminated: Tensor
    truncated: Tensor
    cand_packed: Tensor | None = None
    cand_old_hier_lp: Tensor | None = None
    cand_old_lat_lp: Tensor | None = None

class SemiMarkovContinuousPPO(Module):
    def __init__(
        self,
        state_dim: int,
        catalog: TanhOptionCatalog,
        *,
        hidden_dim = 108,
        critic_hidden_dim = 192,
        trace_power = 1.,
        marginal_ent_coef = 0.03,
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
        self.catalog = catalog
        self.actor = FactorizedContinuousHierarchyActor(state_dim, catalog.latent_dims, hidden_dim = hidden_dim)
        self.critic = Critic(state_dim, hidden_dim = critic_hidden_dim)

        self.trace_power = float(trace_power)
        self.marginal_ent_coef = float(marginal_ent_coef)
        self.num_candidates = max(1, int(num_candidates))
        self.q_lam = float(q_lam)
        self.q_rate_advantage = q_rate_advantage
        self.conservative_weight = float(conservative_weight)
        self.soft_exit_coef = float(soft_exit_coef)
        self.soft_exit_kl_budget = float(soft_exit_kl_budget)
        self.base_lr, self.gamma, self.lam, self.eps_clip = lr, gamma, lam, eps_clip
        self.epochs, self.minibatch_envs = epochs, minibatch_envs

        if use_q_filter:
            self.q_critic = NextLatentQCritic(
                self.critic,
                dim_embed = state_dim,
                action_embed = ContinuousOptionEmbed(catalog, embed_dim = q_action_embed_dim, gamma = gamma),
                action_embed_dim = q_action_embed_dim,
                action_horizons = catalog.horizons_np,
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

    def set_lr_scale(self, scale: float):
        for pg in self.opt_actor.param_groups:
            pg['lr'] = self.base_lr * scale

    @torch.no_grad()
    def sample_rollout_action(self, state: Tensor, filter_prob = 0.):
        if not exists(self.q_critic):
            packed, hier_lp, lat_lp = self.actor.sample(state)
            return packed, hier_lp, lat_lp, None, None, None

        cand_packed, cand_hier_lp, cand_lat_lp = self.actor.sample_candidates(state, self.num_candidates)
        base_packed, base_hier_lp, base_lat_lp = cand_packed[:, 0], cand_hier_lp[:, 0], cand_lat_lp[:, 0]
        if filter_prob <= 0. or self.num_candidates <= 1:
            return base_packed, base_hier_lp, base_lat_lp, cand_packed, cand_hier_lp, cand_lat_lp

        filtered = self.q_critic.filter_actions(
            state, cand_packed, rate_advantage = self.q_rate_advantage, conservative_weight = self.conservative_weight
        )
        b_idx = arange(state.shape[0], device = state.device)
        filt_hier_lp, filt_lat_lp = cand_hier_lp[b_idx, filtered.indices], cand_lat_lp[b_idx, filtered.indices]
        if filter_prob >= 1.:
            return filtered.actions, filt_hier_lp, filt_lat_lp, cand_packed, cand_hier_lp, cand_lat_lp

        use_f = torch.rand(state.shape[0], device = state.device) < filter_prob
        return (
            torch.where(use_f[:, None], filtered.actions, base_packed),
            torch.where(use_f, filt_hier_lp, base_hier_lp),
            torch.where(use_f, filt_lat_lp, base_lat_lp),
            cand_packed, cand_hier_lp, cand_lat_lp
        )

    @torch.no_grad()
    def eval_action(self, state: Tensor, use_q = True, rel_prob_min = 0.2) -> tuple[Tensor, Tensor]:
        if not (use_q and exists(self.q_critic) and self.num_candidates > 1):
            return self.actor.greedy(state)
        cand_packed, cand_mask, hier_probs = self.actor.eval_candidates(state, rel_prob_min = rel_prob_min)
        if int(cand_mask.sum(dim = -1).max().item()) <= 1:
            return self.actor.greedy(state)
        filtered = self.q_critic.filter_actions(
            state, cand_packed, rate_advantage = self.q_rate_advantage,
            conservative_weight = self.conservative_weight, candidate_mask = cand_mask
        )
        return filtered.actions, hier_probs

    def update(self, rollout: ContinuousSMDPRollout, *, hier_entropy_coef = 0.002, beta_entropy_coef = 0.0004, router_scale = 1., distill_active = True):
        num_envs = rollout.states.shape[0]

        with torch.no_grad():
            values, next_values = self.critic(rollout.states), self.critic(rollout.next_states)
            gae = calc_gae(
                rewards = rollout.rewards, values = values, next_values = next_values,
                terminated = rollout.terminated, truncated = rollout.truncated, durations = rollout.durations,
                trace_power = self.trace_power, gamma = self.gamma, lam = self.lam,
                use_accelerated = False, return_advantages = True
            )

            q_gae, cand_adv = None, None
            if exists(self.q_critic):
                q_gae = calc_gae(
                    rewards = rollout.rewards, values = values, next_values = next_values,
                    terminated = rollout.terminated, truncated = rollout.truncated, durations = rollout.durations,
                    trace_power = self.trace_power, gamma = self.gamma, lam = self.q_lam,
                    use_accelerated = False, return_advantages = True
                )
                if distill_active and self.soft_exit_coef > 0. and exists(rollout.cand_packed) and rollout.cand_packed.shape[-2] > 1:
                    scores = self.q_critic.smdp_advantage_scores(
                        rollout.states, rollout.cand_packed,
                        rate_advantage = self.q_rate_advantage, conservative_weight = self.conservative_weight
                    )
                    weights = soft_exit_weights(scores, kl_budget = self.soft_exit_kl_budget)
                    k_cands = rollout.cand_packed.shape[-2]
                    cand_adv = (weights - (1. / k_cands)) * k_cands

        self.train()
        self.catalog.eval()

        for _ in range(self.epochs):
            perm = randperm(num_envs, device = rollout.states.device)
            for start in range(0, num_envs, self.minibatch_envs):
                mb = perm[start:start + self.minibatch_envs]
                mb_s = rearrange(rollout.states[mb], 'b t d -> (b t) d')
                mb_act = rearrange(rollout.packed_actions[mb], 'b t p -> (b t) p')
                mb_old_hlp = rearrange(rollout.old_hier_log_probs[mb], 'b t -> (b t)')
                mb_old_llp = rearrange(rollout.old_lat_log_probs[mb], 'b t -> (b t)')
                mb_adv = rearrange(gae.advantages[mb], 'b t -> (b t)')
                mb_ret = rearrange(gae.returns[mb], 'b t -> (b t)')

                new_hlp, new_llp, hier_probs, within_beta_ent = self.actor.evaluate_packed(mb_s, mb_act)
                lat_eps = self.eps_clip * self.actor.eps_scale[mb_act[:, 0].long()]
                surr_loss = (
                    router_scale * hpo_actor_loss(new_hlp, mb_old_hlp, mb_adv, eps_clip = self.eps_clip, normalize_advantages = True).mean() +
                    hpo_actor_loss(new_llp, mb_old_llp, mb_adv, eps_clip = lat_eps, normalize_advantages = True).mean()
                )

                if exists(cand_adv):
                    mb_cand = rearrange(rollout.cand_packed[mb], 'b t k p -> (b t) k p')
                    mb_c_old_hlp = rearrange(rollout.cand_old_hier_lp[mb], 'b t k -> (b t) k')
                    mb_c_old_llp = rearrange(rollout.cand_old_lat_lp[mb], 'b t k -> (b t) k')
                    mb_c_adv = rearrange(cand_adv[mb], 'b t k -> (b t) k')

                    new_c_hlp, new_c_llp, _, _ = self.actor.evaluate_packed(mb_s, mb_cand)
                    c_lat_eps = self.eps_clip * self.actor.eps_scale[mb_cand[..., 0].long()]
                    exit_loss = (
                        router_scale * hpo_actor_loss(new_c_hlp, mb_c_old_hlp, mb_c_adv, eps_clip = self.eps_clip, normalize_advantages = False).mean() +
                        hpo_actor_loss(new_c_llp, mb_c_old_llp, mb_c_adv, eps_clip = c_lat_eps, normalize_advantages = False).mean()
                    )
                    surr_loss = surr_loss + self.soft_exit_coef * exit_loss

                ent_bonus = (
                    hier_entropy_coef * normalized_entropy(hier_probs).mean() +
                    beta_entropy_coef * within_beta_ent.mean() +
                    self.marginal_ent_coef * normalized_entropy(hier_probs.mean(dim = 0))
                )
                self.opt_actor.zero_grad()
                (surr_loss - ent_bonus).backward()
                clip_grad_norm_(self.actor.parameters(), 0.5)
                self.opt_actor.step()

                if exists(self.q_critic):
                    critic_loss = self.q_critic(
                        mb_s, mb_act,
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
                    critic_loss = clipped_value_loss_fn(self.critic(mb_s), rearrange(values[mb], 'b t -> (b t)'), mb_ret, clip = None)
                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    self.opt_critic.step()

# evaluation, video & plotting

@torch.no_grad()
def evaluate_policy(ppo: SemiMarkovContinuousPPO, env_name = 'LunarLander-v3', max_episode_steps = 600, num_episodes = 48, use_q = True, rel_prob_min = 0.2, seed = 1000) -> dict:
    ppo.eval()
    catalog = ppo.catalog
    env = gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True)

    episodes = []
    dec_by_h = {h: 0 for h in catalog.hierarchies}
    step_by_h = {h: 0 for h in catalog.hierarchies}
    q_overrides = 0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed = ep_seed)
        done, ep_ret, last_r = False, 0., 0.
        segments, dec_hs, latents_padded, latent_dims_used = [], [], [], []
        ep_dec_by_h = {h: 0 for h in catalog.hierarchies}

        while not done:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            packed, _ = ppo.eval_action(obs_t, use_q = use_q, rel_prob_min = rel_prob_min)
            if use_q and exists(ppo.q_critic):
                greedy_packed, _ = ppo.actor.greedy(obs_t)
                q_overrides += int(not torch.allclose(packed, greedy_packed, atol = 1e-4))

            h_idx = int(packed[0, 0].item())
            h, d_h = catalog.hierarchies[h_idx], catalog.latent_dims[h_idx]
            dec_by_h[h] += 1
            ep_dec_by_h[h] += 1
            dec_hs.append(h)
            latents_padded.append(packed[0, 1:].cpu().numpy().tolist())
            latent_dims_used.append(d_h)

            prim_seq = catalog.decode_padded(packed[:, 0].long(), packed[:, 1:])[0, :h].cpu().numpy()
            seg_pts = [(float(obs[0]), float(obs[1]))]
            for step_i in range(h):
                obs, reward, terminated, truncated, _ = env.step(np.clip(prim_seq[step_i], -1., 1.))
                seg_pts.append((float(obs[0]), float(obs[1])))
                last_r = float(reward)
                ep_ret += last_r
                step_by_h[h] += 1
                if terminated or truncated:
                    done = True
                    break
            segments.append((h, seg_pts))

        episodes.append(dict(
            seed = ep_seed, return_val = ep_ret, landed = 1. if last_r >= 50. else 0.,
            used_hier_count = sum(1 for c in ep_dec_by_h.values() if c > 0),
            segments = segments, dec_hs = dec_hs, latents_padded = latents_padded, latent_dims_used = latent_dims_used
        ))

    env.close()
    ranked_ret = sorted(episodes, key = lambda e: e['return_val'], reverse = True)
    ranked_mixed = sorted(episodes, key = lambda e: e['return_val'] + 6. * e['used_hier_count'], reverse = True)
    tot_dec, tot_step = max(1, sum(dec_by_h.values())), max(1, sum(step_by_h.values()))
    returns = [e['return_val'] for e in episodes]

    return dict(
        mean_return = float(np.mean(returns)),
        std_return = float(np.std(returns)),
        best_return = float(ranked_ret[0]['return_val']),
        land_rate = 100. * float(np.mean([e['landed'] for e in episodes])),
        q_override_pct = 100. * q_overrides / tot_dec if (use_q and exists(ppo.q_critic)) else 0.,
        dec_frac = {h: 100. * dec_by_h[h] / tot_dec for h in catalog.hierarchies},
        step_frac = {h: 100. * step_by_h[h] / tot_step for h in catalog.hierarchies},
        top_seeds = [e['seed'] for e in ranked_mixed[:5]],
        best_episode = ranked_mixed[0]
    )

@torch.no_grad()
def render_mixed_video(ppo: SemiMarkovContinuousPPO, seed: int, mp4_path: str | Path, gif_path: str | Path, env_name = 'LunarLander-v3', rel_prob_min = 0.2, max_episode_steps = 600) -> dict:
    ppo.eval()
    catalog = ppo.catalog
    env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_episode_steps, continuous = True)
    obs, _ = env.reset(seed = seed)
    font = ImageFont.load_default()
    frames, done, ep_ret, dec_idx = [], False, 0., 0
    dec_counts = {h: 0 for h in catalog.hierarchies}

    while not done:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        packed, hier_probs_t = ppo.eval_action(obs_t, use_q = True, rel_prob_min = rel_prob_min)
        h_idx = int(packed[0, 0].item())
        h, d_h = catalog.hierarchies[h_idx], catalog.latent_dims[h_idx]
        z_vec = packed[0, 1:1 + d_h].cpu().numpy()
        prim_seq = catalog.decode_padded(packed[:, 0].long(), packed[:, 1:])[0, :h].cpu().numpy()
        p_str = ' '.join(f'dt{ho}:{int(round(100 * p))}%' for ho, p in zip(catalog.hierarchies, hier_probs_t[0].cpu().tolist()))
        z_str = ','.join(f'{v:+.2f}' for v in z_vec[:4]) + ('..' if d_h > 4 else '')
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
            y0 = rgb.shape[0] + 12
            draw.text((10, y0), f'dt={h} ({d_h}D) {step_i + 1}/{h} | z=[{z_str}] | {p_str} | ret={ep_ret:+6.1f}', fill = (240, 244, 252), font = font)
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

def plot_results(histories: dict[str, list[dict]], evals: dict[str, dict], catalog: TanhOptionCatalog, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize = (14.5, 9.6), dpi = 150)
    colors = {'smdp_diverse': '#d62828', 'smdp_q_filter': '#0f766e'}
    ax_curve, ax_mix, ax_traj, ax_lat = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for k, rows in histories.items():
        ax_curve.plot([r['env_steps'] / 1e3 for r in rows], [r['recent_return_32'] for r in rows], color = colors.get(k, '#2b6cb0'), linewidth = 2.2, label = f'{k} (eval {evals[k]["mean_return"]:+.1f})')
    ax_curve.axhline(200., color = 'green', linestyle = '--', linewidth = 1.)
    ax_curve.set(xlabel = 'primitive env steps (k)', ylabel = 'rolling 32-ep return', title = 'Unified Continuous Multi-Horizon Learning Curve')
    ax_curve.legend(loc = 'lower right', fontsize = 8)
    ax_curve.grid(alpha = 0.25)

    primary_key = 'smdp_q_filter' if 'smdp_q_filter' in histories else next(iter(histories))
    div_rows = histories[primary_key]
    ax_mix.stackplot(
        [r['env_steps'] / 1e3 for r in div_rows],
        *[[r['dec_pct'][h] for r in div_rows] for h in catalog.hierarchies],
        labels = [f'dt = {h} ({d}D)' for h, d in zip(catalog.hierarchies, catalog.latent_dims)],
        colors = [HIER_COLORS_HEX.get(h, '#555555') for h in catalog.hierarchies], alpha = 0.85
    )
    ax_mix.set(ylim = (0, 100), xlabel = 'primitive env steps (k)', ylabel = 'decision share (%)', title = f'Continuous Hierarchy Mix ({primary_key})')
    ax_mix.legend(loc = 'upper right', fontsize = 8)

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
    ax_traj.legend(handles = [Patch(color = HIER_COLORS_HEX.get(h, '#555555'), label = f'dt={h} ({d}D): {ev_d["dec_frac"][h]:.1f}% dec') for h, d in zip(catalog.hierarchies, catalog.latent_dims)], loc = 'upper right', fontsize = 8)
    ax_traj.set(xlim = (-0.75, 0.75), ylim = (-0.08, 1.5), xlabel = 'x', ylabel = 'y', title = f'Mixed-Hierarchy Continuous Trajectory ({primary_key})')
    ax_traj.grid(alpha = 0.25)

    lat_arr, dims_used = np.asarray(ep['latents_padded'], dtype = np.float32), np.asarray(ep['latent_dims_used'], dtype = np.int64)
    t_steps = np.arange(lat_arr.shape[0])
    for t_i, h_chosen in enumerate(ep['dec_hs']):
        ax_lat.axvspan(t_i - 0.5, t_i + 0.5, color = HIER_COLORS_HEX.get(h_chosen, '#cccccc'), alpha = 0.16, lw = 0)
    for d_i in range(lat_arr.shape[1]):
        ax_lat.plot(t_steps, np.where(d_i < dims_used, lat_arr[:, d_i], np.nan), linewidth = 1.4, marker = '.', markersize = 3., label = f'z[{d_i}]')
    ax_lat.axhline(0., color = 'gray', linestyle = ':', linewidth = 0.8)
    ax_lat.set(xlim = (-0.5, max(1., lat_arr.shape[0] - 0.5)), ylim = (-1.05, 1.05), xlabel = 'decision index (shaded band = active dt)', ylabel = 'action / tanh latent in (-1, 1)', title = f'Trans-Dimensional Latent Trace ({lat_arr.shape[0]} decisions, d_h in {catalog.latent_dims})')
    ax_lat.legend(loc = 'lower right', fontsize = 7.5, ncol = 2)
    ax_lat.grid(alpha = 0.25)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches = 'tight')
    plt.close(fig)

# stage 2 training loop & main

def train_unified_variant(
    *,
    name: str,
    catalog: TanhOptionCatalog,
    marginal_ent_coef = 0.03,
    trace_power = 1.,
    use_q_filter = True,
    num_candidates = 8,
    q_lam = 0.8,
    q_filter_prob = 0.5,
    q_warmup_frac = 0.08,
    router_warmup_frac = 0.08,
    q_rate_advantage = False,
    conservative_weight = 0.25,
    soft_exit_coef = 0.12,
    soft_exit_kl_budget = 0.25,
    eval_rel_prob_min = 0.2,
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
            [lambda: UnifiedContinuousHierarchyEnv(gym.make(env_name, max_episode_steps = max_episode_steps, continuous = True), catalog.max_hierarchy, catalog.num_actions) for _ in range(num_envs)],
            autoreset_mode = AutoresetMode.SAME_STEP
        )
        state_dim = int(env.single_observation_space.shape[0])
        ppo = SemiMarkovContinuousPPO(
            state_dim, catalog, trace_power = trace_power, marginal_ent_coef = marginal_ent_coef,
            use_q_filter = use_q_filter, num_candidates = num_candidates, q_lam = q_lam,
            q_rate_advantage = q_rate_advantage, conservative_weight = conservative_weight,
            soft_exit_coef = soft_exit_coef, soft_exit_kl_budget = soft_exit_kl_budget
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
            router_scale = 1. if router_warmup_frac <= 0. else min(1., 0.25 + 0.75 * (progress / router_warmup_frac))

            states, next_states, packed_acts, old_hlps, old_llps = [], [], [], [], []
            cand_acts, cand_hlps, cand_llps = [], [], []
            rewards, durations, terms, truncs, ep_rets = [], [], [], [], []
            h_counts = {h: 0 for h in catalog.hierarchies}

            ppo.eval()
            for _ in range(rollout_decisions):
                with torch.no_grad():
                    packed_t, hlp_t, llp_t, cand_p_t, cand_hlp_t, cand_llp_t = ppo.sample_rollout_action(state, filter_prob = active_filter_prob)
                    cmds_np = catalog.to_env_commands(packed_t)

                next_obs, disc_r, terminated, truncated, info = env.step(cmds_np)
                done = terminated | truncated
                raw_r = step_info_array(info, 'raw_reward', done, 0.)
                durs = step_info_array(info, 'duration', done, 1.)

                for h_i in packed_t[:, 0].long().cpu().tolist():
                    h_counts[catalog.hierarchies[h_i]] += 1

                running += raw_r
                env_steps += int(durs.sum())
                for idx in np.flatnonzero(done):
                    ep_rets.append(float(running[idx]))
                    recent.append(float(running[idx]))
                    running[idx] = 0.

                states.append(state)
                next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                packed_acts.append(packed_t)
                old_hlps.append(hlp_t)
                old_llps.append(llp_t)
                if exists(cand_p_t):
                    cand_acts.append(cand_p_t)
                    cand_hlps.append(cand_hlp_t)
                    cand_llps.append(cand_llp_t)
                rewards.append(torch.from_numpy(disc_r / 100.).float().to(device))
                durations.append(torch.from_numpy(durs).float().to(device))
                terms.append(torch.from_numpy(terminated).bool().to(device))
                truncs.append(torch.from_numpy(truncated).bool().to(device))
                state = torch.from_numpy(next_obs).float().to(device)

            ppo.update(
                ContinuousSMDPRollout(
                    states = stack(states, dim = 1), next_states = stack(next_states, dim = 1),
                    packed_actions = stack(packed_acts, dim = 1), old_hier_log_probs = stack(old_hlps, dim = 1),
                    old_lat_log_probs = stack(old_llps, dim = 1), rewards = stack(rewards, dim = 1),
                    durations = stack(durations, dim = 1), terminated = stack(terms, dim = 1),
                    truncated = stack(truncs, dim = 1),
                    cand_packed = stack(cand_acts, dim = 1) if cand_acts else None,
                    cand_old_hier_lp = stack(cand_hlps, dim = 1) if cand_hlps else None,
                    cand_old_lat_lp = stack(cand_llps, dim = 1) if cand_llps else None
                ),
                hier_entropy_coef = schedule_at(HIER_ENTROPY_SCHEDULE, progress),
                beta_entropy_coef = schedule_at(BETA_ENTROPY_SCHEDULE, progress),
                router_scale = router_scale,
                distill_active = warmed_up
            )

            rec32 = float(np.mean(recent[-32:])) if recent else float('nan')
            upd_ret = float(np.mean(ep_rets)) if ep_rets else rec32

            if divisible_by(upd, val_every) or env_steps >= env_step_budget:
                val_ret = evaluate_policy(
                    ppo, env_name = env_name, max_episode_steps = max_episode_steps,
                    num_episodes = 12, use_q = warmed_up, rel_prob_min = eval_rel_prob_min, seed = seed + 5000
                )['mean_return']
                if val_ret >= best_val:
                    best_val = val_ret
                    best_actor_state = deepcopy(ppo.actor.state_dict())
                    if exists(ppo.q_critic):
                        best_q_state = deepcopy(ppo.q_critic.state_dict())

            tot_dec = max(1, sum(h_counts.values()))
            dec_pct = {h: 100. * h_counts[h] / tot_dec for h in catalog.hierarchies}
            history.append(dict(variant = name, seed = seed, update = upd, env_steps = env_steps, recent_return_32 = rec32, update_return = upd_ret, dec_pct = dec_pct))

            if divisible_by(upd, 15) or env_steps >= env_step_budget:
                mix_str = ' '.join(f'dt{h}({d}D):{dec_pct[h]:4.1f}%' for h, d in zip(catalog.hierarchies, catalog.latent_dims))
                print(f'  [{name:14s} upd {upd:3d}] steps = {env_steps // 1000:4d}k | ret32 = {rec32:+6.1f} (val = {best_val:+6.1f}) | {mix_str} [{time.time() - t0:.1f}s]', flush = True)

        env.close()

    ppo.actor.load_state_dict(best_actor_state)
    if exists(ppo.q_critic) and exists(best_q_state):
        ppo.q_critic.load_state_dict(best_q_state)

    eval_stats = evaluate_policy(ppo, env_name = env_name, max_episode_steps = max_episode_steps, use_q = True, rel_prob_min = eval_rel_prob_min)
    if exists(ppo.q_critic):
        eval_stats['actor_only_mean_return'] = evaluate_policy(ppo, env_name = env_name, max_episode_steps = max_episode_steps, use_q = False)['mean_return']

    return history, ppo, eval_stats

def main(
    env_name = 'LunarLander-v3',
    out_folder = './continuous-unified-smdp',
    tanh_ckpt_dir: str | None = None,
    macro_hierarchies: tuple[int, ...] = (2, 4, 8),
    base_updates = 135,
    base_rollout_steps = 384,
    env_step_budget = 1_100_000,
    marginal_ent_coef = 0.03,
    trace_power = 1.,
    use_q_filter = True,
    compare_q_filter = False,
    num_candidates = 8,
    q_lam = 0.8,
    q_filter_prob = 0.5,
    q_warmup_frac = 0.08,
    router_warmup_frac = 0.08,
    q_rate_advantage = False,
    conservative_weight = 0.25,
    soft_exit_coef = 0.12,
    soft_exit_kl_budget = 0.25,
    eval_rel_prob_min = 0.2,
    num_videos = 2,
    seed = 42
):
    out_dir = Path(out_folder)
    out_dir.mkdir(parents = True, exist_ok = True)

    tmp_env = gym.make(env_name, continuous = True)
    num_actions = int(tmp_env.action_space.shape[0])
    tmp_env.close()

    macro_hs = tuple(int(h) for h in macro_hierarchies)
    if exists(tanh_ckpt_dir):
        compressors = load_tanh_compressors(tanh_ckpt_dir, macro_hs, num_actions)
    else:
        compressors = discover_tanh_compressors(
            env_name = env_name, macro_hierarchies = macro_hs,
            base_updates = base_updates, rollout_steps = base_rollout_steps, seed = seed
        )

    catalog = TanhOptionCatalog(compressors, num_actions = num_actions, include_primitive = True).to(device)
    dims_desc = ', '.join(f'dt={h} -> {d}D' for h, d in zip(catalog.hierarchies, catalog.latent_dims))
    print(f'\n[stage 2] unified continuous SMDP training ({dims_desc})', flush = True)

    variants = [('smdp_diverse', False), ('smdp_q_filter', True)] if compare_q_filter else [('smdp_q_filter' if use_q_filter else 'smdp_diverse', use_q_filter)]
    histories, agents, evals = {}, {}, {}

    for name, q_filt in variants:
        hist, agent, ev = train_unified_variant(
            name = name, catalog = catalog, marginal_ent_coef = marginal_ent_coef,
            trace_power = trace_power, use_q_filter = q_filt, num_candidates = num_candidates,
            q_lam = q_lam, q_filter_prob = q_filter_prob, q_warmup_frac = q_warmup_frac,
            router_warmup_frac = router_warmup_frac, q_rate_advantage = q_rate_advantage,
            conservative_weight = conservative_weight, soft_exit_coef = soft_exit_coef,
            soft_exit_kl_budget = soft_exit_kl_budget, eval_rel_prob_min = eval_rel_prob_min,
            env_step_budget = env_step_budget, env_name = env_name, seed = seed
        )
        histories[name], agents[name], evals[name] = hist, agent, ev
        mix = ' '.join(f'dt{h}({d}D):{ev["dec_frac"][h]:.1f}%' for h, d in zip(catalog.hierarchies, catalog.latent_dims))
        extra = f' | actor_only {ev["actor_only_mean_return"]:+.1f} | q_override {ev["q_override_pct"]:.1f}%' if q_filt else ''
        print(f'  [{name:14s} eval] {ev["mean_return"]:+.1f} +/- {ev["std_return"]:.1f} (land {ev["land_rate"]:.1f}%, best {ev["best_return"]:+.1f}) | {mix}{extra}', flush = True)

    with open(out_dir / 'continuous_unified_smdp_curves.csv', 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['variant', 'seed', 'update', 'env_steps', 'recent_return_32', 'update_return', *[f'dec_pct_h{h}' for h in catalog.hierarchies]])
        for rows in histories.values():
            for r in rows:
                writer.writerow([r['variant'], r['seed'], r['update'], r['env_steps'], r['recent_return_32'], r['update_return'], *[r['dec_pct'][h] for h in catalog.hierarchies]])

    primary_key = 'smdp_q_filter' if 'smdp_q_filter' in agents else next(reversed(agents))
    primary_agent = agents[primary_key]
    videos = []
    if num_videos > 0:
        for i, sd in enumerate(evals[primary_key]['top_seeds'][:num_videos]):
            prefix = 'continuous-unified-smdp-q-filter' if primary_key == 'smdp_q_filter' else 'continuous-unified-smdp-mixed'
            stem = f'{prefix}-best' if i == 0 else f'{prefix}-rank{i + 1}-s{sd}'
            videos.append(render_mixed_video(
                primary_agent, sd, out_dir / f'{stem}.mp4', out_dir / f'{stem}.gif',
                env_name = env_name, rel_prob_min = eval_rel_prob_min
            ))

    plot_results(histories, evals, catalog, out_dir / 'continuous_unified_smdp_results.png')

    torch.save(dict(
        hierarchies = catalog.hierarchies,
        latent_dims = catalog.latent_dims,
        compressors = {h: catalog.compressors[str(h)].state_dict() for h in catalog.hierarchies if h > 1},
        evals = evals,
        histories = histories,
        videos = videos,
        actor = primary_agent.actor.state_dict(),
        critic = primary_agent.critic.state_dict(),
        q_critic = primary_agent.q_critic.state_dict() if exists(primary_agent.q_critic) else None
    ), out_dir / 'ppo_continuous_unified_smdp.pt')

if __name__ == '__main__':
    fire.Fire(main)
