# /// script
# dependencies = [
#   "assoc-scan",
#   "einops",
#   "einx",
#   "ema-pytorch",
#   "fire",
#   "gymnasium[box2d,other]",
#   "matplotlib",
#   "numpy",
#   "torch",
#   "torch-einops-utils>=0.2.5",
#   "vector-quantize-pytorch>=1.34.0",
# ]
# ///

"""
Per-Hierarchy World Models & Cross-Hierarchy State Embeddings.

Stage 1 discovers VQ macro-codebooks for h in (2, 4, 8) and trains an action-conditioned
forward dynamics model (FDM) at each hierarchy h in (1, 2, 4, 8) predicting the detached
EMA next encoded state (with SIGReg anti-collapse regularization).
Stage 2 freezes all world model encoders and trains newly initialized macro-actors from
scratch while widening only `proj_in` to accept same- or higher-hierarchy WM state embeds.

uv run train_vq_world_model.py
"""

from __future__ import annotations

import csv
import time
from copy import deepcopy
from pathlib import Path

import fire
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, Tensor, cat, stack, zeros
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.distributions import Categorical

from einops import rearrange
from einops.layers.torch import Rearrange
from ema_pytorch import EMA

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AutoresetMode

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

def sigreg_loss(x: Tensor, num_slices = 64, domain = (-5., 5.), num_knots = 17) -> Tensor:
    dim, dev = x.shape[-1], x.device
    projs = F.normalize(torch.randn((num_slices, dim), device = dev), dim = -1)
    t = torch.linspace(*domain, num_knots, device = dev)
    exp_f = (-0.5 * t.square()).exp()

    x_t = rearrange(torch.einsum('... d, m d -> ... m', x, projs), '... m -> (...) m 1') * t
    err = ((x_t.cos().mean(dim = 0) - exp_f).square() + x_t.sin().mean(dim = 0).square()) * exp_f
    return torch.trapezoid(err, t, dim = -1).mean()

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

    def forward(self, state: Tensor):
        return self.net(state).squeeze(-1)

class WMAugmentedActor(Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim = 64,
        wm_encoders: list[Module] | None = None,
        wm_dim = 16,
        dropout = 0.
    ):
        super().__init__()
        dim = hidden_dim * 2
        encoders = wm_encoders or []

        self.wm_encoders = ModuleList([deepcopy(enc) for enc in encoders])
        for enc in self.wm_encoders:
            enc.eval()
            enc.requires_grad_(False)

        self.in_dim = state_dim + len(self.wm_encoders) * wm_dim

        # isolate widened proj_in RNG so trunk, critic, and rollouts share identical initialization
        self.proj_in = nn.Linear(state_dim, dim)
        if len(self.wm_encoders) > 0:
            with torch.random.fork_rng():
                self.proj_in = nn.Linear(self.in_dim, dim)

        self.trunk = nn.Sequential(
            SimBa(dim, dropout = dropout),
            SimBa(dim, dropout = dropout),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    @torch.no_grad()
    def augment_state(self, state: Tensor) -> Tensor:
        if len(self.wm_encoders) == 0 or state.shape[-1] == self.in_dim:
            return state
        return cat((state, *(enc(state) for enc in self.wm_encoders)), dim = -1)

    def forward(self, state: Tensor):
        return self.trunk(self.proj_in(self.augment_state(state)))

    def dist(self, state: Tensor):
        return Categorical(logits = self.forward(state), validate_args = False)

    def sample(self, state: Tensor):
        dist = self.dist(state)
        action = dist.sample()
        return action, dist.log_prob(action)

    def greedy(self, state: Tensor):
        return self.forward(state).argmax(dim = -1)

# vector quantized action chunks & slower macro env

class ActionChunkVQ(Module):
    def __init__(self, hierarchy: int, num_actions: int, dim = 128, codebook_dim = 16, decay = 0.9):
        super().__init__()
        num_codes = num_actions * hierarchy
        chunk_dim = hierarchy * num_actions

        self.hierarchy = hierarchy
        self.num_actions = num_actions
        self.num_codes = num_codes

        self.encoder = nn.Sequential(nn.Linear(chunk_dim, dim), SimBa(dim), nn.Linear(dim, dim))

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
        flat = rearrange(F.one_hot(chunks.long(), self.num_actions).float(), 'b h a -> b (h a)')
        quantized, _, vq_loss = self.vq(self.encoder(flat))
        logits = self.decoder(quantized)
        recon = F.cross_entropy(rearrange(logits, 'b h a -> (b h) a'), rearrange(chunks.long(), 'b h -> (b h)'))
        seq_acc = (logits.argmax(dim = -1) == chunks).all(dim = -1).float().mean()
        return recon + vq_loss, seq_acc

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
        self.ptr = 0
        self.size = 0

    def add_rollout(self, actions: Tensor, dones: Tensor):
        h = self.hierarchy
        if actions.shape[1] < h:
            return
        a_win = actions.detach().cpu().unfold(1, h, self.stride)
        d_win = dones.detach().cpu().unfold(1, h, self.stride)
        valid = ~d_win[..., :-1].any(dim = -1) if h > 1 else torch.ones_like(d_win[..., 0], dtype = torch.bool)
        chunks = a_win[valid].long()[-self.capacity:]
        n = chunks.shape[0]
        if n == 0:
            return
        idx = (self.ptr + torch.arange(n)) % self.capacity
        self.chunks[idx] = chunks
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def train_steps(self, model: ActionChunkVQ, opt: torch.optim.Optimizer, steps = 80, batch_size = 256):
        if self.size < batch_size:
            return 0.
        dev = next(model.parameters()).device
        model.train()
        with torch.random.fork_rng():
            if not bool(model.vq._codebook.initted.item()):
                init_n = min(self.size, max(1024, model.num_codes * 16))
                with torch.no_grad():
                    model(self.chunks[torch.randint(0, self.size, (init_n,))].to(dev))
            accs = []
            for _ in range(steps):
                loss, acc = model(self.chunks[torch.randint(0, self.size, (batch_size,))].to(dev))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                accs.append(float(acc))
        return float(np.mean(accs[-16:]))

class VQMacroEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, decoded_table: np.ndarray, hierarchy: int, gamma = 0.99):
        super().__init__(env)
        self.decoded_table = decoded_table
        self.hierarchy = hierarchy
        self.gamma = gamma
        self.action_space = gym.spaces.Discrete(int(decoded_table.shape[0]))

    def step(self, code_action):
        prim_seq = self.decoded_table[int(code_action)]
        disc_reward, raw_reward, prim_steps = 0., 0., 0
        for h in range(self.hierarchy):
            obs, reward, terminated, truncated, info = self.env.step(int(prim_seq[h]))
            r = float(reward)
            disc_reward += (self.gamma ** h) * r
            raw_reward += r
            prim_steps += 1
            if terminated or truncated:
                break
        return obs, disc_reward, terminated, truncated, {**info, 'raw_reward': raw_reward, 'prim_steps': prim_steps}

# per-hierarchy world model (action-conditioned fdm predicting detached ema state embed)

class HierarchyWorldModel(Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        wm_dim = 16,
        hidden_dim = 128,
        action_embed_dim = 16,
        ema_beta = 0.99,
        sigreg_weight = 0.15
    ):
        super().__init__()
        self.wm_dim = wm_dim
        self.hidden_dim = hidden_dim
        self.sigreg_weight = sigreg_weight

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            SimBa(hidden_dim),
            nn.Linear(hidden_dim, wm_dim)
        )

        self.ema_state_encoder = EMA(
            self.state_encoder,
            beta = ema_beta,
            include_online_model = False,
            update_after_step = 0,
            update_every = 1
        )

        self.action_embed = nn.Embedding(num_actions, action_embed_dim)

        self.fdm = nn.Sequential(
            nn.Linear(wm_dim + action_embed_dim, hidden_dim),
            SimBa(hidden_dim),
            nn.Linear(hidden_dim, wm_dim)
        )
        nn.init.zeros_(self.fdm[-1].weight)
        nn.init.zeros_(self.fdm[-1].bias)

    def forward(self, state: Tensor, action: Tensor, next_state: Tensor):
        z = self.state_encoder(state)
        z_next_pred = z + self.fdm(cat((z, self.action_embed(action.long())), dim = -1))

        with torch.no_grad():
            z_next_target = self.ema_state_encoder(next_state).detach()
            id_loss = F.smooth_l1_loss(z.detach(), z_next_target)

        fdm_loss = F.smooth_l1_loss(z_next_pred, z_next_target)
        loss = fdm_loss + self.sigreg_weight * sigreg_loss(z)
        return loss, fdm_loss.item(), id_loss.item()

class TransitionBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = zeros((capacity, state_dim), dtype = torch.float32)
        self.next_states = zeros((capacity, state_dim), dtype = torch.float32)
        self.actions = zeros((capacity,), dtype = torch.long)
        self.ptr = 0
        self.size = 0

    def add_rollout(self, states: Tensor, actions: Tensor, next_states: Tensor, valid_mask: Tensor | None = None):
        flat_s = rearrange(states.detach().cpu(), 'b t d -> (b t) d')
        flat_sn = rearrange(next_states.detach().cpu(), 'b t d -> (b t) d')
        flat_a = rearrange(actions.detach().cpu().long(), 'b t -> (b t)')

        if exists(valid_mask):
            v = rearrange(valid_mask.detach().cpu().bool(), 'b t -> (b t)')
            flat_s, flat_a, flat_sn = flat_s[v], flat_a[v], flat_sn[v]

        n = min(self.capacity, flat_s.shape[0])
        if n == 0:
            return
        idx = (self.ptr + torch.arange(n)) % self.capacity
        self.states[idx], self.actions[idx], self.next_states[idx] = flat_s[-n:], flat_a[-n:], flat_sn[-n:]
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def train_steps(self, wm: HierarchyWorldModel, opt: torch.optim.Optimizer, steps = 64, batch_size = 256):
        if self.size < batch_size:
            return 0., 0.
        dev = next(wm.parameters()).device
        wm.train()
        fdms, ids = [], []
        with torch.random.fork_rng():
            for _ in range(steps):
                idx = torch.randint(0, self.size, (batch_size,))
                loss, fdm_l, id_l = wm(self.states[idx].to(dev), self.actions[idx].to(dev), self.next_states[idx].to(dev))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(wm.parameters(), 1.)
                opt.step()
                wm.ema_state_encoder.update()
                fdms.append(fdm_l)
                ids.append(id_l)
        return float(np.mean(fdms[-16:])), float(np.mean(ids[-16:]))

# ppo update & evaluation

def ppo_update(
    actor: WMAugmentedActor,
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
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.2,
    entropy_coef = 0.005,
    epochs = 6,
    minibatch_envs = 4,
    max_grad_norm = 0.5
):
    num_envs = states.shape[0]
    aug_states = actor.augment_state(states)

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

            mb_aug = rearrange(aug_states[mb], 'b t d -> (b t) d')
            mb_raw = rearrange(states[mb], 'b t d -> (b t) d')
            mb_a = rearrange(actions[mb], 'b t -> (b t)')
            mb_old_lp = rearrange(old_log_probs[mb], 'b t -> (b t)')
            mb_adv = rearrange(gae.advantages[mb], 'b t -> (b t)')
            mb_ret = rearrange(gae.returns[mb], 'b t -> (b t)')
            mb_val = rearrange(values[mb], 'b t -> (b t)')

            dist = actor.dist(mb_aug)
            actor_loss = hpo_actor_loss(dist.log_prob(mb_a), mb_old_lp, mb_adv, eps_clip = eps_clip, normalize_advantages = True).mean()
            actor_loss = actor_loss - entropy_coef * dist.entropy().mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt_actor.step()

            critic_loss = clipped_value_loss_fn(critic(mb_raw), mb_val, mb_ret, clip = None)
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

@torch.no_grad()
def evaluate_macro_actor(actor: WMAugmentedActor, table: np.ndarray, hierarchy: int, env_name = 'LunarLander-v3', max_episode_steps = 600, num_episodes = 32, seed = 1000) -> dict:
    actor.eval()
    env = gym.make(env_name, max_episode_steps = max_episode_steps)
    returns, landed = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed = seed + ep)
        done, ep_ret, last_r = False, 0., 0.
        while not done:
            code = int(actor.greedy(torch.from_numpy(obs).float().unsqueeze(0).to(device)).item())
            for prim_a in table[code]:
                obs, reward, terminated, truncated, _ = env.step(int(prim_a))
                last_r = float(reward)
                ep_ret += last_r
                if terminated or truncated:
                    done = True
                    break
        returns.append(ep_ret)
        landed.append(1. if last_r >= 50. else 0.)

    env.close()
    return dict(mean_return = float(np.mean(returns)), std_return = float(np.std(returns)), best_return = float(max(returns)), land_rate = 100. * float(np.mean(landed)))

# stage 1: self-contained vq codebook & per-hierarchy world model training

def train_vq_and_world_models(
    env_name = 'LunarLander-v3',
    macro_hierarchies: tuple[int, ...] = (2, 4, 8),
    wm_dim = 16,
    base_updates = 60,
    macro_wm_updates = 40,
    num_envs = 16,
    rollout_steps = 512,
    max_episode_steps = 600,
    buffer_capacity = 65536,
    seed = 42
):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = SyncVectorEnv([lambda: gym.make(env_name, max_episode_steps = max_episode_steps) for _ in range(num_envs)], autoreset_mode = AutoresetMode.SAME_STEP)
        state_dim = env.single_observation_space.shape[0]
        num_actions = int(env.single_action_space.n)

        base_actor = WMAugmentedActor(state_dim, num_actions, hidden_dim = 108).to(device)
        base_critic = Critic(state_dim).to(device)
        opt_actor = torch.optim.AdamW(base_actor.parameters(), lr = 1e-3)
        opt_critic = torch.optim.AdamW(base_critic.parameters(), lr = 2.5e-3)

        vqs = {h: ActionChunkVQ(h, num_actions).to(device) for h in macro_hierarchies}
        vq_opts = {h: torch.optim.AdamW(vqs[h].parameters(), lr = 2e-3) for h in macro_hierarchies}
        vq_bufs = {h: ChunkBuffer(buffer_capacity, h) for h in macro_hierarchies}

        wms = {1: HierarchyWorldModel(state_dim, num_actions, wm_dim = wm_dim).to(device)}
        for h in macro_hierarchies:
            wms[h] = HierarchyWorldModel(state_dim, vqs[h].num_codes, wm_dim = wm_dim).to(device)
        wm_opts = {h: torch.optim.AdamW([p for p in wms[h].parameters() if p.requires_grad], lr = 2e-3) for h in wms}
        wm_bufs = {h: TransitionBuffer(buffer_capacity, state_dim) for h in wms}

        print(f'[stage 1] learning VQ codebooks {macro_hierarchies} + per-hierarchy world models {(1, *macro_hierarchies)}', flush = True)

        obs, _ = env.reset(seed = seed)
        state = torch.from_numpy(obs).float().to(device)

        for upd in range(1, base_updates + 1):
            progress = (upd - 1) / max(1, base_updates)
            ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)
            for pg in opt_actor.param_groups:
                pg['lr'] = 1e-3 * schedule_at(LR_SCALE_SCHEDULE, progress)

            states, next_states, actions, lps, rews, terms, truncs = [], [], [], [], [], [], []
            base_actor.eval()
            for _ in range(rollout_steps):
                with torch.no_grad():
                    action, lp = base_actor.sample(state)
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                done = terminated | truncated
                states.append(state)
                next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                actions.append(action)
                lps.append(lp)
                rews.append(torch.from_numpy(reward / 100.).float().to(device))
                terms.append(torch.from_numpy(terminated).bool().to(device))
                truncs.append(torch.from_numpy(truncated).bool().to(device))
                state = torch.from_numpy(next_obs).float().to(device)

            s_t, ns_t, a_t = stack(states, dim = 1), stack(next_states, dim = 1), stack(actions, dim = 1)
            term_t, trunc_t = stack(terms, dim = 1), stack(truncs, dim = 1)

            for h in macro_hierarchies:
                vq_bufs[h].add_rollout(a_t, term_t | trunc_t)
            wm_bufs[1].add_rollout(s_t, a_t, ns_t)

            base_actor.train()
            base_critic.train()
            ppo_update(
                base_actor, base_critic, opt_actor, opt_critic,
                states = s_t, next_states = ns_t, actions = a_t, old_log_probs = stack(lps, dim = 1),
                rewards = stack(rews, dim = 1), terminated = term_t, truncated = trunc_t,
                entropy_coef = ent_coef
            )

            if divisible_by(upd, 5):
                for h in macro_hierarchies:
                    vq_bufs[h].train_steps(vqs[h], vq_opts[h], steps = 80)
                wm_bufs[1].train_steps(wms[1], wm_opts[1], steps = 64)

        env.close()

        fdm1, id1 = wm_bufs[1].train_steps(wms[1], wm_opts[1], steps = 200, batch_size = 512)
        print(f'  [wm h=1 frozen] fdm = {fdm1:.4f} (id = {id1:.4f})', flush = True)

        tables = {}
        for h in macro_hierarchies:
            acc = vq_bufs[h].train_steps(vqs[h], vq_opts[h], steps = 250, batch_size = 512)
            tables[h] = vqs[h].codebook_table()

            m_env = SyncVectorEnv([lambda h_val = h, tbl = tables[h]: VQMacroEnv(gym.make(env_name, max_episode_steps = max_episode_steps), tbl, h_val) for _ in range(num_envs)], autoreset_mode = AutoresetMode.SAME_STEP)
            m_actor = WMAugmentedActor(state_dim, vqs[h].num_codes, hidden_dim = 64).to(device)
            m_critic = Critic(state_dim).to(device)
            opt_ma = torch.optim.AdamW(m_actor.parameters(), lr = 1e-3)
            opt_mc = torch.optim.AdamW(m_critic.parameters(), lr = 2.5e-3)

            m_obs, _ = m_env.reset(seed = seed + 100 * h)
            m_state = torch.from_numpy(m_obs).float().to(device)
            m_steps = max(128, rollout_steps // h)

            for upd in range(1, macro_wm_updates + 1):
                states, next_states, codes, lps, rews, terms, truncs, durs = [], [], [], [], [], [], [], []
                m_actor.eval()
                for _ in range(m_steps):
                    with torch.no_grad():
                        code, lp = m_actor.sample(m_state)
                    next_obs, disc_r, terminated, truncated, info = m_env.step(code.cpu().numpy())
                    done = terminated | truncated
                    p_steps = step_info_array(info, 'prim_steps', done, float(h))

                    states.append(m_state)
                    next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                    codes.append(code)
                    lps.append(lp)
                    rews.append(torch.from_numpy(disc_r / 100.).float().to(device))
                    terms.append(torch.from_numpy(terminated).bool().to(device))
                    truncs.append(torch.from_numpy(truncated).bool().to(device))
                    durs.append(torch.from_numpy(p_steps).float().to(device))
                    m_state = torch.from_numpy(next_obs).float().to(device)

                s_t, ns_t, c_t, d_t = stack(states, dim = 1), stack(next_states, dim = 1), stack(codes, dim = 1), stack(durs, dim = 1)
                wm_bufs[h].add_rollout(s_t, c_t, ns_t, valid_mask = d_t >= (float(h) - 0.5))

                m_actor.train()
                m_critic.train()
                ppo_update(
                    m_actor, m_critic, opt_ma, opt_mc,
                    states = s_t, next_states = ns_t, actions = c_t, old_log_probs = stack(lps, dim = 1),
                    rewards = stack(rews, dim = 1), terminated = stack(terms, dim = 1),
                    truncated = stack(truncs, dim = 1), durations = d_t
                )

                if divisible_by(upd, 4):
                    wm_bufs[h].train_steps(wms[h], wm_opts[h], steps = 64)

            m_env.close()
            fdm_h, id_h = wm_bufs[h].train_steps(wms[h], wm_opts[h], steps = 200, batch_size = 512)
            print(f'  [wm h={h} frozen] vq_acc = {100 * acc:.1f}% | fdm = {fdm_h:.4f} (id = {id_h:.4f})', flush = True)

        encoders = {}
        for h, wm in wms.items():
            wm.eval()
            wm.requires_grad_(False)
            encoders[h] = wm.state_encoder

    return state_dim, wm_dim, tables, encoders

# stage 2: train fresh macro-actor with same- or higher-hierarchy wm state embeds

def run_wm_embed_variant(
    *,
    key: str,
    wm_horizons: tuple[int, ...],
    encoders: dict[int, Module],
    wm_dim: int,
    table: np.ndarray,
    target_hierarchy = 2,
    updates = 50,
    rollout_steps = 512,
    num_envs = 16,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    seed = 42
):
    selected = [encoders[h] for h in wm_horizons]

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = SyncVectorEnv([lambda: VQMacroEnv(gym.make(env_name, max_episode_steps = max_episode_steps), table, target_hierarchy) for _ in range(num_envs)], autoreset_mode = AutoresetMode.SAME_STEP)
        state_dim = env.single_observation_space.shape[0]

        actor = WMAugmentedActor(state_dim, int(table.shape[0]), hidden_dim = 64, wm_encoders = selected, wm_dim = wm_dim).to(device)
        critic = Critic(state_dim).to(device)
        opt_actor = torch.optim.AdamW(actor.parameters(), lr = 1e-3)
        opt_critic = torch.optim.AdamW(critic.parameters(), lr = 1e-3)

        obs, _ = env.reset(seed = seed + 100)
        state = torch.from_numpy(obs).float().to(device)
        running = np.zeros(num_envs)
        recent, history = [], []
        best_rolling = -float('inf')
        best_state = deepcopy(actor.state_dict())
        env_steps = 0
        t0 = time.time()

        for upd in range(1, updates + 1):
            progress = (upd - 1) / max(1, updates)
            ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)
            for pg in opt_actor.param_groups:
                pg['lr'] = 1e-3 * schedule_at(LR_SCALE_SCHEDULE, progress)

            states, next_states, codes, lps, rews, terms, truncs, durs = [], [], [], [], [], [], [], []
            actor.eval()
            for _ in range(rollout_steps):
                with torch.no_grad():
                    code, lp = actor.sample(state)
                next_obs, disc_r, terminated, truncated, info = env.step(code.cpu().numpy())
                done = terminated | truncated
                raw_r = step_info_array(info, 'raw_reward', done, 0.)
                p_steps = step_info_array(info, 'prim_steps', done, float(target_hierarchy))

                running += raw_r
                env_steps += int(p_steps.sum())
                for idx in np.flatnonzero(done):
                    recent.append(float(running[idx]))
                    running[idx] = 0.

                states.append(state)
                next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                codes.append(code)
                lps.append(lp)
                rews.append(torch.from_numpy(disc_r / 100.).float().to(device))
                terms.append(torch.from_numpy(terminated).bool().to(device))
                truncs.append(torch.from_numpy(truncated).bool().to(device))
                durs.append(torch.from_numpy(p_steps).float().to(device))
                state = torch.from_numpy(next_obs).float().to(device)

            actor.train()
            critic.train()
            ppo_update(
                actor, critic, opt_actor, opt_critic,
                states = stack(states, dim = 1), next_states = stack(next_states, dim = 1),
                actions = stack(codes, dim = 1), old_log_probs = stack(lps, dim = 1),
                rewards = stack(rews, dim = 1), terminated = stack(terms, dim = 1),
                truncated = stack(truncs, dim = 1), durations = stack(durs, dim = 1),
                entropy_coef = ent_coef
            )

            rec32 = float(np.mean(recent[-32:])) if recent else float('nan')
            if not np.isnan(rec32) and rec32 > best_rolling:
                best_rolling = rec32
                best_state = deepcopy(actor.state_dict())

            history.append(dict(variant = key, update = upd, env_steps = env_steps, recent_return_32 = rec32))
            if divisible_by(upd, 15) or upd == updates:
                print(f'  [{key:10s} upd {upd:2d}/{updates}] steps = {env_steps // 1000:3d}k | ret32 = {rec32:+6.1f} (best = {best_rolling:+6.1f}) [{time.time() - t0:.1f}s]', flush = True)

        env.close()

    actor.load_state_dict(best_state)
    ev = evaluate_macro_actor(actor, table, target_hierarchy, env_name = env_name, max_episode_steps = max_episode_steps)
    ev['in_dim'] = actor.in_dim
    ev['peak_rolling_32'] = best_rolling
    return history, ev

# main

def main(
    env_name = 'LunarLander-v3',
    out_folder = './vq-hierarchical-recording',
    ckpt_dir: str | None = None,
    target_hierarchy = 2,
    macro_hierarchies: tuple[int, ...] = (2, 4, 8),
    wm_dim = 16,
    base_updates = 60,
    macro_wm_updates = 40,
    embed_updates = 50,
    seed = 42
):
    out_dir = Path(out_folder)
    out_dir.mkdir(parents = True, exist_ok = True)

    if exists(ckpt_dir):
        cdir = Path(ckpt_dir)
        tbl = torch.load(cdir / f'ppo_vq_h{target_hierarchy}.pt', map_location = 'cpu', weights_only = False)['decoded_table']
        target_table = (tbl.cpu().numpy() if torch.is_tensor(tbl) else np.asarray(tbl)).astype(np.int64)
        payload = torch.load(cdir / 'vq_world_models.pt', map_location = device, weights_only = False)
        state_dim, wm_dim, wm_hidden = int(payload['state_dim']), int(payload['wm_dim']), int(payload['wm_hidden_dim'])
        encoders = {}
        for k, sd in payload['state_encoders'].items():
            enc = nn.Sequential(nn.Linear(state_dim, wm_hidden), SimBa(wm_hidden), nn.Linear(wm_hidden, wm_dim)).to(device)
            enc.load_state_dict(sd)
            enc.eval()
            enc.requires_grad_(False)
            encoders[int(k)] = enc
    else:
        _, wm_dim, tables, encoders = train_vq_and_world_models(
            env_name = env_name,
            macro_hierarchies = macro_hierarchies,
            wm_dim = wm_dim,
            base_updates = base_updates,
            macro_wm_updates = macro_wm_updates,
            seed = seed
        )
        target_table = tables[target_hierarchy]
        torch.save(dict(state_encoders = {h: enc.state_dict() for h, enc in encoders.items()}, wm_dim = wm_dim), out_dir / 'vq_world_models_self_contained.pt')

    variants = [
        ('none', 'No WM Embed', ()),
        *(
            (f'wm_dt{h}', f'+ WM(dt={h})', (h,))
            for h in macro_hierarchies if h in encoders
        ),
        ('wm_all', f'+ WM{macro_hierarchies}', tuple(h for h in macro_hierarchies if h in encoders))
    ]

    print(f'\n[stage 2] training dt = {target_hierarchy} actor from scratch with frozen WM state embeds', flush = True)
    histories, evals = {}, {}

    for key, label, hs in variants:
        hist, ev = run_wm_embed_variant(
            key = key, wm_horizons = hs, encoders = encoders, wm_dim = wm_dim,
            table = target_table, target_hierarchy = target_hierarchy,
            updates = embed_updates, env_name = env_name, seed = seed
        )
        histories[key], evals[key] = hist, ev
        print(f'  [{label:16s} | proj_in={ev["in_dim"]:2d}] eval = {ev["mean_return"]:+6.1f} +/- {ev["std_return"]:4.1f} (peak32 = {ev["peak_rolling_32"]:+6.1f})', flush = True)

    with open(out_dir / 'cross_wm_curves.csv', 'w', newline = '') as f:
        writer = csv.DictWriter(f, fieldnames = ['variant', 'update', 'env_steps', 'recent_return_32'])
        writer.writeheader()
        for rows in histories.values():
            writer.writerows(rows)

    fig, ax = plt.subplots(figsize = (8.5, 5.2), dpi = 150)
    palette = ['#6c757d', '#2a9d8f', '#e98a15', '#d62828', '#3a5a9b']
    for idx, (key, label, _) in enumerate(variants):
        rows = histories[key]
        ev = evals[key]
        ax.plot([r['env_steps'] / 1e3 for r in rows], [r['recent_return_32'] for r in rows], color = palette[idx % len(palette)], linewidth = 2.2, label = f'{label} [{ev["in_dim"]}->128] (eval {ev["mean_return"]:+.1f})')
    ax.axhline(200., color = 'green', linestyle = '--', linewidth = 1.)
    ax.axhline(0., color = 'gray', linestyle = ':', linewidth = 0.8)
    ax.set(xlabel = 'primitive env steps (k)', ylabel = 'rolling 32-ep return', title = f'dt = {target_hierarchy} Actor From Scratch With Frozen WM State Embeds')
    ax.legend(loc = 'lower right', fontsize = 8.2)
    ax.grid(alpha = 0.25)
    plt.tight_layout()
    fig.savefig(out_dir / 'cross_wm_comparison.png', bbox_inches = 'tight')
    plt.close(fig)

if __name__ == '__main__':
    fire.Fire(main)
