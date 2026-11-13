# /// script
# dependencies = [
#   "assoc-scan",
#   "einops",
#   "fire",
#   "gymnasium[box2d,other]",
#   "moviepy",
#   "numpy",
#   "torch",
#   "torch-einops-utils>=0.2.5",
#   "vector-quantize-pytorch>=1.34.0",
# ]
# ///
"""
Accretive-SMDP: Concentric accretion of frozen VQ codebooks (A_k = A_{k-1} ∪ VQ_{2^k}).

  Layer 0 : A_0 = {dt=1}        -> base PPO stopped early at +100 hover
  Layer 1 : A_1 = A_0 ∪ VQ_2    -> reborn SMDP PPO over {1, 2}
  Layer 2 : A_2 = A_1 ∪ VQ_4    -> reborn SMDP PPO over {1, 2, 4}
  Layer 3 : A_3 = A_2 ∪ VQ_8    -> reborn SMDP PPO over {1, 2, 4, 8}
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

import fire
import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv
import numpy as np

import torch
from torch import Tensor, cat, nn, stack, tensor, zeros
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange
from vector_quantize_pytorch import VectorQuantize

from x_ppo import (
    NextLatentQCritic,
    calc_gae,
    clipped_value_loss_fn,
    hpo_actor_loss,
    soft_exit_weights
)

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ENTROPY_SCHEDULE = (0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002)
LR_SCALE_SCHEDULE = (1., 0.85, 0.55, 0.30, 0.15, 0.08)
PRIM_NAMES = ('N', 'L', 'M', 'R')

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

def annealed_marginal_ent(base_coef: float, progress: float, hold_frac = 0.25, min_ratio = 0.067) -> float:
    if base_coef <= 0. or progress <= hold_frac:
        return base_coef
    p = min(1., (progress - hold_frac) / max(1e-6, 1. - hold_frac))
    return base_coef * (min_ratio + (1. - min_ratio) * 0.5 * (1. + math.cos(math.pi * p)))

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

# action routing (A_k = A_{k-1} ∪ VQ_{2^k})

@dataclass(frozen = True)
class ActionBank:
    name: str
    horizon: int
    sequences: tuple[tuple[int, ...], ...]

    @property
    def count(self) -> int:
        return len(self.sequences)

class ActionRouter:
    def __init__(self, num_prim_actions: int, banks: tuple[ActionBank, ...]):
        self.num_prim_actions = num_prim_actions
        self.banks = banks
        self.hierarchies = tuple(b.horizon for b in banks)
        self.counts = tuple(b.count for b in banks)
        self.total_actions = sum(self.counts)

        self.horizons, self.bank_ids, self.sequences = [], [], []
        for bank_idx, bank in enumerate(banks):
            for seq in bank.sequences:
                self.horizons.append(bank.horizon)
                self.bank_ids.append(bank_idx)
                self.sequences.append(seq)

        self.horizons_np = np.asarray(self.horizons, dtype = np.int64)

    @classmethod
    def from_base(cls, num_prim_actions: int) -> ActionRouter:
        return cls(num_prim_actions, (ActionBank('base_dt1', 1, tuple((a,) for a in range(num_prim_actions))),))

    def accrete(self, decoded_table: np.ndarray, *, mode: str = 'union', name: str | None = None) -> ActionRouter:
        assert mode in ('union', 'override')
        table = np.asarray(decoded_table, dtype = np.int64)
        h = int(table.shape[1])
        seqs = tuple(tuple(int(a) for a in row) for row in table)
        bank = ActionBank(default(name, f'vq_dt{h}'), h, seqs)
        return ActionRouter(self.num_prim_actions, (*self.banks, bank) if mode == 'union' else (bank,))

class ActionRoutingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, router: ActionRouter, gamma = 0.99):
        super().__init__(env)
        self.router = router
        self.gamma = gamma
        self.action_space = gym.spaces.Discrete(router.total_actions)

    def step(self, action):
        seq = self.router.sequences[int(action)]
        disc_reward, raw_reward, duration = 0., 0., 0

        for step_idx, prim_action in enumerate(seq):
            obs, reward, terminated, truncated, info = self.env.step(prim_action)
            r = float(reward)
            disc_reward += (self.gamma ** step_idx) * r
            raw_reward += r
            duration += 1
            if terminated or truncated:
                break

        return obs, disc_reward, terminated, truncated, {**info, 'raw_reward': raw_reward, 'duration': duration}

# networks

class SimBa(Module):
    def __init__(self, dim, expansion = 2):
        super().__init__()
        dim_inner = int(dim * expansion)
        self.net = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.SiLU(),
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
    def __init__(self, state_dim: int, action_counts: tuple[int, ...], hidden_dim = 64, factorized = True):
        super().__init__()
        self.action_counts = action_counts
        self.num_hierarchies = len(action_counts)
        self.total_actions = sum(action_counts)
        self.factorized = factorized and self.num_hierarchies > 1

        offsets = tensor([0, *action_counts[:-1]]).cumsum(0)
        self.register_buffer('offsets', offsets, persistent = False)
        self.register_buffer('log_counts', tensor(action_counts, dtype = torch.float32).clamp(min = 2).log(), persistent = False)

        dim = hidden_dim * 2
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, dim),
            SimBa(dim),
            SimBa(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU()
        )

        if self.factorized:
            self.to_hier_logits = nn.Linear(hidden_dim, self.num_hierarchies)
            nn.init.zeros_(self.to_hier_logits.weight)
            nn.init.zeros_(self.to_hier_logits.bias)
            self.to_code_logits = ModuleList([nn.Linear(hidden_dim, count) for count in action_counts])
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
            return joint_log_probs.argmax(dim = -1)

        best_level = hier_probs.argmax(dim = -1)
        best_local = stack([c.argmax(dim = -1) for c in joint_log_probs.split(self.action_counts, dim = -1)], dim = -1)
        local_code = best_local.gather(-1, best_level[..., None]).squeeze(-1)
        return self.offsets[best_level] + local_code

    def top_candidates(self, state: Tensor, k: int = 6, rel_prob_min: float = 0.2):
        feat = self.backbone(state)
        k_eff = min(max(1, k), self.total_actions)

        if not self.factorized:
            probs = self.to_flat_logits(feat).softmax(dim = -1)
            top_vals, cand_u = probs.topk(k_eff, dim = -1)
            return cand_u, top_vals >= (rel_prob_min * top_vals[..., :1])

        hier_log_probs = self.to_hier_logits(feat).log_softmax(dim = -1)
        code_log_probs = [head(feat).log_softmax(dim = -1) for head in self.to_code_logits]
        plausibility = cat([
            hier_lp + (code_lp - code_lp.amax(dim = -1, keepdim = True))
            for hier_lp, code_lp in zip(hier_log_probs.split(1, dim = -1), code_log_probs)
        ], dim = -1).exp()

        top_vals, cand_u = plausibility.topk(k_eff, dim = -1)
        return cand_u, top_vals >= (rel_prob_min * top_vals[..., :1])

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

# option embedding & semi-markov ppo

class RouterOptionEmbed(Module):
    def __init__(self, router: ActionRouter, embed_dim = 64, gamma = 0.99):
        super().__init__()
        max_h = max(router.hierarchies)
        num_prim = max(4, router.num_prim_actions)
        num_banks = len(router.banks)

        disc_max = sum(gamma ** j for j in range(max_h))
        feat_dim = max_h * num_prim + num_prim + num_banks + 2
        feats = zeros((router.total_actions, feat_dim), dtype = torch.float32)

        for u, (h, b_idx, seq) in enumerate(zip(router.horizons, router.bank_ids, router.sequences)):
            seq_mat = zeros((max_h, num_prim), dtype = torch.float32)
            disc_h = 0.
            for j, prim_a in enumerate(seq):
                w = gamma ** j
                seq_mat[j, int(prim_a)] = w
                disc_h += w

            duty = seq_mat.sum(dim = 0) / max(disc_h, 1e-6)
            b_onehot = zeros(num_banks, dtype = torch.float32)
            b_onehot[b_idx] = 1.
            hor_stats = tensor([disc_h / max(disc_max, 1e-6), gamma ** h], dtype = torch.float32)
            feats[u] = cat((rearrange(seq_mat, 'h a -> (h a)'), duty, b_onehot, hor_stats), dim = 0)

        self.register_buffer('option_features', feats, persistent = False)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )
        self.id_embed = nn.Embedding(router.total_actions, embed_dim)
        nn.init.zeros_(self.id_embed.weight)

    def forward(self, actions: Tensor) -> Tensor:
        return (self.proj(self.option_features) + self.id_embed.weight)[actions.long()]

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
    cand_actions: Tensor | None = None
    cand_old_log_probs: Tensor | None = None

class SemiMarkovPPO(Module):
    def __init__(
        self,
        state_dim: int,
        router: ActionRouter,
        *,
        hidden_dim = 64,
        critic_hidden_dim = 192,
        factorized = True,
        use_smdp_gae = True,
        trace_power = 0.5,
        marginal_ent_coef = 0.03,
        use_q_filter = True,
        num_candidates = 6,
        q_action_embed_dim = 64,
        q_lam = 0.8,
        conservative_weight = 0.25,
        soft_exit_coef = 0.12,
        soft_exit_kl_budget = 0.25,
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
        self.router = router
        self.actor = UnifiedHierarchyActor(state_dim, router.counts, hidden_dim = hidden_dim, factorized = factorized)
        self.critic = Critic(state_dim, hidden_dim = critic_hidden_dim)

        self.use_smdp_gae = use_smdp_gae
        self.trace_power = float(trace_power)
        self.marginal_ent_coef = marginal_ent_coef
        self.num_candidates = max(1, int(num_candidates))
        self.q_lam = float(q_lam)
        self.conservative_weight = float(conservative_weight)
        self.soft_exit_coef = float(soft_exit_coef)
        self.soft_exit_kl_budget = float(soft_exit_kl_budget)

        self.base_lr = lr
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.minibatch_envs = minibatch_envs

        if use_q_filter:
            option_embed = RouterOptionEmbed(router, embed_dim = q_action_embed_dim, gamma = gamma)
            self.q_critic = NextLatentQCritic(
                self.critic,
                dim_embed = state_dim,
                num_actions = None,
                action_embed = option_embed,
                action_embed_dim = q_action_embed_dim,
                action_horizons = router.horizons_np if use_smdp_gae else None,
                discrete_actions = True,
                hidden_dim = critic_hidden_dim,
                depth = 2,
                gamma = gamma,
                next_latent_coef = 0.25,
                q_coef = 1.,
                value_coef = 1.,
                reward_coef = 0.5,
                terminal_coef = 0.25,
                duration_coef = 0.25
            )
            value_param_ids = {id(p) for p in self.critic.parameters()}
            self._q_trunk_params = [p for p in self.q_critic.parameters() if id(p) not in value_param_ids]
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
    def sample_rollout_action(self, state: Tensor, filter_prob: float = 0.):
        if not exists(self.q_critic):
            act, lp = self.actor.sample(state)
            return act, lp, None, None

        joint_lp, *_ = self.actor(state)
        dist = Categorical(logits = joint_lp, validate_args = False)
        cand_u = rearrange(dist.sample((self.num_candidates,)), 'k b -> b k')
        cand_lp = joint_lp.gather(-1, cand_u)

        base_u, base_lp = cand_u[:, 0], cand_lp[:, 0]
        if filter_prob <= 0. or self.num_candidates <= 1:
            return base_u, base_lp, cand_u, cand_lp

        filtered = self.q_critic.filter_actions(
            state,
            cand_u,
            candidate_log_probs = cand_lp,
            conservative_weight = self.conservative_weight
        )

        if filter_prob >= 1.:
            return filtered.actions, filtered.log_probs, cand_u, cand_lp

        use_f = torch.rand(state.shape[0], device = state.device) < filter_prob
        return (
            torch.where(use_f, filtered.actions, base_u),
            torch.where(use_f, filtered.log_probs, base_lp),
            cand_u,
            cand_lp
        )

    def update(
        self,
        rollout: SMDPRollout,
        entropy_coef = 0.005,
        marginal_ent_coef: float | None = None,
        distill_active = True
    ):
        num_envs = rollout.states.shape[0]
        eff_marg_ent = default(marginal_ent_coef, self.marginal_ent_coef)
        durs = rollout.durations if self.use_smdp_gae else None

        with torch.no_grad():
            values = self.critic(rollout.states)
            next_values = self.critic(rollout.next_states)
            gae = calc_gae(
                rewards = rollout.rewards,
                values = values,
                next_values = next_values,
                terminated = rollout.terminated,
                truncated = rollout.truncated,
                durations = durs,
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
                    durations = durs,
                    trace_power = self.trace_power,
                    gamma = self.gamma,
                    lam = self.q_lam,
                    use_accelerated = False,
                    return_advantages = True
                )

                if distill_active and self.soft_exit_coef > 0. and exists(rollout.cand_actions) and rollout.cand_actions.shape[-1] > 1:
                    cand_scores = self.q_critic.smdp_advantage_scores(
                        rollout.states,
                        rollout.cand_actions,
                        conservative_weight = self.conservative_weight
                    )
                    cand_weights = soft_exit_weights(cand_scores, kl_budget = self.soft_exit_kl_budget)
                    k_cands = rollout.cand_actions.shape[-1]
                    cand_adv = (cand_weights - (1. / k_cands)) * k_cands

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

                if exists(cand_adv):
                    mb_cand_u = rearrange(rollout.cand_actions[mb], 'b t k -> (b t) k')
                    mb_cand_old_lp = rearrange(rollout.cand_old_log_probs[mb], 'b t k -> (b t) k')
                    mb_cand_adv = rearrange(cand_adv[mb], 'b t k -> (b t) k')
                    exit_loss = hpo_actor_loss(
                        joint_lp.gather(-1, mb_cand_u),
                        mb_cand_old_lp,
                        mb_cand_adv,
                        eps_clip = self.eps_clip,
                        normalize_advantages = False
                    ).mean()
                    surr_loss = surr_loss + self.soft_exit_coef * exit_loss

                if self.actor.factorized:
                    state_ent = 0.5 * (normalized_entropy(hier_probs) + within_ent).mean()
                    marginal_ent = normalized_entropy(hier_probs.mean(dim = 0))
                    ent_bonus = entropy_coef * state_ent + eff_marg_ent * marginal_ent
                else:
                    ent_bonus = entropy_coef * entropy(joint_lp.exp()).mean()

                self.opt_actor.zero_grad()
                (surr_loss - ent_bonus).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                if exists(self.q_critic):
                    critic_loss = self.q_critic(
                        mb_s,
                        mb_a,
                        rewards = rearrange(rollout.rewards[mb], 'b k -> (b k)'),
                        next_embeddings = rearrange(rollout.next_states[mb], 'b k d -> (b k) d'),
                        returns = mb_ret,
                        dones = rearrange(rollout.terminated[mb], 'b k -> (b k)'),
                        durations = rearrange(rollout.durations[mb], 'b k -> (b k)') if self.use_smdp_gae else None,
                        q_targets = rearrange(q_gae.returns[mb], 'b k -> (b k)')
                    )
                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self._q_trunk_params, 1.)
                    self.opt_critic.step()
                else:
                    critic_loss = clipped_value_loss_fn(self.critic(mb_s), mb_val, mb_ret, clip = None)
                    self.opt_critic.zero_grad()
                    critic_loss.backward()
                    self.opt_critic.step()

# harvesting & pre-VQ hierarchical chunk filtering

def harvest_and_train_vq(
    agent: SemiMarkovPPO,
    router: ActionRouter,
    *,
    target_hierarchy: int,
    filter_chunks: bool = True,
    collect_prim_steps = 100_000,
    vq_steps = 1400,
    vq_lr = 2e-3,
    max_chunks = 65_536,
    env_name = 'LunarLander-v3',
    num_envs = 16,
    max_episode_steps = 600,
    gamma = 0.99,
    seed = 42
) -> tuple[ActionChunkVQ, np.ndarray, float]:
    with torch.random.fork_rng():
        torch.manual_seed(seed + 700 * target_hierarchy)
        rng = np.random.default_rng(seed + 700 * target_hierarchy)

        base_env = SyncVectorEnv(
            [lambda: gym.make(env_name, max_episode_steps = max_episode_steps) for _ in range(num_envs)],
            autoreset_mode = AutoresetMode.SAME_STEP
        )

        agent.eval()
        obs, _ = base_env.reset(seed = seed + 900 * target_hierarchy)

        pending_seqs: list[list[int]] = [[] for _ in range(num_envs)]
        active_hiers = np.ones(num_envs, dtype = np.int64)
        new_traj = lambda: dict(obs = [], next_obs = [], acts = [], rews = [], terms = [], hiers = [], starts = [], ends = [], raw_ret = 0.)
        ep_trajs = [new_traj() for _ in range(num_envs)]
        completed_episodes = []
        prim_collected = 0
        eff_filter = (0.85 if filter_chunks else 0.50) if exists(agent.q_critic) else 0.

        with torch.no_grad():
            while prim_collected < collect_prim_steps:
                need_dec_set = {i for i in range(num_envs) if len(pending_seqs[i]) == 0}
                if need_dec_set:
                    need_dec = sorted(need_dec_set)
                    act_t, *_ = agent.sample_rollout_action(torch.from_numpy(obs[need_dec]).float().to(device), filter_prob = eff_filter)
                    for idx_local, env_i in enumerate(need_dec):
                        u = int(act_t[idx_local].item())
                        pending_seqs[env_i] = list(router.sequences[u])
                        active_hiers[env_i] = int(router.horizons[u])

                prim_acts = np.array([pending_seqs[i].pop(0) for i in range(num_envs)], dtype = np.int64)
                next_obs, reward, terminated, truncated, info = base_env.step(prim_acts)
                done = terminated | truncated
                real_next = true_next_obs(next_obs, done, info)
                prim_collected += num_envs

                for i in range(num_envs):
                    tr = ep_trajs[i]
                    tr['obs'].append(obs[i].copy())
                    tr['next_obs'].append(real_next[i].copy())
                    tr['acts'].append(int(prim_acts[i]))
                    tr['rews'].append(float(reward[i]) / 100.)
                    tr['terms'].append(bool(terminated[i]))
                    tr['hiers'].append(int(active_hiers[i]))
                    tr['starts'].append(i in need_dec_set)
                    tr['ends'].append((len(pending_seqs[i]) == 0) or bool(done[i]))
                    tr['raw_ret'] += float(reward[i])

                    if done[i]:
                        if len(tr['acts']) >= target_hierarchy:
                            completed_episodes.append(tr)
                        ep_trajs[i] = new_traj()
                        pending_seqs[i] = []
                        active_hiers[i] = 1

                obs = next_obs

        base_env.close()

        if filter_chunks and len(completed_episodes) >= 8:
            high_landed = [ep for ep in completed_episodes if ep['raw_ret'] >= 185.]
            if len(high_landed) >= max(12, len(completed_episodes) // 5):
                kept_episodes = high_landed
            else:
                ret_cut = float(np.percentile([ep['raw_ret'] for ep in completed_episodes], 55))
                kept_episodes = [ep for ep in completed_episodes if ep['raw_ret'] >= ret_cut]
        else:
            kept_episodes = completed_episodes

        h = target_hierarchy
        half_h = max(1, h // 2)
        stride = 1 if filter_chunks else h
        gam_pows = np.array([gamma ** j for j in range(h)], dtype = np.float32)
        gam_h, gam_half = float(gamma ** h), float(gamma ** half_h)
        parent_has_macros = max(router.hierarchies) > 1
        max_parent_h = float(max(router.hierarchies))

        chunks, rows = [], []
        with torch.no_grad():
            for ep in kept_episodes:
                T = len(ep['acts'])
                if T < h:
                    continue
                all_s = np.concatenate([np.asarray(ep['obs'], dtype = np.float32), np.asarray(ep['next_obs'][-1], dtype = np.float32)[None]], axis = 0)
                vals = agent.critic(torch.from_numpy(all_s).to(device)).cpu().numpy()

                acts_np = np.asarray(ep['acts'], dtype = np.int64)
                rews_np = np.asarray(ep['rews'], dtype = np.float32)
                terms_np = np.asarray(ep['terms'], dtype = bool)
                hiers_np = np.asarray(ep['hiers'], dtype = np.float32)
                starts_np = np.asarray(ep['starts'], dtype = bool)
                ends_np = np.asarray(ep['ends'], dtype = bool)

                for t in range(0, T - h + 1, stride):
                    if terms_np[t:t + h - 1].any():
                        continue
                    r_disc = float((rews_np[t:t + h] * gam_pows).sum())
                    end_term = bool(terms_np[t + h - 1])
                    v_start, v_mid = float(vals[t]), float(vals[t + half_h])
                    v_end = 0. if end_term else float(vals[t + h])
                    adv_h = r_disc + gam_h * v_end - v_start
                    post_val = r_disc if end_term else v_end

                    r_first = float((rews_np[t:t + half_h] * gam_pows[:half_h]).sum())
                    r_second = float((rews_np[t + half_h:t + h] * gam_pows[:half_h]).sum())
                    sub_adv = min(r_first + gam_half * v_mid - v_start, r_second + gam_half * v_end - v_mid)

                    h_slice = hiers_np[t:t + h]
                    commit_frac = float((h_slice > 1.).mean()) if parent_has_macros else 1.
                    chunks.append(acts_np[t:t + h])
                    rows.append((adv_h, sub_adv, v_start, post_val, commit_frac, float(h_slice.mean()), float(starts_np[t] and ends_np[t + h - 1])))

        chunks_np = np.asarray(chunks, dtype = np.int64)
        stats_np = np.asarray(rows, dtype = np.float32)
        advs_np, sub_advs_np, start_vals_np, end_vals_np, commit_np, mean_h_np, closed_np = stats_np.T
        closed_mask = closed_np > 0.5
        total_carved = len(chunks_np)

        if filter_chunks and total_carved >= 256:
            keep_mask = (
                (start_vals_np >= np.percentile(start_vals_np, 20)) &
                (end_vals_np >= np.percentile(end_vals_np, 25)) &
                (advs_np >= max(-0.002, float(np.percentile(advs_np, 25)))) &
                (sub_advs_np >= max(-0.004, float(np.percentile(sub_advs_np, 20))))
            )
            if parent_has_macros:
                bound_commit = keep_mask & closed_mask & (commit_np >= 0.50)
                if int(bound_commit.sum()) >= 512:
                    keep_mask = bound_commit

            if int(keep_mask.sum()) < 256:
                keep_mask = np.zeros(total_carved, dtype = bool)
                keep_mask[np.argsort(-advs_np)[:min(total_carved, 1024)]] = True

            chunks_np, advs_np, mean_h_np = chunks_np[keep_mask], advs_np[keep_mask], mean_h_np[keep_mask]

        if len(chunks_np) > max_chunks:
            sel = rng.choice(len(chunks_np), size = max_chunks, replace = False)
            chunks_np, advs_np, mean_h_np = chunks_np[sel], advs_np[sel], mean_h_np[sel]

        if filter_chunks:
            z_adv = (advs_np - advs_np.mean()) / max(1e-6, float(advs_np.std()))
            weights_np = np.exp(np.clip(0.45 * z_adv, -0.8, 1.0)).astype(np.float64)
            if parent_has_macros:
                weights_np *= (0.5 + 0.5 * (mean_h_np / max_parent_h))
            sample_probs = torch.from_numpy(weights_np / weights_np.sum()).float()
        else:
            sample_probs = None

        vq = ActionChunkVQ(
            hierarchy = h,
            num_actions = router.num_prim_actions,
            codebook_size = router.num_prim_actions * h
        ).to(device)

        chunks_t = torch.from_numpy(chunks_np).long()
        n_buf = int(chunks_t.shape[0])
        bs = min(512, n_buf)
        opt = torch.optim.AdamW(vq.parameters(), lr = vq_lr)
        vq.train()

        init_n = min(n_buf, max(1024, vq.num_codes * 32))
        init_idx = torch.multinomial(sample_probs, init_n, replacement = True) if exists(sample_probs) else torch.randint(0, n_buf, (init_n,))
        with torch.no_grad():
            vq(chunks_t[init_idx].to(device))

        seq_accs = []
        for step_i in range(vq_steps):
            frac = step_i / max(1, vq_steps - 1)
            cur_lr = vq_lr * (0.1 + 0.9 * 0.5 * (1. + math.cos(math.pi * frac)))
            for pg in opt.param_groups:
                pg['lr'] = cur_lr

            b_idx = torch.multinomial(sample_probs, bs, replacement = True) if exists(sample_probs) else torch.randint(0, n_buf, (bs,))
            loss, seq_acc = vq(chunks_t[b_idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vq.parameters(), 1.)
            opt.step()
            seq_accs.append(float(seq_acc))

        vq.eval()
        vq.requires_grad_(False)
        return vq, vq.codebook_table(), float(np.mean(seq_accs[-32:]))

# evaluation & stage training

@torch.no_grad()
def select_eval_action(
    actor: UnifiedHierarchyActor,
    state: Tensor,
    q_critic: NextLatentQCritic | None = None,
    num_candidates: int = 6,
    rel_prob_min: float = 0.2,
    conservative_weight: float = 0.25
):
    if not exists(q_critic) or num_candidates <= 1:
        return actor.greedy(state)

    cand_u, cand_mask = actor.top_candidates(state, k = num_candidates, rel_prob_min = rel_prob_min)
    if not bool(cand_mask[:, 1:].any()):
        return cand_u[:, 0]

    return q_critic.filter_actions(
        state,
        cand_u,
        conservative_weight = conservative_weight,
        candidate_mask = cand_mask
    ).actions

@torch.no_grad()
def evaluate_policy(
    ppo: SemiMarkovPPO,
    router: ActionRouter,
    *,
    env_name = 'LunarLander-v3',
    max_episode_steps = 600,
    num_episodes = 48,
    video_folder: str | Path | None = None,
    video_prefix: str = 'layer',
    seed = 1000
) -> dict:
    ppo.eval()
    env = gym.make(env_name, render_mode = 'rgb_array' if exists(video_folder) else None, max_episode_steps = max_episode_steps)

    if exists(video_folder):
        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = str(Path(video_folder) / video_prefix),
            name_prefix = video_prefix,
            episode_trigger = lambda ep_idx: ep_idx == 0,
            disable_logger = True
        )

    returns, landed, ep_decs, ep_steps = [], [], [], []
    dec_by_h = {h: 0 for h in router.hierarchies}

    for ep in range(num_episodes):
        obs, _ = env.reset(seed = seed + ep)
        done, ep_ret, last_r, n_dec, n_stp = False, 0., 0., 0, 0

        while not done:
            u = int(select_eval_action(
                ppo.actor,
                torch.from_numpy(obs).float().unsqueeze(0).to(device),
                q_critic = ppo.q_critic,
                num_candidates = ppo.num_candidates,
                conservative_weight = ppo.conservative_weight
            ).item())

            h, seq = router.horizons[u], router.sequences[u]
            dec_by_h[h] += 1
            n_dec += 1

            for prim_a in seq:
                obs, reward, terminated, truncated, _ = env.step(prim_a)
                last_r = float(reward)
                ep_ret += last_r
                n_stp += 1
                if terminated or truncated:
                    done = True
                    break

        returns.append(ep_ret)
        landed.append(1. if last_r >= 50. else 0.)
        ep_decs.append(n_dec)
        ep_steps.append(n_stp)

    env.close()
    tot_dec = max(1, sum(dec_by_h.values()))

    return dict(
        mean_return = float(np.mean(returns)),
        std_return = float(np.std(returns)),
        land_rate = 100. * float(np.mean(landed)),
        mean_decisions = float(np.mean(ep_decs)),
        mean_effective_dt = float(sum(ep_steps)) / max(1., float(sum(ep_decs))),
        dec_frac = {h: 100. * dec_by_h[h] / tot_dec for h in router.hierarchies}
    )

def train_accretion_layer(
    *,
    layer_idx: int,
    router: ActionRouter,
    env_step_budget: int,
    stop_return_32: float | None = None,
    use_q_filter: bool = True,
    marginal_ent_coef: float = 0.03,
    trace_power: float = 0.5,
    q_filter_prob: float = 0.5,
    q_warmup_frac: float = 0.06,
    rollout_decisions: int | None = None,
    target_rollout_env_steps: int = 4800,
    env_name: str = 'LunarLander-v3',
    num_envs: int = 16,
    max_episode_steps: int = 600,
    lr: float = 1e-3,
    critic_lr: float = 1e-3,
    gamma: float = 0.99,
    lam: float = 0.95,
    log_every: int = 25,
    seed: int = 42
) -> SemiMarkovPPO:
    is_multi_hier = len(router.hierarchies) > 1
    label = f"L{layer_idx}:{{{','.join(map(str, router.hierarchies))}}}"

    with torch.random.fork_rng():
        stage_seed = seed + 100 * layer_idx
        torch.manual_seed(stage_seed)
        np.random.seed(stage_seed)

        env = SyncVectorEnv(
            [lambda: ActionRoutingWrapper(gym.make(env_name, max_episode_steps = max_episode_steps), router, gamma = gamma) for _ in range(num_envs)],
            autoreset_mode = AutoresetMode.SAME_STEP
        )
        state_dim = int(env.single_observation_space.shape[0])

        ppo = SemiMarkovPPO(
            state_dim,
            router,
            factorized = is_multi_hier,
            use_smdp_gae = is_multi_hier,
            trace_power = trace_power,
            marginal_ent_coef = marginal_ent_coef if is_multi_hier else 0.,
            use_q_filter = use_q_filter and is_multi_hier,
            lr = lr,
            critic_lr = critic_lr,
            gamma = gamma,
            lam = lam
        ).to(device)

        obs, _ = env.reset(seed = stage_seed + 50)
        state = torch.from_numpy(obs).float().to(device)
        running, recent = np.zeros(num_envs), []
        best_rolling = -float('inf')
        best_state = deepcopy(ppo.state_dict())

        ema_dur = float(np.mean([float(h) for h in router.hierarchies]))
        env_steps, upd = 0, 0

        while env_steps < env_step_budget:
            upd += 1
            progress = min(1., env_steps / max(1, env_step_budget))
            ppo.set_lr_scale(schedule_at(LR_SCALE_SCHEDULE, progress))
            ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)
            cur_marg_ent = annealed_marginal_ent(marginal_ent_coef, progress) if is_multi_hier else 0.

            warmed_up = progress >= q_warmup_frac
            active_filter_prob = q_filter_prob if (exists(ppo.q_critic) and warmed_up) else 0.
            cur_decisions = default(rollout_decisions, int(np.clip(round((target_rollout_env_steps / num_envs) / max(1., ema_dur)), 72, 320)))

            states, next_states, actions, log_probs = [], [], [], []
            cand_actions, cand_log_probs = [], []
            rewards, durations, terms, truncs = [], [], [], []
            h_dec_counts = {h: 0 for h in router.hierarchies}
            batch_env_steps = 0

            ppo.eval()
            for _ in range(cur_decisions):
                with torch.no_grad():
                    act_t, lp_t, cand_u_t, cand_lp_t = ppo.sample_rollout_action(state, filter_prob = active_filter_prob)

                act_np = act_t.cpu().numpy()
                next_obs, disc_r, terminated, truncated, info = env.step(act_np)
                done = terminated | truncated

                raw_r = step_info_array(info, 'raw_reward', done, 0.)
                durs = step_info_array(info, 'duration', done, 1.)

                for h_sel in router.horizons_np[act_np]:
                    h_dec_counts[int(h_sel)] += 1

                running += raw_r
                step_inc = int(durs.sum())
                env_steps += step_inc
                batch_env_steps += step_inc

                for idx in np.flatnonzero(done):
                    recent.append(float(running[idx]))
                    running[idx] = 0.

                states.append(state)
                next_states.append(torch.from_numpy(true_next_obs(next_obs, done, info)).float().to(device))
                actions.append(act_t)
                log_probs.append(lp_t)
                if exists(cand_u_t):
                    cand_actions.append(cand_u_t)
                    cand_log_probs.append(cand_lp_t)
                rewards.append(torch.from_numpy(disc_r / 100.).float().to(device))
                durations.append(torch.from_numpy(durs).float().to(device))
                terms.append(torch.from_numpy(terminated).bool().to(device))
                truncs.append(torch.from_numpy(truncated).bool().to(device))
                state = torch.from_numpy(next_obs).float().to(device)

            ema_dur = 0.7 * ema_dur + 0.3 * (batch_env_steps / max(1, num_envs * cur_decisions))

            ppo.update(
                SMDPRollout(
                    states = stack(states, dim = 1),
                    next_states = stack(next_states, dim = 1),
                    actions = stack(actions, dim = 1),
                    old_log_probs = stack(log_probs, dim = 1),
                    rewards = stack(rewards, dim = 1),
                    durations = stack(durations, dim = 1),
                    terminated = stack(terms, dim = 1),
                    truncated = stack(truncs, dim = 1),
                    cand_actions = stack(cand_actions, dim = 1) if cand_actions else None,
                    cand_old_log_probs = stack(cand_log_probs, dim = 1) if cand_log_probs else None
                ),
                entropy_coef = ent_coef,
                marginal_ent_coef = cur_marg_ent,
                distill_active = warmed_up
            )

            rec32 = float(np.mean(recent[-32:])) if recent else float('nan')
            if not np.isnan(rec32) and rec32 > best_rolling:
                best_rolling = rec32
                best_state = deepcopy(ppo.state_dict())

            hit_stop = exists(stop_return_32) and len(recent) >= 32 and not np.isnan(rec32) and rec32 >= stop_return_32

            if divisible_by(upd, log_every) or env_steps >= env_step_budget or hit_stop:
                tot_dec = max(1, sum(h_dec_counts.values()))
                mix = ' '.join(f'dt{h}:{100. * h_dec_counts[h] / tot_dec:4.1f}%' for h in router.hierarchies)
                print(f'  [{label:11s} upd {upd:3d}] steps = {env_steps // 1000:4d}k | ret32 = {rec32:+6.1f} (best = {best_rolling:+6.1f}) | {mix}', flush = True)

            if hit_stop:
                break

        env.close()

    if not exists(stop_return_32):
        ppo.load_state_dict(best_state)

    return ppo

# entrypoint

def main(
    env_name = 'LunarLander-v3',
    video_folder = './accretive-smdp-recording',
    render = True,
    clear_videos = True,
    macro_hierarchies: tuple[int, ...] = (2, 4, 8),
    base_target_return = 100.,
    base_max_step_budget = 600_000,
    stage_step_budgets: tuple[int, ...] = (800_000, 900_000, 1_000_000),
    target_rollout_env_steps = 4800,
    vq_collect_steps = 100_000,
    vq_steps = 1400,
    filter_chunks: bool = True,
    marginal_ent_coef = 0.03,
    trace_power = 0.5,
    use_q_filter = True,
    seed = 42,
    smoke: bool = False
):
    if smoke:
        base_target_return = 20.
        base_max_step_budget = 80_000
        stage_step_budgets = (90_000, 90_000, 90_000)
        vq_collect_steps = 20_000
        vq_steps = 300

    eval_eps = 12 if smoke else 48

    if render and clear_videos:
        rmtree(video_folder, ignore_errors = True)

    probe_env = gym.make(env_name)
    router = ActionRouter.from_base(int(probe_env.action_space.n))
    probe_env.close()

    print(f'\n[Layer 0] A_0 = {{dt=1}} (early stop at ret32 >= {base_target_return:+.0f})', flush = True)
    agent = train_accretion_layer(
        layer_idx = 0,
        router = router,
        env_step_budget = base_max_step_budget,
        stop_return_32 = base_target_return,
        use_q_filter = False,
        rollout_decisions = 512,
        env_name = env_name,
        critic_lr = 2.5e-3,
        seed = seed
    )
    ev0 = evaluate_policy(agent, router, env_name = env_name, num_episodes = eval_eps, video_folder = video_folder if render else None, video_prefix = 'accretive-layer0')
    print(f"  [L0 eval] {ev0['mean_return']:+.1f} +/- {ev0['std_return']:.1f} | land = {ev0['land_rate']:.1f}% | avg_dt = {ev0['mean_effective_dt']:.2f}x", flush = True)

    for idx_m, h in enumerate(macro_hierarchies, start = 1):
        print(f'\n[Layer {idx_m}] A_{idx_m} = A_{idx_m - 1} ∪ VQ_{h}', flush = True)
        _, table, seq_acc = harvest_and_train_vq(
            agent,
            router,
            target_hierarchy = int(h),
            filter_chunks = filter_chunks,
            collect_prim_steps = int(vq_collect_steps),
            vq_steps = int(vq_steps),
            env_name = env_name,
            seed = seed + 31 * idx_m
        )
        macros = [''.join(PRIM_NAMES[int(a)] for a in row) for row in table]
        print(f'  [VQ_{h}] seq_acc = {100. * seq_acc:.1f}% | unique = {len(set(macros))}/{len(macros)} | macros = {macros}', flush = True)

        router = router.accrete(table, mode = 'union', name = f'vq_dt{h}')
        budget = stage_step_budgets[min(idx_m - 1, len(stage_step_budgets) - 1)]

        agent = train_accretion_layer(
            layer_idx = idx_m,
            router = router,
            env_step_budget = int(budget),
            use_q_filter = use_q_filter,
            marginal_ent_coef = marginal_ent_coef,
            trace_power = trace_power,
            target_rollout_env_steps = int(target_rollout_env_steps),
            env_name = env_name,
            seed = seed
        )

        ev = evaluate_policy(
            agent,
            router,
            env_name = env_name,
            num_episodes = eval_eps,
            video_folder = video_folder if render else None,
            video_prefix = f'accretive-layer{idx_m}'
        )
        mix = ' '.join(f"dt{x}:{ev['dec_frac'][x]:.1f}%" for x in router.hierarchies)
        print(
            f"  [L{idx_m} eval] {ev['mean_return']:+.1f} +/- {ev['std_return']:.1f} | "
            f"land = {ev['land_rate']:.1f}% | avg_dt = {ev['mean_effective_dt']:.2f}x ({ev['mean_decisions']:.0f} decs) | {mix}",
            flush = True
        )

if __name__ == '__main__':
    fire.Fire(main)
