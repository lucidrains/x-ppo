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
#   "vector-quantize-pytorch>=1.34.0",
# ]
# ///

"""
Hierarchical Action Compression via Vector Quantization.

Stage 1 trains a normal dt = 1 PPO policy while intermittently quantizing
contiguous h-step action chunks into discrete VQ codebooks (num_actions * h codes).
Stage 2 freezes each codebook and trains a fresh macro-actor in an h-times-slower
environment that decodes VQ indices and commits to their primitive action sequences.

uv run train_vq_hierarchical.py --sweep
"""

from __future__ import annotations

import csv
import time
from copy import deepcopy
from pathlib import Path

import fire
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, Tensor, cat, stack, zeros
import torch.nn.functional as F
from torch.nn import Module
from torch.distributions import Categorical

from einops import rearrange
from einops.layers.torch import Rearrange

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AutoresetMode

from mean_conc_beta import Beta as MeanConcBeta
from vector_quantize_pytorch import VectorQuantize

from x_ppo import (
    calc_gae,
    clipped_value_loss_fn,
    hpo_actor_loss,
    ppo_actor_loss
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

def format_macro(seq: np.ndarray, continuous = False) -> str:
    if continuous:
        return np.array2string(seq, precision = 2, separator = ',')
    names = ('N', 'L', 'M', 'R')
    if seq.max() < 4:
        return ''.join(names[int(a)] for a in seq)
    return ','.join(str(int(a)) for a in seq)

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

class Actor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        *,
        continuous = False,
        beta_init_conc = 5.,
        beta_eps = 1e-5,
        dropout = 0.
    ):
        super().__init__()
        self.continuous = continuous
        dim = hidden_dim * 2
        out_dim = 2 * num_actions if continuous else num_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, dim),
            SimBa(dim, dropout = dropout),
            SimBa(dim, dropout = dropout),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        if continuous:
            nn.init.orthogonal_(self.net[-1].weight, gain = 0.01)
            nn.init.zeros_(self.net[-1].bias)
            self.beta = MeanConcBeta(bounds = (-1., 1.), init_conc = beta_init_conc, unimodal = True, eps = beta_eps)

    def forward(self, state: Tensor):
        out = self.net(state)
        return rearrange(out, '... (a two) -> ... a two', two = 2) if self.continuous else out

    def dist(self, out: Tensor):
        if self.continuous:
            mean_logits, conc_logits = out.unbind(dim = -1)
            return self.beta(mean_logits, conc_logits)
        return Categorical(logits = out, validate_args = False)

    def sample(self, state: Tensor):
        dist = self.dist(self.forward(state))
        action = dist.sample()
        if self.continuous:
            return action, dist.log_prob(action.clamp(-1. + 1e-5, 1. - 1e-5)).sum(dim = -1)
        return action, dist.log_prob(action)

    def greedy(self, state: Tensor):
        out = self.forward(state)
        return self.dist(out).mean if self.continuous else out.argmax(dim = -1)

    def log_prob(self, out: Tensor, action: Tensor):
        dist = self.dist(out)
        if self.continuous:
            return dist.log_prob(action.clamp(-1. + 1e-5, 1. - 1e-5)).sum(dim = -1)
        return dist.log_prob(action.long())

    def entropy(self, out: Tensor):
        ent = self.dist(out).entropy()
        return ent.sum(dim = -1) if self.continuous else ent

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

# vector quantized action chunks

class ActionChunkVQ(Module):
    def __init__(
        self,
        *,
        hierarchy: int,
        num_actions: int,
        state_dim: int,
        codebook_size: int | None = None,
        dim = 128,
        codebook_dim = 16,
        continuous = False,
        prepend_first_state = False,
        state_recon_weight = 0.1,
        decay = 0.9
    ):
        super().__init__()
        target_codes = default(codebook_size, max(4, num_actions * hierarchy))
        num_codes = min(target_codes, num_actions ** hierarchy) if (not continuous and not prepend_first_state and hierarchy <= 6) else target_codes

        self.hierarchy = hierarchy
        self.num_actions = num_actions
        self.num_codes = num_codes
        self.continuous = continuous
        self.prepend_first_state = prepend_first_state
        self.state_recon_weight = state_recon_weight

        chunk_dim = hierarchy * num_actions
        enc_in_dim = chunk_dim + (state_dim if prepend_first_state else 0)

        self.state_norm = nn.RMSNorm(state_dim) if prepend_first_state else None

        self.encoder = nn.Sequential(
            nn.Linear(enc_in_dim, dim),
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

        self.state_decoder = nn.Linear(dim, state_dim) if prepend_first_state else None

    def forward(self, chunks: Tensor, first_states: Tensor | None = None):
        act_vecs = chunks.float() if self.continuous else F.one_hot(chunks.long(), self.num_actions).float()
        flat = rearrange(act_vecs, 'b h a -> b (h a)')

        if self.prepend_first_state:
            assert exists(first_states)
            flat = cat((self.state_norm(first_states), flat), dim = -1)

        quantized, indices, vq_loss = self.vq(self.encoder(flat))
        decoded = self.decoder(quantized)

        if self.continuous:
            pred = decoded.tanh()
            recon_loss = F.mse_loss(pred, chunks.float())
            token_acc = ((pred - chunks.float()).abs().mean(dim = -1) < 0.15).float().mean()
            seq_acc = token_acc
        else:
            recon_loss = F.cross_entropy(rearrange(decoded, 'b h a -> (b h) a'), rearrange(chunks.long(), 'b h -> (b h)'))
            matches = decoded.argmax(dim = -1) == chunks.long()
            token_acc = matches.float().mean()
            seq_acc = matches.all(dim = -1).float().mean()

        loss = recon_loss + vq_loss

        if self.prepend_first_state and exists(first_states):
            loss = loss + self.state_recon_weight * F.mse_loss(self.state_decoder(quantized), self.state_norm(first_states).detach())

        return loss, dict(token_acc = float(token_acc), seq_acc = float(seq_acc), indices = indices.detach())

    @torch.no_grad()
    def codebook_table(self) -> np.ndarray:
        dev = next(self.parameters()).device
        quantized = self.vq.get_output_from_indices(torch.arange(self.num_codes, device = dev))
        decoded = self.decoder(quantized)
        out = decoded.tanh() if self.continuous else decoded.argmax(dim = -1)
        return out.cpu().numpy()

class ChunkBuffer:
    def __init__(self, *, capacity: int, hierarchy: int, state_dim: int, num_actions: int, continuous = False, stride = 2):
        self.capacity = capacity
        self.hierarchy = hierarchy
        self.continuous = continuous
        self.stride = max(1, stride)

        act_shape = (capacity, hierarchy, num_actions) if continuous else (capacity, hierarchy)
        self.actions = zeros(act_shape, dtype = torch.float32 if continuous else torch.long)
        self.first_states = zeros((capacity, state_dim), dtype = torch.float32)
        self.ptr = 0
        self.size = 0

    def add_rollout(self, states: Tensor, actions: Tensor, dones: Tensor):
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
        act_chunks = (rearrange(a_win[valid], 'n a h -> n h a') if self.continuous else a_win[valid].long())[-self.capacity:]
        n = act_chunks.shape[0]

        idx = (self.ptr + torch.arange(n)) % self.capacity
        self.actions[idx] = act_chunks
        self.first_states[idx] = first_s
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def train_steps(self, model: ActionChunkVQ, opt: torch.optim.Optimizer, steps = 80, batch_size = 256) -> dict:
        if self.size < 1:
            return dict(token_acc = 0., seq_acc = 0., unique_macros = 0)

        dev = next(model.parameters()).device
        bs = min(batch_size, self.size)
        model.train()

        with torch.random.fork_rng():
            if not bool(model.vq._codebook.initted.item()):
                init_idx = torch.randint(0, self.size, (min(self.size, max(1024, model.num_codes * 16)),))
                with torch.no_grad():
                    model(self.actions[init_idx].to(dev), self.first_states[init_idx].to(dev))

            tok_accs, seq_accs = [], []
            for _ in range(steps):
                idx = torch.randint(0, self.size, (bs,))
                loss, stats = model(self.actions[idx].to(dev), self.first_states[idx].to(dev))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                tok_accs.append(stats['token_acc'])
                seq_accs.append(stats['seq_acc'])

        table = model.codebook_table()
        unique_macros = model.num_codes if self.continuous else int(np.unique(table, axis = 0).shape[0])
        return dict(token_acc = float(np.mean(tok_accs[-16:])), seq_acc = float(np.mean(seq_accs[-16:])), unique_macros = unique_macros)

# slower macro environment committing to decoded vq sequences

class VQMacroEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, decoded_table: np.ndarray, hierarchy: int, gamma = 0.99, continuous = False):
        super().__init__(env)
        self.decoded_table = decoded_table
        self.hierarchy = hierarchy
        self.gamma = gamma
        self.continuous = continuous
        self.action_space = gym.spaces.Discrete(int(decoded_table.shape[0]))

    def step(self, code_action):
        prim_seq = self.decoded_table[int(code_action)]
        disc_reward, raw_reward, prim_steps = 0., 0., 0

        for h in range(self.hierarchy):
            prim_act = prim_seq[h] if self.continuous else int(prim_seq[h])
            obs, reward, terminated, truncated, info = self.env.step(prim_act)

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
    actor: Actor,
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
    entropy_coef = 0.005,
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
            trace_power = trace_power,
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
            mb_a = rearrange(actions[mb], 'b t ... -> (b t) ...')
            mb_old_lp = rearrange(old_log_probs[mb], 'b t -> (b t)')
            mb_adv = rearrange(gae.advantages[mb], 'b t -> (b t)')
            mb_ret = rearrange(gae.returns[mb], 'b t -> (b t)')
            mb_val = rearrange(values[mb], 'b t -> (b t)')

            out = actor(mb_s)
            actor_loss = loss_fn(actor.log_prob(out, mb_a), mb_old_lp, mb_adv, eps_clip = eps_clip, normalize_advantages = True).mean()
            actor_loss = actor_loss - entropy_coef * actor.entropy(out).mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt_actor.step()

            critic_loss = clipped_value_loss_fn(critic(mb_s), mb_val, mb_ret, clip = None)
            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

# evaluation & video recording

@torch.no_grad()
def evaluate_macro_policy(
    actor: Actor,
    decoded_table: np.ndarray | None,
    hierarchy: int,
    env_name = 'LunarLander-v3',
    continuous = False,
    max_episode_steps = 600,
    num_episodes = 32,
    seed = 1000,
    video_folder: str | None = None,
    video_prefix: str | None = None
) -> dict:
    actor.eval()
    env = gym.make(env_name, max_episode_steps = max_episode_steps, continuous = continuous)

    returns, prim_lens, macro_lens, landed = [], [], [], []
    best_return, best_seed = -float('inf'), seed

    for ep in range(num_episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed = ep_seed)
        ep_ret, prim_t, macro_t, last_r, done = 0., 0, 0, 0., False

        while not done:
            act = actor.greedy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
            macro_t += 1

            if not exists(decoded_table):
                obs, reward, terminated, truncated, _ = env.step(act.squeeze(0).cpu().numpy() if continuous else int(act.item()))
                last_r = float(reward)
                ep_ret += last_r
                prim_t += 1
                done = terminated or truncated
            else:
                prim_seq = decoded_table[int(act.item())]
                for h in range(hierarchy):
                    prim_a = prim_seq[h] if continuous else int(prim_seq[h])
                    obs, reward, terminated, truncated, _ = env.step(prim_a)
                    last_r = float(reward)
                    ep_ret += last_r
                    prim_t += 1
                    if terminated or truncated:
                        done = True
                        break

        returns.append(ep_ret)
        prim_lens.append(prim_t)
        macro_lens.append(macro_t)
        landed.append(1. if last_r >= 50. else 0.)

        if ep_ret > best_return:
            best_return, best_seed = ep_ret, ep_seed

    env.close()

    video_path = None
    if exists(video_folder):
        render_env = gym.make(env_name, render_mode = 'rgb_array', max_episode_steps = max_episode_steps, continuous = continuous)
        obs, _ = render_env.reset(seed = best_seed)
        frames, done = [render_env.render()], False

        while not done:
            act = actor.greedy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
            if not exists(decoded_table):
                obs, _, terminated, truncated, _ = render_env.step(act.squeeze(0).cpu().numpy() if continuous else int(act.item()))
                frames.append(render_env.render())
                done = terminated or truncated
            else:
                prim_seq = decoded_table[int(act.item())]
                for h in range(hierarchy):
                    prim_a = prim_seq[h] if continuous else int(prim_seq[h])
                    obs, _, terminated, truncated, _ = render_env.step(prim_a)
                    frames.append(render_env.render())
                    if terminated or truncated:
                        done = True
                        break

        render_env.close()

        out_dir = Path(video_folder).resolve()
        out_dir.mkdir(parents = True, exist_ok = True)
        prefix = default(video_prefix, f'vq-macro-h{hierarchy}-best')
        mp4_path = out_dir / f'{prefix}.mp4'
        imageio.mimsave(str(mp4_path), frames, fps = 50, macro_block_size = 1)
        imageio.mimsave(str(out_dir / f'{prefix}.gif'), frames[::2], fps = 25)
        video_path = str(mp4_path)
        print(f'[video] h={hierarchy} best eval = {best_return:+.1f} (seed={best_seed}) -> {mp4_path}', flush = True)

    return dict(
        mean_return = float(np.mean(returns)),
        std_return = float(np.std(returns)),
        best_return = float(best_return),
        best_seed = int(best_seed),
        land_rate = 100. * float(np.mean(landed)),
        mean_prim_steps = float(np.mean(prim_lens)),
        mean_macro_steps = float(np.mean(macro_lens)),
        video_path = video_path
    )

def plot_sweep(base_hist: list[dict], base_eval: dict, macro_runs: list[dict], out_path: Path, env_name: str):
    fig, axes = plt.subplots(1, 2, figsize = (14.5, 5.4), dpi = 150)
    palette = ['#4c72b0', '#2a9d8f', '#e98a15', '#d62828', '#6a4c93']

    series = [(f'dt=1 flat (eval {base_eval["mean_return"]:+.1f}, best {base_eval["best_return"]:+.1f})', base_hist, palette[0])]
    for idx, run in enumerate(macro_runs):
        ev = run['eval']
        lbl = f'dt={run["h"]} ({run["num_codes"]} VQ codes | eval {ev["mean_return"]:+.1f}, best {ev["best_return"]:+.1f})'
        series.append((lbl, run['history'], palette[(idx + 1) % len(palette)]))

    for lbl, hist, col in series:
        decs_k = [r['decisions'] / 1e3 for r in hist]
        steps_k = [r['env_steps'] / 1e3 for r in hist]
        rets = [r['recent_return_32'] for r in hist]
        axes[0].plot(decs_k, rets, color = col, linewidth = 2.2, label = lbl)
        axes[1].plot(steps_k, rets, color = col, linewidth = 2.2, label = lbl)

    for ax, xlabel, title in [
        (axes[0], 'policy decisions (k)', f'{env_name}: Return vs Decisions'),
        (axes[1], 'primitive env steps (k)', f'{env_name}: Return vs Env Steps')
    ]:
        ax.axhline(200., color = 'green', linestyle = '--', linewidth = 1.)
        ax.axhline(0., color = 'gray', linestyle = ':', linewidth = 0.8)
        ax.set(xlabel = xlabel, ylabel = 'rolling 32-ep return', title = title)
        ax.legend(loc = 'lower right', fontsize = 8)
        ax.grid(alpha = 0.25)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches = 'tight')
    plt.close(fig)
    print(f'[plot] saved -> {out_path}', flush = True)

# main training pipeline

def main(
    env_name = 'LunarLander-v3',
    hierarchy = 4,
    sweep = False,
    hierarchies: tuple[int, ...] = (2, 4, 8),
    continuous = False,
    prepend_first_state = False,
    codebook_size: int | None = None,
    vq_dim = 128,
    vq_codebook_dim = 16,
    vq_lr = 2e-3,
    compress_every = 5,
    vq_steps = 80,
    vq_final_steps = 250,
    buffer_capacity = 65536,
    base_updates = 135,
    macro_updates = 160,
    num_envs = 16,
    base_rollout_steps = 384,
    macro_rollout_steps: int | None = None,
    max_episode_steps = 600,
    actor_hidden_dim = 108,
    critic_hidden_dim = 192,
    epochs = 6,
    minibatch_envs = 4,
    lr = 1e-3,
    critic_lr = 2.5e-3,
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.2,
    max_grad_norm = 0.5,
    use_hpo = True,
    seed = 42,
    render = True,
    out_folder = './vq-hierarchical-recording',
    log_every = 10
):
    active_hierarchies = tuple(int(h) for h in hierarchies) if sweep else (int(hierarchy),)

    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = Path(out_folder)
    out_dir.mkdir(parents = True, exist_ok = True)

    # stage 1: dt = 1 ppo + intermittent vq action compression

    base_env = SyncVectorEnv(
        [lambda: gym.make(env_name, max_episode_steps = max_episode_steps, continuous = continuous) for _ in range(num_envs)],
        autoreset_mode = AutoresetMode.SAME_STEP
    )

    state_dim = base_env.single_observation_space.shape[0]
    num_actions = base_env.single_action_space.shape[0] if continuous else int(base_env.single_action_space.n)

    base_actor = Actor(state_dim, actor_hidden_dim, num_actions, continuous = continuous).to(device)
    base_critic = Critic(state_dim, critic_hidden_dim).to(device)
    opt_actor = torch.optim.AdamW(base_actor.parameters(), lr = lr)
    opt_critic = torch.optim.AdamW(base_critic.parameters(), lr = critic_lr)

    with torch.random.fork_rng():
        vqs = {
            h: ActionChunkVQ(
                hierarchy = h, num_actions = num_actions, state_dim = state_dim,
                codebook_size = codebook_size, dim = vq_dim, codebook_dim = vq_codebook_dim,
                continuous = continuous, prepend_first_state = prepend_first_state
            ).to(device)
            for h in active_hierarchies
        }
        vq_opts = {h: torch.optim.AdamW(vqs[h].parameters(), lr = vq_lr) for h in active_hierarchies}
        buffers = {
            h: ChunkBuffer(capacity = buffer_capacity, hierarchy = h, state_dim = state_dim, num_actions = num_actions, continuous = continuous)
            for h in active_hierarchies
        }
        vq_stats = {h: dict(seq_acc = 0., token_acc = 0., unique_macros = 0) for h in active_hierarchies}

    print(f'[stage 1] dt = 1 PPO + VQ compression for hierarchies {active_hierarchies}', flush = True)

    obs, _ = base_env.reset(seed = seed)
    state = torch.from_numpy(obs).float().to(device)
    running = np.zeros(num_envs)
    recent, base_hist = [], []
    best_base_rolling = -float('inf')
    best_base_state = deepcopy(base_actor.state_dict())
    env_steps = 0
    t0 = time.time()

    for upd in range(1, base_updates + 1):
        progress = (upd - 1) / max(1, base_updates)
        ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)
        for pg in opt_actor.param_groups:
            pg['lr'] = lr * schedule_at(LR_SCALE_SCHEDULE, progress)

        states, next_states, actions, log_probs, rewards, terms, truncs = [], [], [], [], [], [], []
        ep_rets = []

        base_actor.eval()
        for _ in range(base_rollout_steps):
            with torch.no_grad():
                action, lp = base_actor.sample(state)

            next_obs, reward, terminated, truncated, info = base_env.step(action.cpu().numpy())
            done = terminated | truncated
            running += reward

            for idx in np.flatnonzero(done):
                ep_rets.append(float(running[idx]))
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
                    vq_stats[h] = buffers[h].train_steps(vqs[h], vq_opts[h], steps = vq_steps)

        rec32 = float(np.mean(recent[-32:])) if recent else float('nan')
        if not np.isnan(rec32) and rec32 > best_base_rolling:
            best_base_rolling = rec32
            best_base_state = deepcopy(base_actor.state_dict())

        base_hist.append(dict(update = upd, env_steps = env_steps, decisions = env_steps, recent_return_32 = rec32))

        if divisible_by(upd, log_every) or upd == base_updates:
            vq_msg = ' | '.join(f'h={h}: {100 * vq_stats[h]["seq_acc"]:4.1f}% ({vq_stats[h]["unique_macros"]}/{vqs[h].num_codes})' for h in active_hierarchies)
            print(f'  [dt=1 upd {upd:3d}/{base_updates}] steps = {env_steps // 1000:4d}k | ret32 = {rec32:+6.1f} (best = {best_base_rolling:+6.1f}) | {vq_msg} [{time.time() - t0:.1f}s]', flush = True)

    base_env.close()

    base_actor.load_state_dict(best_base_state)
    flat_table = None if continuous else np.arange(num_actions, dtype = np.int64)[:, None]
    base_eval = evaluate_macro_policy(
        base_actor, flat_table, 1, env_name = env_name, continuous = continuous,
        max_episode_steps = max_episode_steps, video_folder = out_folder if render else None,
        video_prefix = 'vq-flat-h1-best'
    )

    decoded_tables = {}
    for h in active_hierarchies:
        vq_stats[h] = buffers[h].train_steps(vqs[h], vq_opts[h], steps = vq_final_steps, batch_size = 512)
        vqs[h].eval()
        vqs[h].requires_grad_(False)
        decoded_tables[h] = vqs[h].codebook_table()
        macros = [format_macro(row, continuous) for row in decoded_tables[h]]
        print(f'  [vq frozen h={h}] seq_acc = {100 * vq_stats[h]["seq_acc"]:.1f}% | unique = {vq_stats[h]["unique_macros"]}/{vqs[h].num_codes} | {macros}', flush = True)

    # stage 2: slower macro-environment ppo for each hierarchy h > 1

    macro_runs = []

    for h in active_hierarchies:
        table = decoded_tables[h]
        num_codes = vqs[h].num_codes
        rollout_macro = default(macro_rollout_steps, max(128, base_rollout_steps // h))

        with torch.random.fork_rng():
            torch.manual_seed(seed + 1000 * h)
            np.random.seed(seed + 1000 * h)

            macro_env = SyncVectorEnv(
                [lambda h_val = h, tbl = table: VQMacroEnv(gym.make(env_name, max_episode_steps = max_episode_steps, continuous = continuous), tbl, h_val, gamma = gamma, continuous = continuous) for _ in range(num_envs)],
                autoreset_mode = AutoresetMode.SAME_STEP
            )

            macro_actor = Actor(state_dim, actor_hidden_dim, num_codes, continuous = False).to(device)
            macro_critic = Critic(state_dim, critic_hidden_dim).to(device)
            opt_m_actor = torch.optim.AdamW(macro_actor.parameters(), lr = lr)
            opt_m_critic = torch.optim.AdamW(macro_critic.parameters(), lr = critic_lr)

            obs, _ = macro_env.reset(seed = seed + 100)
            state = torch.from_numpy(obs).float().to(device)
            running = np.zeros(num_envs)
            m_recent, m_hist = [], []
            best_m_rolling = -float('inf')
            best_m_state = deepcopy(macro_actor.state_dict())
            m_steps, m_decs = 0, 0
            t_h = time.time()

            print(f'\n[stage 2] dt = {h} macro-actor training ({num_codes} VQ codes, rollout = {rollout_macro})', flush = True)

            for upd in range(1, macro_updates + 1):
                progress = (upd - 1) / max(1, macro_updates)
                ent_coef = schedule_at(ENTROPY_SCHEDULE, progress)
                for pg in opt_m_actor.param_groups:
                    pg['lr'] = lr * schedule_at(LR_SCALE_SCHEDULE, progress)

                states, next_states, codes, log_probs, rewards, terms, truncs, psteps = [], [], [], [], [], [], [], []

                macro_actor.eval()
                for _ in range(rollout_macro):
                    with torch.no_grad():
                        code, lp = macro_actor.sample(state)

                    next_obs, disc_r, terminated, truncated, info = macro_env.step(code.cpu().numpy())
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
                    codes.append(code)
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
                    actions = stack(codes, dim = 1), old_log_probs = stack(log_probs, dim = 1),
                    rewards = stack(rewards, dim = 1), terminated = stack(terms, dim = 1),
                    truncated = stack(truncs, dim = 1), durations = stack(psteps, dim = 1),
                    gamma = gamma, lam = lam, eps_clip = eps_clip, entropy_coef = ent_coef,
                    epochs = epochs, minibatch_envs = minibatch_envs, max_grad_norm = max_grad_norm, use_hpo = use_hpo
                )

                rec32 = float(np.mean(m_recent[-32:])) if m_recent else float('nan')
                if not np.isnan(rec32) and rec32 > best_m_rolling:
                    best_m_rolling = rec32
                    best_m_state = deepcopy(macro_actor.state_dict())

                m_hist.append(dict(update = upd, env_steps = m_steps, decisions = m_decs, recent_return_32 = rec32))

                if divisible_by(upd, log_every) or upd == macro_updates:
                    print(f'  [dt={h} upd {upd:3d}/{macro_updates}] decs = {m_decs // 1000:3d}k (env = {m_steps // 1000:4d}k) | ret32 = {rec32:+6.1f} (best = {best_m_rolling:+6.1f}) [{time.time() - t_h:.1f}s]', flush = True)

            macro_env.close()

        macro_actor.load_state_dict(best_m_state)
        m_eval = evaluate_macro_policy(
            macro_actor, table, h, env_name = env_name, continuous = continuous,
            max_episode_steps = max_episode_steps, video_folder = out_folder if render else None,
            video_prefix = f'vq-macro-h{h}-best'
        )

        torch.save(dict(
            hierarchy = h,
            decoded_table = torch.from_numpy(table),
            vq_compressor = vqs[h].state_dict(),
            macro_actor = macro_actor.state_dict(),
            macro_eval = m_eval
        ), out_dir / f'ppo_vq_h{h}.pt')

        with open(out_dir / f'vq_h{h}_returns.csv', 'w', newline = '') as f:
            writer = csv.DictWriter(f, fieldnames = ['update', 'env_steps', 'decisions', 'recent_return_32'])
            writer.writeheader()
            writer.writerows(m_hist)

        print(f'  [dt={h} eval] {m_eval["mean_return"]:+.1f} +/- {m_eval["std_return"]:.1f} (best {m_eval["best_return"]:+.1f}) | land = {m_eval["land_rate"]:.1f}%', flush = True)
        macro_runs.append(dict(h = h, num_codes = num_codes, history = m_hist, eval = m_eval))

    if sweep:
        plot_sweep(base_hist, base_eval, macro_runs, out_dir / 'vq_hierarchy_sweep.png', env_name)

if __name__ == '__main__':
    fire.Fire(main)
