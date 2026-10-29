# /// script
# dependencies = [
#   "einops",
#   "fire",
#   "gymnasium[box2d,other]",
#   "mean-conc-beta>=0.1.5",
#   "moviepy",
#   "numpy",
#   "torch",
#   "torch-einops-utils",
#   "tqdm",
# ]
# ///

from __future__ import annotations

from pathlib import Path
from shutil import rmtree
from copy import deepcopy
from collections import deque, namedtuple

import fire
import numpy as np
from tqdm import tqdm
import gymnasium as gym

import torch
from torch import nn, arange, cat, empty, full, randint, randn, randperm, stack, tensor, zeros
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential, SiLU
from torch.optim import Adam
from torch.distributions import Categorical

from einops import einsum, pack, rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from mean_conc_beta import Beta

from torch_einops_utils import (
    cast_tensor,
    exclusive_cumsum,
    masked_mean,
    shift_right,
    z_score
)

from x_ppo import (
    calc_gae,
    clipped_value_loss_fn,
    dones_to_masks,
    hpo_actor_loss
)

torch.set_num_threads(1)

# memory

Memory = namedtuple('Memory', [
    'state',
    'action',
    'log_prob',
    'reward',
    'value',
    'done',
    'elite_action',
    'planned'
])

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def dynamic_rollout_loss_weights(step_losses, decay = 1.):
    cum_step_losses = exclusive_cumsum(step_losses.clamp(min = 0.), dim = -1)
    return (-decay * cum_step_losses).exp()

def mlp(
    dim_in,
    dim_out,
    dim_hidden = 256,
    depth = 2,
    pre_norm = False,
    post_norm = True,
    zero_init_out = False,
    squeeze_out = False
):
    layers = []

    if pre_norm:
        layers.append(LayerNorm(dim_in))

    dim = dim_in

    for _ in range(depth):
        layers.extend([Linear(dim, dim_hidden), ReLU()])
        dim = dim_hidden

    out_linear = Linear(dim, dim_out)

    if zero_init_out:
        nn.init.zeros_(out_linear.weight)
        nn.init.zeros_(out_linear.bias)

    layers.append(out_linear)

    if post_norm:
        layers.append(LayerNorm(dim_out))

    if squeeze_out:
        layers.append(Rearrange('... 1 -> ...'))

    return Sequential(*layers)

# self-predictive representations (SPR) world model + policy-anchored CEM planner
# Schwarzer et al. - https://arxiv.org/abs/2007.05929
# Teoh et al. - https://arxiv.org/abs/2511.05963

class SPRWorldModel(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim = 64,
        encoder = None,
        encoder_hidden_dim = 256,
        encoder_depth = 2,
        hidden_dim = 256,
        depth = 1,
        action_embed_dim = 16,
        k = 5,
        ema_tau = 0.01,
        recon_coef = 1.,
        reward_coef = 0.5,
        done_coef = 0.5,
        latent_value_coef = 0.5,
        disagreement_penalty = 0.5,
        dynamic_loss_decay = 1.,
        cem_iters = 2,
        cem_elites = 16,
        cem_init_std = 0.75,
        cem_min_std = 0.1,
        plan_temp = 0.5,
        prior_weight = 0.5
    ):
        super().__init__()
        assert k >= 1, 'rollout horizon k must be at least 1'

        self.k = k
        self.action_dim = action_dim
        self.ema_tau = ema_tau

        self.recon_coef = recon_coef
        self.reward_coef = reward_coef
        self.done_coef = done_coef
        self.latent_value_coef = latent_value_coef
        self.disagreement_penalty = disagreement_penalty
        self.dynamic_loss_decay = dynamic_loss_decay

        self.cem_iters = cem_iters
        self.cem_elites = cem_elites
        self.cem_init_std = cem_init_std
        self.cem_min_std = cem_min_std
        self.plan_temp = plan_temp
        self.prior_weight = prior_weight

        if not exists(encoder):
            encoder = mlp(state_dim, latent_dim, encoder_hidden_dim, encoder_depth)

        self.encoder = encoder
        self.target_encoder = deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)

        self.latent_norm = LayerNorm(latent_dim)
        self.action_embed = Sequential(Linear(action_dim, action_embed_dim), SiLU())

        dim_cond = latent_dim + action_embed_dim

        self.predictor = mlp(
            dim_cond,
            latent_dim,
            hidden_dim,
            depth,
            pre_norm = True,
            post_norm = False,
            zero_init_out = True
        )

        self.recon_head = mlp(latent_dim, state_dim, hidden_dim, depth, post_norm = False)
        self.reward_head = mlp(dim_cond, 1, hidden_dim, depth, post_norm = False, squeeze_out = True)
        self.done_head = mlp(dim_cond, 1, hidden_dim, depth, post_norm = False, squeeze_out = True)
        self.value_head = mlp(latent_dim, 1, hidden_dim, depth, post_norm = False, squeeze_out = True)

    @torch.no_grad()
    def update_target(self):
        for online_param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.lerp_(online_param, self.ema_tau)

    def transition(self, z, action):
        action_embed = self.action_embed(action)
        cond = cat((z, action_embed), dim = -1)
        next_z = self.latent_norm(z + self.predictor(cond))
        return next_z, cond

    def rollout(self, z, actions):
        action_embeds = self.action_embed(actions)
        latents, conds = [], []

        for action_embed in action_embeds.unbind(dim = 1):
            cond = cat((z, action_embed), dim = -1)
            z = self.latent_norm(z + self.predictor(cond))
            latents.append(z)
            conds.append(cond)

        return stack(latents, dim = 1), stack(conds, dim = 1)

    def forward(self, states, actions, rewards, dones, returns):
        # states: (b, k + 1, d), actions: (b, k, a), rewards: (b, k), dones: (b, k), returns: (b, k + 1)

        z0 = self.encoder(states[:, 0])

        with torch.no_grad():
            target_latents = self.target_encoder(states[:, 1:])

        pred_latents, conds = self.rollout(z0, actions)

        # latent consistency + state reconstruction loss

        latent_losses = reduce(F.smooth_l1_loss(pred_latents, target_latents, reduction = 'none'), 'b k d -> b k', 'mean')
        recon_losses = reduce(F.smooth_l1_loss(self.recon_head(pred_latents), states[:, 1:], reduction = 'none'), 'b k d -> b k', 'mean')

        loss_weights = dynamic_rollout_loss_weights(latent_losses.detach(), decay = self.dynamic_loss_decay)
        step_losses = (latent_losses + self.recon_coef * recon_losses) * loss_weights

        post_done_mask = (~dones).cumprod(dim = -1).bool()
        pre_done_mask = shift_right(post_done_mask, amount = 1, dim = -1, pad_value = True)

        loss = masked_mean(step_losses, mask = post_done_mask)
        loss = loss + self.recon_coef * F.smooth_l1_loss(self.recon_head(z0), states[:, 0])

        # detached latent value head for epistemic disagreement during planning

        v0_loss = 0.5 * F.mse_loss(self.value_head(z0.detach()), returns[:, 0])
        vk_losses = 0.5 * F.mse_loss(self.value_head(pred_latents.detach()), returns[:, 1:], reduction = 'none')
        loss = loss + self.latent_value_coef * (v0_loss + masked_mean(vk_losses, mask = post_done_mask))

        # reward and termination world model losses

        pred_rewards = self.reward_head(conds)
        pred_done_logits = self.done_head(conds)

        reward_losses = F.smooth_l1_loss(pred_rewards, rewards, reduction = 'none')
        done_losses = F.binary_cross_entropy_with_logits(pred_done_logits, dones.float(), reduction = 'none')
        wm_losses = (self.reward_coef * reward_losses + self.done_coef * done_losses) * loss_weights

        loss = loss + masked_mean(wm_losses, mask = pre_done_mask)
        return loss

    def loss_from_rollout(self, obs, actions, rewards, dones, returns, batch_size = 128):
        num_steps, device = obs.shape[0], obs.device
        if num_steps <= self.k:
            return None

        valid_starts = rearrange((~dones[:num_steps - self.k]).nonzero(), 'n 1 -> n')
        if valid_starts.numel() == 0:
            return None

        count = min(batch_size, valid_starts.numel())
        starts = valid_starts[randint(valid_starts.numel(), (count,), device = device)]
        starts = rearrange(starts, 'b -> b 1')

        state_idx = starts + arange(self.k + 1, device = device)
        action_idx = starts + arange(self.k, device = device)

        return self.forward(
            obs[state_idx],
            actions[action_idx],
            rewards = rewards[action_idx],
            dones = dones[action_idx],
            returns = returns[state_idx]
        )

    def bootstrap_value(self, z, value_fn):
        pred_state = self.recon_head(z)
        state_value = value_fn(pred_state)
        latent_value = self.value_head(z)

        scale = state_value.abs().clamp(min = 10.)
        rel_disagreement = ((state_value - latent_value).abs() / scale).clamp(max = 2.)

        return 0.5 * (state_value + latent_value) - self.disagreement_penalty * rel_disagreement

    @torch.no_grad()
    def plan(
        self,
        state,
        policy_fn,
        value_fn,
        horizon = None,
        num_samples = 128,
        gamma = 0.99,
        deterministic = False
    ):
        device = state.device
        horizon = default(horizon, self.k)
        rem_horizon = horizon - 1
        num_elites = min(self.cem_elites, num_samples)

        z0 = self.encoder(state).detach()
        dist0 = policy_fn(state, z0)

        # step 0 action is sampled from the policy; steps 1..horizon-1 start random and are refined via CEM

        first_actions = dist0.sample((num_samples,))

        if deterministic:
            first_actions[0] = dist0.mean

        prior_score = reduce(dist0.log_prob(first_actions), 'n a -> n', 'mean')

        z0_expanded = repeat(z0, 'd -> n d', n = num_samples)
        z1, cond0 = self.transition(z0_expanded, first_actions)
        r0 = self.reward_head(cond0)
        d0 = self.done_head(cond0).sigmoid()

        if rem_horizon == 0:
            returns = r0 + gamma * (1. - d0) * self.bootstrap_value(z1, value_fn)
            total_scores = z_score(returns) + self.prior_weight * z_score(prior_score)
            elite_idx = total_scores.topk(num_elites).indices
        else:
            cem_mean = zeros(rem_horizon, self.action_dim, device = device)
            cem_std = full((rem_horizon, self.action_dim), self.cem_init_std, device = device)
            discounts = gamma ** arange(horizon + 1, device = device)

            total_scores = None
            elite_idx = None

            for cem_iter in range(self.cem_iters):
                if cem_iter == 0:
                    future_actions = empty(num_samples, rem_horizon, self.action_dim, device = device).uniform_(-1., 1.)
                else:
                    noise = randn(num_samples, rem_horizon, self.action_dim, device = device)
                    future_actions = (cem_mean + cem_std * noise).clamp(-1., 1.)

                    if deterministic:
                        future_actions[0] = cem_mean

                latents, conds = self.rollout(z1, future_actions)

                pred_rewards = self.reward_head(conds)
                pred_dones = self.done_head(conds).sigmoid()
                bootstrap = self.bootstrap_value(latents[:, -1], value_fn)

                all_rewards, _ = pack((r0, pred_rewards, bootstrap), 'n *')
                not_dones, _ = pack((1. - d0, 1. - pred_dones), 'n *')
                alive = F.pad(not_dones.cumprod(dim = -1), (1, 0), value = 1.)

                returns = einsum(all_rewards * alive, discounts, 'n t, t -> n')

                total_scores = z_score(returns) + self.prior_weight * z_score(prior_score)
                elite_idx = total_scores.topk(num_elites).indices

                elite_future = future_actions[elite_idx]
                cem_mean = reduce(elite_future, 'e h a -> h a', 'mean')
                cem_std = elite_future.std(dim = 0).clamp(min = self.cem_min_std)

        elite_scores = total_scores[elite_idx]
        elite_first_actions = first_actions[elite_idx]
        elite_weights = F.softmax(z_score(elite_scores) / self.plan_temp, dim = 0)
        elite_mean_action = einsum(elite_weights, elite_first_actions, 'e, e a -> a')

        if deterministic:
            chosen_action = elite_mean_action
        else:
            chosen_elite = Categorical(probs = elite_weights).sample()
            chosen_action = elite_first_actions[chosen_elite]

        log_prob = dist0.log_prob(chosen_action)
        return chosen_action, log_prob, elite_mean_action

# actor-critic hooked to a detached SPR world model

class ActorCritic(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        bounds,
        latent_dim = 64,
        hidden_dim = 256,
        depth = 2,
        use_latent = True
    ):
        super().__init__()
        self.use_latent = use_latent
        self.critic_encoder = mlp(state_dim, latent_dim, hidden_dim, depth)

        self.state_embed = Sequential(
            Linear(state_dim, latent_dim),
            LayerNorm(latent_dim)
        ) if use_latent else mlp(state_dim, latent_dim, hidden_dim, depth)

        actor_in_dim = latent_dim * 2 if use_latent else latent_dim

        self.actor = Sequential(
            Linear(actor_in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, action_dim * 2),
            Rearrange('... (a c) -> ... a c', c = 2)
        )

        self.beta = Beta(bounds = bounds, init_conc = 2.)

        self.critic = Sequential(
            Linear(latent_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1),
            Rearrange('... 1 -> ...')
        )

    def policy(self, state, latent = None):
        actor_in = cat((latent.detach(), self.state_embed(state)), dim = -1) if self.use_latent else self.state_embed(state)
        return self.beta(self.actor(actor_in))

    def value(self, state):
        return self.critic(self.critic_encoder(state))

    def forward(self, state, spr: SPRWorldModel | None = None):
        latent = spr.encoder(state).detach() if (self.use_latent and exists(spr)) else None
        return self.policy(state, latent), self.value(state)

# evaluation and video recording

@torch.no_grad()
def rollout_episode(
    env,
    agent,
    spr,
    seed,
    device,
    use_planning = True,
    plan_horizon = 5,
    max_timesteps = 1000,
    plan_samples = 128,
    gamma = 0.99
):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)

    state, _ = env.reset(seed = seed)
    episode_return = 0.

    for _ in range(max_timesteps):
        state_t = cast_tensor(state, dtype = torch.float32, device = device)

        if use_planning and exists(spr):
            action, _, _ = spr.plan(
                state_t,
                policy_fn = agent.policy,
                value_fn = agent.value,
                horizon = plan_horizon,
                num_samples = plan_samples,
                gamma = gamma,
                deterministic = True
            )
        else:
            dist, _ = agent(state_t, spr)
            action = dist.mean

        env_action = action.clamp(-1., 1.).cpu().numpy()
        state, reward, terminated, truncated, _ = env.step(env_action)
        episode_return += float(reward)

        if terminated or truncated:
            break

    torch.set_rng_state(rng_state)
    return episode_return

@torch.no_grad()
def evaluate(
    agent,
    spr,
    env_name,
    seed,
    num_episodes,
    device,
    **rollout_kwargs
):
    env = gym.make(env_name, continuous = True)
    returns = []
    best_seed = seed
    best_return = -float('inf')

    for ep in range(num_episodes):
        ep_seed = seed + ep
        ep_return = rollout_episode(env, agent, spr, ep_seed, device, **rollout_kwargs)
        returns.append(ep_return)

        if ep_return > best_return:
            best_return = ep_return
            best_seed = ep_seed

    env.close()
    return float(np.mean(returns)), best_seed

@torch.no_grad()
def record_best_trajectory(
    agent,
    spr,
    env_name,
    video_folder,
    seed,
    device,
    **rollout_kwargs
):
    video_folder = Path(video_folder)
    if video_folder.exists():
        rmtree(video_folder)

    env = gym.make(env_name, continuous = True, render_mode = 'rgb_array')
    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = str(video_folder),
        name_prefix = 'winning-combo',
        episode_trigger = lambda ep: ep == 0,
        disable_logger = True
    )

    episode_return = rollout_episode(env, agent, spr, seed, device, **rollout_kwargs)
    env.close()
    return episode_return

# main

def main(
    env_name = 'LunarLander-v3',
    out_dir = './runs/spr_world_model',
    seed = 0,
    device = 'cpu',
    num_episodes = 800,
    max_timesteps = 1000,
    rollout_steps = 1024,
    minibatch_size = 64,
    epochs = 10,
    lr = 3e-4,
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.3,
    advantage_clip = 1.,
    value_clip = 0.2,
    ent_coef = 0.01,
    vf_coef = 0.5,
    max_grad_norm = 0.5,
    hidden_dim = 256,
    latent_dim = 64,
    depth = 2,
    use_spr = True,
    use_planning = True,
    actor_use_latent = True,
    spr_coef = 1.,
    spr_k = 5,
    plan_horizon = 5,
    spr_batch_size = 128,
    plan_samples = 128,
    rollout_plan_samples = 32,
    plan_rollout_prob = 0.35,
    plan_warmup_episodes = 250,
    wm_actor_coef = 0.05,
    disagreement_penalty = 0.5,
    dynamic_loss_decay = 1.,
    mb_norm_advantages = True,
    eval_every = 50,
    eval_episodes = 5,
    eval_seed_base = 10_000,
    record_video = True
):
    device = torch.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    checkpoint_path = out_dir / 'best_model.pt'
    video_dir = out_dir / 'videos'

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name, continuous = True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    bounds = np.stack([env.action_space.low, env.action_space.high], axis = -1)

    use_planning = use_planning and use_spr
    actor_use_latent = actor_use_latent and use_spr

    spr_encoder = mlp(state_dim, latent_dim, hidden_dim, depth) if use_spr else None

    agent = ActorCritic(
        state_dim = state_dim,
        action_dim = action_dim,
        bounds = bounds,
        latent_dim = latent_dim,
        hidden_dim = hidden_dim,
        depth = depth,
        use_latent = actor_use_latent
    ).to(device)

    spr = SPRWorldModel(
        state_dim = state_dim,
        action_dim = action_dim,
        latent_dim = latent_dim,
        encoder = spr_encoder,
        encoder_hidden_dim = hidden_dim,
        encoder_depth = depth,
        k = spr_k,
        disagreement_penalty = disagreement_penalty,
        dynamic_loss_decay = dynamic_loss_decay
    ).to(device) if use_spr else None

    actor_params = [*agent.actor.parameters(), *agent.state_embed.parameters()]
    critic_params = [*agent.critic_encoder.parameters(), *agent.critic.parameters()]
    spr_params = [p for p in spr.parameters() if p.requires_grad] if exists(spr) else []

    optimizer = Adam([*actor_params, *critic_params, *spr_params], lr = lr, eps = 1e-5)

    rollout_kwargs = dict(
        use_planning = use_planning,
        plan_horizon = plan_horizon,
        max_timesteps = max_timesteps,
        plan_samples = plan_samples,
        gamma = gamma
    )

    state, _ = env.reset(seed = seed)

    episode = 0
    total_steps = 0
    episode_return = 0.
    last_returns = deque(maxlen = 20)
    best_eval = -float('inf')
    best_video_seed = eval_seed_base
    last_eval_episode = -eval_every

    pbar = tqdm(total = num_episodes, desc = 'training')

    while episode < num_episodes:
        memories = []

        for _ in range(rollout_steps):
            with torch.no_grad():
                state_t = cast_tensor(state, dtype = torch.float32, device = device)
                dist, value = agent(state_t, spr)

                use_planner = (
                    use_planning and
                    episode >= plan_warmup_episodes and
                    (np.random.rand() < plan_rollout_prob)
                )

                if use_planner:
                    action, log_prob, elite_action = spr.plan(
                        state_t,
                        policy_fn = agent.policy,
                        value_fn = agent.value,
                        horizon = plan_horizon,
                        num_samples = min(plan_samples, rollout_plan_samples),
                        gamma = gamma,
                        deterministic = False
                    )
                else:
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    elite_action = action

            next_state, raw_reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = bool(terminated or truncated)

            # bootstrap value on time-limit truncations

            bootstrap_reward = float(raw_reward)

            if truncated and not terminated:
                with torch.no_grad():
                    next_state_t = cast_tensor(next_state, dtype = torch.float32, device = device)
                    bootstrap_reward += gamma * float(agent.value(next_state_t).item())

            memories.append(Memory(
                state = state_t,
                action = action,
                log_prob = log_prob,
                reward = tensor(bootstrap_reward, dtype = torch.float32, device = device),
                value = value,
                done = tensor(done, dtype = torch.bool, device = device),
                elite_action = elite_action,
                planned = tensor(use_planner, dtype = torch.bool, device = device)
            ))

            state = next_state
            total_steps += 1
            episode_return += float(raw_reward)

            if done:
                episode += 1
                pbar.update(1)

                last_returns.append(episode_return)
                avg_last = float(np.mean(last_returns))

                if divisible_by(episode, 10):
                    pbar.set_postfix(ep_return = f'{episode_return:.1f}', avg20 = f'{avg_last:.1f}')

                episode_return = 0.
                state, _ = env.reset()

                if episode >= num_episodes:
                    break

        # generalized advantage estimation (GAE)

        obs, actions, old_log_probs, rewards, old_values, dones, elite_actions, planned = map(stack, zip(*memories))
        num_steps = obs.shape[0]

        with torch.no_grad():
            next_state_t = cast_tensor(state, dtype = torch.float32, device = device)
            last_value = agent.value(next_state_t)

            returns, advantages = calc_gae(
                rewards,
                old_values,
                masks = dones_to_masks(dones = dones),
                next_value = last_value,
                gamma = gamma,
                lam = lam,
                use_accelerated = False,
                return_advantages = True
            )

            if not mb_norm_advantages:
                advantages = z_score(advantages)

        # HPO policy, clipped value, and SPR world model updates

        for _ in range(epochs):
            perm = randperm(num_steps, device = device)

            for mb_start in range(0, num_steps, minibatch_size):
                mb_idx = perm[mb_start : mb_start + minibatch_size]
                mb_advantages = z_score(advantages[mb_idx]) if mb_norm_advantages else advantages[mb_idx]

                dist, values = agent(obs[mb_idx], spr)
                new_log_probs = dist.log_prob(actions[mb_idx])
                entropy = reduce(dist.entropy(), 'b a -> b', 'sum').mean()

                policy_loss = hpo_actor_loss(
                    new_log_probs,
                    old_log_probs[mb_idx],
                    advantages[mb_idx],
                    eps_clip = eps_clip,
                    advantage_clip = advantage_clip,
                    per_action_dim = True,
                    normalize_advantages = mb_norm_advantages
                ).mean()

                if use_planning and wm_actor_coef > 0.:
                    pos_plan_mask = planned[mb_idx] & (mb_advantages > 0.)
                    if pos_plan_mask.any():
                        wm_actor_loss = F.smooth_l1_loss(dist.mean[pos_plan_mask], elite_actions[mb_idx][pos_plan_mask])
                        policy_loss = policy_loss + wm_actor_coef * wm_actor_loss

                value_loss = 0.5 * clipped_value_loss_fn(
                    values,
                    old_values[mb_idx],
                    returns[mb_idx],
                    clip = value_clip
                )

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                if exists(spr) and spr_coef > 0.:
                    spr_loss = spr.loss_from_rollout(
                        obs,
                        actions,
                        rewards = rewards,
                        dones = dones,
                        returns = returns,
                        batch_size = spr_batch_size
                    )

                    if exists(spr_loss):
                        loss = loss + spr_coef * spr_loss

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
                nn.utils.clip_grad_norm_(critic_params, max_grad_norm)

                if exists(spr):
                    nn.utils.clip_grad_norm_(spr_params, max_grad_norm)

                optimizer.step()

                if exists(spr):
                    spr.update_target()

        if (episode - last_eval_episode) >= eval_every or episode >= num_episodes:
            eval_return, ep_best_seed = evaluate(
                agent,
                spr,
                env_name,
                eval_seed_base,
                eval_episodes,
                device,
                **rollout_kwargs
            )
            last_eval_episode = episode

            if eval_return > best_eval:
                best_eval = eval_return
                best_video_seed = ep_best_seed
                torch.save(
                    dict(
                        agent = agent.state_dict(),
                        spr = spr.state_dict() if exists(spr) else None,
                        video_seed = best_video_seed
                    ),
                    checkpoint_path
                )

            avg_last = float(np.mean(last_returns)) if len(last_returns) > 0 else 0.
            tqdm.write(f'episode {episode} | step {total_steps} | avg20 {avg_last:.1f} | eval {eval_return:.1f} | best {best_eval:.1f}')

    pbar.close()
    env.close()

    if record_video and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location = device, weights_only = True)
        agent.load_state_dict(ckpt['agent'])

        if exists(spr) and exists(ckpt['spr']):
            spr.load_state_dict(ckpt['spr'])

        video_return = record_best_trajectory(
            agent,
            spr,
            env_name,
            video_dir,
            seed = ckpt['video_seed'],
            device = device,
            **rollout_kwargs
        )
        print(f'recorded best checkpoint trajectory video (return {video_return:.1f}) in {video_dir}')

if __name__ == '__main__':
    fire.Fire(main)
