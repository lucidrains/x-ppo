import json
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from mean_conc_beta import Beta

from x_ppo import ppo_actor_loss, hpo_actor_loss, calc_gae

# keep runs single threaded and comparable

torch.set_num_threads(1)


def mlp(n_in, n_out, hidden = 64):
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, n_out)
    )


class Actor(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, continuous, bounds = None):
        super().__init__()
        self.continuous = continuous
        self.net = mlp(obs_dim, action_dim * 2 if continuous else action_dim)
        self.beta = Beta(bounds = bounds, init_conc = 2.) if continuous else None

    def forward(self, state, perturb_mean_scale = 0.):
        out = self.net(state)

        if not self.continuous:
            return Categorical(logits = out)

        params = out.view(*state.shape[:-1], -1, 2)
        return self.beta(params, perturb_mean_scale = perturb_mean_scale)


def actor_loss(
    method,
    log_probs,
    old_log_probs,
    advantages,
    eps_clip,
    advantage_clip = 1.,
    per_action_dim = False
):
    if method == 'ppo':
        return ppo_actor_loss(
            log_probs,
            old_log_probs,
            advantages,
            eps_clip = eps_clip,
            per_action_dim = per_action_dim
        )

    return hpo_actor_loss(
        log_probs,
        old_log_probs,
        advantages,
        eps_clip = eps_clip,
        advantage_clip = advantage_clip,
        per_action_dim = per_action_dim
    )


@torch.no_grad()
def evaluate(actor, eval_seeds, max_steps = 500, env_name = 'CartPole-v1'):
    env = gym.make(env_name)
    returns = []

    continuous = isinstance(env.action_space, gym.spaces.Box)
    action_low = torch.as_tensor(env.action_space.low, dtype = torch.float32) if continuous else None
    action_high = torch.as_tensor(env.action_space.high, dtype = torch.float32) if continuous else None

    for eval_seed in eval_seeds:
        obs, _ = env.reset(seed = eval_seed)
        episode_return = 0.

        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype = torch.float32)
            dist = actor(obs_t)

            if continuous:
                action = dist.mean.clamp(action_low, action_high).numpy()
            else:
                action = dist.probs.argmax().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward

            if terminated or truncated:
                break

        returns.append(episode_return)

    env.close()
    return float(np.mean(returns))


def train(
    method,
    seed,
    env_name = 'CartPole-v1',
    solve_return = 475.,
    total_steps = 300_000,
    rollout = 2_048,
    minibatch = 64,
    epochs = 10,
    lr = 3e-4,
    gamma = 0.99,
    lam = 0.95,
    eps_clip = 0.2,
    advantage_clip = 1.,
    per_action_dim = False,
    rpo_alpha = 0.,
    value_coef = 0.5,
    entropy_coef = 0.01,
    grad_clip = 0.5,
    eval_every = 5,
    eval_episodes = 10,
    checkpoint = None
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    max_steps = env.spec.max_episode_steps or 1_000

    continuous = isinstance(env.action_space, gym.spaces.Box)

    if rpo_alpha > 0. and not continuous:
        raise ValueError('rpo_alpha is only supported for continuous action spaces')

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if continuous else env.action_space.n
    bounds = np.stack([env.action_space.low, env.action_space.high], axis = -1) if continuous else None

    actor = Actor(obs_dim, action_dim, continuous, bounds)
    critic = mlp(obs_dim, 1)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr = lr)

    eval_seeds = [10_000_000 + seed * 1_000 + i for i in range(eval_episodes)]

    obs, _ = env.reset(seed = seed)
    episode_return = 0.
    curve = []
    diagnostics = []
    max_abs_log_ratio = 0.
    clipped_fraction = 0.
    start_time = time.time()

    updates = int(np.ceil(total_steps / rollout))

    for update in range(updates):
        states, actions, old_log_probs, rewards, values, masks = [], [], [], [], [], []
        update_returns = []

        for _ in range(rollout):
            state = torch.as_tensor(obs, dtype = torch.float32)
            dist = actor(state)
            action = dist.sample()

            with torch.no_grad():
                value = critic(state).item()

            env_action = action.numpy() if continuous else action.item()
            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            states.append(state)
            actions.append(action)

            log_prob = dist.log_prob(action)

            if continuous and not per_action_dim:
                log_prob = log_prob.sum(-1)

            old_log_probs.append(log_prob.detach())

            rewards.append(reward)
            values.append(value)
            masks.append(1. - float(terminated))

            episode_return += reward

            if terminated or truncated:
                update_returns.append(episode_return)
                episode_return = 0.
                next_obs, _ = env.reset()

            obs = next_obs

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        rewards = torch.as_tensor(rewards, dtype = torch.float32)
        values = torch.as_tensor(values, dtype = torch.float32)
        masks = torch.as_tensor(masks, dtype = torch.float32)

        with torch.no_grad():
            next_state = torch.as_tensor(obs, dtype = torch.float32)
            last_value = critic(next_state).item()

            gae = calc_gae(
                rewards,
                values,
                masks = masks,
                next_value = last_value,
                gamma = gamma,
                lam = lam,
                use_accelerated = False,
                return_advantages = True
            )

            advantages = (gae.advantages - gae.advantages.mean()) / (gae.advantages.std() + 1e-8)
            returns = gae.returns

        num_samples = rollout
        update_max_log_ratio = 0.
        update_clipped = 0.
        update_grad_norm = 0.
        update_entropy = 0.
        update_value_loss = 0.

        for _ in range(epochs):
            indices = torch.randperm(num_samples)

            for start in range(0, num_samples, minibatch):
                batch_indices = indices[start:start + minibatch]
                state = states[batch_indices]
                action = actions[batch_indices]
                old_log_prob = old_log_probs[batch_indices]
                advantage = advantages[batch_indices]
                ret = returns[batch_indices]

                dist = actor(state, perturb_mean_scale = rpo_alpha)
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                if continuous:
                    if not per_action_dim:
                        log_prob = log_prob.sum(-1)

                    entropy = entropy.sum(-1)

                policy_loss = actor_loss(
                    method,
                    log_prob,
                    old_log_prob,
                    advantage,
                    eps_clip,
                    advantage_clip = advantage_clip,
                    per_action_dim = per_action_dim
                ).mean()

                with torch.no_grad():
                    old_value = critic(state).view(-1)

                value = critic(state).view(-1)
                clipped_value = value.clamp(old_value - eps_clip, old_value + eps_clip)
                value_loss = torch.max(F.mse_loss(value, ret, reduction = 'none'), F.mse_loss(clipped_value, ret, reduction = 'none')).mean()

                entropy = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip) if grad_clip > 0 else torch.nn.utils.clip_grad_norm_(params, float('inf'))
                opt.step()

                with torch.no_grad():
                    log_ratio = log_prob - old_log_prob

                    if per_action_dim and continuous:
                        log_ratio = log_ratio.mean(-1)

                    update_max_log_ratio = max(update_max_log_ratio, log_ratio.abs().max().item())
                    update_clipped += ((log_ratio.abs() > eps_clip).float().mean().item())
                    update_grad_norm += grad_norm.item()
                    update_entropy += entropy.item()
                    update_value_loss += value_loss.item()

        update_clipped /= epochs * int(np.ceil(num_samples / minibatch))
        update_grad_norm /= epochs * int(np.ceil(num_samples / minibatch))
        update_entropy /= epochs * int(np.ceil(num_samples / minibatch))
        update_value_loss /= epochs * int(np.ceil(num_samples / minibatch))

        max_abs_log_ratio = max(max_abs_log_ratio, update_max_log_ratio)
        clipped_fraction += update_clipped

        diagnostics.append({
            'step': (update + 1) * rollout,
            'train_return': float(np.mean(update_returns)) if update_returns else None,
            'max_abs_log_ratio': update_max_log_ratio,
            'clipped_fraction': update_clipped,
            'grad_norm': update_grad_norm,
            'entropy': update_entropy,
            'value_loss': update_value_loss
        })

        if (update + 1) % eval_every == 0:
            eval_return = evaluate(actor, eval_seeds, max_steps, env_name)
            curve.append({
                'step': (update + 1) * rollout,
                'return': eval_return,
                'wall_sec': time.time() - start_time
            })

    env.close()

    if checkpoint is not None:
        torch.save(actor.state_dict(), checkpoint)

    curve_returns = [entry['return'] for entry in curve]
    solved = [entry['step'] for entry in curve if entry['return'] >= solve_return]
    first_solved_step = solved[0] if solved else None

    return {
        'method': method,
        'seed': seed,
        'config': {
            'env_name': env_name,
            'policy': 'beta' if continuous else 'categorical',
            'solve_return': solve_return,
            'total_steps': updates * rollout,
            'rollout': rollout,
            'minibatch': minibatch,
            'epochs': epochs,
            'lr': lr,
            'gamma': gamma,
            'lam': lam,
            'eps_clip': eps_clip,
            'advantage_clip': advantage_clip,
            'per_action_dim': per_action_dim,
            'rpo_alpha': rpo_alpha,
            'value_coef': value_coef,
            'entropy_coef': entropy_coef,
            'grad_clip': grad_clip,
            'eval_every': eval_every,
            'eval_episodes': eval_episodes
        },
        'curve': curve,
        'diagnostics': diagnostics,
        'best': max(curve_returns) if curve_returns else None,
        'final': curve_returns[-1] if curve_returns else None,
        'final_mean': float(np.mean(curve_returns[-3:])) if curve_returns else None,
        'first_solved_step': first_solved_step,
        'post_solve_min': min(
            entry['return'] for entry in curve if first_solved_step is not None and entry['step'] >= first_solved_step
        ) if first_solved_step is not None else None,
        'max_abs_log_ratio': max_abs_log_ratio,
        'mean_clipped_fraction': clipped_fraction / len(diagnostics),
        'wall_sec': time.time() - start_time
    }


def summarize(results, out_dir):
    summary = {}

    for method in sorted(set(result['method'] for result in results)):
        method_results = [result for result in results if result['method'] == method]
        finals = [result['final'] for result in method_results]
        bests = [result['best'] for result in method_results]
        solve_steps = [result['first_solved_step'] for result in method_results if result['first_solved_step'] is not None]
        post_solve_mins = [result['post_solve_min'] for result in method_results if result['post_solve_min'] is not None]

        summary[method] = {
            'runs': len(method_results),
            'solved': len(solve_steps),
            'first_solved_step_median': float(np.median(solve_steps)) if solve_steps else None,
            'final_mean': float(np.mean(finals)),
            'final_std': float(np.std(finals)),
            'final_min': float(np.min(finals)),
            'best_mean': float(np.mean(bests)),
            'post_solve_min': float(np.min(post_solve_mins)) if post_solve_mins else None,
            'max_abs_log_ratio': float(np.max([result['max_abs_log_ratio'] for result in method_results])),
            'mean_clipped_fraction': float(np.mean([result['mean_clipped_fraction'] for result in method_results]))
        }

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent = 2))
    return summary


def plot_curves(results, path, env_name = 'CartPole-v1', solve_return = 475., eval_episodes = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize = (7.2, 4.2))
    colors = {'ppo': '#d62728', 'hpo': '#1f77b4'}

    for method in sorted(set(result['method'] for result in results)):
        method_results = [result for result in results if result['method'] == method]
        max_len = max(len(result['curve']) for result in method_results)
        returns = np.full((len(method_results), max_len), np.nan)

        for i, result in enumerate(method_results):
            returns[i, :len(result['curve'])] = [entry['return'] for entry in result['curve']]

        mean = np.nanmean(returns, axis = 0)
        std = np.nanstd(returns, axis = 0)
        steps = np.array([entry['step'] for entry in method_results[0]['curve']])[:max_len]

        ax.plot(steps, mean, color = colors.get(method, None), lw = 2., label = method)
        ax.fill_between(steps, mean - std, mean + std, color = colors.get(method, None), alpha = 0.16)

    ax.axhline(solve_return, color = 'black', lw = 0.8, ls = ':', label = 'solve threshold')
    ax.set_xlabel('environment steps')
    ax.set_ylabel(f'eval return ({eval_episodes} episodes)' if eval_episodes else 'eval return')
    ax.set_title(f'{env_name} stability - mean +/- std across seeds')
    ax.legend(frameon = False)
    ax.grid(alpha = 0.2)
    fig.tight_layout()
    fig.savefig(path, dpi = 180, bbox_inches = 'tight')
    plt.close(fig)


def run(
    env_name = 'CartPole-v1',
    out = None,
    methods = ('hpo',),
    seeds = (0,),
    steps = 300_000,
    rollout = 2_048,
    minibatch = 64,
    epochs = 10,
    lr = 3e-4,
    eps = 0.2,
    advantage_clip = 1.,
    per_action_dim = False,
    rpo_alpha = 0.,
    grad_clip = 0.5,
    eval_every = 5,
    eval_episodes = 10,
    solve_return = None,
    plot = False,
    save_models = False
):
    methods = (methods,) if isinstance(methods, str) else methods
    seeds = (seeds,) if isinstance(seeds, int) else seeds
    out = out or ('lunarlander-hpo' if 'LunarLander' in env_name else 'cartpole-hpo')
    solve_return = solve_return if solve_return is not None else (200. if 'LunarLander' in env_name else 475.)

    assert all(method in ('ppo', 'hpo') for method in methods), 'methods must be ppo or hpo'

    out_dir = Path(out)
    out_dir.mkdir(parents = True, exist_ok = True)

    results = []

    for method in methods:
        for seed in seeds:
            result = train(
                method,
                seed,
                env_name = env_name,
                solve_return = solve_return,
                total_steps = steps,
                rollout = rollout,
                minibatch = minibatch,
                epochs = epochs,
                lr = lr,
                eps_clip = eps,
                advantage_clip = advantage_clip,
                per_action_dim = per_action_dim,
                rpo_alpha = rpo_alpha,
                grad_clip = grad_clip,
                eval_every = eval_every,
                eval_episodes = eval_episodes,
                checkpoint = out_dir / f'{method}_s{seed}.pt' if save_models else None
            )
            results.append(result)

            path = out_dir / f'{method}_s{seed}.json'
            path.write_text(json.dumps(result, indent = 2))

            print(
                f"{method:6s} seed {seed:2d}  final {result['final']:+7.1f}  "
                f"best {result['best']:+7.1f}  max |log r| {result['max_abs_log_ratio']:.3f}  "
                f"({result['wall_sec']:.0f}s)"
            )

    summary = summarize(results, out_dir)

    print()
    for method, stats in summary.items():
        print(
            f"{method:6s}  solved {stats['solved']}/{stats['runs']}  "
            f"final {stats['final_mean']:+7.1f} +/- {stats['final_std']:.1f}  "
            f"min final {stats['final_min']:+7.1f}  max |log r| {stats['max_abs_log_ratio']:.3f}"
        )

    if plot:
        path = out_dir / 'curves.png'
        plot_curves(results, path, env_name, solve_return, eval_episodes)
        print(f'\nwrote {path}')


if __name__ == '__main__':
    fire.Fire(run)
