"""Toy demonstration: train V and Q together with next-latent prediction,
then use Q to filter K = 2 candidate actions.

The task is a short corridor. The state is the normalized position in [0, 1],
the action is -1 (left) or +1 (right), and the episode ends with reward 1 at
the right end. The behavior policy is deliberately imperfect: it moves toward
the goal 70% of the time.

The critic learns
    z' = z + dynamics([z, action_embed(a)])
    r' = reward_head([z, action_embed(a)])
    d' = terminal_head([z, action_embed(a)])
    Q(s, a) = shared_head([z, action_embed(a)])[-1]
with target
    r + gamma * (1 - done) * V(z').detach()

At test time, draw two actions from the behavior policy and keep the one with
the larger Q. That is best-of-2 action filtering.

Next-latent prediction - Teoh et al.,
*Next-Latent Prediction Transformers Learn Compact World Models*,
https://arxiv.org/abs/2511.05963
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from x_ppo import NextLatentQCritic

# toy task

CORRIDOR_LEN = 6
HORIZON = CORRIDOR_LEN
GAMMA = 0.99
P_RIGHT = 0.7
STEP = 1. / CORRIDOR_LEN

def sample_action(num_samples = 1) -> Tensor:
    return torch.where(torch.rand(num_samples) < P_RIGHT, 1., -1.)

def transition(x: float, action: float) -> tuple[float, Tensor, Tensor]:
    next_x = min(max(x + STEP * action, 0.), 1.)
    reward = torch.tensor(float(next_x >= 1. - 1e-6))
    done = reward > 0.5
    return next_x, reward, done

def make_dataset(num_episodes = 2000):
    xs, actions, rewards, next_xs, dones, returns = [], [], [], [], [], []
    rets = []

    for _ in range(num_episodes):
        x = 0.
        episode = []

        for _ in range(HORIZON):
            action = sample_action().item()
            next_x, reward, done = transition(x, action)
            episode.append((x, action, reward.item(), next_x, done.item()))
            x = next_x

            if done:
                break

        ret = 0.
        rets.clear()

        for _, _, reward, _, done in reversed(episode):
            ret = reward + GAMMA * ret * (0. if done else 1.)
            rets.append(ret)

        for (x, action, reward, next_x, done), ret in zip(episode, reversed(rets)):
            xs.append([x])
            actions.append([action])
            rewards.append(reward)
            next_xs.append([next_x])
            dones.append(done)
            returns.append(ret)

    to_tensor = lambda values: torch.tensor(values, dtype = torch.float32)

    return dict(
        embeddings = to_tensor(xs),
        actions = to_tensor(actions),
        rewards = to_tensor(rewards),
        next_embeddings = to_tensor(next_xs),
        dones = to_tensor(dones),
        returns = to_tensor(returns)
    )

def value_net() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 1)
    )

def train_toy_critic(
    num_episodes = 2000,
    train_steps = 1500,
    batch_size = 256,
    lr = 3e-3,
    seed = 0
):
    torch.manual_seed(seed)

    data = make_dataset(num_episodes)
    critic = NextLatentQCritic(
        value_net(),
        dim_embed = 1,
        dim_action = 1,
        action_embed_dim = 16,
        hidden_dim = 128,
        depth = 2,
        gamma = GAMMA
    )

    optimizer = torch.optim.Adam(critic.parameters(), lr = lr)
    num_transitions = data['embeddings'].shape[0]

    for _ in range(train_steps):
        indices = torch.randint(num_transitions, (batch_size,))
        batch = {key: value[indices] for key, value in data.items()}

        loss = critic(
            batch['embeddings'],
            batch['actions'],
            rewards = batch['rewards'],
            next_embeddings = batch['next_embeddings'],
            returns = batch['returns'],
            dones = batch['dones']
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return critic, data

# evaluation

@torch.no_grad()
def evaluate(
    critic: NextLatentQCritic,
    mode = 'filter',              # 'sample' | 'filter' | 'greedy'
    num_candidates = 2,
    num_episodes = 1000,
    seed = 1
):
    torch.manual_seed(seed)

    successes = 0
    total_return = 0.
    total_steps = 0

    for _ in range(num_episodes):
        x = 0.
        episode_return = 0.

        for step in range(HORIZON):
            if mode == 'greedy':
                candidates = torch.tensor([[-1.], [1.]])
                action = critic.filter_actions(torch.tensor([[x]]), candidates[None]).actions[0, 0].item()
            elif mode == 'filter':
                candidates = sample_action(num_candidates)[:, None]
                action = critic.filter_actions(torch.tensor([[x]]), candidates[None]).actions[0, 0].item()
            else:
                action = sample_action().item()

            x, reward, done = transition(x, action)
            episode_return += GAMMA ** step * reward.item()

            if done:
                successes += 1
                break

        total_return += episode_return
        total_steps += step + 1

    return dict(
        success = successes / num_episodes,
        return_ = total_return / num_episodes,
        steps = total_steps / num_episodes
    )

@torch.no_grad()
def q_table(critic: NextLatentQCritic):
    states = torch.arange(CORRIDOR_LEN, dtype = torch.float32)[:, None] / CORRIDOR_LEN
    right = torch.ones_like(states)
    left = -torch.ones_like(states)

    return dict(
        states = states.squeeze(-1),
        values = critic.forward_value(states),
        q_right = critic.q_values(states, right),
        q_left = critic.q_values(states, left),
        pred_right = critic.next_latents(states, right),
        pred_left = critic.next_latents(states, left),
        reward_right = critic.predicted_rewards(states, right),
        reward_left = critic.predicted_rewards(states, left),
        done_right = critic.predicted_dones(states, right),
        done_left = critic.predicted_dones(states, left)
    )

def main():
    critic, data = train_toy_critic()

    print()
    print(f'transitions: {data["embeddings"].shape[0]}')
    print()
    print('state      V(s)   Q(s, left)  Q(s, right)  argmax    z\' left   z\' right')
    print('--------------------------------------------------------------------------')

    table = q_table(critic)
    for i in range(CORRIDOR_LEN):
        q_left = table['q_left'][i].item()
        q_right = table['q_right'][i].item()
        chosen = 'right' if q_right > q_left else 'left'

        print(
            f'{table["states"][i].item():5.2f}  '
            f'{table["values"][i].item():7.3f}  '
            f'{q_left:10.3f}  '
            f'{q_right:11.3f}  '
            f'{chosen:>7}  '
            f'{table["pred_left"][i].item():8.3f}  '
            f'{table["pred_right"][i].item():8.3f}'
        )

    print()
    print('world model heads:')
    print('state   r\' left  r\' right   p(done) left  p(done) right')
    print('--------------------------------------------------------')

    for i in range(CORRIDOR_LEN):
        print(
            f'{table["states"][i].item():5.2f}  '
            f'{table["reward_left"][i].item():8.3f}  '
            f'{table["reward_right"][i].item():9.3f}  '
            f'{table["done_left"][i].item():13.3f}  '
            f'{table["done_right"][i].item():14.3f}'
        )

    print()
    print('action filtering on the corridor (1000 episodes):')
    print('-------------------------------------------------')

    for mode, label in (('sample', 'policy sample (K = 1)'), ('filter', 'Q filter (K = 2)'), ('greedy', 'greedy Q (K = 2 exhaustive)')):
        stats = evaluate(critic, mode = mode)
        print(f'{label:28} success {stats["success"]:.3f}   return {stats["return_"]:.3f}   steps {stats["steps"]:.2f}')

if __name__ == '__main__':
    main()
