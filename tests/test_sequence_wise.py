import pytest
import torch
from torch.distributions import Categorical
from torch_einops_utils import lens_to_mask, z_score

from x_ppo import hpo_actor_loss, ppo_actor_loss, spo_actor_loss

param = pytest.mark.parametrize

def sequence_inputs(action_dim = None):
    torch.manual_seed(0)

    shape = (3, 6, action_dim) if action_dim else (3, 6)
    log_probs = torch.randn(*shape)
    old_log_probs = torch.randn(*shape)
    cumulative_rewards = torch.tensor([3., -1., 2.])
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0]
    ]).bool()

    return log_probs, old_log_probs, cumulative_rewards, mask

# sequence wise - GSPO

@param('loss_fn', (ppo_actor_loss, spo_actor_loss, hpo_actor_loss))
@param('per_action_dim', (False, True))
@param('delightful', (False, True))
def test_actor_loss_sequence_wise(loss_fn, per_action_dim, delightful):
    torch.manual_seed(0)

    batch, seq, action_dim = 4, 7, 3
    mask = lens_to_mask(torch.tensor([3, 7, 1, 5]), seq)

    shape = (batch, seq, action_dim) if per_action_dim else (batch, seq)

    action_log_probs = torch.randn(*shape, requires_grad = True)
    old_action_log_probs = torch.randn(*shape)

    for advantages in (torch.randn(batch), torch.randn(batch, seq)):
        loss = loss_fn(
            action_log_probs,
            old_action_log_probs,
            advantages,
            mask = mask,
            normalize_advantages = True,
            delightful = delightful,
            per_action_dim = per_action_dim,
            sequence_wise = True
        )

        assert torch.isfinite(loss)

        loss.backward()
        assert torch.isfinite(action_log_probs.grad).all()

        action_log_probs.grad = None

@param('loss_fn', (ppo_actor_loss, spo_actor_loss, hpo_actor_loss))
def test_actor_loss_sequence_wise_stats_and_skip(loss_fn):
    action_log_probs = torch.randn(3, 4, requires_grad = True)
    old_action_log_probs = torch.randn(3, 4)
    cumulative_rewards = torch.tensor([1., -1., 2.])
    mask = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 0, 0]
    ]).bool()

    loss, stats = loss_fn(
        action_log_probs,
        old_action_log_probs,
        cumulative_rewards,
        mask = mask,
        normalize_advantages = True,
        sequence_wise = True,
        return_stats = True
    )

    expected = z_score(cumulative_rewards, return_only_stats = True)

    assert torch.allclose(stats.mean, expected.mean)
    assert torch.allclose(stats.var, expected.var)

    loss, stats = loss_fn(
        action_log_probs,
        old_action_log_probs,
        torch.ones(3),
        mask = mask,
        min_advantage_std = 1e-3,
        sequence_wise = True,
        return_stats = True
    )

    assert torch.all(loss == 0.)
    assert torch.allclose(stats.var, torch.zeros(1, 1))

@param('loss_fn', (ppo_actor_loss, spo_actor_loss, hpo_actor_loss))
def test_actor_loss_sequence_wise_padding_invariance(loss_fn):
    torch.manual_seed(0)

    batch, seq = 3, 6
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0]
    ]).bool()

    log_probs = torch.randn(batch, seq)
    old_log_probs = torch.randn(batch, seq)
    advantages = torch.randn(batch)

    loss = loss_fn(log_probs, old_log_probs, advantages, mask = mask, sequence_wise = True)

    padded_log_probs = log_probs.clone()
    padded_old_log_probs = old_log_probs.clone()
    padded_log_probs.masked_fill_(~mask, 1e3)
    padded_old_log_probs.masked_fill_(~mask, -1e3)

    padded_loss = loss_fn(padded_log_probs, padded_old_log_probs, advantages, mask = mask, sequence_wise = True)

    assert torch.allclose(loss, padded_loss)

def test_ppo_sequence_wise_matches_reference():
    log_probs, old_log_probs, cumulative_rewards, mask = sequence_inputs()

    eps_low, eps_high = 0.2, 0.3

    loss = ppo_actor_loss(
        log_probs,
        old_log_probs,
        cumulative_rewards,
        eps_clip = (eps_low, eps_high),
        mask = mask,
        normalize_advantages = True,
        sequence_wise = True
    )

    advantages = z_score(cumulative_rewards)
    log_ratios = ((log_probs - old_log_probs) * mask).sum(dim = -1) / mask.sum(dim = -1)
    ratios = log_ratios.exp()

    surr = torch.min(ratios * advantages, ratios.clamp(1. - eps_low, 1. + eps_high) * advantages)
    expected = -surr.mean()

    assert torch.allclose(loss, expected)

def test_spo_sequence_wise_matches_reference():
    log_probs, old_log_probs, cumulative_rewards, mask = sequence_inputs()

    eps_low, eps_high = 0.2, 0.3

    loss = spo_actor_loss(
        log_probs,
        old_log_probs,
        cumulative_rewards,
        eps_clip = (eps_low, eps_high),
        mask = mask,
        normalize_advantages = True,
        sequence_wise = True
    )

    advantages = z_score(cumulative_rewards)
    log_ratios = ((log_probs - old_log_probs) * mask).sum(dim = -1) / mask.sum(dim = -1)
    ratios = log_ratios.exp()
    eps = torch.where(ratios >= 1., eps_high, eps_low)

    surr = ratios * advantages - advantages.abs() * (ratios - 1.).square() / (2 * eps)
    expected = -surr.mean()

    assert torch.allclose(loss, expected)

def test_hpo_sequence_wise_matches_reference():
    log_probs, old_log_probs, cumulative_rewards, mask = sequence_inputs()

    eps_low, eps_high = 0.2, 0.3
    advantage_clip = 2.

    loss = hpo_actor_loss(
        log_probs,
        old_log_probs,
        cumulative_rewards,
        eps_clip = (eps_low, eps_high),
        advantage_clip = advantage_clip,
        mask = mask,
        normalize_advantages = True,
        sequence_wise = True
    )

    advantages = z_score(cumulative_rewards)
    log_ratios = ((log_probs - old_log_probs) * mask).sum(dim = -1) / mask.sum(dim = -1)

    eps = torch.where(advantages >= 0., eps_high, eps_low)
    target = eps * advantages.clamp(-advantage_clip, advantage_clip)
    weight = advantages.abs()

    error = log_ratios - target
    abs_error = error.abs()
    huber = torch.where(abs_error < eps, 0.5 * error.square() / eps, abs_error - 0.5 * eps)

    expected = (weight * huber).mean()

    assert torch.allclose(loss, expected)

def test_ppo_sequence_wise_per_action_dim_matches_reference():
    log_probs, old_log_probs, cumulative_rewards, mask = sequence_inputs(action_dim = 4)

    eps_low, eps_high = 0.2, 0.3

    loss = ppo_actor_loss(
        log_probs,
        old_log_probs,
        cumulative_rewards,
        eps_clip = (eps_low, eps_high),
        mask = mask,
        normalize_advantages = True,
        per_action_dim = True,
        sequence_wise = True
    )

    advantages = z_score(cumulative_rewards)
    log_ratios = ((log_probs - old_log_probs) * mask[..., None]).sum(dim = -2) / mask.sum(dim = -1, keepdim = True)
    ratios = log_ratios.exp()

    surr = torch.min(
        ratios * advantages[..., None],
        ratios.clamp(1. - eps_low, 1. + eps_high) * advantages[..., None]
    )

    expected = -surr.mean(dim = -1).mean()

    assert torch.allclose(loss, expected)


# fast end-to-end convergence on a toy stabilizer

class ToyStabilizerPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2, 2))

    def forward(self, state):
        return self.logits[state]

def collect_stabilizer(policy, batch, horizon = 4):
    # state 0 is upright, 1 is tilted - action 1 balances, action 0 recovers

    state = torch.zeros(batch, dtype = torch.long)
    states, actions, old_log_probs, rewards = [], [], [], []

    for _ in range(horizon):
        dist = Categorical(logits = policy(state))
        action = dist.sample()

        states.append(state)
        actions.append(action)
        old_log_probs.append(dist.log_prob(action).detach())

        upright = state == 0
        next_state = torch.where(upright, (action == 0).long(), (action == 1).long())
        reward = torch.where(upright & (action == 1), 1., torch.where(~upright & (action == 1), -1., 0.))

        rewards.append(reward)
        state = next_state

    return (
        torch.stack(states, dim = 1),
        torch.stack(actions, dim = 1),
        torch.stack(old_log_probs, dim = 1),
        torch.stack(rewards, dim = 1)
    )

@param('loss_fn', (ppo_actor_loss, spo_actor_loss, hpo_actor_loss))
def test_sequence_wise_converges(loss_fn):
    torch.manual_seed(0)

    policy = ToyStabilizerPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr = 0.05)

    for _ in range(300):
        states, actions, old_log_probs, rewards = collect_stabilizer(policy, batch = 64)

        advantages = z_score(rewards.sum(dim = -1))
        log_probs = Categorical(logits = policy(states)).log_prob(actions)

        loss = loss_fn(
            log_probs,
            old_log_probs,
            advantages,
            eps_clip = 0.2,
            sequence_wise = True
        ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = policy(torch.tensor([0, 1])).softmax(dim = -1)

    # upright -> balance, tilted -> recover

    assert probs[0, 1] > 0.9
    assert probs[1, 0] > 0.9
