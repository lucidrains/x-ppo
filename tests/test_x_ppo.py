import copy
import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch import cat, nn, tensor
from torch.distributions import Bernoulli, Categorical, Distribution, Normal, OneHotCategorical, kl_divergence
from torch_einops_utils import lens_to_mask, z_score

from x_ppo import (
    DaggerDistiller,
    kl_loss,
    mse_loss,
    nll_loss,
    ppo_actor_loss,
    spo_actor_loss,
    hpo_actor_loss,
    calc_gae,
    calc_returns,
    combine_grouped_advantages,
    dones_to_masks,
    clipped_value_loss_fn,
    improved_value_clipping_fn
)
from x_ppo.dagger_distill import (
    DISCRETE_DISTRIBUTIONS,
    best_action,
    is_discrete,
    log_prob_sum,
    policy_kind,
    to_log_prob_target,
)

param = pytest.mark.parametrize

# helpers

def exists(val):
    return val is not None

@param('loss_fn', (ppo_actor_loss, spo_actor_loss, partial(spo_actor_loss, asymmetric = True)))
@param('eps_clip', (0.2, (0.2, 0.3)))
@param('dual_clip', (False, True))
@param('normalize_advantages', (False, True))
@param('delightful', (False, True))
@param('log_ratio_clamp', (None, (-20., 20.)))
@param('use_mask', (False, True))
def test_actor_loss(
    loss_fn,
    eps_clip,
    dual_clip,
    normalize_advantages,
    delightful,
    log_ratio_clamp,
    use_mask
):
    batch, seq = 2, 4

    rewards = torch.randn(batch, seq)
    values = torch.randn(batch, seq)
    mask = lens_to_mask(torch.tensor([2, 4]), seq) if use_mask else None

    returns, advantages = calc_gae(
        rewards,
        values,
        masks = mask,
        return_advantages = True,
        use_accelerated = False
    )

    action_log_probs = torch.randn(batch, seq, requires_grad = True)
    old_action_log_probs = torch.randn(batch, seq)

    loss = loss_fn(
        action_log_probs,
        old_action_log_probs,
        advantages,
        eps_clip = eps_clip,
        dual_clip = dual_clip,
        mask = mask,
        normalize_advantages = normalize_advantages,
        delightful = delightful,
        log_ratio_clamp = log_ratio_clamp
    )

    loss.sum().backward()

    assert exists(action_log_probs.grad)

@param('loss_fn', (ppo_actor_loss, spo_actor_loss, hpo_actor_loss))
def test_actor_loss_stats_and_skip(loss_fn):
    action_log_probs = torch.randn(2, 4, requires_grad = True)
    old_action_log_probs = torch.randn(2, 4)
    advantages = torch.tensor([
        [1., -1., 2., 4.],
        [0.5, -0.5, 1., -2.]
    ])
    mask = torch.tensor([
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]
    ]).bool()

    loss, stats = loss_fn(
        action_log_probs,
        old_action_log_probs,
        advantages,
        mask = mask,
        normalize_advantages = True,
        return_stats = True
    )

    expected = z_score(advantages, mask = mask, return_only_stats = True)

    assert torch.allclose(stats.mean, expected.mean)
    assert torch.allclose(stats.var, expected.var)

    loss, stats = loss_fn(
        action_log_probs,
        old_action_log_probs,
        torch.ones(2, 4),
        min_advantage_std = 1e-3,
        return_stats = True
    )

    assert torch.all(loss == 0.)
    assert torch.allclose(stats.var, torch.zeros(1, 1))

def test_log_ratio_clamp_prevents_overflow():
    action_log_probs = torch.tensor([200., -200.], requires_grad = True)
    old_action_log_probs = torch.zeros(2)
    advantages = torch.tensor([-1., 1.])

    loss = ppo_actor_loss(action_log_probs, old_action_log_probs, advantages, log_ratio_clamp = (-20., 20.))

    assert torch.isfinite(loss).all()

    loss.sum().backward()

    assert torch.isfinite(action_log_probs.grad).all()

def grad_magnitude(loss_fn, log_ratios, advantages):
    x = log_ratios.clone().requires_grad_()
    loss = loss_fn(x, torch.zeros_like(x), advantages).sum()
    return torch.autograd.grad(loss, x)[0].abs()

def test_hpo():
    # quadratic gate below 1 sigma, linear above at init

    advantages = torch.tensor([0.2, -0.4, 3., -3.])
    log_probs = torch.zeros(4, requires_grad = True)

    hpo_actor_loss(log_probs, torch.zeros(4), advantages, eps_clip = 0.2).sum().backward()

    assert torch.allclose(log_probs.grad, -advantages * advantages.abs().clamp(max = 1.))

    # fixed target inside trust band, linear restore outside

    log_ratios = torch.tensor([0.12, -0.12, 0.5, -0.5], requires_grad = True)
    hpo_actor_loss(log_ratios, torch.zeros(4), torch.tensor([0.3, -0.3, 0.3, -0.3]), eps_clip = 0.2).sum().backward()

    assert torch.allclose(log_ratios.grad, torch.tensor([0.09, -0.09, 0.3, -0.3]))

    advantages = torch.tensor([10., -10.])
    log_probs = torch.tensor([-5., 5.], requires_grad = True)

    hpo_actor_loss(log_probs, torch.zeros(2), advantages).sum().backward()

    assert torch.allclose(log_probs.grad.abs(), advantages.abs())

    # asymmetric clip and Huber transition

    eps_low, eps_high = 0.2, 0.28
    log_ratios = torch.tensor([2 * eps_high, -2 * eps_low], requires_grad = True)

    hpo_actor_loss(log_ratios, torch.zeros(2), torch.tensor([1., -1.]), eps_clip = (eps_low, eps_high)).sum().backward()

    assert torch.allclose(log_ratios.grad, torch.tensor([1., -1.]))

    # gradient decays smoothly to the target, PPO clips

    eps = 0.2
    distances = torch.tensor([0., 0.05, 0.1, 0.15, 0.2, 0.5, 1.])
    log_ratios = eps - distances
    ones = torch.ones_like(log_ratios)

    hpo = grad_magnitude(partial(hpo_actor_loss, eps_clip = eps, min_advantage_std = None), log_ratios, ones)
    ppo = grad_magnitude(partial(ppo_actor_loss, eps_clip = eps), log_ratios, ones)

    assert hpo[0] < 1e-6 and torch.all(hpo[1:] >= hpo[:-1] - 1e-6) and torch.allclose(hpo[4], torch.ones(()))
    assert ppo[0] < 1e-6 and ppo[1] > 1. and ppo[1] > ppo[4]

    # clamp guard and clamp gradient

    with pytest.raises(ValueError):
        hpo_actor_loss(torch.zeros(1), torch.zeros(1), torch.ones(1), log_ratio_clamp = (-0.2, 0.2))

    log_probs = torch.tensor([5., -5.], requires_grad = True)

    hpo_actor_loss(log_probs, torch.zeros(2), torch.tensor([1., -1.]), log_ratio_clamp = (-1., 1.)).sum().backward()

    assert torch.allclose(log_probs.grad, torch.tensor([1., -1.]))

    # advantage_clip saturates the target, asymmetric low / high

    log_ratios = torch.tensor([0.25, -0.25], requires_grad = True)

    hpo_actor_loss(log_ratios, torch.zeros(2), torch.tensor([1.5, -1.5]), eps_clip = 0.2, advantage_clip = (1., 2.)).sum().backward()

    assert torch.allclose(log_ratios.grad, torch.full((2,), -0.375))

    # zero variance advantages are skipped by default

    zero_var_loss = hpo_actor_loss(torch.zeros(4, requires_grad = True), torch.zeros(4), torch.ones(4))

    assert torch.all(zero_var_loss == 0.)

@param('loss_fn', (ppo_actor_loss, spo_actor_loss, hpo_actor_loss))
def test_actor_loss_fp32(loss_fn):
    action_log_probs = torch.randn(2, 4)
    old_action_log_probs = torch.randn(2, 4)
    advantages = torch.randn(2, 4)

    with torch.autocast(device_type = 'cpu', dtype = torch.bfloat16):
        loss = loss_fn(action_log_probs, old_action_log_probs, advantages)

    assert loss.dtype == torch.float32

    loss = loss_fn(action_log_probs.half(), old_action_log_probs.half(), advantages.half())

    assert loss.dtype == torch.float32

# GAE vs Sequential Ground Truth

def sequential_gae(rewards, values, masks, gamma = 0.99, lam = 0.95, next_val = 0.):
    b, t = rewards.shape
    advantages = torch.zeros(b, t)
    last_gae = torch.zeros(b)

    for step in reversed(range(t)):
        v_next = next_val if step == t - 1 else values[:, step + 1]
        delta = rewards[:, step] + gamma * v_next * masks[:, step] - values[:, step]
        last_gae = delta + gamma * lam * masks[:, step] * last_gae
        advantages[:, step] = last_gae

    returns = advantages + values
    return returns, advantages

def test_gae_matches_sequential():
    b, t = 3, 10
    rewards = torch.randn(b, t)
    values = torch.randn(b, t)
    masks = torch.tensor([
        [1., 1., 1., 0., 1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 1., 0., 1., 1., 0.]
    ])
    gamma, lam = 0.99, 0.95

    expected_returns, expected_advantages = sequential_gae(rewards, values, masks, gamma, lam)

    out = calc_gae(rewards, values, masks, gamma = gamma, lam = lam, use_accelerated = False, return_advantages = True)

    assert torch.allclose(out.returns, expected_returns, atol = 1e-5)
    assert torch.allclose(out.advantages, expected_advantages, atol = 1e-5)

# Shapes: 1D, 2D, 3D

@param('shape', [(8,), (2, 8), (2, 3, 8)])
def test_gae_shapes(shape):
    rewards = torch.randn(*shape)
    values = torch.randn(*shape)
    masks = torch.ones(*shape)

    returns = calc_gae(rewards, values, masks, use_accelerated = False)
    assert returns.shape == rewards.shape

    out = calc_gae(rewards, values, masks, use_accelerated = False, return_advantages = True)
    assert out.returns.shape == rewards.shape
    assert out.advantages.shape == rewards.shape

# Bootstrapping modes

def test_gae_t_plus_one_values():
    b, t = 2, 6
    rewards = torch.randn(b, t)
    values = torch.randn(b, t + 1)
    masks = torch.ones(b, t)

    returns, advantages = calc_gae(rewards, values, masks, use_accelerated = False, return_advantages = True)

    expected_returns, expected_advantages = calc_gae(
        rewards, values[:, :-1], masks, next_value = values[:, -1], use_accelerated = False, return_advantages = True
    )

    assert torch.allclose(returns, expected_returns, atol = 1e-5)
    assert torch.allclose(advantages, expected_advantages, atol = 1e-5)

def test_gae_scalar_and_tensor_next_value():
    b, t = 2, 5
    rewards = torch.randn(b, t)
    values = torch.randn(b, t)
    masks = torch.ones(b, t)

    ret_scalar = calc_gae(rewards, values, masks, next_value = 2.5, use_accelerated = False)

    next_val_tensor = torch.full((b,), 2.5)
    ret_tensor = calc_gae(rewards, values, masks, next_value = next_val_tensor, use_accelerated = False)

    assert torch.allclose(ret_scalar, ret_tensor, atol = 1e-5)

def test_gae_explicit_next_values():
    b, t = 2, 5
    rewards = torch.randn(b, t)
    values = torch.randn(b, t)
    next_values = torch.randn(b, t)
    masks = torch.ones(b, t)

    returns, advantages = calc_gae(
        rewards, values, masks, next_values = next_values, use_accelerated = False, return_advantages = True
    )

    delta = rewards + 0.99 * next_values * masks - values
    assert torch.allclose(advantages[:, -1], delta[:, -1], atol = 1e-5)

# Dones vs Masks

def test_gae_dones():
    b, t = 2, 6
    rewards = torch.randn(b, t)
    values = torch.randn(b, t)
    dones = torch.tensor([[False, False, True, False, False, True], [False, True, False, False, False, False]])

    ret_dones = calc_gae(rewards, values, masks = dones_to_masks(dones), use_accelerated = False)
    ret_masks = calc_gae(rewards, values, masks = (~dones).float(), use_accelerated = False)

    assert torch.allclose(ret_dones, ret_masks, atol = 1e-5)

# Gymnasium Terminated vs Truncated Semantics

def test_gae_terminated_vs_truncated():
    b, t = 1, 4
    rewards = torch.zeros(b, t)
    values = torch.ones(b, t)
    next_values = torch.full((b, t), 10.0)

    # Step 1 is terminated
    terminated = torch.tensor([[False, True, False, False]])
    truncated = torch.tensor([[False, False, False, False]])

    term_mask, done_mask = dones_to_masks(terminated, truncated)
    out_term = calc_gae(
        rewards, values, masks = term_mask, done_masks = done_mask,
        next_values = next_values, gamma = 1.0, lam = 1.0, use_accelerated = False, return_advantages = True
    )
    assert torch.allclose(out_term.advantages[0, 1], torch.tensor(-1.0), atol = 1e-5)

    # Step 1 is truncated
    terminated_f = torch.tensor([[False, False, False, False]])
    truncated_t = torch.tensor([[False, True, False, False]])

    term_mask_t, done_mask_t = dones_to_masks(terminated_f, truncated_t)
    out_trunc = calc_gae(
        rewards, values, masks = term_mask_t, done_masks = done_mask_t,
        next_values = next_values, gamma = 1.0, lam = 1.0, use_accelerated = False, return_advantages = True
    )
    assert torch.allclose(out_trunc.advantages[0, 1], torch.tensor(9.0), atol = 1e-5)

# Grouped Advantages (Host-pytorch use case)

def test_grouped_advantages():
    # group_dim = 1
    b, g, n = 2, 3, 8
    rewards = torch.randn(b, g, n)
    values = torch.randn(b, g, n)
    masks = torch.ones(b, n)
    masks[:, -1] = 0.

    returns, advantages = calc_gae(
        rewards, values, masks = masks, use_accelerated = False, return_advantages = True
    )
    assert returns.shape == (b, g, n)
    assert advantages.shape == (b, g, n)

    weights = torch.tensor([0.5, 0.3, 0.2])
    combined = combine_grouped_advantages(
        advantages, weights = weights, normalize = True, group_dim = 1
    )
    assert combined.shape == (b, n)

    out_with_weights = calc_gae(
        rewards, values, masks = masks, weights = weights, group_dim = 1, use_accelerated = False, return_advantages = True
    )
    assert out_with_weights.advantages.shape == (b, n)

    # group_dim = 0
    rewards_g0 = torch.randn(g, b, n)
    values_g0 = torch.randn(g, b, n)
    masks_g0 = torch.ones(b, n)

    combined_g0 = combine_grouped_advantages(
        rewards_g0, weights = weights, normalize = True, group_dim = 0
    )
    assert combined_g0.shape == (b, n)

    out_g0 = calc_gae(
        rewards_g0, values_g0, masks = masks_g0, weights = weights, group_dim = 0, use_accelerated = False, return_advantages = True
    )
    assert out_g0.advantages.shape == (b, n)

# Variable lengths

def test_variable_lengths():
    lens = torch.tensor([3, 5])
    rewards = torch.randn(2, 5)
    values = torch.randn(2, 5)

    out = calc_gae(rewards, values, lens = lens, use_accelerated = False, return_advantages = True)

    assert torch.all(out.returns[0, 3:] == 0.)
    assert torch.all(out.advantages[0, 3:] == 0.)

    expected_out = calc_gae(rewards[:1, :3], values[:1, :3], use_accelerated = False, return_advantages = True)
    assert torch.allclose(out.returns[:1, :3], expected_out.returns, atol = 1e-5)
    assert torch.allclose(out.advantages[:1, :3], expected_out.advantages, atol = 1e-5)

# Bare returns

def test_calc_returns():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    gamma = 0.9

    expected = torch.tensor([[5.23, 4.7, 3.0]])

    returns = calc_returns(rewards, gamma = gamma, use_accelerated = False)
    assert torch.allclose(returns, expected, atol = 1e-5)

    returns_gae = calc_gae(rewards, values = torch.zeros_like(rewards), gamma = gamma, lam = 1.0, use_accelerated = False)
    assert torch.allclose(returns_gae, expected, atol = 1e-5)

def test_calc_returns_with_next_value():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    gamma = 0.9
    next_value = 10.0

    expected = torch.tensor([[12.52, 12.8, 12.0]])

    returns = calc_returns(rewards, gamma = gamma, next_value = next_value, use_accelerated = False)
    assert torch.allclose(returns, expected, atol = 1e-5)

def test_calc_returns_dones_and_lens():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    dones = torch.tensor([[False, True, False, False]])
    gamma = 0.9

    expected = torch.tensor([[2.8, 2.0, 6.6, 4.0]])

    returns = calc_returns(rewards, masks = dones_to_masks(dones), gamma = gamma, use_accelerated = False)
    assert torch.allclose(returns, expected, atol = 1e-5)

    lens = torch.tensor([2])
    returns_lens = calc_returns(rewards[:, :2], gamma = gamma, use_accelerated = False)
    returns_padded = calc_returns(rewards, lens = lens, gamma = gamma, use_accelerated = False)
    assert torch.allclose(returns_padded[:, :2], returns_lens, atol = 1e-5)
    assert torch.all(returns_padded[:, 2:] == 0.)

def test_grouped_advantages_b_n_g():
    # shape (b, n, g) with group_dim = -1
    b, n, g = 2, 8, 3
    rewards = torch.randn(b, n, g)
    values = torch.randn(b, n, g)
    masks = torch.ones(b, n)
    weights = torch.tensor([0.5, 0.3, 0.2])

    out = calc_gae(
        rewards, values, masks = masks, weights = weights, group_dim = -1, use_accelerated = False, return_advantages = True
    )
    assert out.advantages.shape == (b, n)
    assert out.returns.shape == (b, n, g)

    # without weights, both stay (b, n, g)
    out_no_weights = calc_gae(
        rewards, values, masks = masks, group_dim = -1, use_accelerated = False, return_advantages = True
    )
    assert out_no_weights.advantages.shape == (b, n, g)
    assert out_no_weights.returns.shape == (b, n, g)

# dones / truncated edge cases

def test_dones_to_masks_truncated_only():
    truncated = torch.tensor([[False, True, False]])

    term_mask, done_mask = dones_to_masks(truncated = truncated)

    assert torch.all(term_mask == 1.)
    assert torch.equal(done_mask.bool(), ~truncated)

def test_dones_to_masks_requires_an_input():
    with pytest.raises(ValueError):
        dones_to_masks()

# shared value head across groups

@param('group_dim', (-2, 0))
def test_gae_shared_values_across_groups(group_dim):
    b, g, n = 2, 3, 5
    rewards = torch.randn(b, g, n) if group_dim == -2 else torch.randn(g, b, n)
    values = torch.randn(b, n)
    masks = torch.ones(b, n)
    masks[:, -1] = 0.

    shared = calc_gae(rewards, values, masks = masks, group_dim = group_dim, use_accelerated = False, return_advantages = True)

    expand = (values.unsqueeze(1) if group_dim == -2 else values.unsqueeze(0)).expand_as(rewards)
    per_group = calc_gae(rewards, expand, masks = masks, group_dim = group_dim, use_accelerated = False, return_advantages = True)

    assert torch.allclose(shared.returns, per_group.returns, atol = 1e-5)
    assert torch.allclose(shared.advantages, per_group.advantages, atol = 1e-5)

@param('group_dim', (-2, 0))
def test_gae_next_value_grouped_shapes(group_dim):
    b, g, n = 4, 3, 5
    rewards = torch.randn(b, g, n) if group_dim == -2 else torch.randn(g, b, n)
    values = torch.randn(b, g, n) if group_dim == -2 else torch.randn(g, b, n)
    masks = torch.ones(b, n)
    next_value = torch.randn(b)

    out = calc_gae(rewards, values, masks = masks, next_value = next_value, group_dim = group_dim, use_accelerated = False)
    assert out.shape == rewards.shape

    # (b,) and (b, g) next values must agree

    next_values_flat = next_value[:, None].expand(b, g) if group_dim == -2 else next_value[None, :].expand(g, b)
    out_2d = calc_gae(rewards, values, masks = masks, next_value = next_values_flat, group_dim = group_dim, use_accelerated = False)

    assert torch.allclose(out, out_2d, atol = 1e-5)

# bootstrapping with t + 1 values

def test_gae_t_plus_one_values_with_next_values():
    b, t = 2, 6
    rewards = torch.randn(b, t)
    values = torch.randn(b, t + 1)
    next_values = torch.randn(b, t)
    masks = torch.ones(b, t)

    out = calc_gae(rewards, values, masks, next_values = next_values, use_accelerated = False, return_advantages = True)
    expected = calc_gae(rewards, values[:, :-1], masks, next_values = next_values, use_accelerated = False, return_advantages = True)

    assert torch.allclose(out.returns, expected.returns, atol = 1e-5)
    assert torch.allclose(out.advantages, expected.advantages, atol = 1e-5)

def test_gae_t_plus_one_values_with_next_value():
    b, t = 2, 6
    rewards = torch.randn(b, t)
    values = torch.randn(b, t + 1)
    next_value = torch.randn(b)
    masks = torch.ones(b, t)

    out = calc_gae(rewards, values, masks, next_value = next_value, use_accelerated = False, return_advantages = True)
    expected = calc_gae(rewards, values[:, :-1], masks, next_value = next_value, use_accelerated = False, return_advantages = True)

    assert torch.allclose(out.returns, expected.returns, atol = 1e-5)
    assert torch.allclose(out.advantages, expected.advantages, atol = 1e-5)

# variable lengths with explicit bootstrapping

def test_gae_variable_lengths_explicit_bootstrap():
    rewards = torch.tensor([[1., 2., 3., 4.]])
    values = torch.zeros_like(rewards)
    lens = torch.tensor([2])
    gamma = 0.9

    # next_value is the bootstrap at the last valid timestep

    out = calc_gae(rewards, values, lens = lens, gamma = gamma, lam = 1.0, next_value = 10., use_accelerated = False, return_advantages = True)
    assert torch.allclose(out.returns, torch.tensor([[10.9, 11.0, 0., 0.]]), atol = 1e-5)

    # explicit next values are respected at the last valid timestep

    next_values = torch.tensor([[0., 10., 0., 0.]])
    out = calc_gae(rewards, values, lens = lens, gamma = gamma, lam = 1.0, next_values = next_values, use_accelerated = False, return_advantages = True)
    assert torch.allclose(out.returns, torch.tensor([[10.9, 11.0, 0., 0.]]), atol = 1e-5)

    # but a terminated last timestep should still not bootstrap

    masks = torch.tensor([[1., 0., 1., 1.]])
    out = calc_gae(rewards, values, masks = masks, lens = lens, gamma = gamma, lam = 1.0, next_values = next_values, use_accelerated = False, return_advantages = True)
    assert torch.allclose(out.returns, torch.tensor([[2.8, 2.0, 0., 0.]]), atol = 1e-5)

def test_gae_variable_lengths_nan_padding_does_not_leak():
    rewards = torch.ones(2, 4)
    values = torch.tensor([
        [1., 1., float('nan'), float('nan')],
        [1., 1., 1., 1.]
    ])
    lens = torch.tensor([2, 4])

    out = calc_gae(rewards, values, lens = lens, use_accelerated = False, return_advantages = True)

    assert torch.isfinite(out.returns).all()
    assert torch.isfinite(out.advantages).all()
    assert torch.all(out.returns[0, 2:] == 0.)
    assert torch.allclose(out.returns[0, :2], torch.tensor([1.99, 1.]), atol = 1e-5)

def test_calc_returns_variable_lengths_next_value():
    b, n = 3, 6
    rewards = torch.randn(b, n)
    masks = torch.ones(b, n)
    masks[0, 3] = 0.
    lens = torch.tensor([4, 6, 2])
    next_value = torch.randn(b)

    returns = calc_returns(rewards, masks = masks, gamma = 0.99, next_value = next_value, lens = lens, use_accelerated = False)

    # equivalent to gae with zero values and lam = 1

    expected = calc_gae(
        rewards, torch.zeros_like(rewards), masks = masks, gamma = 0.99, lam = 1.0,
        next_value = next_value, lens = lens, use_accelerated = False
    )

    assert torch.allclose(returns, expected, atol = 1e-5)

def test_calc_returns_nan_padding_does_not_leak():
    rewards = torch.tensor([
        [1., 1., float('nan'), float('nan')],
        [1., 1., 1., 1.]
    ])
    lens = torch.tensor([2, 4])

    returns = calc_returns(rewards, lens = lens, use_accelerated = False)

    assert torch.isfinite(returns).all()
    assert torch.all(returns[0, 2:] == 0.)

def test_lens_zero_and_beyond_sequence_length():
    b, n = 3, 5
    rewards = torch.randn(b, n)
    values = torch.randn(b, n)
    masks = torch.ones(b, n)
    lens = torch.tensor([0, 3, n + 2])
    next_value = torch.randn(b)

    out = calc_gae(rewards, values, masks = masks, lens = lens, next_value = next_value, use_accelerated = False, return_advantages = True)

    assert torch.isfinite(out.returns).all()
    assert torch.isfinite(out.advantages).all()
    assert torch.all(out.returns[0] == 0.)
    assert torch.all(out.advantages[0] == 0.)

    returns = calc_returns(rewards, masks = masks, next_value = next_value, lens = lens, use_accelerated = False)

    assert torch.isfinite(returns).all()
    assert torch.all(returns[0] == 0.)

def test_gae_grouped_lens_next_value():
    b, g, n = 3, 2, 5
    rewards = torch.randn(g, b, n)
    values = torch.randn(g, b, n)
    masks = torch.ones(b, n)
    lens = torch.tensor([2, 5, 3])
    next_value = torch.randn(b)

    out = calc_gae(rewards, values, masks = masks, lens = lens, next_value = next_value, group_dim = 0, use_accelerated = False, return_advantages = True)

    # equivalent to broadcasting the next value across the groups

    out_broadcast = calc_gae(
        rewards, values, masks = masks, lens = lens, next_value = next_value[None].expand(g, b),
        group_dim = 0, use_accelerated = False, return_advantages = True
    )

    assert torch.allclose(out.returns, out_broadcast.returns, atol = 1e-5)
    assert torch.allclose(out.advantages, out_broadcast.advantages, atol = 1e-5)

# misc robustness

def test_combine_grouped_advantages_requires_group_dim():
    with pytest.raises(AssertionError):
        combine_grouped_advantages(torch.randn(8), weights = torch.tensor([0.5, 0.5]))

@pytest.mark.skipif(torch.cuda.is_available(), reason = 'accelerated scan is used when cuda is available')
def test_use_accelerated_falls_back_on_cpu():
    rewards = torch.randn(2, 5)
    values = torch.randn(2, 5)

    out = calc_gae(rewards, values, use_accelerated = True)

    assert out.shape == rewards.shape

# value clipping tests

def test_value_clipping_losses():
    values = torch.randn(8)
    old_values = torch.randn(8)
    returns = torch.randn(8)

    loss_std = clipped_value_loss_fn(values, old_values, returns, clip = 0.2)
    assert loss_std.ndim == 0 and loss_std >= 0.

    loss_adv = improved_value_clipping_fn(values, old_values, returns, clip = 0.2)
    assert loss_adv.ndim == 0 and loss_adv >= 0.

def test_value_clipping_masked_and_lens():
    values = torch.randn(2, 6, requires_grad = True)
    old_values = torch.randn(2, 6)
    returns = torch.randn(2, 6)
    lens = torch.tensor([3, 6])

    loss_std = clipped_value_loss_fn(values, old_values, returns, lens = lens)
    assert loss_std.ndim == 0 and loss_std >= 0.
    loss_std.backward()
    assert exists(values.grad)

    values.grad = None
    loss_adv = improved_value_clipping_fn(values, old_values, returns, lens = lens)
    assert loss_adv.ndim == 0 and loss_adv >= 0.
    loss_adv.backward()
    assert exists(values.grad)


# single sequence, no batch dimension

def test_gae_single_sequence():
    seq_len = 16
    rewards = torch.randn(seq_len)
    values = torch.randn(seq_len)
    masks = (torch.rand(seq_len) > 0.1).float()
    next_value = 2.5

    returns, advantages = calc_gae(rewards, values, masks = masks, next_value = next_value, use_accelerated = False, return_advantages = True)

    batched = calc_gae(rewards[None], values[None], masks = masks[None], next_value = next_value, use_accelerated = False, return_advantages = True)

    assert returns.shape == rewards.shape
    assert advantages.shape == rewards.shape

    assert torch.allclose(returns, batched.returns[0], atol = 1e-5)
    assert torch.allclose(advantages, batched.advantages[0], atol = 1e-5)

    # a tensor next value with batch size 1 is also valid

    tensor_next = calc_gae(rewards, values, masks = masks, next_value = torch.tensor([next_value]), use_accelerated = False)
    assert torch.allclose(tensor_next, batched.returns[0], atol = 1e-5)

def test_calc_returns_single_sequence():
    seq_len = 16
    rewards = torch.randn(seq_len)
    masks = (torch.rand(seq_len) > 0.1).float()
    next_value = 2.5

    returns = calc_returns(rewards, masks = masks, next_value = next_value, use_accelerated = False)
    batched_returns = calc_returns(rewards[None], masks = masks[None], next_value = next_value, use_accelerated = False)

    assert returns.shape == rewards.shape
    assert torch.allclose(returns, batched_returns[0], atol = 1e-5)

    tensor_next_returns = calc_returns(rewards, masks = masks, next_value = torch.tensor([next_value]), use_accelerated = False)
    assert torch.allclose(tensor_next_returns, batched_returns[0], atol = 1e-5)

# per sequence discount factors

@param('grouped', (False, True))
def test_gae_per_sample_discounts(grouped):
    b, n = 3, 6
    shape = (b, 2, n) if grouped else (b, n)
    rewards = torch.randn(*shape)
    values = torch.randn(*shape)
    masks = torch.ones(b, n)
    gammas = tensor([0.9, 0.99, 0.999])
    lams = tensor([0.5, 0.8, 0.95])
    kwargs = dict(group_dim = 1) if grouped else dict()

    out = calc_gae(rewards, values, masks = masks, gamma = gammas, lam = lams, use_accelerated = False, return_advantages = True, **kwargs)

    per_sample = [
        calc_gae(rewards[i:i + 1], values[i:i + 1], masks = masks[i:i + 1], gamma = gammas[i].item(), lam = lams[i].item(), use_accelerated = False, return_advantages = True, **kwargs)
        for i in range(b)
    ]

    assert torch.allclose(out.returns, cat([p.returns for p in per_sample]), atol = 1e-5)
    assert torch.allclose(out.advantages, cat([p.advantages for p in per_sample]), atol = 1e-5)

def test_calc_returns_per_sample_gamma():
    b, t = 3, 8
    rewards = torch.randn(b, t)
    masks = (torch.rand(b, t) > 0.2).float()
    gammas = tensor([0.9, 0.99, 0.999])
    next_value = torch.randn(b)

    out = calc_returns(rewards, masks = masks, gamma = gammas, next_value = next_value, use_accelerated = False)

    expected = cat([calc_returns(rewards[i:i + 1], masks = masks[i:i + 1], gamma = gammas[i].item(), next_value = next_value[i:i + 1], use_accelerated = False) for i in range(b)])

    assert torch.allclose(out, expected, atol = 1e-5)

def test_value_clipping_none_or_false():
    values = torch.randn(8, requires_grad = True)
    old_values = torch.randn(8)
    returns = torch.randn(8)

    for fn in (clipped_value_loss_fn, improved_value_clipping_fn):
        for clip in (None, False):
            loss = fn(values, old_values, returns, clip = clip)
            assert torch.allclose(loss, torch.nn.functional.mse_loss(values, returns))

        loss_no_old = fn(values, None, returns)
        assert torch.allclose(loss_no_old, torch.nn.functional.mse_loss(values, returns))

# dagger distillation - env free

class ToyVecEnv:
    # a minimal standardized vector env (autoreset semantics)

    def __init__(self, num_envs, state_dim, action_dim, step_fn, episode_len = None, seed = 0):
        self.num_envs, self.state_dim, self.action_dim = num_envs, state_dim, action_dim
        self.step_fn = step_fn
        self.episode_len = episode_len
        self.generator = torch.Generator().manual_seed(seed)
        self.state = torch.randn(num_envs, state_dim, generator = self.generator) * 0.5
        self.t = 0

    def reset(self):
        self.state = torch.randn(self.num_envs, self.state_dim, generator = self.generator) * 0.5
        self.t = 0
        return self.state.clone(), {}

    def step(self, actions):
        self.state = self.step_fn(self.state, actions)
        self.t += 1
        reward = torch.zeros(self.num_envs)
        term = torch.zeros(self.num_envs, dtype = torch.bool)
        trunc = torch.zeros(self.num_envs, dtype = torch.bool)

        if self.episode_len is not None:
            done = self.t >= self.episode_len
            trunc = torch.full((self.num_envs,), done, dtype = torch.bool)

            if done:
                self.state = torch.randn(self.num_envs, self.state_dim, generator = self.generator) * 0.5
                self.t = 0

        return self.state.clone(), reward, term, trunc, {}

class NormalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, dim = 32, std = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, dim), nn.Tanh(), nn.Linear(dim, action_dim))
        self.std = std

    def forward(self, state):
        return Normal(self.net(state), torch.full_like(self.net(state), self.std))

class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, num_actions, dim = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, dim), nn.Tanh(), nn.Linear(dim, num_actions))

    def forward(self, state):
        return Categorical(logits = self.net(state))

def normal_teacher(action_dim, std = 1.):
    return lambda state: Normal(torch.zeros(*state.shape[:-1], action_dim), std)

# 1. action space helpers - labels, modes and sampling, per family

def test_dagger_action_spaces():
    # discrete families carry class labels, multi-dim bernoulli one 0/1 per dim

    onehot_dist = OneHotCategorical(probs = torch.softmax(torch.randn(16, 4), -1))
    onehot_target = to_log_prob_target(torch.randint(0, 4, (16,)), onehot_dist)

    assert onehot_target.shape == (16, 4)
    assert torch.isfinite(log_prob_sum(onehot_dist, onehot_target)).all()

    bernoulli_dist = Bernoulli(probs = torch.rand(16, 3))
    bern_target = to_log_prob_target(torch.randint(0, 2, (16, 3)).float(), bernoulli_dist)

    assert torch.isfinite(log_prob_sum(bernoulli_dist, bern_target)).all()

    assert is_discrete(onehot_dist) and is_discrete(bernoulli_dist) and is_discrete(Categorical(logits = torch.randn(4, 2)))
    assert not is_discrete(Normal(torch.zeros(4, 2), torch.ones(4, 2)))

    # bernoulli's mode is per-dimension, not an argmax over dims

    assert best_action(Bernoulli(probs = tensor([[0.9, 0.1, 0.7]]))).tolist() == [[1., 0., 1.]]

    # the mode is the class index, and a sampled one-hot draw collapses to it

    assert best_action(onehot_dist).shape == (16,)
    assert best_action(onehot_dist, sample = True).shape == (16,)
    assert best_action(onehot_dist, sample = True).dtype == torch.long

    # labels stored as (b, 1) land right - no silent (b, b) broadcast

    categorical = Categorical(logits = torch.randn(16, 4))
    labels = torch.randint(0, 4, (16,))

    assert torch.allclose(log_prob_sum(categorical, to_log_prob_target(labels[:, None], categorical)), categorical.log_prob(labels))

    # composite policies are pytrees of heads - discrete / continuous / mixed

    composite = (Bernoulli(probs = torch.rand(4, 2)), Normal(torch.zeros(4, 2), torch.ones(4, 2)))

    assert policy_kind(composite) == 'mixed' and policy_kind(composite[0]) == 'discrete' and policy_kind(composite[1]) == 'continuous'

    button_actions, arm_actions = best_action(composite)

    assert button_actions.shape == (4, 2) and arm_actions.shape == (4, 2)

    # student / teacher contradictions fail early

    mismatch = DaggerDistiller(CategoricalPolicy(2, 3), NormalPolicy(2, 1), action_kind = 'continuous')

    with pytest.raises(ValueError):
        mismatch(torch.randn(4, 2))

    # custom discrete families - append to DISCRETE_DISTRIBUTIONS

    class TinyCategorical(Distribution):
        arg_constraints = {}

        def __init__(self, logits):
            super().__init__(batch_shape = logits.shape[:-1])
            self.logits = logits

        def sample(self, sample_shape = torch.Size()):
            return Categorical(logits = self.logits).sample(sample_shape)

        def log_prob(self, value):
            return Categorical(logits = self.logits).log_prob(value)

    DISCRETE_DISTRIBUTIONS.append(TinyCategorical)
    tiny = TinyCategorical(torch.randn(4, 3))

    assert is_discrete(tiny) and best_action(tiny).shape == (4,)
    assert to_log_prob_target(torch.randint(0, 3, (4, 1)), tiny).shape == (4,)

    DISCRETE_DISTRIBUTIONS.remove(TinyCategorical)

# 2. env-free wrapper - student + teacher in, one distill loss out

def test_dagger_env_free():
    # the teacher never builds a graph

    torch.manual_seed(0)
    state_dim, action_dim = 4, 3

    teacher_w = torch.randn(state_dim, action_dim) * 0.5
    teacher_grad_modes = []

    def teacher(state):
        teacher_grad_modes.append(torch.is_grad_enabled())
        return Normal(state @ teacher_w, torch.full_like(state @ teacher_w, 0.1))

    student = NormalPolicy(state_dim, action_dim)
    snapshot = copy.deepcopy(student)
    optimizer = torch.optim.Adam(student.parameters(), lr = 5e-3)

    distiller = DaggerDistiller(student, teacher)

    for _ in range(40):
        states = torch.randn(64, state_dim)
        loss, out = distiller(states)

        assert loss.dim() == 0 and torch.isfinite(loss) and loss.requires_grad
        assert isinstance(out, Normal) and out.mean.shape == (64, action_dim)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probe = torch.randn(256, state_dim)
        start_err = (snapshot(probe).mean - teacher(probe).mean).pow(2).mean().item()
        end_err = (student(probe).mean - teacher(probe).mean).pow(2).mean().item()

    assert not any(teacher_grad_modes) and end_err < start_err * 0.2, f'{start_err=:.4f} {end_err=:.4f}'

    # hooks - the teacher's own state view, a label transform, info / step both

    wide_w = torch.randn(state_dim + 1, action_dim) * 0.5
    wide_teacher = lambda state: Normal(state @ wide_w, torch.full_like(state @ wide_w, 0.1))

    derive_steps, transform_steps = [], []

    def teacher_state_fn(state, info, step):
        derive_steps.append(step)
        return torch.cat([state, torch.full_like(state[:, :1], info['theta'])], -1)

    def label_transform(state, action, info, step):
        transform_steps.append((step, info.get('theta')))
        return action * 0.5

    hooked = DaggerDistiller(student, wide_teacher, teacher_state_fn = teacher_state_fn, label_transform = label_transform)

    states = torch.randn(8, state_dim)
    loss, out = hooked(states, info = dict(theta = 1.5), step = 7)
    loss.backward()

    assert derive_steps == [7] and transform_steps == [(7, 1.5)] and out.mean.shape == (8, action_dim)

    # explicit teacher data skips teacher_state_fn

    hooked(states, torch.randn(8, state_dim + 1), info = dict(theta = 1.5), step = 8)

    assert derive_steps == [7] and transform_steps[-1] == (8, 1.5)

    # research seam - a custom loss_fn sees both distributions, per-sample

    seen = []

    def mean_mse(student_distr, teacher_distr, label, info):
        seen.append((teacher_distr is not None, label.shape, info))
        return (student_distr.mean - teacher_distr.mean.detach()).pow(2).mean(dim = 1)

    custom = DaggerDistiller(student, teacher, loss_fn = mean_mse)
    loss, _ = custom(states, info = dict(tag = 'env-free'))

    assert seen == [(True, (8, action_dim), dict(tag = 'env-free'))]

    with torch.no_grad():
        manual = (student(states).mean - teacher(states).mean).pow(2).mean()

    assert torch.allclose(loss, manual, atol = 1e-6)

    # built-in objectives - nll is the default, kl matches dists, mse matches means

    states = torch.randn(16, state_dim)

    default_loss, _ = DaggerDistiller(student, teacher)(states)
    nll_value, _ = DaggerDistiller(student, teacher, loss_fn = nll_loss)(states)
    mse_value, _ = DaggerDistiller(student, teacher, loss_fn = mse_loss)(states)
    kl_value, _ = DaggerDistiller(student, teacher, loss_fn = kl_loss)(states)

    with torch.no_grad():
        teacher_out, student_out = teacher(states), student(states)
        manual_mse = (student_out.mean - teacher_out.mean).pow(2).mean()
        manual_kl = torch.distributions.kl_divergence(teacher_out, student_out).sum(-1).mean()

    assert torch.allclose(default_loss, nll_value, atol = 1e-6)
    assert torch.allclose(mse_value, manual_mse, atol = 1e-6)
    assert torch.allclose(kl_value, manual_kl, atol = 1e-6)

    # loss breakdown - the total by default, the components on demand

    breakdown, _ = DaggerDistiller(student, teacher, entropy_weight = 0.1)(states, return_loss_breakdown = True)

    assert torch.allclose(breakdown.loss, breakdown.policy - 0.1 * breakdown.entropy)
    assert breakdown.entropy.dim() == 0 and breakdown.alignment is None

    plain_breakdown, _ = DaggerDistiller(student, teacher)(states, return_loss_breakdown = True)

    assert plain_breakdown.entropy is None and torch.allclose(plain_breakdown.loss, plain_breakdown.policy)

    # representation alignment - hook a layer of each, 1 - cosine by default

    class TinyNet(nn.Module):
        def __init__(self, state_dim, hidden = 8):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
            self.head = nn.Linear(hidden, 3)

        def forward(self, state):
            return Normal(self.head(self.trunk(state)), 0.2)

    torch.manual_seed(0)
    student_net, teacher_net = TinyNet(state_dim), TinyNet(state_dim)

    aligned_breakdown, _ = DaggerDistiller(student_net, teacher_net, student_layer = 'trunk.1', teacher_layer = 'trunk.1', rep_align_weight = 0.25)(states, return_loss_breakdown = True)

    with torch.no_grad():
        manual_align = (1. - F.cosine_similarity(student_net.trunk(states), teacher_net.trunk(states), dim = -1)).mean()

    assert torch.allclose(aligned_breakdown.alignment, manual_align, atol = 1e-6)
    assert torch.allclose(aligned_breakdown.loss, aligned_breakdown.policy + 0.25 * manual_align, atol = 1e-6)

    # a projector bridges mismatched widths, and its grads flow

    wide_net = TinyNet(state_dim, hidden = 16)
    projector = nn.Linear(8, 16)

    projected_breakdown, _ = DaggerDistiller(student_net, wide_net, student_layer = 'trunk.1', teacher_layer = 'trunk.1', rep_align_projector = projector)(states, return_loss_breakdown = True)

    projected_breakdown.loss.backward()

    assert torch.isfinite(projected_breakdown.alignment) and projector.weight.grad is not None

    # layer names resolve at init, and both sides are required

    with pytest.raises(ValueError):
        DaggerDistiller(student_net, teacher_net, student_layer = 'nope', teacher_layer = 'trunk.1')

    with pytest.raises(AssertionError):
        DaggerDistiller(student_net, teacher_net, student_layer = 'trunk.1')

    # composite action spaces - a pytree of heads, the loss sums over them

    button_w, arm_w = torch.randn(state_dim, 2) * 0.5, torch.randn(state_dim, 2) * 0.5

    def composite_teacher(state):
        return Bernoulli(logits = state @ button_w), Normal(state @ arm_w, 0.2)

    class CompositeStudent(nn.Module):
        def __init__(self, state_dim, dim = 32):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(state_dim, dim), nn.Tanh(), nn.Linear(dim, 4))

        def forward(self, state):
            out = self.net(state)
            return Bernoulli(logits = out[..., :2]), Normal(out[..., 2:], 0.2)

    composite_student = CompositeStudent(state_dim)
    composite_optim = torch.optim.Adam(composite_student.parameters(), lr = 1e-2)
    composite = DaggerDistiller(composite_student, composite_teacher)

    actions = best_action(composite(states)[1])

    assert isinstance(actions, tuple) and actions[0].shape == (16, 2) and actions[1].shape == (16, 2)

    for _ in range(60):
        loss, _ = composite(torch.randn(32, state_dim))
        composite_optim.zero_grad()
        loss.backward()
        composite_optim.step()

    with torch.no_grad():
        probe = torch.randn(256, state_dim)
        t_buttons, t_arm = composite_teacher(probe)
        s_buttons, s_arm = composite_student(probe)

    assert ((s_buttons.probs > 0.5) == (t_buttons.probs > 0.5)).float().mean() > 0.85
    assert (s_arm.mean - t_arm.mean).pow(2).mean() < 0.05

    # soft kl sums over the heads too

    composite_kl, _ = DaggerDistiller(composite_student, composite_teacher, loss_fn = kl_loss)(states)

    with torch.no_grad():
        s_buttons, s_arm = composite_student(states)
        t_buttons, t_arm = composite_teacher(states)
        manual_kl = (torch.distributions.kl_divergence(t_buttons, s_buttons).sum(-1)
                     + torch.distributions.kl_divergence(t_arm, s_arm).sum(-1)).mean()

    assert torch.allclose(composite_kl, manual_kl, atol = 1e-6)

    # non-finite labels parked on the student's mean - a fully unusable batch is nan

    def bad_transform(state, action, info, step):
        action = action.clone()
        action[::2] = float('nan')
        return action

    loss, _ = DaggerDistiller(student, teacher, label_transform = bad_transform)(states)

    assert torch.isfinite(loss)

    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters())

    all_nan = DaggerDistiller(student, teacher, label_transform = lambda state, action, info, step: action * float('nan'))

    assert torch.isnan(all_nan(states)[0])

    # discrete heads ride the same wrapper

    discrete_w = torch.randn(4, 3)
    discrete_teacher = lambda state: Categorical(logits = state @ discrete_w)
    discrete_student = CategoricalPolicy(4, 3)
    discrete_optim = torch.optim.Adam(discrete_student.parameters(), lr = 1e-2)
    discrete = DaggerDistiller(discrete_student, discrete_teacher)

    assert discrete.kind is None

    for _ in range(100):
        loss, out = discrete(torch.randn(32, 4))
        discrete_optim.zero_grad()
        loss.backward()
        discrete_optim.step()

    assert discrete.kind == 'discrete' and out.probs.shape == (32, 3)

    with torch.no_grad():
        probe = torch.randn(256, 4)
        agree = (discrete_student(probe).probs.argmax(-1) == discrete_teacher(probe).probs.argmax(-1)).float().mean().item()

    assert agree > 0.85, f'{agree=:.2f}'

# 3. the DAgger loop - roll the student out, label, fit

def test_dagger_env_loop():
    # episode bookkeeping - 3-step episodes cycle per-env counters 0, 1, 2, 0, ...

    seen = []

    def step_fn(state, actions):
        return 0.9 * state + 0.1 * actions

    env = ToyVecEnv(4, 2, 1, step_fn, episode_len = 3)
    student = NormalPolicy(2, 1)
    optimizer = torch.optim.Adam(student.parameters(), lr = 1e-3)

    def teacher_action_fn(teacher_state, info, step):
        seen.append(tuple(info['episode_steps'].tolist()))
        return torch.zeros(teacher_state.shape[0], 1)

    distiller = DaggerDistiller(student, teacher_action_fn = teacher_action_fn, env = env, optimizer = optimizer)
    distiller.collect(6)

    assert seen[0] == (0, 0, 0, 0) and seen[3] == (0, 0, 0, 0)
    assert distiller.count == 6 * 4

    # teacher-rollout pretraining - the teacher acts, the student only labels

    teacher_actions = []

    class RecordingEnv(ToyVecEnv):
        def step(self, actions):
            teacher_actions.append(torch.as_tensor(actions).clone())
            return super().step(actions)

    pretrain_student = NormalPolicy(2, 1)
    pretrain_optimizer = torch.optim.Adam(pretrain_student.parameters(), lr = 1e-3)
    pretrain = DaggerDistiller(
        pretrain_student,
        normal_teacher(1, std = 0.1),
        env = RecordingEnv(4, 2, 1, step_fn, episode_len = 3),
        optimizer = pretrain_optimizer,
    )

    pretrain_stats = pretrain.pretrain(4, epochs = 1)

    assert pretrain_stats['labeled'] == 16 and pretrain_stats['samples'] == 16
    assert len(teacher_actions) == 4 and all((actions == 0.).all() for actions in teacher_actions)

    # ring buffer wraps, subsampling is respected

    ring = DaggerDistiller(student, normal_teacher(1), env = ToyVecEnv(4, 2, 1, lambda s, a: s), optimizer = optimizer, buffer_size = 10, max_samples = 6)

    for _ in range(5):
        ring.collect(1)

    assert ring.count == 20 and ring.train(epochs = 1)['samples'] == 6

    # deterministic collect uses modes, custom env step + action conversion hooks

    calls = []
    hooked_env = ToyVecEnv(4, 2, 1, step_fn)

    def env_step_fn(actions):
        calls.append(torch.as_tensor(actions).dtype)
        return hooked_env.step(actions)

    hooked = DaggerDistiller(student, normal_teacher(1), env = hooked_env, optimizer = optimizer, env_step_fn = env_step_fn)

    assert hooked.collect(3, sample = False)['labeled'] == 12 and len(calls) == 3

    # representation alignment rides the loop - teacher reps buffer up at collect

    teacher_policy = NormalPolicy(2, 1)
    aligned = DaggerDistiller(student, teacher_policy, env = ToyVecEnv(4, 2, 1, step_fn), optimizer = optimizer, student_layer = 'net.2', teacher_layer = 'net.2')

    stats = aligned.step(collect_steps = 4, epochs = 1, return_loss_breakdown = True)

    assert aligned.rep_buffer.shape == (aligned.buffer_size, 1)
    assert distiller.rep_buffer is None
    assert math.isfinite(stats['loss']) and torch.isfinite(stats['loss_breakdown'].alignment)

    # the teacher's own state view inside the loop (a clock, here)

    phase_steps = []

    def teacher_state_fn(state, info, step):
        phase_steps.append(step)
        clock = torch.full_like(state[:, :1], 0.05 * info['episode_steps'][0])
        return torch.cat([state, clock], -1)

    clocked = DaggerDistiller(student, lambda teacher_state: Normal(0.1 * teacher_state[..., :1], 1.), env = ToyVecEnv(4, 2, 1, step_fn), optimizer = optimizer, teacher_state_fn = teacher_state_fn)
    clocked.collect(3)

    assert phase_steps == [0, 1, 2]

    # a dict observation (numpy, as a vec env hands back) keeps its fields -
    # the student reads its subset, the teacher its own

    class DictObsEnv(ToyVecEnv):
        def _wrap(self, obs):
            return dict(state = obs.numpy(), privileged = 2. * obs.numpy())

        def reset(self):
            obs, info = super().reset()
            return self._wrap(obs), info

        def step(self, actions):
            obs, reward, term, trunc, info = super().step(actions)
            return self._wrap(obs), reward, term, trunc, info

    views, teacher_inputs = [], []

    def dict_student_state_fn(obs, info, step):
        views.append(sorted(obs))
        return obs['state']

    def dict_teacher_obs_fn(obs, info, step):
        views.append(sorted(obs))
        return obs['privileged']

    def wide_teacher(teacher_state):
        teacher_inputs.append(teacher_state)
        return Normal(0.1 * teacher_state[..., :1], 1.)

    dict_obs = DaggerDistiller(student, wide_teacher, env = DictObsEnv(4, 2, 1, lambda s, a: s), optimizer = optimizer, student_state_fn = dict_student_state_fn, teacher_obs_fn = dict_teacher_obs_fn, loss_fn = kl_loss)
    dict_obs.collect(2)

    assert all(keys == ['privileged', 'state'] for keys in views)
    assert torch.allclose(torch.cat(teacher_inputs), 2. * dict_obs.state_buffers[0][:8])
    assert torch.allclose(dict_obs.teacher_input_buffers[0][:8], torch.cat(teacher_inputs))

    # teacher_state_fn still refines the student's state, as it did before

    raws, chained = [], []

    def scaling_student_state_fn(state, info, step):
        raws.append(state)
        return state * 3.

    def chaining_teacher_state_fn(state, info, step):
        chained.append(state)
        return state

    chain_distiller = DaggerDistiller(student, normal_teacher(1), env = ToyVecEnv(4, 2, 1, lambda s, a: s), optimizer = optimizer, student_state_fn = scaling_student_state_fn, teacher_state_fn = chaining_teacher_state_fn)
    chain_distiller.collect(1)

    assert torch.allclose(torch.cat(chained), 3. * torch.cat(raws))
    assert torch.allclose(chain_distiller.state_buffers[0][:4], 3. * torch.cat(raws))

    # a multi-key subset is a state too - the buffer carries the pytree

    class DictStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(4, 1)

        def forward(self, state):
            return Normal(self.net(torch.cat([state['state'], state['privileged']], -1)), 1.)

    subset_student = DictStudent()
    subset = DaggerDistiller(subset_student, normal_teacher(1), env = DictObsEnv(4, 2, 1, lambda s, a: s), optimizer = torch.optim.Adam(subset_student.parameters()), student_state_fn = lambda obs, info, step: dict(state = obs['state'], privileged = obs['privileged']), teacher_obs_fn = dict_teacher_obs_fn)
    subset_stats = subset.step(collect_steps = 2, epochs = 1)

    assert math.isfinite(subset_stats['loss']) and len(subset.state_buffers) == 2

    # non-finite teacher labels never poison the student

    def bad_label_transform(state, action, info, step):
        action = action.clone()
        action[::2] = float('nan')
        return action

    nan_distiller = DaggerDistiller(student, normal_teacher(1), env = ToyVecEnv(8, 2, 1, lambda s, a: s), optimizer = optimizer, label_transform = bad_label_transform, batch_size = 64)

    for _ in range(3):
        nan_distiller.step(collect_steps = 4)

    assert all(torch.isfinite(p).all() for p in student.parameters())

    # custom objectives fit from the labels - soft built-ins re-run the teacher

    label_only = []

    def label_mse(student_distr, teacher_distr, label, info):
        label_only.append(teacher_distr is None and info is None)
        return (student_distr.mean - label).pow(2).mean(dim = 1)

    for loss_fn in (label_mse, kl_loss, partial(kl_loss, reverse = True)):
        loop = DaggerDistiller(student, normal_teacher(1), env = ToyVecEnv(4, 2, 1, step_fn), optimizer = optimizer, loss_fn = loss_fn)
        assert math.isfinite(loop.step(collect_steps = 4, epochs = 1)['loss'])
        assert loop.needs_teacher_out == (loss_fn is not label_mse)

    assert all(label_only)

    # kl in the loop matches the teacher distributions, not just its actions

    teacher = normal_teacher(1)
    kl_distiller = DaggerDistiller(student, teacher, env = ToyVecEnv(4, 2, 1, step_fn), optimizer = optimizer, loss_fn = kl_loss, batch_size = 16, max_samples = 16)
    kl_distiller.collect(4)

    states = kl_distiller.state_buffers[0][:16]

    with torch.no_grad():
        expected_kl = kl_divergence(teacher(states), student(states)).sum(-1).mean()

    assert torch.allclose(tensor(kl_distiller.train(epochs = 1)['loss']), expected_kl, atol = 1e-5)

    # requirements fail early and loudly

    with pytest.raises(RuntimeError, match = 'no env'):
        DaggerDistiller(student, normal_teacher(1)).collect(1)

    with pytest.raises(AssertionError, match = 'optimizer'):
        DaggerDistiller(student, normal_teacher(1), env = env)


# dagger distillation - end to end

class ToyGoalEnv:
    # vel' = 0.8 vel + control, pos' = pos + 0.1 vel'

    def __init__(self, num_envs, control_dim, seed = 0):
        self.num_envs, self.control_dim = num_envs, control_dim
        self.generator = torch.Generator().manual_seed(seed)
        self.state = self._sample()

    def _sample(self):
        return torch.randn(self.num_envs, 2 * self.control_dim, generator = self.generator)

    def reset(self):
        self.state = self._sample()
        return self.state.clone(), {}

    def step(self, control):
        control = torch.as_tensor(control, dtype = torch.float32)
        pos, vel = self.state[:, :self.control_dim], self.state[:, self.control_dim:]
        vel = 0.8 * vel + control
        pos = pos + 0.1 * vel
        self.state = torch.cat([pos, vel], -1)

        zeros = torch.zeros(self.num_envs)
        return self.state.clone(), zeros, zeros.bool(), zeros.bool(), {}

def make_net(state_dim, out_dim, dim = 32):
    return nn.Sequential(nn.Linear(state_dim, dim), nn.Tanh(), nn.Linear(dim, out_dim))

class NormalHead(nn.Module):
    def __init__(self, state_dim, action_dim, std = 0.2):
        super().__init__()
        self.net, self.std = make_net(state_dim, action_dim), std

    def forward(self, state):
        mean = self.net(state)
        return Normal(mean, torch.full_like(mean, self.std))

class ChunkNormalHead(nn.Module):
    def __init__(self, state_dim, chunk, action_dim, std = 0.2):
        super().__init__()
        self.chunk, self.action_dim, self.std = chunk, action_dim, std
        self.net = make_net(state_dim, chunk * action_dim)

    def forward(self, state):
        mean = self.net(state).reshape(*state.shape[:-1], self.chunk, self.action_dim)
        return Normal(mean, torch.full_like(mean, self.std))

class CategoricalHead(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.net = make_net(state_dim, num_actions)

    def forward(self, state):
        return Categorical(logits = self.net(state))

class OneHotHead(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.net = make_net(state_dim, num_actions)

    def forward(self, state):
        return OneHotCategorical(probs = torch.softmax(self.net(state), -1))

class BernoulliHead(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = make_net(state_dim, action_dim)

    def forward(self, state):
        return Bernoulli(logits = self.net(state))

def build_policy_family(family, state_dim, control_dim, num_actions = 5, chunk = 2, seed = 0):
    # one PD controller wearing each family's policy head -> (student, teacher, action_to_env_fn)

    torch.manual_seed(seed)

    scale = 2.0

    def desired(state):
        pos, vel = state[..., :control_dim], state[..., control_dim:]
        return -2.0 * pos - vel

    if family == 'normal':
        student = NormalHead(state_dim, control_dim)

        def teacher(state):
            mean = desired(state)
            return Normal(mean, torch.full_like(mean, 0.05))

        return student, teacher, None

    if family == 'normal_chunk':
        student = ChunkNormalHead(state_dim, chunk, control_dim)

        def teacher(state):
            mean = desired(state).unsqueeze(-2).expand(-1, chunk, -1)
            return Normal(mean, torch.full_like(mean, 0.05))

        return student, teacher, lambda action: action[..., 0, :]

    if family in ('categorical', 'onehot'):
        # action 0 coasts, the rest push along fixed directions
        directions = torch.zeros(num_actions, control_dim)
        directions[1:] = F.normalize(torch.randn(num_actions - 1, control_dim), dim = -1)

        def teacher_logits(state):
            goal = desired(state)
            fit = (goal.unsqueeze(-2) - scale * directions).pow(2).sum(-1)
            return -2.0 * fit

        if family == 'categorical':
            student = CategoricalHead(state_dim, num_actions)
            teacher = lambda state: Categorical(logits = teacher_logits(state))
        else:
            student = OneHotHead(state_dim, num_actions)
            teacher = lambda state: OneHotCategorical(probs = torch.softmax(teacher_logits(state), -1))

        return student, teacher, lambda action: scale * directions[action.long()]

    if family == 'bernoulli':
        student = BernoulliHead(state_dim, control_dim)

        def teacher(state):
            return Bernoulli(logits = 4.0 * desired(state))

        return student, teacher, lambda action: 2.0 * action.float() - 1.0

    if family in ('beta', 'hybrid'):
        from mean_conc_beta import Beta

        beta = Beta(init_conc = 20.)

    if family == 'beta':
        class BetaHead(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.net = make_net(state_dim, action_dim * 2)

            def forward(self, state):
                params = self.net(state).reshape(*state.shape[:-1], -1, 2)
                return beta(params, temperature = 1.)

        student = BetaHead(state_dim, control_dim)

        def teacher(state):
            goal = desired(state)
            return beta(torch.stack([goal, torch.zeros_like(goal)], -1), temperature = 1.)

        return student, teacher, lambda action: 3.0 * action

    if family == 'hybrid':
        num_binary, num_continuous = control_dim // 2, control_dim - control_dim // 2

        class HybridHead(nn.Module):
            def __init__(self, state_dim, num_binary, num_continuous):
                super().__init__()
                self.binary_net = make_net(state_dim, num_binary)
                self.arm_net = make_net(state_dim, num_continuous * 2)

            def forward(self, state):
                buttons = Bernoulli(logits = self.binary_net(state))
                params = self.arm_net(state).reshape(*state.shape[:-1], -1, 2)
                return buttons, beta(params, temperature = 1.)

        student = HybridHead(state_dim, num_binary, num_continuous)

        def teacher(state):
            goal = desired(state)
            binary_goal, continuous_goal = goal[..., :num_binary], goal[..., num_binary:]
            buttons = Bernoulli(logits = 4.0 * binary_goal)
            params = torch.stack([continuous_goal, torch.zeros_like(continuous_goal)], -1)
            return buttons, beta(params, temperature = 1.)

        def to_env(action):
            buttons, arm = action
            return torch.cat([2.0 * buttons.float() - 1.0, 3.0 * arm], -1)

        return student, teacher, to_env

    raise ValueError(f'{family=}')

@param('loss_fn', (nll_loss, kl_loss, mse_loss))
@param('family', ('normal', 'normal_chunk', 'categorical', 'onehot', 'bernoulli', 'beta', 'hybrid'))
def test_dagger_e2e_action_spaces(family, loss_fn):
    if family in ('beta', 'hybrid'):
        pytest.importorskip('mean_conc_beta')

    if loss_fn is mse_loss and family in ('categorical', 'onehot'):
        pytest.skip('categorical has no mean - use kl_loss')

    torch.manual_seed(0)
    num_envs, cycles, collect_steps = 16, 18, 4
    control_dim = 4 if family == 'hybrid' else 2
    state_dim = 2 * control_dim

    student, teacher, action_to_env_fn = build_policy_family(family, state_dim, control_dim)
    snapshot = copy.deepcopy(student)
    optimizer = torch.optim.Adam(student.parameters(), lr = 5e-3)

    distiller = DaggerDistiller(
        student, teacher,
        env = ToyGoalEnv(num_envs, control_dim, seed = 1),
        optimizer = optimizer,
        action_to_env_fn = action_to_env_fn,
        loss_fn = loss_fn,
        batch_size = 128,
        max_samples = 2048
    )

    for _ in range(cycles):
        stats = distiller.step(collect_steps = collect_steps)

    assert distiller.count == cycles * collect_steps * num_envs
    assert math.isfinite(stats['loss'])

    # imitation - mean error for continuous heads, per-dim agreement for discrete

    probe = torch.randn(512, state_dim)

    with torch.no_grad():
        teacher_out, student_out, snapshot_out = teacher(probe), student(probe), snapshot(probe)

    if family in ('normal', 'normal_chunk', 'beta'):
        error = lambda out: (out.mean - teacher_out.mean).pow(2).mean().item()
        assert error(student_out) < 0.1 and error(student_out) < error(snapshot_out) * 0.25, f'{family=} {error(snapshot_out)=:.4f} {error(student_out)=:.4f}'
    elif family == 'hybrid':
        agree = lambda out: ((out[0].probs > 0.5) == (teacher_out[0].probs > 0.5)).float().mean().item()
        error = lambda out: (out[1].mean - teacher_out[1].mean).pow(2).mean().item()
        assert agree(student_out) > 0.85 and agree(student_out) > agree(snapshot_out), f'{family=} {agree(snapshot_out)=:.2f} {agree(student_out)=:.2f}'
        assert error(student_out) < 0.05 and error(student_out) < error(snapshot_out) * 0.25, f'{family=} {error(snapshot_out)=:.4f} {error(student_out)=:.4f}'
    else:
        if family == 'bernoulli':
            agree = lambda out: ((out.probs > 0.5) == (teacher_out.probs > 0.5)).float().mean().item()
        else:
            agree = lambda out: (out.probs.argmax(-1) == teacher_out.probs.argmax(-1)).float().mean().item()

        assert agree(student_out) > 0.85 and agree(student_out) > agree(snapshot_out), f'{family=} {agree(snapshot_out):.2f} {agree(student_out):.2f}'

    # behaviour - the distilled student reaches the goal, the snapshot wanders

    def goal_distance(policy_fn, seed = 123, steps = 40):
        env = ToyGoalEnv(num_envs, control_dim, seed = seed)
        obs, _ = env.reset()
        total = 0.

        with torch.no_grad():
            for _ in range(steps):
                action = best_action(policy_fn(obs))

                if action_to_env_fn is not None:
                    action = action_to_env_fn(action)

                obs, *_ = env.step(action)
                total += obs[:, :control_dim].pow(2).sum(-1)

        return (total / steps).mean().item()

    student_goal = goal_distance(student)

    assert student_goal < goal_distance(snapshot) * 0.5, f'{family=} {student_goal=:.3f}'
    assert student_goal < goal_distance(teacher) * 2.0 + 0.05, f'{family=} {student_goal=:.3f}'

class PrivilegedGoalEnv(ToyGoalEnv):
    # env hands back a dict observation - the student reads its subset `obs`
    # (position only), the teacher's view adds `privileged` (true velocity)

    def _split(self, state):
        return dict(obs = state[:, :self.control_dim].clone(), privileged = state.clone())

    def reset(self):
        state, info = super().reset()
        return self._split(state), info

    def step(self, control):
        state, reward, term, trunc, info = super().step(control)
        return self._split(state), reward, term, trunc, info

@param('loss_fn', (nll_loss, kl_loss, mse_loss))
def test_dagger_e2e_privileged(loss_fn):
    # the env gives back a dict observation and each side reads its own subset -
    # the teacher keeps the field the student cannot see

    torch.manual_seed(0)
    num_envs, cycles, collect_steps = 16, 24, 4
    control_dim = 2

    student = NormalHead(control_dim, control_dim)   # position only
    snapshot = copy.deepcopy(student)
    optimizer = torch.optim.Adam(student.parameters(), lr = 5e-3)

    def teacher(full_state):
        goal = -2.0 * full_state[..., :control_dim] - full_state[..., control_dim:]
        return Normal(goal, torch.full_like(goal, 0.05))

    def student_state_fn(obs, info, step):
        return obs['obs']

    def teacher_obs_fn(obs, info, step):
        return obs['privileged']

    distiller = DaggerDistiller(
        student, teacher,
        env = PrivilegedGoalEnv(num_envs, control_dim, seed = 1),
        optimizer = optimizer,
        student_state_fn = student_state_fn,
        teacher_obs_fn = teacher_obs_fn,
        loss_fn = loss_fn,
        batch_size = 128,
        max_samples = 2048
    )

    for _ in range(cycles):
        stats = distiller.step(collect_steps = collect_steps)

    assert distiller.count == cycles * collect_steps * num_envs
    assert math.isfinite(stats['loss'])

    # behaviour - the position-only student still drives to the goal

    def goal_distance(policy_fn, seed = 123, steps = 40):
        env = PrivilegedGoalEnv(num_envs, control_dim, seed = seed)
        obs, _ = env.reset()
        total = 0.

        with torch.no_grad():
            for _ in range(steps):
                obs, *_ = env.step(best_action(policy_fn(obs['obs'])))
                total += obs['obs'].pow(2).sum(-1)

        return (total / steps).mean().item()

    student_goal = goal_distance(student)

    assert student_goal < goal_distance(snapshot) * 0.5, f'{student_goal=:.3f}'
