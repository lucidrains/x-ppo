from __future__ import annotations
from typing import NamedTuple, Callable

import torch
from torch import Tensor, as_tensor, cat, is_tensor
import torch.nn.functional as F

from einops import rearrange, repeat
import einx

from assoc_scan import AssocScan

from torch_einops_utils import (
    z_score,
    pack_with_inverse,
    lens_to_mask,
    pad_right_at_dim,
    pad_right_ndim_to,
    batched_index_select,
    tree_map_tensor,
    tree_map_tensor_to_device,
    masked_reduce,
    exclusive_cumprod
)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)

def align_group_dim(t, group_dim):
    if not is_tensor(t):
        return t

    if t.ndim == 2:
        return rearrange(t, 'b n -> b 1 n')

    return t.movedim(group_dim, -2)

def align_grouped_next_value(next_value, group_dim, ndim):
    if not is_tensor(next_value) or next_value.ndim <= 1:
        return next_value

    # group first layouts have a trailing batch dimension

    return next_value.movedim(0, -1) if group_dim in (0, -ndim) else next_value

def align_grouped_mask(
    mask: Tensor | None,
    ndim: int,
    group_dim: int = -2
) -> Tensor | None:
    # (b n) -> (b 1 n) or (b n 1), broadcastable against values / advantages grouped over critics

    if not is_tensor(mask) or mask.ndim >= ndim:
        return mask

    return mask.unsqueeze(group_dim)

def lens_to_last_mask(lens, max_len):
    # (b n) one hot mask at each sequence's last valid timestep

    last_idx = (lens - 1).clamp(min = 0, max = max_len - 1)
    return F.one_hot(last_idx.long(), max_len).bool()

class GAE(NamedTuple):
    returns: Tensor
    advantages: Tensor

# dones to masks helper (e.g. gymnasium)

def dones_to_masks(
    terminated: Tensor | None = None,         # (b n)
    truncated: Tensor | None = None,          # (b n)
    dones: Tensor | None = None               # (b n)
) -> tuple[Tensor, Tensor] | Tensor:
    terminated = default(terminated, dones)

    if not exists(terminated) and not exists(truncated):
        raise ValueError('at least one of `terminated`, `truncated` or `dones` must be passed in')

    if not exists(terminated):
        terminated = torch.zeros_like(truncated, dtype = torch.bool)

    term_mask = (~terminated.bool()).float()

    if not exists(truncated):
        return term_mask

    done_mask = (~(terminated.bool() | truncated.bool())).float()
    return term_mask, done_mask

# grouped advantages (multi-critic)
# each reward group gets its own value network - normalize each group before combining, else reward scale dominates
# https://openreview.net/forum?id=rJvY_5OzoI
# https://arxiv.org/abs/2507.08656

def resolve_grouped_mask(
    advantages: Tensor,
    action_log_probs: Tensor,
    mask: Tensor | None = None,
    group_dim: int = -2
) -> Tensor | None:
    # grouped advantages must first be combined into a single advantage, unless log probs are grouped in the same layout

    if advantages.ndim > action_log_probs.ndim:
        raise ValueError('advantages is grouped over multiple critics - combine with `combine_grouped_advantages` first')

    return align_grouped_mask(mask, advantages.ndim, group_dim)

def combine_grouped_advantages(
    advantages: Tensor,                       # (b n g) or (b g n)
    weights: Tensor | None = None,            # (g) - weights each group before 'sum' / callable reductions, renormalized for 'mean'
    normalize: bool = True,
    group_dim: int = -2,
    mask: Tensor | None = None,               # (b n)
    reduce: str | Callable = 'mean',          # 'sum' | 'mean' | 'max' | 'min', or callable receiving (g ...) and returning (...)
    eps: float = 1e-5
) -> Tensor:                                  # (b n)
    assert advantages.ndim >= 2, 'advantages must have at least 2 dimensions'

    mask = align_grouped_mask(mask, advantages.ndim, group_dim)

    if normalize:
        adv_groups = advantages.movedim(group_dim, 0)
        mask_groups = mask.expand_as(advantages).movedim(group_dim, 0).bool() if is_tensor(mask) else None
        normed = z_score(adv_groups, dim = tuple(range(1, adv_groups.ndim)), mask = mask_groups, eps = eps)
        advantages = normed.movedim(0, group_dim)

    # group first

    grouped = advantages.movedim(group_dim, 0)

    weight = None
    if exists(weights):
        weight = pad_right_ndim_to(as_tensor(weights, device = grouped.device, dtype = grouped.dtype), grouped.ndim)

    if callable(reduce):
        return reduce(grouped if not exists(weight) else grouped * weight)

    # masked reductions accept continuous weights
    # for 'mean' the weights act as importance and are renormalized, making it invariant to their scale

    if reduce in ('sum', 'mean'):
        return masked_reduce(grouped, mode = reduce, mask = weight, dim = 0, eps = eps)

    if reduce in ('max', 'min'):
        return grouped.amax(dim = 0) if reduce == 'max' else grouped.amin(dim = 0)

    raise ValueError(f'unknown reduce mode {reduce}')

# scans

accelerated_scan = AssocScan(reverse = True, use_accelerated = True, has_no_feature_dim = True)
cpu_scan = AssocScan(reverse = True, use_accelerated = False, has_no_feature_dim = True)

def reverse_assoc_scan(
    gates: Tensor,                            # (* n)
    inputs: Tensor,                           # (* n)
    use_accelerated: bool | None = None
) -> Tensor:
    # accelerated scan is only available on cuda

    use_accelerated = default(use_accelerated, inputs.is_cuda) and inputs.is_cuda
    scan = accelerated_scan if use_accelerated else cpu_scan

    inputs, inverse_pack = pack_with_inverse(inputs, '* n')
    gates, _ = pack_with_inverse(gates, '* n')

    return inverse_pack(scan(gates, inputs))

# finite-time gae - renormalize geometric weights over the remaining rollout

def finite_time_gae_ratio(
    horizon: Tensor,                          # (* n) remaining trace in segment
    lam: float | Tensor,                      # () or (b)
    step: float | Tensor = 1.                 # () or (* n) per-step trace increment
) -> Tensor:
    # (1 - lam ** (h - step)) / (1 - lam ** h), stable near lam = 1 and overflow-free for lam > 1
    # gates already carry gamma * lam ** step, so only the ratio is returned

    lam = as_tensor(lam, device = horizon.device, dtype = horizon.dtype)
    rem = horizon - step

    neg_abs_log_lam = -lam.clamp(min = torch.finfo(lam.dtype).tiny).log().abs()
    neg_abs_log_lam = torch.where(lam == 1., -torch.ones_like(lam), neg_abs_log_lam)

    ratio = torch.expm1(rem * neg_abs_log_lam) / torch.expm1(horizon * neg_abs_log_lam)
    ratio = ratio / lam.clamp(min = 1.).pow(step)

    return torch.where(lam == 1., rem / horizon, ratio)

# generalized advantage estimation - Schulman et al.

def calc_gae(
    rewards: Tensor,                          # (b n) or (b n g)
    values: Tensor,                           # (b n) or (b n + 1) or (b n g)
    masks: Tensor | None = None,              # (b n)
    gamma: float | Tensor = 0.99,             # () or (b)
    lam: float | Tensor = 0.95,               # () or (b)
    done_masks: Tensor | None = None,         # (b n)
    next_value: float | Tensor | None = None, # () or (b) or (b g)
    next_values: Tensor | None = None,        # (b n) or (b n g)
    lens: Tensor | None = None,               # (b)
    weights: Tensor | None = None,            # (g)
    normalize_grouped: bool = True,
    group_reduce: str | Callable = 'mean',
    group_dim: int = -2,
    use_accelerated: bool | None = None,
    return_advantages: bool = False,
    terminated: Tensor | None = None,         # (b n) gymnasium termination - no bootstrap, cut gae
    truncated: Tensor | None = None,          # (b n) gymnasium truncation - bootstrap, cut gae
    chunk_size: int | tuple[int, ...] | list[int] = 1,
    durations: Tensor | None = None,          # (b n) semi-markov option durations tau_k >= 1
    trace_power: float = 1.,                  # 1.: lam^tau clock trace; 0.: lam^1 decision trace; (0, 1): interpolated
    finite_time: bool = True                  # normalize geometric weights over the remaining rollout
) -> Tensor | GAE | tuple[Tensor | GAE, ...]:
    if exists(durations) and (isinstance(chunk_size, (tuple, list)) or chunk_size > 1):
        raise ValueError('`durations` (semi-markov gae) cannot be combined with `chunk_size` > 1')

    if isinstance(chunk_size, (tuple, list)):
        return tuple(
            calc_gae(
                rewards = rewards,
                values = values,
                masks = masks,
                gamma = gamma,
                lam = lam,
                done_masks = done_masks,
                next_value = next_value,
                next_values = next_values,
                lens = lens,
                weights = weights,
                normalize_grouped = normalize_grouped,
                group_reduce = group_reduce,
                group_dim = group_dim,
                use_accelerated = use_accelerated,
                return_advantages = return_advantages,
                terminated = terminated,
                truncated = truncated,
                chunk_size = k,
                finite_time = finite_time
            )
            for k in chunk_size
        )

    if chunk_size < 1:
        raise ValueError(f'chunk_size must be >= 1, got {chunk_size}')

    device, dtype = rewards.device, rewards.dtype

    # gymnasium signals -> masks: terminated cuts bootstrap, truncated bootstraps from final state

    if exists(terminated) or exists(truncated):
        if exists(masks) or exists(done_masks):
            raise ValueError(
                'pass either `terminated` / `truncated` (gymnasium) or `masks` / `done_masks` (low-level), not both'
            )

        truncated = as_tensor(truncated, dtype = torch.bool) if exists(truncated) else None
        terminated = as_tensor(terminated, dtype = torch.bool) if exists(terminated) else torch.zeros_like(truncated, dtype = torch.bool)

        masks = (~terminated).float()
        done_masks = (~(terminated | truncated)).float() if exists(truncated) else masks

    # handle single sequence input, with no batch dimension

    is_single_sequence = rewards.ndim == 1

    if is_single_sequence:
        rewards, values, masks, done_masks, next_values, gamma, lam, durations = tree_map_tensor(
            lambda t: t[None] if t.ndim == 1 else t,
            (rewards, values, masks, done_masks, next_values, gamma, lam, durations)
        )

    grouped = rewards.ndim > 2

    # canonicalize grouped inputs to (b g n), with shared value heads broadcast across groups

    masks = default(masks, 1.)
    done_masks = default(done_masks, masks)

    if grouped:
        next_value = align_grouped_next_value(next_value, group_dim, rewards.ndim)

        rewards = rewards.movedim(group_dim, -2)
        values = align_group_dim(values, group_dim)
        next_values = align_group_dim(next_values, group_dim) if exists(next_values) else next_values
        durations = align_group_dim(durations, group_dim) if exists(durations) else durations

        masks = align_group_dim(masks, group_dim)
        done_masks = align_group_dim(done_masks, group_dim)

    seq_len = rewards.shape[-1]

    values, next_values, masks, done_masks, gamma, lam, durations, lens = tree_map_tensor_to_device(
        (values, next_values, masks, done_masks, gamma, lam, durations, lens),
        device
    )

    if chunk_size > 1:
        num_chunks = seq_len // chunk_size
        if num_chunks < 1:
            raise ValueError(f'sequence length ({seq_len}) must be at least chunk_size ({chunk_size})')

        if values.shape[-1] in (num_chunks, num_chunks + 1) and num_chunks != seq_len:
            if values.shape[-1] == num_chunks + 1 and not exists(next_value):
                next_value = values[..., -1]

            values = repeat(values[..., :num_chunks], '... c -> ... (c k)', k = chunk_size)
            seq_len = num_chunks * chunk_size
            rewards, masks, done_masks, next_values = tuple(
                t[..., :seq_len] if is_tensor(t) else t
                for t in (rewards, masks, done_masks, next_values)
            )

    assert values.shape[-1] in (seq_len, seq_len + 1), f'values must have sequence length of {seq_len} or {seq_len + 1}, but received {values.shape[-1]}'

    has_explicit_bootstrap = exists(next_values) or exists(next_value)

    # derive next values from values if not explicitly provided

    values_next = None
    if values.shape[-1] == seq_len + 1:
        values, values_next = values[..., :-1], values[..., 1:]

    if exists(next_values):
        assert next_values.shape[-1] == seq_len, f'next_values must have sequence length of {seq_len}, but received {next_values.shape[-1]}'
    elif exists(values_next) and not exists(next_value):
        next_values = values_next
    else:
        next_val = as_tensor(default(next_value, 0.), device = device, dtype = dtype)
        next_val = pad_right_ndim_to(next_val, values.ndim - 1).expand_as(values[..., 0])

        next_values = cat((values[..., 1:], next_val[..., None]), dim = -1)

        # with variable lengths, the bootstrap value belongs at the last valid timestep, not the padded end

        if exists(next_value) and exists(lens):
            last_mask = lens_to_last_mask(lens, seq_len)
            pattern = 'b n, b g, b g n -> b g n' if grouped else 'b n, b, b n -> b n'
            next_values = einx.where(pattern, last_mask, next_val, next_values)

    # variable length masking
    # with an explicit bootstrap, the last valid timestep stays unmasked, but recurrence is always cut across padding

    len_mask = None
    if exists(lens):
        len_mask = lens_to_mask(lens, max_len = seq_len)
        is_next = pad_right_at_dim(len_mask[..., 1:], 1, value = False)

        if grouped:
            len_mask = rearrange(len_mask, 'b n -> b 1 n')
            is_next = rearrange(is_next, 'b n -> b 1 n')

        masks = masks * (len_mask if has_explicit_bootstrap else is_next)
        done_masks = done_masks * is_next

    # deltas and gates
    # mask next values first so NaNs at padded timesteps cannot leak in through `0. * nan`

    if is_tensor(masks):
        next_values = next_values.masked_fill(masks == 0., 0.)

    # semi-markov option durations: discount by gamma^tau and decay trace by lam^(tau ^ trace_power)

    base_lam = pad_right_ndim_to(as_tensor(lam, device = device, dtype = dtype), rewards.ndim)
    trace_step = 1.

    if exists(durations):
        durations = durations.to(dtype = dtype)
        if exists(len_mask):
            durations = durations.masked_fill(~len_mask, 1.)

        gamma = pad_right_ndim_to(as_tensor(gamma, device = device, dtype = dtype), durations.ndim) ** durations

        if trace_power != 0.:
            trace_step = durations if trace_power == 1. else durations.pow(trace_power)
            lam = base_lam ** trace_step

    # discount factors and lambdas may differ across the batch

    gamma, lam = tree_map_tensor(lambda t: pad_right_ndim_to(t, rewards.ndim), (gamma, lam))

    delta = rewards + gamma * next_values * masks - values

    if exists(len_mask):
        delta = delta.masked_fill(~len_mask, 0.)

    # chunked gae: fold 1-step TD residuals across macro-chunks of length `chunk_size`

    if chunk_size > 1:
        trim = num_chunks * chunk_size
        step_discounts = as_tensor(gamma * done_masks, device = device, dtype = dtype).expand_as(delta)

        delta_chunks, step_discounts = tuple(
            rearrange(t[..., :trim], '... (c k) -> ... c k', k = chunk_size)
            for t in (delta, step_discounts)
        )

        delta = (delta_chunks * exclusive_cumprod(step_discounts)).sum(dim = -1)
        gates = as_tensor(lam * step_discounts.prod(dim = -1), device = device, dtype = dtype).expand_as(delta)
        values = values[..., 0:trim:chunk_size]

        if is_tensor(done_masks):
            done_masks = rearrange(done_masks[..., :trim], '... (c k) -> ... c k', k = chunk_size).prod(dim = -1)

        if exists(len_mask):
            len_mask = len_mask[..., 0:trim:chunk_size]
    else:
        gates = as_tensor(gamma * lam * done_masks, device = device, dtype = dtype).expand_as(delta)

    # finite-time gae: count remaining trace per segment, then renormalize the geometric tail

    if finite_time:
        continuation = as_tensor(done_masks, device = device, dtype = dtype)
        step_inc = as_tensor(trace_step, device = device, dtype = dtype)
        scan_shape = torch.broadcast_shapes(continuation.shape, step_inc.shape, (1,) * (delta.ndim - 1) + (delta.shape[-1],))

        horizon = reverse_assoc_scan(continuation.expand(scan_shape), step_inc.expand(scan_shape), use_accelerated)
        gates = gates * finite_time_gae_ratio(horizon, base_lam, step = step_inc)

    # scan

    gae = reverse_assoc_scan(gates, delta, use_accelerated)
    returns = gae + values

    if exists(len_mask):
        gae = gae.masked_fill(~len_mask, 0.)
        returns = returns.masked_fill(~len_mask, 0.)

    if grouped:
        returns = returns.movedim(-2, group_dim)

    remove_batch_dim = lambda t: t[0] if is_single_sequence else t

    # early return if advantages are not needed

    if not return_advantages:
        return remove_batch_dim(returns)

    # combine grouped advantages, if weights are given

    if exists(weights):
        gae = combine_grouped_advantages(gae, weights = weights, normalize = normalize_grouped, group_dim = -2, mask = done_masks, reduce = group_reduce)
    elif grouped:
        gae = gae.movedim(-2, group_dim)

    returns, gae = tree_map_tensor(remove_batch_dim, (returns, gae))

    return GAE(returns, gae)

# semi-markov generalized advantage estimation - Sutton et al. + Schulman et al.

def calc_smdp_gae(
    rewards: Tensor,                          # (b n) within-option discounted reward sum_{j=0}^{tau-1} gamma^j r_{t+j}
    values: Tensor,                           # (b n) or (b n + 1)
    durations: Tensor,                        # (b n) elapsed primitive steps tau_k >= 1
    *,
    gamma: float | Tensor = 0.99,
    lam: float | Tensor = 0.95,
    trace_power: float = 1.,                  # 1.: clock trace lam^tau; 0.: decision trace lam^1; (0, 1): interpolated
    return_advantages: bool = True,
    **kwargs
) -> Tensor | GAE:
    return calc_gae(
        rewards = rewards,
        values = values,
        durations = durations,
        gamma = gamma,
        lam = lam,
        trace_power = trace_power,
        return_advantages = return_advantages,
        **kwargs
    )

# returns

def calc_returns(
    rewards: Tensor,                          # (b n)
    masks: Tensor | None = None,              # (b n)
    gamma: float | Tensor = 0.99,             # () or (b)
    next_value: float | Tensor | None = None, # () or (b)
    lens: Tensor | None = None,               # (b)
    use_accelerated: bool | None = None
) -> Tensor:                                  # (b n)
    # handle single sequence input, with no batch dimension

    is_single_sequence = rewards.ndim == 1

    if is_single_sequence:
        rewards, masks, gamma = tree_map_tensor(
            lambda t: t[None] if t.ndim == 1 else t,
            (rewards, masks, gamma)
        )

    device, dtype, seq_len = rewards.device, rewards.dtype, rewards.shape[-1]

    masks = default(masks, 1.)
    masks, gamma, lens = tree_map_tensor_to_device((masks, gamma, lens), device)
    raw_masks = masks

    # variable length masking

    len_mask = None
    if exists(lens):
        len_mask = lens_to_mask(lens, max_len = seq_len)
        masks = masks * pad_right_at_dim(len_mask[..., 1:], 1, value = False)

    # bootstrap next value at each sequence's last valid timestep

    if exists(next_value) and not exists(lens):
        next_val = as_tensor(next_value, device = device, dtype = dtype)
        next_mask = raw_masks[..., -1] if is_tensor(raw_masks) else raw_masks
        rewards = rewards.clone()
        rewards[..., -1] = rewards[..., -1] + gamma * next_val * next_mask

    elif exists(next_value) and exists(lens):
        next_val = as_tensor(next_value, device = device, dtype = dtype)
        last_idx = (lens - 1).clamp(min = 0, max = seq_len - 1)
        last_mask = batched_index_select(raw_masks, last_idx) if is_tensor(raw_masks) else raw_masks
        last_pos = lens_to_last_mask(lens, seq_len).type_as(rewards)

        rewards = rewards.clone() + pad_right_ndim_to(gamma * next_val * last_mask, 2) * last_pos

    if exists(len_mask):
        rewards = rewards.masked_fill(~len_mask, 0.)

    # scan

    if is_tensor(gamma):
        gamma = pad_right_ndim_to(gamma, rewards.ndim)

    gates = as_tensor(gamma * masks, device = device, dtype = dtype).expand_as(rewards)
    returns = reverse_assoc_scan(gates, rewards, use_accelerated)

    if exists(len_mask):
        returns = returns.masked_fill(~len_mask, 0.)

    if is_single_sequence:
        return returns[0]

    return returns

# value clipping functions

def resolve_value_mask(
    values: Tensor,
    mask: Tensor | None,
    lens: Tensor | None,
    group_dim: int
) -> Tensor | None:
    if exists(lens):
        # time is the last dimension, unless the group dimension trails it

        max_len = values.shape[-2] if group_dim == -1 and values.ndim > 2 else values.shape[-1]
        mask = default(mask, lens_to_mask(lens, max_len = max_len))

    # grouped values - one value head per reward group - align the mask against the group dimension

    return align_grouped_mask(mask, values.ndim, group_dim)

# classic clipped value loss - Schulman et al.

def clipped_value_loss_fn(
    values: Tensor,
    old_values: Tensor | None = None,
    returns: Tensor = None,
    clip: float | None = 0.2,
    loss_fn: Callable = F.mse_loss,
    mask: Tensor | None = None,
    lens: Tensor | None = None,
    group_dim: int = -2,
    reduction: str = 'mean'
) -> Tensor:
    mask = resolve_value_mask(values, mask, lens, group_dim)

    if not exists(clip) or clip is False or not exists(old_values):
        loss = loss_fn(values, returns, reduction = 'none')
        return masked_reduce(loss, mode = reduction, mask = mask)

    clipped_values = old_values + (values - old_values).clamp(-clip, clip)

    loss = loss_fn(values, returns, reduction = 'none')
    clipped_loss = loss_fn(clipped_values, returns, reduction = 'none')

    value_loss = torch.max(loss, clipped_loss)

    return masked_reduce(value_loss, mode = reduction, mask = mask)

# improved value clipping - https://www.authorea.com/users/855021/articles/1240083-on-analysis-of-clipped-critic-loss-in-proximal-policy-gradient

def improved_value_clipping_fn(
    values: Tensor,
    old_values: Tensor | None = None,
    returns: Tensor = None,
    clip: float | None = 0.2,
    loss_fn: Callable = F.mse_loss,
    scalar_values: Tensor | None = None,
    scalar_old_values: Tensor | None = None,
    mask: Tensor | None = None,
    lens: Tensor | None = None,
    group_dim: int = -2,
    reduction: str = 'mean'
) -> Tensor:
    mask = resolve_value_mask(values, mask, lens, group_dim)

    if not exists(clip) or clip is False or not exists(old_values):
        loss = loss_fn(values, returns, reduction = 'none')
        return masked_reduce(loss, mode = reduction, mask = mask)

    scalar_values = default(scalar_values, values)
    scalar_old_values = default(scalar_old_values, old_values)

    clipped_returns = returns.clamp(scalar_old_values - clip, scalar_old_values + clip)

    loss = loss_fn(values, returns, reduction = 'none')
    clipped_loss = loss_fn(values, clipped_returns, reduction = 'none')

    old_values_lo = scalar_old_values - clip
    old_values_hi = scalar_old_values + clip

    def is_between(mid, lo, hi):
        return (lo < mid) & (mid < hi)

    value_loss = torch.where(
        is_between(scalar_values, returns, old_values_lo) |
        is_between(scalar_values, old_values_hi, returns),
        0.,
        torch.min(loss, clipped_loss)
    )

    return masked_reduce(value_loss, mode = reduction, mask = mask)
