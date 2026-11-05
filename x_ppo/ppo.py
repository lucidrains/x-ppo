from functools import wraps, partial
from typing import NamedTuple

import torch
from torch import is_tensor, Tensor
from torch_einops_utils import z_score, masked_mean, ZScoreStats

from x_ppo.values import resolve_grouped_mask

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def force_fp32(fn):
    # always run in fp32 - the importance ratio readily overflows under amp

    @wraps(fn)
    def inner(*args, **kwargs):
        device_type = next((t.device.type for t in (*args, *kwargs.values()) if is_tensor(t)), 'cpu')

        to_fp32 = lambda t: t.float() if is_tensor(t) and t.is_floating_point() else t

        with torch.autocast(device_type = device_type, enabled = False):
            return fn(*map(to_fp32, args), **{k: to_fp32(v) for k, v in kwargs.items()})

    return inner

def calc_ratios(
    action_log_probs,
    old_action_log_probs,
    log_ratio_clamp = None
):
    log_ratios = action_log_probs - old_action_log_probs

    # optional numerical guard on the log ratio, e.g. (-20., 20.)

    if exists(log_ratio_clamp):
        log_ratios = log_ratios.clamp(*log_ratio_clamp)

    return log_ratios.exp()

# sequence wise - GSPO https://arxiv.org/abs/2507.18071

def calc_sequence_log_ratios(
    action_log_probs,
    old_action_log_probs,
    mask = None,
    dim = -1
):
    # length normalized log ratio - one per trajectory

    return masked_mean(action_log_probs - old_action_log_probs, mask = mask, dim = dim)

# advantages

def z_score_advantages(
    advantages,
    mask = None,
    normalize = False,
    return_stats = False,
    min_std = None
):
    stats = None
    z_mask = mask.bool() if is_tensor(mask) else mask

    if normalize:
        advantages, stats = z_score(advantages, mask = z_mask, return_stats = True)
    elif return_stats or exists(min_std):
        stats = z_score(advantages, mask = z_mask, return_only_stats = True)

    # zero-variance advantages carry no policy gradient - DAPO dynamic sampling https://arxiv.org/abs/2503.14476

    skip_update = exists(min_std) and bool((stats.var <= min_std ** 2).all())

    return advantages, stats, skip_update

def prepare_advantages(
    advantages,
    mask = None,
    normalize = False,
    return_stats = False,
    min_std = None
):
    return z_score_advantages(advantages, mask, normalize, return_stats, min_std)

def prepare_sequence_advantages(
    advantages,
    action_log_probs,
    mask = None,
    dim = -1,
    normalize = False,
    return_stats = False,
    min_std = None
):
    # per token advantages collapse to one score per trajectory - per trajectory inputs pass through

    if advantages.ndim == action_log_probs.ndim:
        advantages = masked_mean(advantages, mask = mask, dim = dim)

    # any valid token marks the trajectory as valid

    seq_mask = None
    if exists(mask):
        seq_mask = mask.bool()

        if seq_mask.ndim > 1:
            mask_dim = dim if seq_mask.ndim == action_log_probs.ndim else -1
            seq_mask = seq_mask.any(dim = mask_dim)
        elif action_log_probs.ndim == 1:
            seq_mask = seq_mask.any()

    advantages, stats, skip_update = z_score_advantages(advantages, seq_mask, normalize, return_stats, min_std)

    return advantages, seq_mask, stats, skip_update

# per action dim - keep per-dim log probs, otherwise sum to the usual joint log prob

def resolve_action_dims(
    action_log_probs,
    old_action_log_probs,
    advantages,
    per_action_dim = False,
    sequence_wise = False
):
    # sequence wise advantages may lack the time dimension, so infer the action dim from the log probs

    has_action_dim = (
        action_log_probs.ndim >= 3 if sequence_wise
        else action_log_probs.ndim == (advantages.ndim + 1)
    )

    if not has_action_dim:
        return action_log_probs, old_action_log_probs, advantages, False

    if not per_action_dim:
        return action_log_probs.sum(dim = -1), old_action_log_probs.sum(dim = -1), advantages, False

    if not sequence_wise or advantages.ndim < action_log_probs.ndim:
        advantages = advantages[..., None]

    return action_log_probs, old_action_log_probs, advantages, True

class ActorLossInputs(NamedTuple):
    log_ratios: Tensor
    advantages: Tensor
    mask: Tensor | None
    has_action_dim: bool
    stats: ZScoreStats | None
    skip_update: bool

def prepare_actor_inputs(
    action_log_probs,
    old_action_log_probs,
    advantages,
    mask = None,
    normalize_advantages = False,
    delightful = False,
    delight_temp = 1.,
    per_action_dim = False,
    sequence_wise = False,
    min_advantage_std = None,
    return_stats = False
):
    resolve = partial(resolve_action_dims, per_action_dim = per_action_dim, sequence_wise = sequence_wise)

    if sequence_wise:
        action_log_probs, old_action_log_probs, advantages, has_action_dim = resolve(
            action_log_probs, old_action_log_probs, advantages
        )

        dim = -2 if has_action_dim else -1

        advantages, seq_mask, stats, skip_update = prepare_sequence_advantages(
            advantages,
            action_log_probs,
            mask = mask,
            dim = dim,
            normalize = normalize_advantages,
            return_stats = return_stats,
            min_std = min_advantage_std
        )

        log_ratios = calc_sequence_log_ratios(action_log_probs, old_action_log_probs, mask = mask, dim = dim)
        delight_log_probs = log_ratios
        mask = seq_mask
    else:
        advantages, stats, skip_update = prepare_advantages(
            advantages,
            mask = mask,
            normalize = normalize_advantages,
            return_stats = return_stats,
            min_std = min_advantage_std
        )

        action_log_probs, old_action_log_probs, advantages, has_action_dim = resolve(
            action_log_probs, old_action_log_probs, advantages
        )

        log_ratios = action_log_probs - old_action_log_probs
        delight_log_probs = action_log_probs

    # delightful policy gradient - Ian Osband https://arxiv.org/abs/2603.14608

    if delightful:
        gate = (-delight_log_probs * advantages / delight_temp).sigmoid().detach()
        advantages = advantages * gate

    return ActorLossInputs(log_ratios, advantages, mask, has_action_dim, stats, skip_update)

@force_fp32
def ppo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    eps_clip = 0.2,
    mask = None,
    normalize_advantages = False,
    delightful = False,
    delight_temp = 1.,
    dual_clip = False,
    dual_clip_threshold = 3.,
    log_ratio_clamp = None,
    per_action_dim = False,
    group_dim = -2,
    min_advantage_std = None,
    return_stats = False,
    sequence_wise = False
):
    mask = resolve_grouped_mask(advantages, action_log_probs, mask, group_dim)

    inputs = prepare_actor_inputs(
        action_log_probs,
        old_action_log_probs,
        advantages,
        mask = mask,
        normalize_advantages = normalize_advantages,
        delightful = delightful,
        delight_temp = delight_temp,
        per_action_dim = per_action_dim,
        sequence_wise = sequence_wise,
        min_advantage_std = min_advantage_std,
        return_stats = return_stats
    )

    log_ratios = inputs.log_ratios

    # optional numerical guard on the log ratio, e.g. (-20., 20.)

    if exists(log_ratio_clamp):
        log_ratios = log_ratios.clamp(*log_ratio_clamp)

    ratios = log_ratios.exp()

    # decoupled clipping - Zhou et al. https://arxiv.org/abs/2503.14476

    eps_clip_low, eps_clip_high = cast_tuple(eps_clip, 2)

    surr1 = ratios * inputs.advantages
    surr2 = ratios.clamp(1. - eps_clip_low, 1. + eps_clip_high) * inputs.advantages

    surr = torch.min(surr1, surr2)

    # dual-clip ppo - Ye et al. https://arxiv.org/abs/1912.09729

    if dual_clip:
        surr = torch.where(inputs.advantages < 0, torch.max(surr, dual_clip_threshold * inputs.advantages), surr)

    loss = -surr

    if inputs.has_action_dim:
        loss = loss.mean(dim = -1)

    if exists(inputs.mask):
        loss = masked_mean(loss, mask = inputs.mask)

    if inputs.skip_update:
        loss = loss * 0.

    return (loss, inputs.stats) if return_stats else loss
