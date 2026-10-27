from functools import wraps

import torch
from torch import is_tensor
from torch_einops_utils import z_score, masked_mean

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

def prepare_advantages(
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

# per action dim - keep per-dim log probs, otherwise sum to the usual joint log prob

def resolve_action_dims(
    action_log_probs,
    old_action_log_probs,
    advantages,
    per_action_dim = False
):
    has_action_dim = action_log_probs.ndim == (advantages.ndim + 1)

    if not has_action_dim:
        return action_log_probs, old_action_log_probs, advantages, False

    if not per_action_dim:
        return action_log_probs.sum(dim = -1), old_action_log_probs.sum(dim = -1), advantages, False

    return action_log_probs, old_action_log_probs, advantages[..., None], True

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
    return_stats = False
):
    mask = resolve_grouped_mask(advantages, action_log_probs, mask, group_dim)

    advantages, stats, skip_update = prepare_advantages(
        advantages,
        mask = mask,
        normalize = normalize_advantages,
        return_stats = return_stats,
        min_std = min_advantage_std
    )

    action_log_probs, old_action_log_probs, advantages, has_action_dim = resolve_action_dims(
        action_log_probs,
        old_action_log_probs,
        advantages,
        per_action_dim = per_action_dim
    )

    # delightful policy gradient - Ian Osband https://arxiv.org/abs/2603.14608

    if delightful:
        gate = (-action_log_probs * advantages / delight_temp).sigmoid().detach()
        advantages = advantages * gate

    ratios = calc_ratios(action_log_probs, old_action_log_probs, log_ratio_clamp)

    # decoupled clipping - Zhou et al. https://arxiv.org/abs/2503.14476

    eps_clip_low, eps_clip_high = cast_tuple(eps_clip, 2)

    surr1 = ratios * advantages
    surr2 = ratios.clamp(1. - eps_clip_low, 1. + eps_clip_high) * advantages

    surr = torch.min(surr1, surr2)

    # dual-clip ppo - Ye et al. https://arxiv.org/abs/1912.09729

    if dual_clip:
        surr = torch.where(advantages < 0, torch.max(surr, dual_clip_threshold * advantages), surr)

    loss = -surr

    if has_action_dim:
        loss = loss.mean(dim = -1)

    if exists(mask):
        loss = masked_mean(loss, mask = mask)

    if skip_update:
        loss = loss * 0.

    return (loss, stats) if return_stats else loss
