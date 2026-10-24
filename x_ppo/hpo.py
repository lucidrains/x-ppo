import torch
from torch_einops_utils import masked_mean

from x_ppo.ppo import cast_tuple, force_fp32, exists, prepare_advantages
from x_ppo.values import resolve_grouped_mask

@force_fp32
def hpo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    eps_clip = 0.2,
    mask = None,
    normalize_advantages = False,
    delightful = False,
    delight_temp = 1.,
    log_ratio_clamp = None,
    advantage_clip = 2.,
    group_dim = -2,
    min_advantage_std = 1e-6,
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

    if delightful:
        gate = (-action_log_probs * advantages / delight_temp).sigmoid().detach()
        advantages = advantages * gate

    log_ratios = action_log_probs - old_action_log_probs
    eps_clip_low, eps_clip_high = cast_tuple(eps_clip, 2)

    if exists(log_ratio_clamp):
        clamp_low, clamp_high = log_ratio_clamp
        bound = 2 * max(eps_clip_low, eps_clip_high)

        if clamp_low > -bound or clamp_high < bound:
            raise ValueError(f'log_ratio_clamp too tight for hpo - needs ({-bound}, {bound})')

        # straight-through, preserve restoring gradient
        clamped = log_ratios.clamp(clamp_low, clamp_high)
        log_ratios = log_ratios + (clamped - log_ratios).detach()

    eps = torch.where(advantages >= 0., eps_clip_high, eps_clip_low)

    # fixed target
    advantage_clip_low, advantage_clip_high = cast_tuple(advantage_clip, 2)
    target = eps * advantages.clamp(-advantage_clip_low, advantage_clip_high)
    weight = advantages.abs()

    # huber - smooth gradient decay at the target, no clip cliff
    error = log_ratios - target
    abs_error = error.abs()
    loss = weight * torch.where(abs_error < eps, 0.5 * error.square() / eps, abs_error - 0.5 * eps)

    if exists(mask):
        loss = masked_mean(loss, mask = mask)

    if skip_update:
        loss = loss * 0.

    return (loss, stats) if return_stats else loss
