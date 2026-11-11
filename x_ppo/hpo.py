import torch
from torch_einops_utils import masked_mean

from x_ppo.ppo import cast_tuple, exists, force_fp32, prepare_actor_inputs
from x_ppo.values import resolve_grouped_mask

@force_fp32
def hpo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    eps_clip = 0.2,
    target_scale = 1.25,
    mask = None,
    normalize_advantages = False,
    delightful = False,
    delight_temp = 1.,
    log_ratio_clamp = None,
    per_action_dim = False,
    group_dim = -2,
    min_advantage_std = 1e-6,
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
    advantages = inputs.advantages

    eps_clip_low, eps_clip_high = cast_tuple(eps_clip, 2)
    target_scale_low, target_scale_high = cast_tuple(target_scale, 2)

    if min(target_scale_low, target_scale_high) < 1.:
        raise ValueError('target_scale must be >= 1. to preserve exact on-policy gradient')

    if exists(log_ratio_clamp):
        clamp_low, clamp_high = log_ratio_clamp
        min_low = -(target_scale_low + 1.) * eps_clip_low
        min_high = (target_scale_high + 1.) * eps_clip_high

        if clamp_low > min_low or clamp_high < min_high:
            raise ValueError(f'log_ratio_clamp too tight for hpo - needs ({min_low}, {min_high})')

        # straight-through, preserve restoring gradient
        clamped = log_ratios.clamp(clamp_low, clamp_high)
        log_ratios = log_ratios + (clamped - log_ratios).detach()

    is_pos = advantages >= 0.
    eps = torch.where(is_pos, eps_clip_high, eps_clip_low)
    scale = torch.where(is_pos, target_scale_high, target_scale_low)
    sign = torch.where(is_pos, 1., -1.)

    # mollified sign target - exact on-policy gradient A, smooth ramp to zero at target, bounded restore beyond
    target = scale * eps * sign
    weight = advantages.abs()

    # huber - smooth gradient decay at the target, no clip cliff
    error = log_ratios - target
    abs_error = error.abs()
    loss = weight * torch.where(abs_error < eps, 0.5 * error.square() / eps, abs_error - 0.5 * eps)

    if inputs.has_action_dim:
        loss = loss.mean(dim = -1)

    if exists(inputs.mask):
        loss = masked_mean(loss, mask = inputs.mask)

    if inputs.skip_update:
        loss = loss * 0.

    return (loss, inputs.stats) if return_stats else loss
