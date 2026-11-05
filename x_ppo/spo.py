import torch
from torch_einops_utils import masked_mean

from x_ppo.ppo import cast_tuple, force_fp32, prepare_actor_inputs
from x_ppo.values import resolve_grouped_mask

# helpers

def exists(val):
    return val is not None

# simple policy optimization - Xie et al. https://arxiv.org/abs/2401.16025

@force_fp32
def spo_actor_loss(
    action_log_probs,
    old_action_log_probs,
    advantages,
    eps_clip = 0.2,
    mask = None,
    normalize_advantages = False,
    delightful = False,
    delight_temp = 1.,
    asymmetric = False,
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

    eps = torch.where(ratios >= 1., eps_clip_high, eps_clip_low)

    surr = (
        ratios * inputs.advantages -
        (inputs.advantages.abs() * (ratios - 1.).square()) / (2 * eps)
    )

    # dual-clip ppo - Ye et al. https://arxiv.org/abs/1912.09729

    if dual_clip:
        surr = torch.where(inputs.advantages < 0, torch.max(surr, dual_clip_threshold * inputs.advantages), surr)

    if asymmetric:
        # asymmetric spo - https://arxiv.org/abs/2510.06062v1
        clipped_ratios = ratios.clamp(1. - eps_clip_low, 1. + eps_clip_high)
        ppo_surr = torch.min(ratios * inputs.advantages, clipped_ratios * inputs.advantages)
        surr = torch.where(inputs.advantages > 0, ppo_surr, surr)

    loss = -surr

    if inputs.has_action_dim:
        loss = loss.mean(dim = -1)

    if exists(inputs.mask):
        loss = masked_mean(loss, mask = inputs.mask)

    if inputs.skip_update:
        loss = loss * 0.

    return (loss, inputs.stats) if return_stats else loss
