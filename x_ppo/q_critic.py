"""Next-latent Q critic.

Wraps a value network `V(embedding)` and learns Q, reward and terminal heads
from one action-conditioned trunk:

    z' = z + dynamics([z, action_embed(a)])
    r' = reward_head([z, action_embed(a)])
    d' = terminal_head([z, action_embed(a)])
    Q_target = r + (gamma ** tau) * (1 - d) * V(z').detach()

`durations` (tau >= 1, defaulting to 1) and `q_targets` (e.g. `calc_smdp_gae` returns)
generalize the critic and candidate action filter to multi-horizon semi-MDPs.
`next_values` can replace `V(z')` for real transitions.

Soft expert iteration distills a Q-ranked candidate set into the policy:

    w_i = softmax(Q(s, a_i) / beta)
    loss = -sum_i w_i log pi(a_i | s)

`kl_budget` adapts beta per state. `beta -> 0` recovers hard argmax
distillation; `beta -> inf` recovers behavior cloning.

Next-latent prediction - Teoh et al.
https://arxiv.org/abs/2511.05963
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import repeat
from torch import Tensor, arange, as_tensor, cat, is_tensor
from torch.distributions import Normal
from torch.nn import Embedding, LayerNorm, Linear, Module, Sequential, SiLU
from torch_einops_utils import masked_mean

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def mlp(dim_in, dim_hidden, depth):
    layers = []
    dim = dim_in

    for _ in range(depth):
        layers.extend((Linear(dim, dim_hidden), SiLU()))
        dim = dim_hidden

    return Sequential(*layers)

# output containers

class QCriticLosses(NamedTuple):
    loss: Tensor
    value: Tensor
    q: Tensor
    next_latent: Tensor
    reward: Tensor
    terminal: Tensor
    q_values: Tensor | None = None
    duration: Tensor | None = None

class FilteredActions(NamedTuple):
    actions: Tensor
    indices: Tensor
    q_values: Tensor
    log_probs: Tensor | None = None
    scores: Tensor | None = None

class SoftExitStats(NamedTuple):
    weights: Tensor
    beta: Tensor
    kl: Tensor
    ess: Tensor

class QHeads(NamedTuple):
    next_latent: Tensor
    next_latent_mean: Tensor
    next_latent_std: Tensor | None
    q_values: Tensor
    pred_rewards: Tensor
    pred_dones: Tensor
    done_logits: Tensor
    pred_durations: Tensor | None = None

# main class

class NextLatentQCritic(Module):
    def __init__(
        self,
        value_net: Module,
        *,
        dim_embed: int,
        dim_action: int | None = None,
        num_actions: int | None = None,
        action_embed: Module | None = None,
        action_embed_dim: int | None = None,
        action_horizons: Tensor | tuple[int, ...] | list[int] | None = None,
        discrete_actions: bool | None = None,
        hidden_dim: int | None = None,
        depth: int = 2,
        gamma: float | Tensor = 0.99,
        next_latent_coef: float = 1.,
        q_coef: float = 1.,
        value_coef: float = 1.,
        reward_coef: float = 1.,
        terminal_coef: float = 1.,
        duration_coef: float = 0.25,
        probabilistic_next_latent: bool = False,
        next_latent_init_std: float = 0.1,
        next_latent_logvar_range: tuple[float, float] = (-10., 2.)
    ):
        super().__init__()
        assert depth >= 1, 'depth must be at least 1'
        assert next_latent_init_std > 0., 'next_latent_init_std must be positive'
        assert next_latent_logvar_range[0] < next_latent_logvar_range[1], 'next_latent_logvar_range must be increasing'

        self.value_net = value_net
        self.dim_embed = dim_embed
        self.gamma = gamma
        self.probabilistic_next_latent = probabilistic_next_latent
        self.next_latent_logvar_range = next_latent_logvar_range

        self.next_latent_coef = next_latent_coef
        self.q_coef = q_coef
        self.value_coef = value_coef
        self.reward_coef = reward_coef
        self.terminal_coef = terminal_coef
        self.duration_coef = duration_coef

        hidden_dim = default(hidden_dim, dim_embed)
        self.discrete_actions = default(discrete_actions, exists(num_actions) or (exists(action_horizons) and not exists(dim_action)))

        if exists(action_embed):
            assert exists(action_embed_dim), 'action_embed_dim must be provided when passing a custom action_embed module'
            self.action_embed = action_embed
        elif exists(num_actions):
            action_embed_dim = default(action_embed_dim, 32)
            self.action_embed = Embedding(num_actions, action_embed_dim)
        else:
            assert exists(dim_action), 'one of dim_action, num_actions, or action_embed must be provided'
            action_embed_dim = default(action_embed_dim, dim_action)
            self.action_embed = Sequential(Linear(dim_action, action_embed_dim), SiLU())

        self.net = Sequential(
            LayerNorm(dim_embed + action_embed_dim),
            mlp(dim_embed + action_embed_dim, hidden_dim, depth)
        )

        self.to_next_latent_and_q = Linear(hidden_dim, dim_embed + 1)
        self.to_reward = Linear(hidden_dim, 1)
        self.to_done = Linear(hidden_dim, 1)

        if exists(action_horizons):
            horizons_t = as_tensor(action_horizons, dtype = torch.float32)
            assert bool((horizons_t >= 1.).all()), 'action_horizons must all be >= 1'
            self.register_buffer('action_horizons', horizons_t, persistent = False)
            self.to_duration = Linear(hidden_dim, 1)
        else:
            self.register_buffer('action_horizons', None, persistent = False)
            self.to_duration = None

        self.to_next_latent_logvar = Linear(hidden_dim, dim_embed) if probabilistic_next_latent else None

        # start the dynamics as an identity, and predicted option duration near nominal horizon

        with torch.no_grad():
            self.to_next_latent_and_q.weight[:dim_embed].zero_()
            self.to_next_latent_and_q.bias[:dim_embed].zero_()

            if exists(self.to_duration):
                self.to_duration.weight.zero_()
                self.to_duration.bias.fill_(3.)

            if exists(self.to_next_latent_logvar):
                self.to_next_latent_logvar.weight.zero_()
                self.to_next_latent_logvar.bias.fill_(2. * math.log(next_latent_init_std))

    def _align_gamma(self, ref: Tensor) -> Tensor:
        gamma = as_tensor(self.gamma, device = ref.device, dtype = ref.dtype)
        while gamma.ndim < ref.ndim:
            gamma = gamma[..., None]
        return gamma

    def _discount_factor(self, ref: Tensor, durations: Tensor | None = None) -> Tensor:
        gamma = self._align_gamma(ref)
        if not exists(durations):
            return gamma
        return gamma ** durations.to(device = ref.device, dtype = ref.dtype)

    def _discounted_duration(self, ref: Tensor, durations: Tensor) -> Tensor:
        gamma = self._align_gamma(ref)
        durs = durations.to(device = ref.device, dtype = ref.dtype)
        log_g = gamma.clamp_min(1e-12).log()
        denom = -torch.expm1(log_g)
        return torch.where((1. - gamma).abs() < 1e-6, durs, -torch.expm1(durs * log_g) / denom.clamp_min(1e-12))

    def _nominal_horizons(self, actions: Tensor, dtype: torch.dtype) -> Tensor:
        horizon_idx = actions.long() if self.discrete_actions else actions[..., 0].long()
        return self.action_horizons[horizon_idx].to(dtype = dtype)

    def forward_value(self, embeddings: Tensor) -> Tensor:
        values = self.value_net(embeddings)

        if values.ndim == embeddings.ndim and values.shape[-1] == 1:
            values = values.squeeze(-1)

        return values

    def _heads(
        self,
        embeddings: Tensor,
        actions: Tensor,
        sample_next_latent: bool = True
    ) -> QHeads:
        act_emb = self.action_embed(actions.long() if self.discrete_actions else actions)
        cond = cat((embeddings, act_emb), dim = -1)
        hidden = self.net(cond)

        next_latent_and_q = self.to_next_latent_and_q(hidden)
        next_latent_mean = embeddings + next_latent_and_q[..., :self.dim_embed]
        q_values = next_latent_and_q[..., self.dim_embed]

        next_latent_std = None
        next_latent = next_latent_mean

        if exists(self.to_next_latent_logvar):
            logvar = self.to_next_latent_logvar(hidden).clamp(*self.next_latent_logvar_range)
            next_latent_std = (0.5 * logvar).exp()

            if sample_next_latent:
                next_latent = next_latent_mean + next_latent_std * torch.randn_like(next_latent_mean)

        pred_rewards = self.to_reward(hidden)[..., 0]
        done_logits = self.to_done(hidden)[..., 0]
        pred_dones = done_logits.sigmoid()

        pred_durations = None
        if exists(self.to_duration) and exists(self.action_horizons):
            nom_h = self._nominal_horizons(actions, embeddings.dtype)
            pred_durations = 1. + (nom_h - 1.) * self.to_duration(hidden)[..., 0].sigmoid()

        return QHeads(
            next_latent = next_latent,
            next_latent_mean = next_latent_mean,
            next_latent_std = next_latent_std,
            q_values = q_values,
            pred_rewards = pred_rewards,
            pred_dones = pred_dones,
            done_logits = done_logits,
            pred_durations = pred_durations
        )

    def _model_q_from_heads(
        self,
        heads: QHeads,
        durations: Tensor | None = None
    ) -> Tensor:
        eff_dur = default(durations, heads.pred_durations)
        step_gamma = self._discount_factor(heads.pred_rewards, eff_dur)
        return heads.pred_rewards + step_gamma * (1. - heads.pred_dones) * self.forward_value(heads.next_latent)

    def _effective_q_from_heads(
        self,
        heads: QHeads,
        conservative_weight: float = 0.,
        durations: Tensor | None = None
    ) -> Tensor:
        if conservative_weight <= 0.:
            return heads.q_values

        q_model = self._model_q_from_heads(heads, durations)
        return heads.q_values - (2. * conservative_weight) * (heads.q_values - q_model).clamp_min(0.)

    def q_values(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        return self._heads(embeddings, actions, sample_next_latent = False).q_values

    def model_q_values(
        self,
        embeddings: Tensor,
        actions: Tensor,
        durations: Tensor | None = None
    ) -> Tensor:
        heads = self._heads(embeddings, actions, sample_next_latent = False)
        return self._model_q_from_heads(heads, durations)

    def conservative_q_values(
        self,
        embeddings: Tensor,
        actions: Tensor,
        conservative_weight: float = 0.25,
        durations: Tensor | None = None
    ) -> Tensor:
        heads = self._heads(embeddings, actions, sample_next_latent = False)
        return self._effective_q_from_heads(heads, conservative_weight, durations)

    def smdp_advantage_scores(
        self,
        embeddings: Tensor,
        actions: Tensor,
        *,
        durations: Tensor | None = None,
        rate_advantage: bool = False,
        conservative_weight: float = 0.
    ) -> Tensor:
        # supports embeddings of shape (..., d) against actions of shape (..., k) or (..., k, a)

        has_cand_dim = (
            actions.ndim == embeddings.ndim if self.discrete_actions
            else actions.ndim == embeddings.ndim + 1
        )

        if has_cand_dim:
            num_candidates = actions.shape[-1] if self.discrete_actions else actions.shape[-2]
            values = self.forward_value(embeddings)[..., None]
            rep_embeddings = repeat(embeddings, '... d -> ... k d', k = num_candidates)
        else:
            values = self.forward_value(embeddings)
            rep_embeddings = embeddings

        heads = self._heads(rep_embeddings, actions, sample_next_latent = False)
        eff_dur = default(durations, heads.pred_durations)
        q_eff = self._effective_q_from_heads(heads, conservative_weight, eff_dur)
        adv = q_eff - values

        if rate_advantage and exists(eff_dur):
            disc_dur = self._discounted_duration(adv, eff_dur)
            return torch.where(adv > 0., adv / disc_dur, adv)

        return adv

    def next_latents(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        return self._heads(embeddings, actions, sample_next_latent = False).next_latent

    def next_latent_std(self, embeddings: Tensor, actions: Tensor) -> Tensor | None:
        return self._heads(embeddings, actions, sample_next_latent = False).next_latent_std

    def next_latent_distribution(self, embeddings: Tensor, actions: Tensor) -> Normal:
        if not exists(self.to_next_latent_logvar):
            raise ValueError('next latent prediction is deterministic - enable probabilistic_next_latent')

        heads = self._heads(embeddings, actions, sample_next_latent = False)
        return Normal(heads.next_latent_mean, heads.next_latent_std)

    def predicted_rewards(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        return self._heads(embeddings, actions, sample_next_latent = False).pred_rewards

    def predicted_dones(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        return self._heads(embeddings, actions, sample_next_latent = False).pred_dones

    def predicted_durations(self, embeddings: Tensor, actions: Tensor) -> Tensor | None:
        return self._heads(embeddings, actions, sample_next_latent = False).pred_durations

    def forward(
        self,
        embeddings: Tensor,
        actions: Tensor | None = None,
        rewards: Tensor | None = None,
        next_embeddings: Tensor | None = None,
        next_values: Tensor | None = None,
        returns: Tensor | None = None,
        masks: Tensor | None = None,
        dones: Tensor | None = None,
        durations: Tensor | None = None,
        q_targets: Tensor | None = None,
        return_loss_breakdown: bool = False
    ):
        dtype = embeddings.dtype
        zero = embeddings.new_zeros(())

        values = self.forward_value(embeddings)

        value_loss = q_loss = next_latent_loss = reward_loss = terminal_loss = duration_loss = zero
        q_values = None

        if exists(actions):
            heads = self._heads(embeddings, actions, sample_next_latent = self.training)
            next_latent, q_values = heads.next_latent, heads.q_values
            pred_rewards, pred_dones, done_logits = heads.pred_rewards, heads.pred_dones, heads.done_logits

            if exists(next_embeddings):
                next_embeddings = next_embeddings.detach()

                if exists(heads.next_latent_std):
                    next_latent_loss = masked_mean(
                        F.gaussian_nll_loss(
                            heads.next_latent_mean,
                            next_embeddings,
                            heads.next_latent_std.square().clamp_min(1e-6),
                            reduction = 'none',
                            eps = 1e-6
                        ),
                        mask = masks
                    )
                else:
                    next_latent_loss = masked_mean(
                        F.smooth_l1_loss(next_latent, next_embeddings, reduction = 'none'),
                        mask = masks
                    )

            if exists(rewards):
                reward_loss = masked_mean(
                    F.smooth_l1_loss(pred_rewards, rewards, reduction = 'none'),
                    mask = masks
                )

            if exists(dones):
                terminal_loss = masked_mean(
                    F.binary_cross_entropy_with_logits(done_logits, dones.float(), reduction = 'none'),
                    mask = masks
                )

            if exists(durations) and exists(heads.pred_durations) and exists(self.action_horizons):
                nom_h = self._nominal_horizons(actions, dtype).clamp_min(1.)
                duration_loss = masked_mean(
                    F.smooth_l1_loss(
                        heads.pred_durations / nom_h,
                        durations.to(device = embeddings.device, dtype = dtype) / nom_h,
                        reduction = 'none'
                    ),
                    mask = masks
                )

            if exists(q_targets):
                target_q = q_targets.to(device = embeddings.device, dtype = dtype).detach()
            else:
                target_reward = rewards if exists(rewards) else pred_rewards.detach()

                if exists(dones):
                    continuation = default(masks, 1.) * (~dones.bool()).to(dtype)
                elif exists(masks):
                    continuation = masks
                else:
                    continuation = 1. - pred_dones.detach()

                eff_dur = default(durations, heads.pred_durations.detach() if exists(heads.pred_durations) else None)
                step_gamma = self._discount_factor(target_reward, eff_dur)

                # bootstrap from supplied values when available, else the predicted latent

                if exists(next_values):
                    bootstrap_values = next_values if is_tensor(next_values) else as_tensor(next_values)
                    bootstrap_values = bootstrap_values.to(device = embeddings.device, dtype = dtype)

                    if bootstrap_values.ndim == target_reward.ndim - 1:
                        bootstrap_values = bootstrap_values[..., None]
                    elif bootstrap_values.ndim == target_reward.ndim + 1 and bootstrap_values.shape[-1] == 1:
                        bootstrap_values = bootstrap_values[..., 0]
                else:
                    bootstrap_values = self.forward_value(next_latent)

                target_q = target_reward + step_gamma * continuation * bootstrap_values.detach()

            q_loss = masked_mean(
                F.smooth_l1_loss(q_values, target_q, reduction = 'none'),
                mask = masks
            )

        if exists(returns):
            value_loss = masked_mean(
                F.mse_loss(values, returns, reduction = 'none'),
                mask = masks
            )

        loss = (
            self.value_coef * value_loss +
            self.q_coef * q_loss +
            self.next_latent_coef * next_latent_loss +
            self.reward_coef * reward_loss +
            self.terminal_coef * terminal_loss +
            self.duration_coef * duration_loss
        )

        if not return_loss_breakdown:
            return loss

        return loss, QCriticLosses(
            loss = loss,
            value = value_loss,
            q = q_loss,
            next_latent = next_latent_loss,
            reward = reward_loss,
            terminal = terminal_loss,
            q_values = q_values,
            duration = duration_loss
        )

    @torch.no_grad()
    def filter_actions(
        self,
        embeddings: Tensor,             # (b d) or (b k d)
        candidate_actions: Tensor,      # (b k a) or (b k) for discrete options
        candidate_log_probs: Tensor | None = None,
        *,
        durations: Tensor | None = None,
        rate_advantage: bool = False,
        conservative_weight: float = 0.,
        candidate_mask: Tensor | None = None
    ) -> FilteredActions:
        batch, num_candidates = candidate_actions.shape[:2]

        raw_embeddings = embeddings[:, 0] if embeddings.ndim == 3 else embeddings

        if embeddings.ndim == 2:
            rep_embeddings = repeat(embeddings, 'b d -> b k d', k = num_candidates)
        elif embeddings.shape[1] == 1 and num_candidates > 1:
            rep_embeddings = repeat(embeddings, 'b 1 d -> b k d', k = num_candidates)
        else:
            rep_embeddings = embeddings

        heads = self._heads(rep_embeddings, candidate_actions, sample_next_latent = False)
        q_values = heads.q_values

        if rate_advantage or conservative_weight > 0.:
            eff_dur = default(durations, heads.pred_durations)
            q_eff = self._effective_q_from_heads(heads, conservative_weight, eff_dur)
            values = self.forward_value(raw_embeddings)[..., None]
            adv = q_eff - values

            if rate_advantage and exists(eff_dur):
                disc_dur = self._discounted_duration(adv, eff_dur)
                scores = torch.where(adv > 0., adv / disc_dur, adv)
            else:
                scores = adv
        else:
            scores = q_values

        masked_scores = scores.masked_fill(~candidate_mask.bool(), -torch.inf) if exists(candidate_mask) else scores
        indices = masked_scores.argmax(dim = -1)

        batch_indices = arange(batch, device = candidate_actions.device)
        actions = candidate_actions[batch_indices, indices]

        log_probs = None
        if exists(candidate_log_probs):
            log_probs = candidate_log_probs[batch_indices, indices]

        return FilteredActions(
            actions = actions,
            indices = indices,
            q_values = q_values,
            log_probs = log_probs,
            scores = scores
        )

# soft expert iteration

def _softmax_with_mask(
    q_values: Tensor,
    mask: Tensor | None,
    eps: float
) -> Tensor:
    if not exists(mask):
        return q_values.softmax(dim = -1)

    q_values = q_values.masked_fill(~mask, -torch.inf)
    no_valid = ~mask.any(dim = -1)

    weights = q_values.softmax(dim = -1)
    weights = weights * mask
    weights = weights / weights.sum(dim = -1, keepdim = True).clamp_min(eps)

    # all-invalid rows fall back to uniform, the caller masks their loss

    if no_valid.any():
        uniform = torch.full_like(weights, 1. / weights.shape[-1])
        weights = torch.where(no_valid[..., None], uniform, weights)

    return weights

def _candidate_kl(
    q_values: Tensor,
    log_beta: Tensor,
    mask: Tensor | None,
    log_num_candidates: Tensor,
    eps: float
) -> Tensor:
    weights = _softmax_with_mask(q_values / log_beta.exp()[..., None], mask, eps)
    return (weights * (weights.clamp_min(eps).log() + log_num_candidates[..., None])).sum(dim = -1)

def soft_exit_weights(
    q_values: Tensor,                     # (..., k)
    *,
    beta: float | Tensor = 1.,
    kl_budget: float | Tensor | None = None,
    mask: Tensor | None = None,           # (..., k)
    eps: float = 1e-6,
    num_bisect_steps: int = 20,
    return_stats: bool = False
) -> Tensor | tuple[Tensor, SoftExitStats]:
    assert q_values.ndim >= 1, 'q_values must have a candidate dimension'
    assert q_values.shape[-1] > 0, 'q_values must have at least one candidate'

    dtype, device = q_values.dtype, q_values.device
    num_candidates = q_values.shape[-1]

    q_values = q_values.detach()

    if exists(mask):
        mask = mask.bool()
        assert mask.shape == q_values.shape, 'mask must match q_values'
        q_values = q_values.masked_fill(~mask, -torch.inf)

    # center before dividing by beta

    max_q = q_values.max(dim = -1, keepdim = True).values
    max_q = torch.where(torch.isfinite(max_q), max_q, torch.zeros_like(max_q))
    q_values = q_values - max_q

    if exists(mask):
        valid_count = mask.sum(dim = -1).to(dtype)
    else:
        valid_count = torch.full(q_values.shape[:-1], num_candidates, device = device, dtype = dtype)

    # all-invalid rows are uniform and contribute zero kl

    valid_count_for_kl = torch.where(valid_count > 0, valid_count, torch.full_like(valid_count, num_candidates))
    log_num_candidates = valid_count_for_kl.log()

    if exists(kl_budget):
        target_kl = kl_budget if is_tensor(kl_budget) else as_tensor(kl_budget)
        target_kl = target_kl.to(device = device, dtype = dtype).expand(q_values.shape[:-1])

        low = torch.full_like(target_kl, -20.)
        high = torch.full_like(target_kl, 20.)

        for _ in range(num_bisect_steps):
            mid = (low + high) * 0.5
            kl_mid = _candidate_kl(q_values, mid, mask, log_num_candidates, eps)

            too_high = kl_mid > target_kl
            low = torch.where(too_high, mid, low)
            high = torch.where(too_high, high, mid)

        log_beta = high
        beta = log_beta.exp()
    else:
        beta = as_tensor(beta, device = device, dtype = dtype).expand(q_values.shape[:-1])
        assert bool((beta > 0).all()), 'beta must be positive'
        log_beta = beta.log()

    weights = _softmax_with_mask(q_values / beta[..., None], mask, eps)
    kl = _candidate_kl(q_values, log_beta, mask, log_num_candidates, eps)
    ess = 1. / weights.square().sum(dim = -1).clamp_min(eps)

    stats = SoftExitStats(weights = weights, beta = beta, kl = kl, ess = ess)
    return (weights, stats) if return_stats else weights

def soft_exit_loss(
    candidate_log_probs: Tensor,          # (..., k) - sum action dims before passing
    q_values: Tensor,                     # (..., k)
    *,
    beta: float | Tensor = 1.,
    kl_budget: float | Tensor | None = None,
    mask: Tensor | None = None,           # (..., k)
    reduction: str = 'mean',
    eps: float = 1e-6,
    num_bisect_steps: int = 20,
    return_stats: bool = False
) -> Tensor | tuple[Tensor, SoftExitStats]:
    assert candidate_log_probs.shape == q_values.shape, 'candidate_log_probs and q_values must match'

    if exists(mask):
        mask = mask.bool()
        assert mask.shape == q_values.shape, 'mask must match q_values'

    weights, stats = soft_exit_weights(
        q_values,
        beta = beta,
        kl_budget = kl_budget,
        mask = mask,
        eps = eps,
        num_bisect_steps = num_bisect_steps,
        return_stats = True
    )

    if exists(mask):
        candidate_log_probs = candidate_log_probs.masked_fill(~mask, 0.)

    per_state = -(weights * candidate_log_probs).sum(dim = -1)

    if reduction == 'none':
        loss = per_state
    elif reduction == 'sum':
        loss = per_state.sum()
    elif reduction == 'mean':
        if exists(mask):
            state_mask = mask.any(dim = -1)
            loss = masked_mean(per_state, mask = state_mask) if bool(state_mask.any()) else per_state.sum() * 0.
        else:
            loss = per_state.mean()
    else:
        raise ValueError(f'unknown reduction {reduction}')

    return (loss, stats) if return_stats else loss
