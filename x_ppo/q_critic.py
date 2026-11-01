"""Next-latent Q critic.

Wraps a value network `V(embedding)` and learns Q, reward and terminal heads
from one action-conditioned trunk:

    z' = z + dynamics([z, action_embed(a)])
    r' = reward_head([z, action_embed(a)])
    d' = terminal_head([z, action_embed(a)])
    Q_target = r + gamma * (1 - d) * V(z').detach()

Next-latent prediction - Teoh et al.
https://arxiv.org/abs/2511.05963
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import repeat
from torch import Tensor, arange, cat, is_tensor
from torch.nn import LayerNorm, Linear, Module, Sequential, SiLU
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

class FilteredActions(NamedTuple):
    actions: Tensor
    indices: Tensor
    q_values: Tensor
    log_probs: Tensor | None = None

# main class

class NextLatentQCritic(Module):
    def __init__(
        self,
        value_net: Module,
        *,
        dim_embed: int,
        dim_action: int,
        action_embed_dim: int | None = None,
        hidden_dim: int | None = None,
        depth: int = 2,
        gamma: float | Tensor = 0.99,
        next_latent_coef: float = 1.,
        q_coef: float = 1.,
        value_coef: float = 1.,
        reward_coef: float = 1.,
        terminal_coef: float = 1.
    ):
        super().__init__()
        assert depth >= 1, 'depth must be at least 1'

        self.value_net = value_net
        self.dim_embed = dim_embed
        self.gamma = gamma

        self.next_latent_coef = next_latent_coef
        self.q_coef = q_coef
        self.value_coef = value_coef
        self.reward_coef = reward_coef
        self.terminal_coef = terminal_coef

        action_embed_dim = default(action_embed_dim, dim_action)
        hidden_dim = default(hidden_dim, dim_embed)

        self.action_embed = Sequential(Linear(dim_action, action_embed_dim), SiLU())
        self.net = Sequential(
            LayerNorm(dim_embed + action_embed_dim),
            mlp(dim_embed + action_embed_dim, hidden_dim, depth)
        )

        self.to_next_latent_and_q = Linear(hidden_dim, dim_embed + 1)
        self.to_reward = Linear(hidden_dim, 1)
        self.to_done = Linear(hidden_dim, 1)

        # start the dynamics as an identity

        with torch.no_grad():
            self.to_next_latent_and_q.weight[:dim_embed].zero_()
            self.to_next_latent_and_q.bias[:dim_embed].zero_()

    def forward_value(self, embeddings: Tensor) -> Tensor:
        values = self.value_net(embeddings)

        if values.ndim == embeddings.ndim and values.shape[-1] == 1:
            values = values.squeeze(-1)

        return values

    def _heads(self, embeddings: Tensor, actions: Tensor):
        cond = cat((embeddings, self.action_embed(actions)), dim = -1)
        hidden = self.net(cond)

        next_latent_and_q = self.to_next_latent_and_q(hidden)
        next_latent = embeddings + next_latent_and_q[..., :self.dim_embed]
        q_values = next_latent_and_q[..., self.dim_embed]

        pred_rewards = self.to_reward(hidden)[..., 0]
        done_logits = self.to_done(hidden)[..., 0]
        pred_dones = done_logits.sigmoid()

        return next_latent, q_values, pred_rewards, pred_dones, done_logits

    def q_values(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        _, q_values, *_ = self._heads(embeddings, actions)
        return q_values

    def next_latents(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        next_latent, *_ = self._heads(embeddings, actions)
        return next_latent

    def predicted_rewards(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        _, _, pred_rewards, *_ = self._heads(embeddings, actions)
        return pred_rewards

    def predicted_dones(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        _, _, _, pred_dones, _ = self._heads(embeddings, actions)
        return pred_dones

    def forward(
        self,
        embeddings: Tensor,
        actions: Tensor | None = None,
        rewards: Tensor | None = None,
        next_embeddings: Tensor | None = None,
        returns: Tensor | None = None,
        masks: Tensor | None = None,
        dones: Tensor | None = None,
        return_loss_breakdown: bool = False
    ):
        dtype = embeddings.dtype
        zero = embeddings.new_zeros(())

        values = self.forward_value(embeddings)

        value_loss = q_loss = next_latent_loss = reward_loss = terminal_loss = zero
        q_values = None

        if exists(actions):
            next_latent, q_values, pred_rewards, pred_dones, done_logits = self._heads(embeddings, actions)

            if exists(next_embeddings):
                next_latent_loss = masked_mean(
                    F.smooth_l1_loss(next_latent, next_embeddings.detach(), reduction = 'none'),
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

            target_reward = rewards if exists(rewards) else pred_rewards.detach()

            if exists(dones):
                continuation = default(masks, 1.) * (~dones.bool()).to(dtype)
            elif exists(masks):
                continuation = masks
            else:
                continuation = 1. - pred_dones.detach()

            gamma = self.gamma
            if is_tensor(gamma):
                gamma = gamma.to(device = embeddings.device, dtype = dtype)
                while gamma.ndim < target_reward.ndim:
                    gamma = gamma[..., None]

            target_q = target_reward + gamma * continuation * self.forward_value(next_latent).detach()

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
            self.terminal_coef * terminal_loss
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
            q_values = q_values
        )

    @torch.no_grad()
    def filter_actions(
        self,
        embeddings: Tensor,             # (b d) or (b k d)
        candidate_actions: Tensor,      # (b k a)
        candidate_log_probs: Tensor | None = None
    ) -> FilteredActions:
        batch, num_candidates = candidate_actions.shape[:2]

        if embeddings.ndim == 2:
            embeddings = repeat(embeddings, 'b d -> b k d', k = num_candidates)
        elif embeddings.shape[1] == 1 and num_candidates > 1:
            embeddings = repeat(embeddings, 'b 1 d -> b k d', k = num_candidates)

        q_values = self.q_values(embeddings, candidate_actions)
        indices = q_values.argmax(dim = -1)

        batch_indices = arange(batch, device = candidate_actions.device)
        actions = candidate_actions[batch_indices, indices]

        log_probs = None
        if exists(candidate_log_probs):
            log_probs = candidate_log_probs[batch_indices, indices]

        return FilteredActions(
            actions = actions,
            indices = indices,
            q_values = q_values,
            log_probs = log_probs
        )
