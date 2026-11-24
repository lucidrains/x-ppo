"""DAgger distillation - one module, any action space.

A policy is any callable `state -> torch Distribution`, or a pytree (tuple,
dict, nested) of distributions for composite action spaces - anything with
`log_prob` / `sample` works (categorical, one-hot, bernoulli, normal, beta, a
chunk of actions, a mixture of heads). The teacher may just as well hand over
raw actions. The student is the caller's - the distiller only ever calls it,
so gradients only ever flow through it, and the teacher is always evaluated
under `torch.no_grad()`.

The whole ladder, simplest first:

    from x_ppo.dagger_distill import DaggerDistiller, kl_loss, mse_loss

    # 1. student and teacher in, one distill loss out - no env, no optimizer

    distiller = DaggerDistiller(student, teacher)

    distill_loss, student_distr = distiller(states)
    distill_loss.backward()

    # 2. swap the objective (default: nll of the teacher's action)

    DaggerDistiller(student, teacher, loss_fn = kl_loss)     # soft - match distributions
    DaggerDistiller(student, teacher, loss_fn = mse_loss)    # deterministic - match means

    # 3. representation alignment - hook any layer of each, default 1 - cosine

    DaggerDistiller(student, teacher, student_layer = 'trunk.2', teacher_layer = 'blocks.4')

    # 4. composite action spaces need nothing new - return a pytree of heads,
    #    the nll sums over them and `best_action` maps the same structure

    def student(state):
        return dict(
            buttons = Bernoulli(logits = button_net(state)),
            arm = beta(arm_net(state).reshape(*state.shape[:-1], -1, 2))
        )

    DaggerDistiller(student, teacher)

    # 5. the env's observation may be a dict - each side reads its own subset,
    #    or they disagree on what an action means (a phase, an action scaling)

    def student_view(obs, info, step):     # (raw obs, info, step) -> the student's state
        return obs['state']                # deployable fields only

    def teacher_view(obs, info, step):     # (raw obs, info, step) -> the teacher's state
        return obs                         # privileged fields and all

    DaggerDistiller(student, teacher,
        student_state_fn = student_view,
        teacher_obs_fn = teacher_view,     # teacher_state_fn instead refines the student's state
        label_transform = my_transform)    # (student state, action, info, step) -> student-space action

    # 6. hand it the env and it runs the whole DAgger loop itself - roll the
    #    student out, label every visited state, fit

    distiller = DaggerDistiller(student, teacher, env = env, optimizer = optimizer)
    distiller.pretrain(128, epochs = 4)   # teacher-rollout BC warmup

    for cycle in range(cycles):
        stats = distiller.step(collect_steps = 128, epochs = 2)

`kl_loss` and `mse_loss` re-evaluate the teacher on its buffered inputs at fit
time - keep the teacher in eval mode. A raw-action teacher or a
`label_transform` falls back to the stored action.

`state` is batched and may be a pytree - a dict observation rides through the
buffer; `info` carries the env's info plus `episode_steps` (per env steps
since the last done) and `collect_step` - pass your own dict in env-free use.
The env's observation reaches the views raw, a dict keeping its fields:
`student_state_fn` picks the student's state, `teacher_state_fn` refines it,
`teacher_obs_fn` reads the raw observation instead - a privileged teacher.
Non-finite teacher labels are parked on the student's own mean
and masked out of the mean, so a bad teacher never poisons the graph (an
entirely unusable batch comes back as a nan loss the caller can skip). Pass
`return_loss_breakdown = True` for the components on top of the total -
`LossBreakdown(loss, policy, entropy, alignment)` from the env-free call,
`stats['loss_breakdown']` from the loop.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from torch import Tensor, is_tensor, nn
from torch.distributions import Bernoulli, Categorical, Distribution, OneHotCategorical
from torch.utils import _pytree as pytree
from torch_einops_utils import cast_tensor, cross_entropy, kl_divergence, masked_mean

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def masked_loss_mean(losses: Tensor) -> Tensor:
    """Mean over finite entries, nan when nothing is usable."""

    finite = torch.isfinite(losses)

    if not finite.any():
        return losses.new_tensor(float('nan'))

    return masked_mean(torch.where(finite, losses, 0.), finite)

# action spaces

DISCRETE_DISTRIBUTIONS = [Categorical, OneHotCategorical, Bernoulli]   # append custom families - probs or logits, sample(), log_prob()

def is_discrete(distr: Distribution) -> bool:
    return isinstance(distr, tuple(DISCRETE_DISTRIBUTIONS))

def is_policy_out(x) -> bool:
    """A distribution, or a pytree of them - composite action spaces (tuples,
    dicts, nested)."""

    if isinstance(x, Distribution):
        return True

    leaves = pytree.tree_leaves(x)
    return len(leaves) > 0 and all(isinstance(leaf, Distribution) for leaf in leaves)

def policy_kind(policy_out) -> str:
    """`discrete` / `continuous` / `mixed` - across every head."""

    heads = [leaf for leaf in pytree.tree_leaves(policy_out) if isinstance(leaf, Distribution)]
    assert len(heads) > 0, 'a policy output must hold at least one distribution'

    discrete = [is_discrete(head) for head in heads]
    return 'discrete' if all(discrete) else 'continuous' if not any(discrete) else 'mixed'

def best_action(policy_out, sample = False) -> Tensor:
    """The action to hand an environment - sampled, or the mode / mean, mapped
    over a whole policy pytree if need be. Discrete families return class
    indices, bernoulli one 0/1 per binary dimension."""

    if is_tensor(policy_out):
        return policy_out

    if not isinstance(policy_out, Distribution):
        return pytree.tree_map(lambda head: best_action(head, sample = sample), policy_out)

    distr = policy_out

    if isinstance(distr, Bernoulli):
        return distr.sample() if sample else (distr.probs > 0.5).to(distr.probs.dtype)

    if is_discrete(distr):
        if sample:
            action = distr.sample()
            return action.argmax(dim = -1) if isinstance(distr, OneHotCategorical) else action

        probs = getattr(distr, 'probs', None)
        return probs.argmax(dim = -1) if exists(probs) else distr.logits.argmax(dim = -1)

    return distr.sample() if sample else default(getattr(distr, 'mode', None), distr.mean)

def to_log_prob_target(label: Tensor, distr: Distribution) -> Tensor:
    """What `log_prob` wants per distribution family - reshapes the label first,
    so class indices stored as `(b, 1)` and 1-dim actions both land right."""

    if isinstance(distr, OneHotCategorical):
        return F.one_hot(label.reshape(distr.batch_shape).long(), distr.probs.shape[-1]).to(distr.probs.dtype)

    if isinstance(distr, Bernoulli):
        return label.reshape(distr.batch_shape + distr.event_shape).to(distr.probs.dtype)

    if is_discrete(distr):
        return label.reshape(distr.batch_shape).long()

    return label.reshape(distr.batch_shape + distr.event_shape).float()

def log_prob_sum(distr: Distribution, label: Tensor) -> Tensor:
    # one (batch,) log-prob per transition, whatever the event shape

    if isinstance(distr, (Categorical, OneHotCategorical)):
        if isinstance(distr, OneHotCategorical):
            label = label.argmax(dim = -1)

        label = label.reshape(distr.batch_shape).long()
        return reduce(-cross_entropy(distr.logits, label, reduction = 'none'), 'b ... -> b', 'sum')

    return reduce(distr.log_prob(label), 'b ... -> b', 'sum')

# objectives

def _to_categorical(distr: Distribution) -> Distribution:
    return distr._categorical if isinstance(distr, OneHotCategorical) else distr

def _leaf_kl(student_distr: Distribution, teacher_distr: Distribution, reverse: bool) -> Tensor:
    p, q = (student_distr, teacher_distr) if reverse else (teacher_distr, student_distr)
    return reduce(kl_divergence(_to_categorical(p), _to_categorical(q)), 'b ... -> b', 'sum')

def nll_loss(student_out, teacher_out = None, label = None, info = None) -> Tensor:
    """Per-sample negative log-likelihood of the teacher's action under the
    student, summed over every head."""

    head_losses = pytree.tree_map(lambda distr, target: -log_prob_sum(distr, to_log_prob_target(target, distr)), student_out, label)
    return sum(pytree.tree_leaves(head_losses))

def kl_loss(student_out, teacher_out = None, label = None, info = None, *, reverse = False) -> Tensor:
    """Per-sample KL between the two policies, summed over every head - soft
    distillation. Default is KL(teacher || student); `reverse` flips to
    KL(student || teacher). Falls back to nll when only actions are present."""

    if not exists(teacher_out):
        return nll_loss(student_out, teacher_out, label, info)

    head_losses = pytree.tree_map(lambda s, t: _leaf_kl(s, t, reverse), student_out, teacher_out)
    return sum(pytree.tree_leaves(head_losses))

def mse_loss(student_out, teacher_out = None, label = None, info = None) -> Tensor:
    """Per-sample squared error between the two policies' mean actions, summed
    over every head."""

    targets = pytree.tree_map(lambda d: d.mean, teacher_out) if exists(teacher_out) else label
    head_losses = pytree.tree_map(
        lambda distr, target: reduce(F.mse_loss(distr.mean, target.detach(), reduction = 'none'), 'b ... -> b', 'mean'),
        student_out,
        targets
    )
    return sum(pytree.tree_leaves(head_losses))

# kl / mse need the teacher's distribution - the loop re-evaluates it at fit time

def uses_teacher_out(loss_fn) -> bool:
    fn = loss_fn.func if isinstance(loss_fn, partial) else loss_fn
    return fn in (kl_loss, mse_loss)

# alignment (repa)

class LossBreakdown(NamedTuple):
    loss: Tensor
    policy: Tensor
    entropy: Tensor | None = None
    alignment: Tensor | None = None

def cosine_rep_loss(student_rep: Tensor, teacher_rep: Tensor) -> Tensor:
    """Per-sample 1 - cosine similarity between aligned representations."""

    return reduce(1. - F.cosine_similarity(student_rep, teacher_rep, dim = -1), 'b ... -> b', 'mean')

class CaptureActivation:
    """Grabs whatever a hooked module emits on its next forward."""

    def __init__(self, module: nn.Module):
        self.value = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(self, _, __, output):
        self.value = output

    def pop(self):
        value, self.value = self.value, None
        return value

def resolve_layer(model, layer, source = 'student') -> nn.Module:
    """A module, or a submodule name resolved against the model."""

    assert isinstance(layer, (str, nn.Module)), f'{source}_layer must be an nn.Module or a submodule name'

    if isinstance(layer, str):
        assert isinstance(model, nn.Module), f'cannot resolve {source}_layer {layer!r} - the {source} is not an nn.Module'
        try:
            return model.get_submodule(layer)
        except AttributeError:
            raise ValueError(f'{source}_layer {layer!r} not found in the {source}')

    return layer

# pytree ring buffer

class _TreeBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.leaves = None
        self.treedef = None

    def write(self, tree, idx: Tensor):
        leaves, treedef = pytree.tree_flatten(tree)

        if not exists(self.leaves):
            self.treedef = treedef
            self.leaves = [
                torch.zeros(
                    self.capacity,
                    *leaf.shape[1:],
                    dtype = torch.float32 if leaf.is_floating_point() else leaf.dtype,
                    device = self.device
                )
                for leaf in leaves
            ]

        for buf, leaf in zip(self.leaves, leaves):
            buf[idx] = leaf.detach().to(device = self.device, dtype = buf.dtype)

    def get(self, idx: Tensor):
        if not exists(self.leaves):
            return None

        return pytree.tree_unflatten([buf[idx] for buf in self.leaves], self.treedef)

    def finite_mask(self, n: int) -> Tensor:
        mask = torch.ones(n, dtype = torch.bool, device = self.device)

        for buf in self.leaves:
            if buf.is_floating_point():
                mask &= reduce(torch.isfinite(buf[:n]), 'n ... -> n', 'all')

        return mask

# distiller

class DaggerDistiller:
    """Student + teacher -> `(distill loss, student out)`.

    Without an env it is a pure wrapper: the teacher labels whatever data it is
    handed and the loss comes back differentiable, the caller owning
    `.backward()` and the optimizer. With an env (+ optimizer) the same object
    runs the whole DAgger loop itself - `collect` rolls the student out and
    labels every visited state, `train` fits the student on the labelled
    buffer, `step` does both."""

    def __init__(
        self,
        student: Callable,
        teacher: Callable | None = None,
        *,
        env = None,
        teacher_action_fn: Callable | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        params = None,
        lr: float = 5e-4,
        # views - the raw observation to each side's own state, and labels
        student_state_fn: Callable | None = None,
        teacher_state_fn: Callable | None = None,
        teacher_obs_fn: Callable | None = None,
        label_transform: Callable | None = None,
        state_to_torch_fn: Callable | None = None,
        # objective
        loss_fn: Callable | None = None,
        entropy_weight: float = 0.0,
        teacher_sample: bool = False,
        action_kind: str = 'auto',
        # representation alignment - hook a layer of each, default 1 - cosine
        student_layer = None,
        teacher_layer = None,
        rep_align_weight: float = 1.,
        rep_align_fn: Callable = cosine_rep_loss,
        rep_align_projector: nn.Module | None = None,
        # rollout and fitting
        buffer_size: int = 200_000,
        max_samples: int | None = 65_536,
        batch_size: int = 1024,
        epochs: int = 2,
        max_grad_norm: float | None = 1.0,
        env_step_fn: Callable | None = None,
        action_to_env_fn: Callable | None = None,
        device = None,
    ):
        assert callable(student), 'student must be a callable state -> Distribution'
        assert exists(teacher) or exists(teacher_action_fn), 'teacher or teacher_action_fn must be given'
        assert not exists(loss_fn) or callable(loss_fn), 'loss_fn must be callable'
        assert action_kind in ('auto', 'continuous', 'discrete', 'mixed'), f'unknown action_kind {action_kind}'

        if exists(env):
            assert exists(optimizer) or exists(params), 'the env loop needs an optimizer - pass one, or params (+ lr) to build it'

        self.env = env
        self.student = student
        self.teacher = teacher
        self.teacher_action_fn = teacher_action_fn

        self.student_state_fn = student_state_fn
        self.teacher_state_fn = teacher_state_fn
        self.teacher_obs_fn = teacher_obs_fn
        self.label_transform = label_transform
        self.state_to_torch_fn = state_to_torch_fn

        self.loss_fn = loss_fn
        self.entropy_weight = entropy_weight

        # kl / mse need the teacher's distribution, not just its action

        self.needs_teacher_out = uses_teacher_out(loss_fn) and exists(teacher) and not exists(teacher_action_fn) and not exists(label_transform)
        self.buffer_teacher_input = self.needs_teacher_out and (exists(teacher_obs_fn) or exists(teacher_state_fn))
        self.teacher_sample = teacher_sample

        # representation alignment - resolve hooked layers up front

        self.student_capture = CaptureActivation(resolve_layer(student, student_layer)) if exists(student_layer) else None
        self.teacher_capture = CaptureActivation(resolve_layer(teacher, teacher_layer, 'teacher')) if exists(teacher_layer) else None

        assert exists(self.student_capture) == exists(self.teacher_capture), 'representation alignment needs both student_layer and teacher_layer'

        if exists(self.teacher_capture):
            assert exists(teacher) and not exists(teacher_action_fn), 'representation alignment needs the teacher called each step - pass it as `teacher`, not teacher_action_fn'

        self.rep_align_weight = rep_align_weight
        self.rep_align_fn = rep_align_fn
        self.rep_align_projector = rep_align_projector

        self.env_step_fn = env_step_fn
        self.action_to_env_fn = action_to_env_fn

        self.kind = None if action_kind == 'auto' else action_kind
        self.optimizer = optimizer if exists(optimizer) else (torch.optim.Adam(params, lr = lr) if exists(params) else None)

        # device - explicit, else the optimizer's, else leave tensors alone (env-free)

        opt_params = self._params()
        self.device = torch.device(device) if exists(device) else (opt_params[0].device if opt_params else None)
        self.buffer_device = default(self.device, torch.device('cpu'))

        self.buffer_size = buffer_size
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

        self.num_envs = int(getattr(env, 'num_envs', 1)) if exists(env) else 0
        self.episode_steps = np.zeros(self.num_envs, dtype = np.int64)

        self._state_buf = _TreeBuffer(buffer_size, self.buffer_device)
        self._action_buf = _TreeBuffer(buffer_size, self.buffer_device)
        self._teacher_input_buf = _TreeBuffer(buffer_size, self.buffer_device)
        self._rep_buf = _TreeBuffer(buffer_size, self.buffer_device)
        self.count = 0

        self._last_obs = None
        self._last_info = {}

    @property
    def state_buffers(self):
        return self._state_buf.leaves

    @property
    def action_buffers(self):
        return self._action_buf.leaves

    @property
    def teacher_input_buffers(self):
        return self._teacher_input_buf.leaves

    @property
    def rep_buffer(self):
        return self._rep_buf.leaves[0] if exists(self._rep_buf.leaves) else None

    # helpers

    def _to_torch(self, obs):
        if exists(self.state_to_torch_fn):
            obs = self.state_to_torch_fn(obs)

        def convert(value):
            if is_tensor(value):
                return value.to(self.device) if exists(self.device) else value
            return cast_tensor(value, dtype = torch.float32, device = self.device)

        return pytree.tree_map(convert, obs)

    def _resolve_kind(self, policy_out, source = 'student'):
        kind = policy_kind(policy_out)

        if exists(self.kind) and self.kind != kind:
            raise ValueError(f'{source} policy is {kind}, but the distiller already locked {self.kind}')

        self.kind = kind

    def _params(self):
        if not exists(self.optimizer):
            return []

        return [p for group in self.optimizer.param_groups for p in group['params']]

    # teacher labels

    def _teacher_input(self, obs, state, info, step):
        if exists(self.teacher_obs_fn):
            return self.teacher_obs_fn(obs, info, step)

        return self.teacher_state_fn(state, info, step) if exists(self.teacher_state_fn) else state

    @torch.no_grad()
    def _teacher_label(self, teacher_input, state, info, step):
        action = self.teacher_action_fn(teacher_input, info, step) if exists(self.teacher_action_fn) else self.teacher(teacher_input)
        teacher_out = None

        if is_policy_out(action):
            teacher_out = action
            self._resolve_kind(teacher_out, source = 'teacher')
            action = best_action(teacher_out, sample = self.teacher_sample)
        elif any(isinstance(leaf, Distribution) for leaf in pytree.tree_leaves(action)):
            raise ValueError('the teacher output mixes distributions and raw actions - return all distributions, or all raw actions')

        if exists(self.label_transform):
            action = self.label_transform(state, action, info, step)

        return action, teacher_out

    @torch.no_grad()
    def _teacher_distribution(self, teacher_input):
        out = self.teacher(teacher_input)

        if exists(self.teacher_capture):
            self.teacher_capture.pop()

        return out if is_policy_out(out) else None

    # loss

    def _leaf_nll(self, student_distr: Distribution, label: Tensor) -> Tensor:
        # per-head nll - non-finite rows parked on the student's mean

        target = to_log_prob_target(label, student_distr)
        finite = None

        if target.is_floating_point():
            mask = reduce(torch.isfinite(target), 'b ... -> b', 'all')

            if not mask.all():
                if not mask.any():
                    return student_distr.mean.sum() * float('nan')

                finite = mask
                target = target.clone()
                target[~finite] = student_distr.mean[~finite].detach()

        nll = -log_prob_sum(student_distr, target)
        return masked_mean(nll, finite) if exists(finite) else nll.mean()

    def _policy_loss(self, student_out, label, teacher_out = None, info = None) -> Tensor:
        if exists(self.loss_fn):
            loss = self.loss_fn(student_out, teacher_out, label, info)
            return masked_loss_mean(loss) if loss.dim() > 0 else loss

        head_losses = pytree.tree_map(self._leaf_nll, student_out, label)
        return sum(pytree.tree_leaves(head_losses))

    def _pop(self, capture: CaptureActivation, source: str) -> Tensor:
        value = capture.pop()
        assert exists(value), f'the hooked {source} layer did not fire - is it on the {source}\'s forward path?'
        return value

    def _alignment_loss(self, student_rep: Tensor, teacher_rep: Tensor) -> Tensor:
        if exists(self.rep_align_projector):
            student_rep = self.rep_align_projector(student_rep)

        assert student_rep.shape == teacher_rep.shape, f'alignment shapes {tuple(student_rep.shape)} vs {tuple(teacher_rep.shape)} - pick matching layers or pass rep_align_projector'

        return masked_loss_mean(self.rep_align_fn(student_rep, teacher_rep.detach()))

    def _loss(self, student_out, label, teacher_out = None, info = None, teacher_rep = None) -> LossBreakdown:
        policy = self._policy_loss(student_out, label, teacher_out, info)
        total = policy
        entropy = alignment = None

        if self.entropy_weight > 0.:
            entropy = sum(reduce(distr.entropy(), 'b ... -> b', 'sum').mean() for distr in pytree.tree_leaves(student_out))
            total = total - self.entropy_weight * entropy

        if exists(self.student_capture):
            assert exists(teacher_rep), 'alignment has no teacher representation - the teacher layer must fire'
            alignment = self._alignment_loss(self._pop(self.student_capture, 'student'), teacher_rep)
            total = total + self.rep_align_weight * alignment

        return LossBreakdown(total, policy, entropy, alignment)

    # env-free

    def __call__(self, state, teacher_state = None, *, info = None, step = 0, return_loss_breakdown = False) -> tuple[Tensor, Distribution]:
        """Distill one batch directly - no env involved. Returns
        `(distill_loss, student_out)` (or `(LossBreakdown, student_out)`)."""

        info = default(info, {})

        obs = self._to_torch(state)
        state = self.student_state_fn(obs, info, step) if exists(self.student_state_fn) else obs

        student_out = self.student(state)
        self._resolve_kind(student_out)

        teacher_input = self._to_torch(teacher_state) if exists(teacher_state) else self._teacher_input(obs, state, info, step)
        label, teacher_out = self._teacher_label(teacher_input, state, info, step)

        teacher_rep = self._pop(self.teacher_capture, 'teacher') if exists(self.teacher_capture) else None
        breakdown = self._loss(student_out, label, teacher_out, info, teacher_rep)

        return (breakdown, student_out) if return_loss_breakdown else (breakdown.loss, student_out)

    # rollout & buffer

    @torch.no_grad()
    def collect(self, steps: int, sample = True, teacher_acts = False) -> dict:
        """Roll a policy for `steps`, labelling every visited state with the teacher."""

        if not exists(self.env):
            raise RuntimeError('no env was given - the env-free distiller is called with data: distiller(state) -> (distill_loss, student_out)')

        if teacher_acts:
            assert not exists(self.label_transform), 'teacher rollout needs untransformed actions'

        before = self.count

        for step in range(steps):
            self._collect_step(step, sample, teacher_acts)

        return dict(labeled = self.count - before, buffer = min(self.count, self.buffer_size))

    def pretrain(self, steps: int, epochs = None, return_loss_breakdown = False) -> dict:
        """Teacher-rollout behavior cloning, then one supervised fit."""

        return {**self.collect(steps, teacher_acts = True), **self.train(epochs = epochs, return_loss_breakdown = return_loss_breakdown)}

    @torch.no_grad()
    def _collect_step(self, step: int, sample: bool, teacher_acts = False):
        if not exists(self._last_obs):
            out = self.env.reset()
            self._last_obs = out[0] if isinstance(out, (tuple, list)) else out

        info = dict(self._last_info, episode_steps = self.episode_steps.copy(), collect_step = step)

        obs = self._to_torch(self._last_obs)
        state = self.student_state_fn(obs, info, step) if exists(self.student_state_fn) else obs

        student_out = self.student(state)
        self._resolve_kind(student_out)

        teacher_input = self._teacher_input(obs, state, info, step)
        label, _ = self._teacher_label(teacher_input, state, info, step)

        teacher_rep = self._pop(self.teacher_capture, 'teacher') if exists(self.teacher_capture) else None
        self._write(state, label, teacher_rep = teacher_rep, teacher_input = teacher_input if self.buffer_teacher_input else None)

        actions = label if teacher_acts else best_action(student_out, sample = sample)

        if exists(self.action_to_env_fn):
            actions = self.action_to_env_fn(actions)

        step_fn = default(self.env_step_fn, self.env.step)
        obs, _, term, trunc, step_info = step_fn(actions)

        self._last_obs = obs
        self._last_info = dict(step_info) if isinstance(step_info, dict) else {}

        done = np.asarray(term, dtype = bool).reshape(-1) | np.asarray(trunc, dtype = bool).reshape(-1)
        self.episode_steps += 1
        self.episode_steps[done] = 0

    def _write(self, state, target, teacher_rep: Tensor | None = None, teacher_input = None):
        n = pytree.tree_leaves(state)[0].shape[0]
        idx = (self.count + torch.arange(n, device = self.buffer_device)) % self.buffer_size

        self._state_buf.write(state, idx)
        self._action_buf.write(target, idx)

        if exists(teacher_input):
            self._teacher_input_buf.write(teacher_input, idx)

        if exists(teacher_rep):
            self._rep_buf.write(teacher_rep, idx)

        self.count += n

    # train

    def train(self, epochs = None, batch_size = None, max_samples = None, return_loss_breakdown = False) -> dict:
        """Supervised fit of the student on the labelled buffer."""

        epochs = default(epochs, self.epochs)
        batch_size = default(batch_size, self.batch_size)
        max_samples = default(max_samples, self.max_samples)

        loss_val, last_breakdown = float('nan'), None
        n = min(self.count, self.buffer_size)

        if not exists(self.optimizer) or n == 0:
            stats = dict(loss = loss_val, samples = n)
            return {**stats, 'loss_breakdown': last_breakdown} if return_loss_breakdown else stats

        indices = torch.arange(n, device = self.buffer_device)

        # default nll validates targets - drop non-finite labels (a custom loss_fn owns its own)

        if not exists(self.loss_fn):
            indices = indices[self._action_buf.finite_mask(n)]

        if exists(max_samples) and indices.numel() > max_samples:
            indices = indices[torch.randperm(indices.numel(), device = self.buffer_device)[:max_samples]]

        n = indices.numel()
        params = self._params()

        for _ in range(max(1, epochs)):
            perm = indices[torch.randperm(n, device = self.buffer_device)]

            for idx in perm.split(batch_size):
                states = self._state_buf.get(idx)
                label = self._action_buf.get(idx)
                student_out = self.student(states)

                teacher_out = None
                if self.needs_teacher_out:
                    teacher_inputs = default(self._teacher_input_buf.get(idx), states)
                    teacher_out = self._teacher_distribution(teacher_inputs)

                breakdown = self._loss(student_out, label, teacher_out = teacher_out, teacher_rep = self._rep_buf.get(idx))
                loss = breakdown.loss

                # beta nll gradients spike - skip bad batches, clip the rest

                self.optimizer.zero_grad(set_to_none = True)

                if not torch.isfinite(loss):
                    continue

                loss.backward()

                if exists(self.max_grad_norm):
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

                if all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                    self.optimizer.step()
                    loss_val, last_breakdown = float(loss.detach()), breakdown

        stats = dict(loss = loss_val, samples = n)
        return {**stats, 'loss_breakdown': last_breakdown} if return_loss_breakdown else stats

    def step(self, collect_steps: int, epochs = None, sample = True, return_loss_breakdown = False) -> dict:
        """One full DAgger round: roll the student, label with the teacher, fit."""

        return {**self.collect(collect_steps, sample = sample), **self.train(epochs = epochs, return_loss_breakdown = return_loss_breakdown)}

# demo

if __name__ == '__main__':
    from torch.distributions import Bernoulli, Normal

    torch.manual_seed(0)

    student_net, teacher_net = nn.Linear(2, 2), nn.Linear(2, 2)

    with torch.no_grad():
        teacher_net.weight.mul_(2.)

    def student(state):
        return Normal(student_net(state), 1.)

    def teacher(state):
        return Normal(teacher_net(state), 0.1)

    # 1. the whole API - student and teacher in, one distill loss out

    distiller = DaggerDistiller(student, teacher)
    optimizer = torch.optim.Adam(student_net.parameters(), lr = 1e-2)

    for _ in range(100):
        states = torch.randn(32, 2)
        distill_loss, student_distr = distiller(states)

        optimizer.zero_grad()
        distill_loss.backward()
        optimizer.step()

    print(f'[wrap]      distill loss {distill_loss.item():.3f} | student out {tuple(student_distr.mean.shape)}')

    # 2. the objective is one argument - nll default, soft kl, deterministic mse

    soft = DaggerDistiller(student, teacher, loss_fn = kl_loss)
    soft_loss, _ = soft(torch.randn(32, 2))

    deterministic = DaggerDistiller(student, teacher, loss_fn = mse_loss)
    mse, _ = deterministic(torch.randn(32, 2))

    print(f'[losses]    soft kl {soft_loss.item():.3f} | mse {mse.item():.3f}')

    # 3. representation alignment + loss breakdown - hook any layer of each

    class Tiny(nn.Module):
        def __init__(self, dim = 8):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(2, dim), nn.Tanh())
            self.head = nn.Linear(dim, 2)

        def forward(self, state):
            return Normal(self.head(self.trunk(state)), 0.2)

    breakdown, _ = DaggerDistiller(Tiny(), Tiny(), student_layer = 'trunk.1', teacher_layer = 'trunk.1')(torch.randn(16, 2), return_loss_breakdown = True)

    print(f'[align]     loss {breakdown.loss.item():.3f} | policy {breakdown.policy.item():.3f} | alignment {breakdown.alignment.item():.3f}')

    # 4. composite action spaces - a pytree of heads works just the same

    button_net, arm_net = nn.Linear(2, 2), nn.Linear(2, 2)

    def composite_student(state):
        return dict(
            buttons = Bernoulli(logits = button_net(state)),
            arm = Normal(arm_net(state), 0.2)
        )

    def composite_teacher(state):
        return dict(
            buttons = Bernoulli(logits = teacher_net(state)),
            arm = Normal(arm_net(state), 0.1)
        )

    composite = DaggerDistiller(composite_student, composite_teacher)
    composite_loss, composite_out = composite(torch.randn(16, 2))
    composite_action = best_action(composite_out)

    print(f'[composite] loss {composite_loss.item():.3f} | actions {tuple(composite_action["buttons"].shape)} {tuple(composite_action["arm"].shape)}')

    # 5. views - the env hands back a dict observation and each side reads its
    #    own subset of it: the student drops the privileged field, the teacher
    #    keeps it, and labels map across action spaces

    wider_net = nn.Linear(3, 2)

    def wider_teacher(teacher_state):
        return Normal(wider_net(teacher_state), 0.1)

    def student_state_fn(obs, info, step):
        return obs['state']                                     # deployable at test time

    def teacher_obs_fn(obs, info, step):
        return torch.cat([obs['state'], obs['privileged']], -1)  # privileged teacher

    def label_transform(state, action, info, step):
        return action * 0.5

    hooked = DaggerDistiller(student, wider_teacher, student_state_fn = student_state_fn, teacher_obs_fn = teacher_obs_fn, label_transform = label_transform)
    hook_loss, _ = hooked(dict(state = torch.randn(32, 2), privileged = torch.randn(32, 1)), info = dict(phase = 0.3), step = 5)

    print(f'[hooks]     hook loss {hook_loss.item():.3f}')

    # 6. hand it the env and it runs the whole DAgger loop itself

    class ToyEnv:
        num_envs = 8

        def __init__(self):
            self.state = torch.randn(self.num_envs, 2)

        def reset(self):
            self.state = torch.randn(self.num_envs, 2)
            return self.state, {}

        def step(self, actions):
            self.state = 0.9 * self.state + 0.1 * actions
            zeros = torch.zeros(self.num_envs)
            return self.state, zeros, zeros.bool(), zeros.bool(), {}

    auto = DaggerDistiller(student, teacher, env = ToyEnv(), optimizer = optimizer)

    for cycle in range(5):
        stats = auto.step(collect_steps = 32, epochs = 2)
        print(f'[auto]      cycle {cycle} | labeled {stats["labeled"]} | loss {stats["loss"]:.3f}')
