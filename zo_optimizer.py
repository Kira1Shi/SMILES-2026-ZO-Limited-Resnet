"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args:
        model:            The ``nn.Module`` to optimize.
        lr:               Step size / learning rate.
        eps:              Perturbation magnitude for the finite-difference
                          estimator.
        perturbation_mode: Distribution used to sample the perturbation
                          direction. ``"gaussian"`` draws from N(0, I);
                          ``"uniform"`` draws from U(-1, 1) and normalises.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
           Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
           implement dynamic layer selection strategies.

    Example — tune only the final linear layer::

        optimizer = ZeroOrderOptimizer(model)
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        eps: float = 3e-4,
        perturbation_mode: str = "gaussian",
        momentum: float = 0.9,
        betas: tuple[float, float] = (0.95, 0.999),
        adam_eps: float = 1e-8
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.momentum = momentum
        self.momentum_buffer: dict[str, torch.Tensor] = {}
        self.betas = betas  # (beta1, beta2)
        self.adam_eps = adam_eps
        self.step_count = 0

        self.m_buffer: dict[str, torch.Tensor] = {}  # First moment (mean)
        self.v_buffer: dict[str, torch.Tensor] = {}  # Second moment (uncentered variance)

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        # ------------------------------------------------------------------
        # STUDENT: Set self.layer_names to the parameters you want to tune.
        #
        # The default below selects only the final classification head.
        # You may replace this with any subset of named parameters, e.g.:
        #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        #
        # You can also update self.layer_names inside .step() to implement
        # a dynamic schedule (e.g. gradually unfreeze deeper layers).
        # ------------------------------------------------------------------
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate a pseudo-gradient for each active parameter.

        Skeleton: 2-point central-difference estimator.
        For each active parameter ``p`` independently:
            1. Sample a random unit vector ``u`` of the same shape as ``p``.
            2. Evaluate  f_plus  = loss_fn() with ``p ← p + eps * u``
            3. Evaluate  f_minus = loss_fn() with ``p ← p - eps * u``
            4. Restore ``p`` to its original value.
            5. Pseudo-gradient ← ``(f_plus - f_minus) / (2 * eps) * u``

        This is an unbiased estimator of the directional derivative along ``u``
        scaled back to parameter space.

        Args:
            loss_fn: Callable that evaluates the objective on the current batch
                     and returns a scalar ``float``. May be called multiple
                     times; each call must use the *same* batch.
            params:  Dict of active parameter name → tensor (from
                     ``_active_params``).

        Returns:
            Dict mapping each parameter name to its estimated pseudo-gradient
            tensor (same shape as the parameter).

        Student task:
            Replace this with a more efficient or accurate estimator:
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the gradient estimation below.
        # ------------------------------------------------------------------

        original_vals = {name: param.data.clone() for name, param in params.items()}

        deltas = {}
        for name, param in params.items():
            deltas[name] = torch.empty_like(param).bernoulli_(0.5) * 2 - 1

        with torch.no_grad():
            for name, param in params.items():
                param.data = original_vals[name] + self.eps * deltas[name]
            f_plus = loss_fn()
            
            for name, param in params.items():
                param.data = original_vals[name] - self.eps * deltas[name]
            f_minus = loss_fn()
            
            for name, param in params.items():
                param.data = original_vals[name]

        delta_loss = f_plus - f_minus
        grads = {}
        for name, param in params.items():
            grads[name] = (delta_loss / (2.0 * self.eps)) * deltas[name]

        return grads        # ------------------------------------------------------------------

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Skeleton: vanilla gradient *descent* step (minimising the loss).
            ``p ← p - lr * grad``

        Args:
            params: Dict of active parameter name → tensor.
            grads:  Dict of pseudo-gradient name → tensor (same keys as
                    ``params``).

        Student task:
            Replace with a more sophisticated update rule, e.g.:
              - Momentum: accumulate an exponential moving average of gradients.
              - Adam-style: maintain first and second moment estimates.
              - Clipped update: ``p ← p - lr * clip(grad, max_norm)``.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the parameter update below.
        # ------------------------------------------------------------------
        with torch.no_grad():
            for name, param in params.items():
                param.data.sub_(self.lr * grads[name])

        #momentum realisation
        # with torch.no_grad():
        #     for name, param in params.items():
        #         if name not in self.momentum_buffer:
        #             self.momentum_buffer[name] = torch.zeros_like(param.data)
        #         self.momentum_buffer[name] = self.momentum * self.momentum_buffer[name] + grads[name]
        #         param.data.sub_(self.lr * self.momentum_buffer[name])

        #Adam realisation
        # self.step_count += 1
        
        # beta1, beta2 = self.betas
        # bias_correction1 = 1 - beta1 ** self.step_count
        # bias_correction2 = 1 - beta2 ** self.step_count
        
        # with torch.no_grad():
        #     for name, param in params.items():
        #         grad = grads[name]
                
        #         if name not in self.m_buffer:
        #             self.m_buffer[name] = torch.zeros_like(param.data)
        #             self.v_buffer[name] = torch.zeros_like(param.data)

        #         self.m_buffer[name] = beta1 * self.m_buffer[name] + (1 - beta1) * grad
        #         self.v_buffer[name] = beta2 * self.v_buffer[name] + (1 - beta2) * (grad * grad)
        #         m_hat = self.m_buffer[name] / bias_correction1
        #         v_hat = self.v_buffer[name] / bias_correction2
        #         param.data.sub_(self.lr * m_hat / (torch.sqrt(v_hat) + self.adam_eps))
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. Parameters *not* in ``self.layer_names`` are never touched.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)