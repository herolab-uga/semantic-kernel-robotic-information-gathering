"""Attentive Kernel (AK) baseline [Chen et al., RSS 2022].

An attention network produces input-dependent weights over a bank of RBF kernels
with log-spaced base lengthscales, yielding locally adaptive lengthscales -- the
learned non-stationarity the paper compares against.  Unlike the Semantic Kernel,
AK infers heterogeneity from (noisy) observations rather than scene semantics.

Requires the ``baselines`` extra: ``pip install "torch>=2.0" "gpytorch>=1.11"``.
"""

from __future__ import annotations

import numpy as np

try:
    import gpytorch
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The AK baseline requires torch and gpytorch. Install with: "
        'pip install "torch>=2.0" "gpytorch>=1.11"'
    ) from exc

__all__ = [
    "RBFGPRModel",
    "AttentiveKernel",
    "AKGPRModel",
    "train_exact_gp",
    "predict_posterior",
    "ak_effective_lengthscale_map",
]


class RBFGPRModel(gpytorch.models.ExactGP):
    """Stationary RBF (ARD) exact-GP baseline."""

    def __init__(self, train_x, train_y, likelihood, input_dim=2):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class AttentiveKernel(gpytorch.kernels.Kernel):
    is_stationary = False

    def __init__(self, input_dim=2, m_kernels=15, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.m_kernels = m_kernels
        self.base_lengthscales = np.logspace(-2.0, 0.5, m_kernels)
        self.base_kernels = torch.nn.ModuleList(
            [gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)) for _ in range(m_kernels)]
        )
        for i, l in enumerate(self.base_lengthscales):
            self.base_kernels[i].base_kernel.initialize(lengthscale=float(l))
        self.attention_nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, m_kernels),
            torch.nn.Softmax(dim=-1),
        )
        self.register_parameter(name="raw_amplitude", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("raw_amplitude", gpytorch.constraints.Positive())

    @property
    def amplitude(self):
        return self.raw_amplitude_constraint.transform(self.raw_amplitude)

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            w1 = self.attention_nn(x1)
            k_diag = torch.zeros(x1.shape[0], device=x1.device)
            for m in range(self.m_kernels):
                k_diag = k_diag + (w1[:, m] ** 2) * self.base_kernels[m](x1, diag=True)
            return self.amplitude * k_diag
        w1 = self.attention_nn(x1)
        w2 = self.attention_nn(x2)
        Ksum = torch.zeros(x1.shape[0], x2.shape[0], device=x1.device)
        for m in range(self.m_kernels):
            Km = self.base_kernels[m](x1, x2).to_dense()
            Ksum = Ksum + (w1[:, m:m + 1] @ w2[:, m:m + 1].transpose(-2, -1)) * Km
        return self.amplitude * Ksum


class AKGPRModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim=2, m_kernels=15):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = AttentiveKernel(input_dim=input_dim, m_kernels=m_kernels)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def train_exact_gp(model, likelihood, train_x, train_y, steps=200, lr=0.05, wd=1e-4):
    model.train()
    likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(steps):
        opt.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        opt.step()


@torch.no_grad()
def predict_posterior(model, test_x):
    model.eval()
    with gpytorch.settings.fast_pred_var():
        pred = model(test_x)
    return pred.mean.detach().cpu().numpy(), pred.variance.detach().cpu().numpy()


@torch.no_grad()
def ak_effective_lengthscale_map(model, test_x, res, device="cpu"):
    """Effective lengthscale ``sum_m w_m(x)^2 * ls_m`` learned by AK."""
    if not hasattr(model.covar_module, "attention_nn"):
        raise ValueError("Model does not look like an AK model.")
    w = model.covar_module.attention_nn(test_x.to(device))
    ls = torch.tensor(model.covar_module.base_lengthscales, dtype=torch.float32, device=device)
    return torch.matmul(w ** 2, ls).detach().cpu().numpy().reshape(res, res)
