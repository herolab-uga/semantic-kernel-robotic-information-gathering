"""Deep Kernel Learning (DKL) baseline [Wilson et al., 2016].

A neural feature extractor maps inputs into a latent space on which a base RBF
kernel operates, inducing input-dependent (non-stationary) correlations.  Like
AK, DKL learns heterogeneity from observations -- it is data-intensive and brittle
under noise, which the paper contrasts with the semantics-driven SK.

Requires the ``baselines`` extra: ``pip install "torch>=2.0" "gpytorch>=1.11"``.
"""

from __future__ import annotations

try:
    import gpytorch
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "The DKL baseline requires torch and gpytorch. Install with: "
        'pip install "torch>=2.0" "gpytorch>=1.11"'
    ) from exc

__all__ = ["FeatureExtractor", "DKLGPRModel"]


class FeatureExtractor(torch.nn.Sequential):
    """MLP feature extractor phi(x) for deep kernel learning."""

    def __init__(self, input_dim: int = 2, hidden: int = 64, out_dim: int = 2):
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(input_dim, hidden))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(hidden, hidden))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear3", torch.nn.Linear(hidden, out_dim))


class DKLGPRModel(gpytorch.models.ExactGP):
    """Exact GP with a deep-kernel feature extractor over a base RBF kernel."""

    def __init__(self, train_x, train_y, likelihood, input_dim=2, hidden=64, feature_dim=2):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)
        )
        self.feature_extractor = FeatureExtractor(input_dim=input_dim, hidden=hidden, out_dim=feature_dim)
        # Squash features to [-1, 1] for GP-input stability.
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x):
        projected = self.scale_to_bounds(self.feature_extractor(x))
        return gpytorch.distributions.MultivariateNormal(self.mean_module(projected), self.covar_module(projected))
