"""base_warp.py
This file is nothing more than a copy of the Warp class from BoTorch.
See the link for details:
https://github.com/pytorch/botorch/blob/main/botorch/models/transforms/input.py
"""

import torch
import gpytorch


class BaseWarp(gpytorch.Module):
    _min_concentration_level = 1e-4
    def __init__(
        self,
        indices,
        eps = 1e-7,
        concentration1_prior = None,
        concentration0_prior = None
    ):
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self._X_min = eps
        self._X_range = 1 - 2 * eps

        for i in (0, 1):
            p_name = f"raw_concentration{i}"
            self.register_parameter(
                p_name,
                torch.nn.Parameter(torch.full(self.indices.shape, 1.0)),
            )
        
        if concentration0_prior is not None:
            self.register_prior(
                "concentration0_prior",
                concentration0_prior,
                lambda m: m.concentration0,
                lambda m, v: m._set_concentration(i=0, value=v),
            )
        if concentration1_prior is not None:
            self.register_prior(
                "concentration1_prior",
                concentration1_prior,
                lambda m: m.concentration1,
                lambda m, v: m._set_concentration(i=1, value=v),
            )
        for i in (0, 1):
            p_name = f"raw_concentration{i}"
            constraint = gpytorch.constraints.GreaterThan(
                self._min_concentration_level,
                initial_value=1.0
            )
            self.register_constraint(param_name=p_name, constraint=constraint)
        value = torch.as_tensor(1.0)
        self.initialize(raw_concentration0=self.raw_concentration0_constraint.inverse_transform(value))
        self.initialize(raw_concentration1=self.raw_concentration1_constraint.inverse_transform(value))

    @property
    def concentration0(self):
        return self.raw_concentration0_constraint.transform(self.raw_concentration0)

    @property
    def concentration1(self):
        return self.raw_concentration1_constraint.transform(self.raw_concentration1)

    def _set_concentration(self, i, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.concentration0)
        self.initialize(**{f"concentration{i}": value})

    def _transform(self, X):
        X_tf = X.clone()
        k = torch.distributions.Kumaraswamy(
            concentration1=self.concentration1, concentration0=self.concentration0
        )
        # normalize to [eps, 1-eps]
        X_tf[..., self.indices] = k.cdf(
            torch.clamp(
                X_tf[..., self.indices] * self._X_range + self._X_min,
                self._X_min,
                1.0 - self._X_min,
            )
        )
        return X_tf
