# This provides the implementation of negative binomial distributions used in SURE.
# The code is modified from NegativeBinomial of scvi-tools.

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)




def torch_lgamma_mps(x: torch.Tensor) -> torch.Tensor:
    """Used in mac Mx devices while broadcasting a tensor

    Parameters
    ----------
    x
        Data

    Returns
    -------
    lgamma tensor that perform on a copied version of the tensor
    """
    return torch.lgamma(x.contiguous())


def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a zinb model.

    Parameters
    ----------
    x
        Data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi
        logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function

    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    log = log_fn
    lgamma = lgamma_fn
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless
    # of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)
    log_theta_eps = log(theta + eps)
    log_theta_mu_eps = log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return res


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def _convert_mean_disp_to_counts_logits(
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""NB parameterizations conversion.

    Parameters
    ----------
    mu
        mean of the NB distribution.
    theta
        inverse overdispersion.
    eps
        constant used for numerical log stability. (Default value = 1e-6)

    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    if not (mu is None) == (theta is None):
        raise ValueError(
            "If using the mu/theta NB parameterization, both parameters must be specified"
        )
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits


def _convert_counts_logits_to_mean_disp(
    total_count: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NB parameterizations conversion.

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    logits
        success logits.

    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.

    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


def _gamma(theta: torch.Tensor, mu: torch.Tensor, on_mps: bool = False) -> Gamma:
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = (
        Gamma(concentration=concentration.to("cpu"), rate=rate.to("cpu"))
        if on_mps  # TODO: NEED TORCH MPS FIX for 'aten::_standard_gamma'
        else Gamma(concentration=concentration, rate=rate)
    )
    return gamma_d


class NegativeBinomial(Distribution):
    r"""Negative binomial distribution.

    Parameters
    ----------
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "scale": constraints.greater_than_eq(0),
    }
    #support = constraints.nonnegative_integer
    support = constraints.nonnegative

    def __init__(
        self,
        mu: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        validate_args: bool = False,
    ):
        self.on_mps = (
            mu.device.type == "mps"
        )  # TODO: This is used until torch will solve the MPS issues
        self._eps = 1e-8

        mu, theta = broadcast_all(mu, theta)
        if self.on_mps:
            mu, theta = mu.contiguous(), theta.contiguous()
        self.mu = mu
        self.theta = theta
        self.scale = scale
        
        batch_shape = self.mu.shape
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    def get_normalized(self, key) -> torch.Tensor:
        if key == "mu":
            return self.mu
        elif key == "scale":
            return self.scale
        else:
            raise ValueError(f"normalized key {key} not recognized")

    @property
    def variance(self) -> torch.Tensor:
        return self.mean + (self.mean**2) / self.theta

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: torch.Size | tuple | None = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()  # TODO: TORCH MPS FIX - DONE ON CPU CURRENTLY
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = (
            PoissonTorch(l_train).sample().to("mps")
            if self.on_mps  # TODO: NEED TORCH MPS FIX for 'aten::poisson'
            else PoissonTorch(l_train).sample()
        )  # Shape : (n_samples, n_cells_batch, n_vars)
        return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                )

        lgamma_fn = torch_lgamma_mps if self.on_mps else torch.lgamma  # TODO: TORCH MPS FIX
        return log_nb_positive(
            value, mu=self.mu, theta=self.theta, eps=self._eps, lgamma_fn=lgamma_fn
        )

    def _gamma(self) -> Gamma:
        return _gamma(self.theta, self.mu, self.on_mps)

    def to_event(self, n):
        if n == 0:
            return self
        return dist.Independent(self, n)
    
    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: "
                f"{self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
                if self.__dict__[p] is not None
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"



class ZeroInflatedNegativeBinomial(NegativeBinomial):
    r"""Zero-inflated negative binomial distribution.

    Parameters
    ----------
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    zi_logits
        Logits scale of zero inflation probability.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "zi_logits": constraints.real,
        "scale": constraints.greater_than_eq(0),
    }
    #support = constraints.nonnegative_integer
    support = constraints.nonnegative

    def __init__(
        self,
        mu: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        zi_logits: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        validate_args: bool = False,
    ):
        super().__init__(
            mu=mu,
            theta=theta,
            scale=scale,
            validate_args=validate_args,
        )
        self.zi_logits, self.mu, self.theta = broadcast_all(zi_logits, self.mu, self.theta)

    @property
    def mean(self) -> torch.Tensor:
        pi = self.zi_probs
        return (1 - pi) * self.mu

    @property
    def variance(self) -> None:
        pi = self.zi_probs
        return (1 - pi) * self.mu * (self.mu + self.theta + pi * self.mu * self.theta) / self.theta

    @lazy_property
    def zi_logits(self) -> torch.Tensor:
        """ZI logits."""
        return probs_to_logits(self.zi_probs, is_binary=True)

    @lazy_property
    def zi_probs(self) -> torch.Tensor:
        return logits_to_probs(self.zi_logits, is_binary=True)

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: torch.Size | tuple | None = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        samp = super().sample(sample_shape=sample_shape)
        is_zero = torch.rand_like(samp) <= self.zi_probs
        samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
        return samp_

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log probability."""
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within the support of the distribution",
                UserWarning,
            )
        lgamma_fn = torch_lgamma_mps if self.on_mps else torch.lgamma  # TODO: TORCH MPS FIX
        return log_zinb_positive(
            value, self.mu, self.theta, self.zi_logits, eps=1e-08, lgamma_fn=lgamma_fn
        )
        
    def to_event(self, n):
        if n == 0:
            return self
        return dist.Independent(self, n)