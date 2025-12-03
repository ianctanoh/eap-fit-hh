"""
Diagonal approximation to Extended Kalman filtering and smoothing for nonlinear Gaussian state-space models.
"""
import jax.numpy as jnp
import jax.random as jr
from jax import ops
from jax import lax
from jax import jacfwd
from jax import jacobian
from jax import ensure_compile_time_eval
from jax import experimental
from jax import device_get
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import Callable, List, Optional, Tuple, NamedTuple, Union

from jax.scipy.linalg import cho_factor, cho_solve

from sparsejac import jacrev

from jaxley.utils.jax_utils import nested_checkpoint_scan

PRNGKeyT = Array

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float

def psd_solve(A, b, diagonal_boost=1e-9):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1])
    L, lower = cho_factor(A, lower=True)
    x = cho_solve((L, lower), b)
    return x

def symmetrize(A):
    """Symmetrize one or more matrices."""
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))

FnStateToState = Callable[ [Float[Array, " state_dim"]], Float[Array, " state_dim"]]
FnStateAndInputToState = Callable[ [Float[Array, " state_dim"], Float[Array, " input_dim"]], Float[Array, " state_dim"]]
FnStateToEmission = Callable[ [Float[Array, " state_dim"]], Float[Array, " emission_dim"]]
FnStateAndInputToEmission = Callable[ [Float[Array, " state_dim"], Float[Array, " input_dim"] ], Float[Array, " emission_dim"]]


class diagParamsNLGSSM(NamedTuple):
    """Parameters for a NLGSSM model.

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param dynamics_function: $f$
    :param dynamics_covariance: $Q$
    :param emissions_function: $h$
    :param emissions_covariance: $R$
    :param initial_mean: $m$
    :param initial_covariance: $S$

    """

    initial_mean: Float[Array, " state_dim"]
    initial_covariance_diag: Float[Array, "state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_covariance_diag: Float[Array, "state_dim"]
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_covariance_diag: Float[Array, "emission_dim"]

class PosteriorGSSMFiltered(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior.

    :param marginal_loglik: marginal log likelihood, $p(y_{1:T} \mid u_{1:T})$
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$

    """
    marginal_loglik: Union[Scalar, Float[Array, " ntime"]]
    filtered_means: Optional[Float[Array, "ntime state_dim"]] = None
    filtered_variances: Optional[Float[Array, "ntime state_dim"]] = None
    predicted_means: Optional[Float[Array, "ntime state_dim"]] = None
    predicted_covariances_diag: Optional[Float[Array, "ntime state_dim"]] = None


class PosteriorGSSMSmoothed(NamedTuple):
    r"""Marginals of the Gaussian filtering and smoothing posterior.

    :param marginal_loglik: marginal log likelihood, $p(y_{1:T} \mid u_{1:T})$
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    :param smoothed_means: array of smoothed means $\mathbb{E}[z_t \mid y_{1:T}, u_{1:T}]$
    :param smoothed_covariances: array of smoothed marginal covariances, $\mathrm{Cov}[z_t \mid y_{1:T}, u_{1:T}]$
    :param smoothed_cross_covariances: array of smoothed cross products, $\mathbb{E}[z_t z_{t+1}^T \mid y_{1:T}, u_{1:T}]$

    """
    marginal_loglik: Scalar
    filtered_means: Float[Array, "ntime state_dim"]
    filtered_variances: Float[Array, "ntime state_dim"]
    smoothed_means: Float[Array, "ntime state_dim"]
    smoothed_covariances_diag: Float[Array, "ntime state_dim"]
    smoothed_cross_covariances: Optional[Float[Array, "ntime_minus1 state_dim state_dim"]] = None


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x


def _predict(prior_mean: Float[Array, " state_dim"], 
             prior_cov_diag: Float[Array, "state_dim"],
             dynamics_func: Callable, 
             dynamics_jacobian: Callable, 
             dynamics_cov_diag: Float[Array, "state_dim"],
             inpt: Float[Array, " input_dim"]
             ) -> Tuple[Float[Array, " state_dim"], Float[Array, "state_dim state_dim"]]:
    r"""Predict next mean and covariance using first-order additive EKF

        p(z_{t+1}) = \int N(z_t | m, S) N(z_{t+1} | f(z_t, u), Q)
                    = N(z_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    F_x = dynamics_jacobian(prior_mean, inpt)
    mu_pred = dynamics_func(prior_mean, inpt)

    ### Sparse diagonal computation:

    # For each nonzero in F_x, scale it by prior_diag[col]
    scaled_data = F_x.data * prior_cov_diag[F_x.indices[:, 1]]
    
    # Square it
    squared_data = scaled_data * F_x.data  # same as data^2 * prior_diag[col]
    
    # Sum all contributions per row
    row_idx = F_x.indices[:, 0]
    Sigma_pred_diag = ops.segment_sum(squared_data, row_idx, F_x.shape[0]) + dynamics_cov_diag 
    return mu_pred, Sigma_pred_diag

def _condition_on(prior_mean: Float[Array, " state_dim"],
                  prior_cov: Float[Array, "state_dim state_dim"],
                  emission_func: Callable, 
                  emission_jacobian: Callable, 
                  emission_cov_diag: Float[Array, "emission_dim"],
                  inpt: Float[Array, " input_dim"],
                  emission: Float[Array, " emission_dim"],
                  num_iter: int):
    r"""Condition a Gaussian potential on a new observation.

       p(z_t | y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(z_t | y_{1:t-1}, u_{1:t-1}) p(y_t | z_t, u_t)
         = N(z_t | m, S) N(y_t | h_t(z_t, u_t), R_t)
         = N(z_t | mm, SS)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = h(m, u)
         S = R + H(m,u) * P * H(m,u)'
         K = P * H(m, u)' * S^{-1}
         SS = P - K * S * K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

     Returns:
         mu_cond (D_hid,): filtered mean.
         Sigma_cond (D_hid,D_hid): filtered covariance.
    """
    def _step(carry, _):
        """Iteratively re-linearize around posterior mean and covariance."""
        prior_mean, prior_cov_diag = carry
        H_x = emission_jacobian(prior_mean, inpt)

        ### Approx covariance matrix
        r_scaled_H_x = (H_x.T)/emission_cov_diag  # shape (D, N)
        diag_HRH = jnp.sum(r_scaled_H_x * H_x.T, axis=1)

        posterior_cov_diag = 1./(1./(prior_cov_diag)+diag_HRH)

        ### Approx mean
        K_approx = posterior_cov_diag[:,None] * r_scaled_H_x
        posterior_mean = prior_mean + K_approx @ (emission - emission_func(prior_mean, inpt))
        return (posterior_mean, posterior_cov_diag), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (prior_mean, prior_cov)
    (mu_cond, Sigma_cond_diag), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_cond_diag


def get_sparsity(dynamics_function: Union[FnStateToState, FnStateAndInputToState],
                 initial_state: Float[Array, " state_dim"],
                 inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
                ):
    
    F = jacobian(dynamics_function)
    new_state = dynamics_function(initial_state, inputs[0])
    Jac = F(new_state, inputs[1])
    sparsity = device_get((Jac != 0))
    sparsity = experimental.sparse.BCOO.fromdense(sparsity)
    return sparsity
    

def diag_extended_kalman_filter(params: diagParamsNLGSSM,
                           emissions: Float[Array, "num_timesteps emission_dim"],
                           sparsity,
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
                           num_iter: int = 1,
                           output_fields: Optional[List[str]]=["filtered_means", 
                                                               "filtered_variances", 
                                                               "predicted_means", 
                                                               "predicted_covariances_diag"],
                           ) -> PosteriorGSSMFiltered:
    r"""Run an (iterated) extended Kalman filter to produce the
    marginal likelihood and filtered state estimates.

    Args:
        params: model parameters.
        emissions: observation sequence.
        num_iter: number of linearizations around posterior for update step (default 1).
        inputs: optional array of inputs.
        output_fields: list of fields to return in posterior object.
            These can take the values "filtered_means", "filtered_covariances",
            "predicted_means", "predicted_covariances", and "marginal_loglik".

    Returns:
        post: posterior object.

    """
    num_timesteps = len(emissions)

    # Dynamics and emission functions and their Jacobians
    f, h = params.dynamics_function, params.emission_function
    H = jacfwd(h)
    # Sparse Jacobian for f
    with ensure_compile_time_eval():
        F = jacrev(f,sparsity=sparsity)
    f, h, F, H = (_process_fn(fn, inputs) for fn in (f, h, F, H))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, t):
        """Iteratively update the state estimate and log likelihood."""
        ll, pred_mean, pred_cov_diag = carry

        # Get parameters and inputs for time index t
        Q_diag = _get_params(params.dynamics_covariance_diag, 1, t)
        R_diag = _get_params(params.emission_covariance_diag, 1, t)
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood
        H_x = H(pred_mean, u)

        ll += MVN(h(pred_mean, u), H_x @ jnp.diag(pred_cov_diag) @ H_x.T + jnp.diag(R_diag)).log_prob(jnp.atleast_1d(y))

        # Condition on this emission
        filtered_mean, filtered_cov_diag = _condition_on(pred_mean, pred_cov_diag, h, H, R_diag, u, y, num_iter)

        # Predict the next state
        pred_mean, pred_cov_diag = _predict(filtered_mean, filtered_cov_diag, f, F, Q_diag, u)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov_diag)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_variances": filtered_cov_diag,
            "predicted_means": pred_mean,
            "predicted_covariances_diag": pred_cov_diag,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, params.initial_covariance_diag)
    (ll, _, _), outputs = lax.scan(_step, carry, jnp.arange(num_timesteps))
    outputs = {"marginal_loglik": ll, **outputs}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered

def extended_kalman_smoother(params: diagParamsNLGSSM,
                             emissions:  Float[Array, "num_timesteps emission_dim"],
                             filtered_posterior: Optional[PosteriorGSSMFiltered] = None,
                             inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
                             ) -> PosteriorGSSMSmoothed:
    r"""Run an extended Kalman (RTS) smoother.

    Args:
        params: model parameters.
        emissions: observation sequence.
        filtered_posterior: optional output from filtering step.
        inputs: optional array of inputs.

    Returns:
        post: posterior object.

    """
    num_timesteps = len(emissions)

    # Get filtered posterior
    if filtered_posterior is None:
        filtered_posterior = diag_extended_kalman_filter(params, emissions, inputs=inputs)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # Dynamics and emission functions and their Jacobians
    f = params.dynamics_function
    F = jacfwd(f)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        """One step of the extended Kalman smoother."""
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        R = _get_params(params.emission_covariance, 2, t)
        u = inputs[t]
        F_x = F(filtered_mean, u)

        # Prediction step
        m_pred = f(filtered_mean, u)
        S_pred = Q + F_x @ filtered_cov @ F_x.T
        G = psd_solve(S_pred, F_x @ filtered_cov).T

        # Compute smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - m_pred)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - S_pred) @ G.T

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov)

    # Run the extended Kalman smoother
    _, (smoothed_means, smoothed_covs) = lax.scan(
        _step,
        (filtered_means[-1], filtered_covs[-1]),
        (jnp.arange(num_timesteps - 1), filtered_means[:-1], filtered_covs[:-1]),
        reverse=True,
    )

    # Concatenate the arrays and return
    smoothed_means = jnp.vstack((smoothed_means, filtered_means[-1][None, ...]))
    smoothed_covs = jnp.vstack((smoothed_covs, filtered_covs[-1][None, ...]))

    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
    )


def extended_kalman_posterior_sample(key: PRNGKeyT,
                                     params: diagParamsNLGSSM,
                                     emissions:  Float[Array, "num_timesteps emission_dim"],
                                     inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
                                     ) -> Float[Array, "num_timesteps state_dim"]:
    r"""Run forward-filtering, backward-sampling to draw samples.

    Args:
        key: random number key.
        params: model parameters.
        emissions: observation sequence.
        inputs: optional array of inputs.

    Returns:
        Float[Array, "ntime state_dim"]: one sample of $z_{1:T}$ from the posterior distribution on latent states.
    """
    num_timesteps = len(emissions)

    # Get filtered posterior
    filtered_posterior = diag_extended_kalman_filter(params, emissions, inputs=inputs)
    ll = filtered_posterior.marginal_loglik
    filtered_means = filtered_posterior.filtered_means
    filtered_covs = filtered_posterior.filtered_covariances

    # Dynamics and emission functions and their Jacobians
    f = params.dynamics_function
    F = jacfwd(f)
    f, F = (_process_fn(fn, inputs) for fn in (f, F))
    inputs = _process_input(inputs, num_timesteps)

    def _step(carry, args):
        """One step of the extended Kalman sampler."""
        # Unpack the inputs
        next_state = carry
        key, filtered_mean, filtered_cov, t = args

        # Get parameters and inputs for time index t
        Q = _get_params(params.dynamics_covariance, 2, t)
        u = inputs[t]

        # Condition on next state
        smoothed_mean, smoothed_cov = _condition_on(filtered_mean, filtered_cov, f, F, Q, u, next_state, 1)
        state = MVN(smoothed_mean, smoothed_cov).sample(seed=key)
        return state, state

    # Initialize the last state
    key, this_key = jr.split(key, 2)
    last_state = MVN(filtered_means[-1], filtered_covs[-1]).sample(seed=this_key)

    _, states = lax.scan(
        _step,
        last_state,
        (
            jr.split(key, num_timesteps - 1),
            filtered_means[:-1],
            filtered_covs[:-1],
            jnp.arange(num_timesteps - 1),
        ),
        reverse=True,
    )
    return jnp.vstack([states, last_state])