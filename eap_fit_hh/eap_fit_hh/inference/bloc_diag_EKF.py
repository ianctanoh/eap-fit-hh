"""
Diagonal approximation to Extended Kalman filtering and smoothing for nonlinear Gaussian state-space models.
"""
import jax.numpy as jnp
import jax.random as jr
from jax import ops
from jax import lax
from jax import jacfwd, grad
from jax.tree_util import tree_map
from jax import vmap
from jax import jacobian
from jax import ensure_compile_time_eval
from jax import experimental
from jax import device_get
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import Callable, List, Optional, Tuple, NamedTuple, Union
from jaxley.utils.jax_utils import nested_checkpoint_scan
import numpy as np
from math import prod

from jax.scipy.linalg import cho_factor, cho_solve

from sparsejac import jacrev

from jaxley.utils.jax_utils import nested_checkpoint_scan

PRNGKeyT = Array

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float

def psd_solve(A, b, diagonal_boost=1e-9):
    """A wrapper for coordinating the linalg solvers used in the library for psd matrices."""
    A = symmetrize(A) + diagonal_boost * jnp.eye(A.shape[-1], dtype=A.dtype)
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


class bdParamsNLGSSM(NamedTuple):
    """Parameters for a NLGSSM model.

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    If you have no inputs, the dynamics and emission functions do not to take $u_t$ as an argument.

    :param dynamics_function: $f$
    :param dynamics_covariance_diag: $Q$
    :param emissions_function: $h$
    :param emissions_covariance_diag: $R$
    :param initial_mean: $m$
    :param initial_covariance_diag: $S$

    """

    initial_mean: Float[Array, " state_dim"]
    initial_covariance_diag: Float[Array, "state_dim"]
    dynamics_function: Union[FnStateToState, FnStateAndInputToState]
    dynamics_covariance_diag: Float[Array, "state_dim"]
    emission_function: Union[FnStateToEmission, FnStateAndInputToEmission]
    emission_jacobian: Float[Array, " emission_dim n_comps"]
    emission_covariance_diag: Float[Array, "emission_dim"]

class PosteriorGSSMFiltered(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior.

    :param marginal_loglik: marginal log likelihood, $p(y_{1:T} \mid u_{1:T})$
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$

    """
    marginal_loglik: Union[Scalar, Float[Array, " ntime"]]
    filtered_means: Optional[Float[Array, "ntime state_dim"]] = None
    filtered_cov_bloc: Optional[Float[Array, "ntime n_comps n_comps"]] = None
    filtered_cov_diag_rest: Optional[Float[Array, "ntime state_dim-n_comps"]] = None
    predicted_means: Optional[Float[Array, "ntime state_dim"]] = None
    predicted_cov_bloc: Optional[Float[Array, "ntime n_comps n_comps"]] = None
    predicted_cov_diag_rest: Optional[Float[Array, "ntime state_dim-n_comps"]] = None


# Helper functions
_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_process_fn = lambda f, u: (lambda x, y: f(x)) if u is None else f
_process_input = lambda x, y: jnp.zeros((y,1)) if x is None else x


def _predict(prior_mean: Float[Array, " state_dim"], 
             prior_cov_bloc: Float[Array, "num_comp num_comp"],
             prior_cov_diag_rest: Float[Array, "state_dim - num_comp"],
             dynamics_func: Callable, 
             F_blocks: Callable, 
             dynamics_cov_bloc: Float[Array, "num_comp num_comp"],
             dynamics_cov_diag_rest: Float[Array, "state_dim - num_comp"],
             inpt: Float[Array, " input_dim"]
             ) -> Tuple[Float[Array, " state_dim"], Float[Array, "state_dim state_dim"]]:
    r"""Predict next mean and covariance using first-order additive EKF

        p(z_{t+1}) = \int N(z_t | m, S) N(z_{t+1} | f(z_t, u), Q)
                    = N(z_{t+1} | f(m, u), F(m, u) S F(m, u)^T + Q)

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    ### function to compute jacobian
    F_x_bloc, F_x_diag_rest = F_blocks(prior_mean, inpt)
    mu_pred = dynamics_func(prior_mean, inpt)

    ### Get leading bloc of predict covariance
    Sigma_pred_bloc = F_x_bloc @ prior_cov_bloc @ F_x_bloc.T + dynamics_cov_bloc
    #Sigma_pred_bloc = symmetrize(Sigma_pred_bloc) + 1e-8 * jnp.eye(Sigma_pred_bloc.shape[0], dtype=Sigma_pred_bloc.dtype)

    ### Get rest of the diagonal of predict covariance
    Sigma_pred_diag_rest = (F_x_diag_rest ** 2) * prior_cov_diag_rest + dynamics_cov_diag_rest
    
    return mu_pred, Sigma_pred_bloc,  Sigma_pred_diag_rest

def _condition_on(prior_mean: Float[Array, " state_dim"],
                  prior_cov_bloc: Float[Array, "num_comp num_comp"],
                  prior_cov_diag_rest: Float[Array, "state_dim - num_comp"],
                  emission_func: Callable, 
                  H_volt: Float[Array, "emission_dim num_comp"], 
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
    n_comps = prior_cov_bloc.shape[0]
    s   = 1.0 / jnp.sqrt(emission_cov_diag)          # (m,)
    G   = H_volt * s[:, None]                        # (m, K)
    I_m = jnp.eye(G.shape[0], dtype = H_volt.dtype)

    def _step(carry, _):
        prior_mean, prior_cov_bloc, prior_cov_diag_rest = carry  # prior_cov_diag_rest unchanged here

        # Build A and S̃ in measurement space
        A = prior_cov_bloc @ G.T                 # (K, m) = P G^T
        S_tilde = I_m + G @ A                    # (m, m) = I + G P G^T (SPD)

        # Factor/solve
        X = psd_solve(S_tilde, A.T)        # (m, K) = S̃^{-1} A^T

        # Posterior covariance (block)
        posterior_cov_bloc = prior_cov_bloc - A @ X   # (K, K)
       #posterior_cov_bloc = symmetrize(posterior_cov_bloc) + 1e-8 * jnp.eye(posterior_cov_bloc.shape[0], dtype=posterior_cov_bloc.dtype)

        # Innovation (measurement residual)
        innov = emission - emission_func(prior_mean, inpt)   # (m,)

        # Delta without forming K: delta = K @ innov = X.T @ (s * innov)
        delta_voltage = X.T @ (s * innov)             # (K,)

        # Update mean in-place on the first K components
        posterior_mean = prior_mean.at[:n_comps].add(delta_voltage)

        # If the rest of the mean/cov stays unchanged in this update:
        posterior_cov_diag_rest = prior_cov_diag_rest

        return (posterior_mean, posterior_cov_bloc, posterior_cov_diag_rest), None

    # Iterate re-linearization over posterior mean and covariance
    carry = (prior_mean, prior_cov_bloc, prior_cov_diag_rest)
    (mu_cond, Sigma_bloc_cond, Sigma_diag_rest_cond), _ = lax.scan(_step, carry, jnp.arange(num_iter))
    return mu_cond, Sigma_bloc_cond, Sigma_diag_rest_cond


# def get_sparsity(dynamics_function: Union[FnStateToState, FnStateAndInputToState],
#                  initial_state: Float[Array, " state_dim"],
#                  inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
#                 ):
    
#     F = jacobian(dynamics_function)
#     new_state = dynamics_function(initial_state, inputs[0])
#     Jac = F(new_state, inputs[1])
#     sparsity = device_get((Jac != 0))
#     sparsity = experimental.sparse.BCOO.fromdense(sparsity)
#     return sparsity
    

def bd_extended_kalman_filter(params: bdParamsNLGSSM,
                           emissions: Float[Array, "num_timesteps emission_dim"],
                           inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
                           num_iter: int = 1,
                           output_fields: Optional[List[str]]=["marginal_loglik"],
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
    
    f, h, H = params.dynamics_function, params.emission_function, params.emission_jacobian
    n_comps = H.shape[1]

    def F_blocks(x, u):
        
        K = n_comps
        x1, x2 = x[:K], x[K:]                 # (K,), (M,)
        n_tail = x2.shape[0]

        # ---- Leading KxK block: ∂f_{1:K}/∂x_{1:K} ----
        def f_head(x1_local):
            x_full = jnp.concatenate([x1_local, x2])
            return f(x_full, u)[:K]

        F11 = jacfwd(f_head)(x1)          # (K, K)

        # ---- Tail diagonal: diag( ∂f_{K+i}/∂x_{K+i} ) ----
        # Each tail output depends only on its own tail state (by your structure assumption).
        def grad_tail_i(i):
            def g_i(z_i):
                z = x2.at[i].set(z_i)
                x_full = jnp.concatenate([x1, z])
                return f(x_full, u)[K + i]    # scalar
            return grad(g_i)(x2[i])       # scalar derivative wrt x_{K+i}

        idx = jnp.arange(n_tail)
        F22_diag = vmap(grad_tail_i)(idx) # (M,)

        return F11, F22_diag


    f, h = (_process_fn(fn, inputs) for fn in (f, h))
    inputs = _process_input(inputs, num_timesteps)

    init_cov_diag = params.initial_covariance_diag
    init_cov_bloc = jnp.diag(init_cov_diag[:n_comps])
    init_cov_diag_rest = init_cov_diag[n_comps:]
    Q_diag = params.dynamics_covariance_diag
    Q_bloc = jnp.diag(Q_diag[:n_comps])
    Q_diag_rest = Q_diag[n_comps:]
    R_diag = params.emission_covariance_diag

    def _step(carry, t):
        """Iteratively update the state estimate and log likelihood."""
        ll, pred_mean, pred_cov_bloc, pred_cov_diag_rest = carry

        # Get parameters and inputs for time index t
        u = inputs[t]
        y = emissions[t]

        # Update the log likelihood

        ll += MVN(h(pred_mean, u), H @ pred_cov_bloc @ H.T + jnp.diag(R_diag)).log_prob(jnp.atleast_1d(y))

        # Condition on this emission
        filtered_mean, filtered_cov_bloc, filtered_cov_diag_rest = _condition_on(pred_mean, pred_cov_bloc, pred_cov_diag_rest, h, H, R_diag, u, y, num_iter)

        # Predict the next state
        pred_mean, pred_cov_bloc, pred_cov_diag_rest = _predict(filtered_mean, filtered_cov_bloc, filtered_cov_diag_rest, f, F_blocks, Q_bloc, Q_diag_rest, u)

        # Build carry and output states
        carry = (ll, pred_mean, pred_cov_bloc, pred_cov_diag_rest)
        outputs = {
            "filtered_means": filtered_mean,
            "filtered_cov_bloc": filtered_cov_bloc,
            "filtered_cov_diag_rest": filtered_cov_diag_rest,
            "predicted_means": pred_mean,
            "predicted_cov_bloc": pred_cov_bloc,
            "predicted_cov_diag_rest": pred_cov_diag_rest,
            "marginal_loglik": ll,
        }
        outputs = {key: val for key, val in outputs.items() if key in output_fields}

        return carry, outputs

    # Run the extended Kalman filter
    carry = (0.0, params.initial_mean, init_cov_bloc, init_cov_diag_rest)
    levels = 2
    checkpoint_lengths = [int(np.ceil(num_timesteps ** (1/levels))) for _ in range(levels)]
    length = prod(checkpoint_lengths)
    size_diff = length - num_timesteps
    xs = jnp.concatenate((jnp.arange(num_timesteps), jnp.zeros(size_diff, dtype=jnp.int32)))
    _, outputs = nested_checkpoint_scan(_step, carry, xs, length=length, nested_lengths=checkpoint_lengths)
    outputs = tree_map(lambda x: x[:num_timesteps], outputs) #remove extra dummy timesteps
    ll = outputs["marginal_loglik"][-1]
    outputs = {**outputs, "marginal_loglik": ll}
    posterior_filtered = PosteriorGSSMFiltered(
        **outputs,
    )
    return posterior_filtered