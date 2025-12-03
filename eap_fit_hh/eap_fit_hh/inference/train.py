import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax import jit, vmap, value_and_grad
from jax.tree_util import tree_map, tree_leaves
import jax.random as random
from jax import config
config.update("jax_enable_x64", True)
import tensorflow_probability.substrates.jax as tfp
tf = tfp.tf2jax
tfd = tfp.distributions

from typing import Dict, Tuple, List, Callable, Optional, Union, Any
from jax.scipy.special import logit
from jax.scipy.special import expit
import jaxley as jx
from jaxley.optimize.transforms import SoftplusTransform, SigmoidTransform
from jaxley.utils.cell_utils import params_to_pstate

from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter, extended_kalman_smoother

import optax

from eap_fit_hh.inference.diag_EKF import diagParamsNLGSSM, diag_extended_kalman_filter, get_sparsity

from eap_fit_hh.inference.bloc_diag_EKF import bdParamsNLGSSM, bd_extended_kalman_filter

from eap_fit_hh.utils import compute_comp_xyz, rotate_translate_positions, branch_cell_pos_recons, update_dict, regroup_learned_params, distance, get_surface_areas, is_nested, compute_eap, build_axial_matrix, build_eap_M

from tqdm import tqdm

import warnings

Bounds = Union[Dict[str, Dict[str, Tuple[float, float]]], Dict[str, Tuple[float, float]]]
Trainable = Union[Dict[str, List[str]], List[str]]
Noise = Dict[str, Union[float, jnp.ndarray]]
Param = Union[Dict[str, Dict[str, float]], Dict[str, float]]
Jaxley_Param = List[Dict[str, jnp.ndarray]]

#CHANNEL_STATES = ("HH_m", "HH_h", "HH_n")
CHANNEL_STATES = ()

def _pack_state(jaxley_state: Dict[str, jnp.ndarray], state_names: List[str]) -> jnp.ndarray:
    parts = []
    for name in state_names:
        x = jaxley_state[name].reshape(-1)
        if name in CHANNEL_STATES:
            x = logit(x)
        parts.append(x)
    return jnp.concatenate(parts, axis=0)

def _unpack_state(flat: jnp.ndarray, state_names: List[str], sizes: List[int]) -> Dict[str, jnp.ndarray]:
    out = {}
    offset = 0
    for name, sz in zip(state_names, sizes):
        # Python ints for offsets/sizes keep this jit-friendly
        chunk = flat[offset: offset + sz]
        out[name] = expit(chunk) if name in CHANNEL_STATES else chunk
        #out[name] = chunk
        offset += sz
    return out

def build_gate_mask(state_names, sizes):
    mask = []
    for name, sz in zip(state_names, sizes):
        mask.extend([name in CHANNEL_STATES] * sz)
    return jnp.array(mask, dtype=bool)


class EKFTrainer:
    def __init__(
        self,
        cell,
        trainable_params: Trainable,
        dt: float,
        grid: jnp.ndarray,
        extracellular_resistivity: float,
        dynamics_std: Optional[Noise] = None,
        opt_bounds: Optional[Bounds] = None,
        position_training: Optional[str] = None,
        voltage_solver: str = "jaxley.stone",
        base_cell_positions: Optional[jnp.ndarray] = None
        ):
        
        assert cell.initialized, "Module is not initialized, run `.initialize()`."
        self.cell = cell
        self.dt = float(dt)
        self.ncomps = self.cell.nodes.shape[0]
        self.voltage_solver = voltage_solver

        # prepare cell/JAX view once
        self.cell.to_jax()
        self.external_inds = self.cell.external_inds.copy()

        # Set trainable params
        self.trainable_params = trainable_params
        self.setup_trainables(trainable_params)
        print(self.trainable_params)
        # initial params / transforms
        ref_jaxley_params = self.cell.get_parameters()
        self.ref_cond_params = regroup_learned_params(self.trainable_params, ref_jaxley_params)

        if opt_bounds is None:
            # your existing default transform
            self.transform = jx.ParamTransform([{k: SoftplusTransform(0.)} for p in ref_jaxley_params for k in p])
        else:
            # build from bounds if you want; leaving as exercise
            self.transform = jx.ParamTransform(self.bounded_transform(opt_bounds))

        ref_opt_params = self.transform.inverse(ref_jaxley_params)
        self.ref_opt_params, self._to_params = ravel_pytree(ref_opt_params)

        # initial jaxley state (voltage & gates only depend on dt, kinetics, init V)
        pstate = params_to_pstate(ref_jaxley_params, self.cell.indices_set_by_trainables)
        ref_all_params = self.cell.get_all_parameters(pstate, voltage_solver=self.voltage_solver)
        init_state = self.cell.get_all_states(pstate, ref_all_params, self.dt)

        self.ref_all_params = ref_all_params
        
        self.init_state = init_state

        self.state_names = list(init_state.keys())
        self.sizes = [init_state[n].size for n in self.state_names]
        self.gate_mask = build_gate_mask(self.state_names, self.sizes)

        self.init_dynamax_state = _pack_state(init_state, self.state_names)
        self.num_states = int(self.init_dynamax_state.size)

        self.position_training = position_training
        self.true_cell_positions = compute_comp_xyz(cell)
        if self.position_training == "rigid":
            self.base_cell_positions = base_cell_positions
            if base_cell_positions is None:
                raise ValueError(f'Missing base_cell_positions.')
        elif self.position_training == "branch model":
            n_branches = self.cell.nodes["local_branch_index"].nunique()
            self.n_nodes = 2 * n_branches
            ncomps_per_branch_df = cell.nodes["local_branch_index"].value_counts()
            self.ncomps_per_branch = ncomps_per_branch_df.to_numpy()
            
        self.dynamics_std = dynamics_std

        self.grid = grid
        self.extracellular_resistivity = extracellular_resistivity

        self.axial_cond_matrix = build_axial_matrix(self.cell)

    def setup_trainables(self, trainable_params: Trainable):
        true_group_names = list(self.cell.groups.keys())
        if isinstance(trainable_params, dict):
            for group_name, params in trainable_params.items():
                if group_name not in true_group_names:
                    raise ValueError(f"Invalid element: {group_name}. Allowed values are {true_group_names}")
                for p in params:
                    getattr(self.cell, group_name).make_trainable(p)
        elif isinstance(trainable_params, list):
            if len(true_group_names) == 0:
                # if no group, assume parameters are shared across the whole cell
                for param in trainable_params:
                    self.cell.make_trainable(param)
            else:
                # if there are groups, assume we learn the parameters listed for each group seperately
                trainables_dict = {}
                for group_name in true_group_names:
                    trainables_dict[group_name] = trainable_params
                    for param in trainable_params:
                        getattr(self.cell, group_name).make_trainable(param)
                self.trainable_params = trainables_dict
        else:
            raise TypeError(f"Argument must be a dictionary or a list, got {type(trainable_params).__name__}")
        print(self.trainable_params)

    def bounded_transform(self, bounds: Bounds):

        out: List[dict] = []
        # Case 1: group-specific parameters.
        if isinstance(self.trainable_params, dict):
            # each parameter has its own bounds.
            if is_nested(bounds):
                for region, params in self.trainable_params.items():
                    if region not in bounds:
                        raise KeyError(f"Region '{region}' not found in bounds.")
                    region_bounds = bounds[region]

                    for p in params:
                        if p not in region_bounds:
                            raise KeyError(f"Param '{p}' missing in bounds['{region}'].")
                        lo, hi = region_bounds[p]
                        if not (hi > lo):
                            raise ValueError(f"Invalid bounds for {region}.{p}: ({lo}, {hi})")
                        out.append({p: SigmoidTransform(lo, hi)})
            # parameters share bounds across different groups and bounds therefore given as a flat dict.
            # e.g. HH_gNa will take different values depending on the group but the bounds will be the same.            
            else:
                for region, params in self.trainable_params.items():
                    for p in params:
                        if p not in bounds:
                            raise KeyError(f"Param '{p}' missing in bounds.")
                        lo, hi = bounds[p]
                        if not (hi > lo):
                            raise ValueError(f"Invalid bounds for {region}.{p}: ({lo}, {hi})")
                        out.append({p: SigmoidTransform(lo, hi)})
        # Case 2: parameters are shared across the cell and bounds should therefore be given as a flat dict.  
        else:
            if is_nested(bounds):
                raise TypeError('bounds should be a flat dictionary.')
            for param in self.trainable_params:
                if param not in bounds:
                    raise KeyError(f"Param '{param}' missing in bounds.")
                lo, hi = bounds[param]
                if not (hi > lo):
                    raise ValueError(f"Invalid bounds for {param}: ({lo}, {hi})")
                out.append({param: SigmoidTransform(lo, hi)})

        return out
    
    def _build_dyn_cov_diag(self, noise: Noise, default: float = 0.01):
        parts = []
        for i, name in enumerate(self.state_names):
            aval = jnp.asarray(noise.get(name, default))
            if aval.ndim == 0 or aval.size == 1:
                # scalar or length-1 array -> broadcast
                x = jnp.broadcast_to(aval.reshape(()), (self.sizes[i],))
            elif aval.ndim == 1 and aval.size == self.sizes[i]:
                x = aval.reshape(-1)
            else:
                raise ValueError(
                    f"Noise for '{name}' must be scalar/len-1 or length {self.sizes[i]}, "
                    f"got shape {aval.shape}."
                )

            parts.append(x)

        return jnp.square(jnp.concatenate(parts, axis=0))

    def _dynamics_for_params(self, all_params: Dict[str, jnp.ndarray]) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        dt = self.dt
        external_inds = self.external_inds
        state_names = self.state_names
        sizes = self.sizes
        cell = self.cell
        voltage_solver = self.voltage_solver

        def dynamics(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            jstate = _unpack_state(x, state_names, sizes)
            next_state = cell.step(
                jstate,
                dt,
                external_inds,
                {"i": u},
                params = all_params,
                solver = "bwd_euler",
                voltage_solver = voltage_solver,
            )
            return _pack_state(next_state, state_names)
        return dynamics
    
    # def _emission_fn(self, cell_positions):

    #     distances = distance(self.grid, cell_positions)
    #     distances_CM = distances * 10**(-4)
    #     state_names = self.state_names
    #     sizes = self.sizes

    #     def emission(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    #         '''
    #         Returns (num_electrodes,) array of EAPs at fixed time
    #         '''
    #         scaling = self.resistivity/(4.*jnp.pi) 
    #         jstate = _unpack_state(x, state_names, sizes)
    #         surf_currents_states = [value for key, value in jstate.items() if key.startswith("i_")]
    #         surf_currents_array = jnp.stack(surf_currents_states) # (num_currents, Ncomps)
    #         currents = surf_currents_array * self.surf_areas_CM2
    #         currents_per_comp = currents.sum(axis=0)
    #         extr_voltage = jnp.sum(currents_per_comp / distances_CM, axis=1) * scaling
    #         return extr_voltage
        
    #     return emission
    
    def _emission_fn(self, cell_positions):

        distances = distance(self.grid, cell_positions)
        distances_CM = distances * 10**(-4)
        state_names = self.state_names
        sizes = self.sizes
        axial_cond_matrix = self.axial_cond_matrix
        idx_stimulus = self.cell.external_inds['i']
        M = build_eap_M(axial_cond_matrix, distances_CM, self.extracellular_resistivity)

        def emission(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            '''
            Returns (num_electrodes,) array of EAPs at fixed time
            '''
            jstate = _unpack_state(x, state_names, sizes)
            v = jstate['v']
            extr_voltage = compute_eap(v, M, u, idx_stimulus, distances_CM, self.extracellular_resistivity)
            return extr_voltage
        
        return emission, M

    def _make_loss(self,
                true_eap: jnp.ndarray,
                current: jnp.ndarray
                ) -> Callable[[jnp.ndarray], Tuple[jnp.ndarray, Dict[str, Any]]]:
        
        num_emissions = int(true_eap.shape[1])

        if self.diagonal=="diagonal":
            ref_dynamics_function = self._dynamics_for_params(self.ref_all_params)
            sparsity = get_sparsity(ref_dynamics_function, self.init_dynamax_state, current)

            def loss(opt_params: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:

                if self.noise_training is None:
                    dyn_cov_diag = self._build_dyn_cov_diag(self.dynamics_std)
                else:
                    noise_params = opt_params[-self.number_noise_params:]
                    opt_params = opt_params[:-self.number_noise_params]
                    current_dynamics_std = update_dict(self.noise_training, self.noise_lengths, self.dynamics_std, noise_params)
                    dyn_cov_diag = self._build_dyn_cov_diag(current_dynamics_std)

                if self.position_training == "rigid":
                    #retrieve conductance param
                    opt_params_cond = opt_params[:-6]
                    #retrieve position param
                    opt_params_pos = opt_params[-6:]
                    theta_x = opt_params_pos[0]
                    theta_y = opt_params_pos[1]
                    theta_z = opt_params_pos[2]
                    translation = opt_params_pos[-3:]
                    #reconstruct cell positions
                    cell_positions = rotate_translate_positions(self.base_cell_positions, theta_x, theta_y, theta_z, translation)
                
                elif self.position_training == "branch model":
                    #retrieve conductance param
                    opt_params_cond = opt_params[:-3*self.n_nodes]
                    #retrieve pos param
                    cell_xyz = opt_params[-3*self.n_nodes:]
                    cell_xyz = cell_xyz.reshape((self.n_nodes,3))
                    cell_positions = branch_cell_pos_recons(cell_xyz, self.ncomps_per_branch)

                elif self.position_training is None:
                    cell_positions = self.true_cell_positions
                    opt_params_cond = opt_params

                emission, _ = self._emission_fn(cell_positions) #emission_fn will depend on positions (set to true if we're not learning them)

                unconstrained_params_cond = self._to_params(opt_params_cond)
                params_cond = self.transform.forward(unconstrained_params_cond)
                pstate = params_to_pstate(params_cond, self.cell.indices_set_by_trainables)
                all_params = self.cell.get_all_parameters(pstate, voltage_solver=self.voltage_solver)
                dynamics = self._dynamics_for_params(all_params)

                model = diagParamsNLGSSM(
                    initial_mean = self.init_dynamax_state,
                    initial_covariance_diag = 0.1 * jnp.ones(self.num_states),
                    dynamics_function = dynamics,
                    dynamics_covariance_diag = self.dt * dyn_cov_diag,
                    emission_function = emission,
                    emission_covariance_diag = (self.obs_std ** 2) * jnp.ones(num_emissions),
                )

                post = diag_extended_kalman_filter(params = model, emissions = true_eap, inputs = current, sparsity = sparsity)
                aux = {'post_means' : post.filtered_means, 'post_covs': post.filtered_variances}
                return -post.marginal_loglik, aux
            return loss

        if self.diagonal in ["bd", "bloc-diagonal"]:
            def loss(opt_params: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:

                if self.noise_training is None:
                    dyn_cov_diag = self._build_dyn_cov_diag(self.dynamics_std)
                else:
                    noise_params = opt_params[-self.number_noise_params:]
                    opt_params = opt_params[:-self.number_noise_params]
                    current_dynamics_std = update_dict(self.noise_training, self.noise_lengths, self.dynamics_std, noise_params)
                    dyn_cov_diag = self._build_dyn_cov_diag(current_dynamics_std)

                if self.position_training == "rigid":
                    #retrieve conductance param
                    opt_params_cond = opt_params[:-6]
                    #retrieve position param
                    opt_params_pos = opt_params[-6:]
                    theta_x = opt_params_pos[0]
                    theta_y = opt_params_pos[1]
                    theta_z = opt_params_pos[2]
                    translation = opt_params_pos[-3:]
                    #reconstruct cell positions
                    cell_positions = rotate_translate_positions(self.base_cell_positions, theta_x, theta_y, theta_z, translation)
                
                elif self.position_training == "branch model":
                    #retrieve conductance param
                    opt_params_cond = opt_params[:-3*self.n_nodes]
                    #retrieve pos param
                    cell_xyz = opt_params[-3*self.n_nodes:]
                    cell_xyz = cell_xyz.reshape((self.n_nodes,3))
                    cell_positions = branch_cell_pos_recons(cell_xyz, self.ncomps_per_branch)

                elif self.position_training is None:
                    cell_positions = self.true_cell_positions
                    opt_params_cond = opt_params

                emission, H = self._emission_fn(cell_positions) #emission_fn will depend on positions (set to true if we're not learning them)

                unconstrained_params_cond = self._to_params(opt_params_cond)
                params_cond = self.transform.forward(unconstrained_params_cond)
                pstate = params_to_pstate(params_cond, self.cell.indices_set_by_trainables)
                all_params = self.cell.get_all_parameters(pstate, voltage_solver=self.voltage_solver)
                dynamics = self._dynamics_for_params(all_params)

                model = bdParamsNLGSSM(
                    initial_mean = self.init_dynamax_state,
                    initial_covariance_diag = 0.1 * jnp.ones(self.num_states),
                    dynamics_function = dynamics,
                    dynamics_covariance_diag = self.dt * dyn_cov_diag,
                    emission_function = emission,
                    emission_jacobian = H, 
                    emission_covariance_diag = (self.obs_std ** 2) * jnp.ones(num_emissions),
                )

                post = bd_extended_kalman_filter(params = model, 
                                                 emissions = true_eap, 
                                                 inputs = current)
                aux = {'post_means' : post.filtered_means}
                return -post.marginal_loglik, aux
            return loss

        else:
            def loss(opt_params: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, Any]]:

                if self.noise_training is None:
                    dyn_cov_diag = self._build_dyn_cov_diag(self.dynamics_std)
                else:
                    noise_params = opt_params[-self.number_noise_params:]
                    opt_params = opt_params[:-self.number_noise_params]
                    current_dynamics_std = update_dict(self.noise_training, self.noise_lengths, self.dynamics_std, noise_params)
                    dyn_cov_diag = self._build_dyn_cov_diag(current_dynamics_std)
                
                if self.position_training == "rigid":
                    #retrieve conductance param
                    opt_params_cond = opt_params[:-6]
                    #retrieve position param
                    opt_params_pos = opt_params[-6:]
                    theta_x = opt_params_pos[0]
                    theta_y = opt_params_pos[1]
                    theta_z = opt_params_pos[2]
                    translation = opt_params_pos[-3:]
                    #reconstruct cell positions
                    cell_positions = rotate_translate_positions(self.base_cell_positions, theta_x, theta_y, theta_z, translation)
                
                elif self.position_training == "branch model":
                    #retrieve conductance param
                    opt_params_cond = opt_params[:-3*self.n_nodes]
                    #retrieve pos param
                    cell_xyz = opt_params[-3*self.n_nodes:]
                    cell_xyz = cell_xyz.reshape((self.n_nodes,3))
                    cell_positions = branch_cell_pos_recons(cell_xyz, self.ncomps_per_branch)

                elif self.position_training is None:
                    cell_positions = self.true_cell_positions
                    opt_params_cond = opt_params

                emission, _ = self._emission_fn(cell_positions) #emission_fn will depend on positions (set to true if we're not learning them)

                unconstrained_params_cond = self._to_params(opt_params_cond)
                params_cond = self.transform.forward(unconstrained_params_cond)
                pstate = params_to_pstate(params_cond, self.cell.indices_set_by_trainables)
                all_params = self.cell.get_all_parameters(pstate, voltage_solver=self.voltage_solver)
                dynamics = self._dynamics_for_params(all_params)

                model = ParamsNLGSSM(
                    initial_mean = self.init_dynamax_state,
                    initial_covariance = jnp.diag(0.1 * jnp.ones(self.num_states)),
                    dynamics_function = dynamics,
                    dynamics_covariance = self.dt * jnp.diag(dyn_cov_diag),
                    emission_function = emission,
                    emission_covariance = (self.obs_std ** 2) * jnp.eye(num_emissions),
                )

                if self.smoother:
                    post = extended_kalman_smoother(params = model, emissions = true_eap, inputs = current)
                    aux = {'post_means' : post.smoothed_means, 'post_covs': post.smoothed_covariances}
                else:   
                    post = extended_kalman_filter(params = model, emissions = true_eap, inputs = current)
                    aux = {'post_means' : post.filtered_means, 'post_covs': post.filtered_covariances}
                return - post.marginal_loglik, aux
            return loss

    # --------- public API ---------

    def _to_jaxley_params(self, cond_param: Param) -> Jaxley_Param:
        
        out: Jaxley_Param = []

        # Case 1: group-specific parameters.
        if isinstance(self.trainable_params, dict):
            # each parameter has its own value.
            if is_nested(cond_param):
                for region, params in self.trainable_params.items():
                    if region not in cond_param:
                        raise KeyError(f"Region '{region}' not found in cond_param.")
                    region_values = cond_param[region]
                    for p in params:
                        if p not in region_values:
                            raise KeyError(f"Param '{p}' missing in cond_param['{region}'].")
                        cond_value = region_values[p]
                        out.append({p: jnp.array([cond_value])})
            # parameters share bounds across different groups but values given as a flat dict.
            # e.g. HH_gNa's are technically different between the groups but we assign the same value.             
            else:
                for region, params in self.trainable_params.items():
                    for p in params:
                        if p not in cond_param:
                            raise KeyError(f"Param '{p}' missing in cond_param.")
                        cond_value = cond_param[p]
                        out.append({p: jnp.array([cond_value])})
        # Case 2: parameters are shared across the cell and values should be given as a flat dict.              
        else:
            if is_nested(cond_param):
                raise TypeError('cond_param should be a flat dictionary.')
            for param in self.trainable_params:
                cond_value = cond_param[param]
                out.append({param: jnp.array([cond_value])})

        return out


    def sample_cond_params(self, bounds: Bounds, key: int) -> Jaxley_Param:
        """
        Samples values uniformly from `bounds` for each parameter listed in `trainable_params`,
        preserving the order (and duplicates) specified in `trainable_params`.

        Args:
            bounds: Bounds
            seed: Optional int for reproducibility.

        Returns:
            A list like [{"paramA": valA}, {"paramB": valB}, ...] following the order
            in trainable_params.
        """
        rng = random.PRNGKey(key)
        out: Jaxley_Param = []

        # Case 1: group-specific parameters.
        if isinstance(self.trainable_params, dict):
            # each parameter has its own bounds.
            if is_nested(bounds):
                for region, params in self.trainable_params.items():
                    if region not in bounds:
                        raise KeyError(f"Region '{region}' not found in bounds.")
                    region_bounds = bounds[region]

                    for p in params:
                        if p not in region_bounds:
                            raise KeyError(f"Param '{p}' missing in bounds['{region}'].")
                        lo, hi = region_bounds[p]
                        if not (hi > lo):
                            raise ValueError(f"Invalid bounds for {region}.{p}: ({lo}, {hi})")
                        rng, sk = random.split(rng)
                        out.append({p: tfd.Uniform(lo, hi).sample(sample_shape=(1,), seed = sk)})
            # parameters share bounds across different groups and bounds therefore given as a flat dict.
            # e.g. HH_gNa takes different values depending on the group but the bounds are the same.                        
            else:
                for region, params in self.trainable_params.items():
                    for p in params:
                        if p not in bounds:
                            raise KeyError(f"Param '{p}' missing in bounds.")
                        lo, hi = bounds[p]
                        if not (hi > lo):
                            raise ValueError(f"Invalid bounds for {region}.{p}: ({lo}, {hi})")
                        rng, sk = random.split(rng)
                        out.append({p: tfd.Uniform(lo, hi).sample(sample_shape=(1,), seed = sk)})
        # Case 2: parameters are shared across the cell and bounds should be given as a flat dict.              
        else:
            if is_nested(bounds):
                raise TypeError('bounds should be a flat dictionary.')
            for param in self.trainable_params:
                if param not in bounds:
                    raise KeyError(f"Param '{param}' missing in bounds.")
                lo, hi = bounds[param]
                if not (hi > lo):
                    raise ValueError(f"Invalid bounds for {param}: ({lo}, {hi})")
                rng, sk = random.split(rng)
                out.append({param: tfd.Uniform(lo, hi).sample(sample_shape=(1,), seed = sk)})

        return out
    
    def sample_pos_params(self, radial_sd: float, key: int) -> jnp.ndarray:
        """
        Sample initial position parameters for cell morphology.

        Args:
            radial_sd (float): Standard deviation of Gaussian noise applied to positions.
            key (int): Random seed for reproducibility.

        Returns:
            jnp.ndarray: 
                - If self.position_training == "rigid": shape (6,), containing three rotation angles and a translation vector.
                - If self.position_training == "branch model": shape (3 * self.n_nodes,), containing noisy 3D coordinates of all branch endpoints.
        """
        rng = random.PRNGKey(key)

        if self.position_training == "rigid":

            rng, sk_rot, sk_trans = random.split(rng, 3)

            # sample rotation angles
            theta_x, theta_y, theta_z = tfd.Uniform(0, 2 * jnp.pi).sample(sample_shape=(3,), seed = sk_rot)
            # sample a translation vector
            translation = tfd.Normal(0, radial_sd).sample((3,), seed = sk_trans)
            # concatenate to pos_params array
            pos_params = jnp.concatenate(
                (jnp.array([theta_x]), jnp.array([theta_y]), jnp.array([theta_z]), translation)
            )

        elif self.position_training == "branch model":
            # sampling positions of branch endpoints (nodes)
            nodes_pos = []
            idx = 0
            for ncomps in self.ncomps_per_branch:  # array of ints, one per branch
                nodes_pos.extend(self.true_cell_positions[idx])  # first compartment
                nodes_pos.extend(self.true_cell_positions[idx + ncomps - 1])  # last compartment
                idx += ncomps  # move to the next branch start

            # add Gaussian noise to each coordinate
            rng, sk_noise = random.split(rng)
            noise = tfd.Normal(0, radial_sd).sample((3 * self.n_nodes,), seed = sk_noise)
            pos_params = jnp.array(nodes_pos) + noise

        return pos_params
    
    def train(self,
            data: Tuple[jnp.ndarray, jnp.ndarray],
            init_cond_params : Jaxley_Param,
            obs_std = float,
            init_pos_params : Optional[jnp.ndarray] = None,
            init_noise_params : Optional[Noise] = None,
            lr: float = 0.1,
            num_epochs: int = 2000,
            diagonal: Optional[str] = None,
            smoother: bool = False,
            output: Optional[List[str]] = None):
        
        
        if output is None:
            output = ['cond_params']
        
        self.diagonal = diagonal

        self.smoother = bool(smoother)
    
        self.obs_std = obs_std

        if init_noise_params is not None:
            self.noise_training = list(init_noise_params.keys()) 
            init_noise_values = jnp.concatenate([jnp.atleast_1d(jnp.asarray(v)) for v in init_noise_params.values()])
            self.noise_lengths = []
            for v in init_noise_params.values():
                arr = jnp.asarray(v)
                self.noise_lengths.append(int(arr.size))
            self.number_noise_params = len(init_noise_values)
        else:
            self.noise_training = None
            if self.dynamics_std is None:
                raise ValueError(f'Missing dynamics_std')
       
        true_eap, current = data
        loss_fn = self._make_loss(true_eap, current)   # returns (loss_scalar, aux)
        loss_and_grad = jit(value_and_grad(loss_fn, has_aux=True))
        
        # transform init_params to unconstrained array
        unconstrained_cond_params = self.transform.inverse(init_cond_params)
        opt_cond_params, _ = ravel_pytree(unconstrained_cond_params)
    
        if self.position_training is None:
            opt_params = opt_cond_params
            if init_pos_params is not None:
                warnings.warn(f'init_pos_params discarded since self.position_training is None')
        else:
            if init_pos_params is None:
                raise ValueError(f'Missing initial position params.')
            opt_params = jnp.concatenate((opt_cond_params, init_pos_params))

        if self.noise_training is not None:
            opt_params = jnp.concatenate((opt_params, init_noise_values))

        opt = optax.adam(lr)
        opt_state = opt.init(opt_params)

        mll_history = []
        opt_params_history = []
        #print(opt_params)
        for epoch in tqdm(range(num_epochs), desc="EKF train"):
            (loss_val, aux), g = loss_and_grad(opt_params)
            #print(-loss_val)

            # record current values (before the update)
            mll_history.append(-loss_val)
            opt_params_history.append(opt_params)

            # update parameters
            updates, opt_state = opt.update(g, opt_state)
            opt_params = optax.apply_updates(opt_params, updates)

        results = {}

        # convert Python list into JAX array
        opt_params_history = jnp.stack(opt_params_history, axis=0)  # shape (num_epochs, num_params)

        if self.noise_training is None:
            if "noise_params" in output:
                results["noise_params"] = None
        else:
            opt_noise_params_history = opt_params_history[:,-self.number_noise_params:]
            opt_params_history = opt_params_history[:,:-self.number_noise_params]
            if "noise_params" in output:
                dynamics_std_history = vmap(update_dict, in_axes = (None, None, None, 0))(
                self.noise_training, self.noise_lengths, self.dynamics_std, opt_noise_params_history)
                results["noise_params"] = dynamics_std_history

        if self.position_training == "rigid":
            opt_cond_params_history = opt_params_history[:,:-6]
            opt_pos_params_history = opt_params_history[:,-6:]
            if "pos_params" in output:
                results["pos_params"] = opt_pos_params_history
        elif self.position_training == "branch model":
            opt_cond_params_history = opt_params_history[:,:-3 * self.n_nodes]
            opt_pos_params_history = opt_params_history[:,-3 * self.n_nodes:]
            if "pos_params" in output:
                results["pos_params"] = opt_pos_params_history
        elif self.position_training is None:
            opt_cond_params_history = opt_params_history
            opt_pos_params_history = None
            if "pos_params" in output:
                results["pos_params"] = opt_pos_params_history

        # map opt_params_history back to Jaxley-params-like object
        if "cond_params" in output or "jaxley_params" in output:
            opt_cond_params_seq = vmap(self._to_params)(opt_cond_params_history)
            cond_params_seq = self.transform.forward(opt_cond_params_seq)
            if "jaxley_params" in output:
                results["jaxley_params"] = cond_params_seq
            if "cond_params" in output:
                # map params to Param-type object
                cond_params = regroup_learned_params(self.trainable_params, cond_params_seq)
                results["cond_params"] = cond_params

        if "cell_positions" in output:
            # reconstruct compartments 3D location from pos_params
            if self.position_training == "rigid":
                thetas_x = opt_pos_params_history[:,0]
                thetas_y = opt_pos_params_history[:,1]
                thetas_z = opt_pos_params_history[:,2]
                translations = opt_pos_params_history[:,-3:]
                cell_positions_history = vmap(rotate_translate_positions, in_axes=(None, 0, 0, 0, 0))(
                    self.base_cell_positions, thetas_x, thetas_y, thetas_z, translations)
            elif self.position_training == "branch model":
                inferred_cell_xyz = opt_pos_params_history.reshape((-1, self.n_nodes, 3))
                cell_positions_history = vmap(branch_cell_pos_recons, in_axes = (0, None))(
                    inferred_cell_xyz, self.ncomps_per_branch)
            else:
                cell_positions_history = None
            results["cell_positions"] = cell_positions_history
        
        if "mll" in output:
            mll_history = jnp.stack(mll_history, axis=0)  # shape (num_epochs,)
            results["mll"] = mll_history
        
        if "last post means" in output: 
            # map post_means_array back to Jaxley-states-like object
            last_post_means_array = aux['post_means']  # (T, state_dim)
            last_post_means_unpacked = vmap(lambda vec: _unpack_state(vec, self.state_names, self.sizes))(last_post_means_array) # post_means_unpacked is a dict {state_i: (T, size_i)}
            results["last post means"] = last_post_means_unpacked

        if "last post covariances" in output:
            last_post_covs = aux['post_covs']  # (T, state_dim, state_dim) or (num_epochs, T, state_dim)
            results["last post covariances"] = last_post_covs # variances only if diagonal is True

        return results

    
    def integrate_from_params(self,
                            cond_param: dict,
                            ) -> jnp.ndarray:
        """
        Batch-integrate the cell using per-epoch parameter trees.

        """
            
        jaxley_params = self._to_jaxley_params(cond_param)

        trajectories = jx.integrate(self.cell, params = jaxley_params)

        return trajectories    
    
    # def integrate_from_params(self,
    #                         train_output: dict,
    #                         subset: Optional[Union[jnp.ndarray, List[int]]] = None
    #                         ) -> jnp.ndarray:
    #     """
    #     Batch-integrate the cell using per-epoch parameter trees.

    #     Expects train_output['jaxley_params'] to be a pytree whose leaves have shape (N, ...),
    #     where N = number of epochs/snapshots.
    #     """
        
    #     jaxley_params_history = train_output['jaxley_params']

    #     first_leaf = tree_leaves(jaxley_params_history)[0]
    #     N = first_leaf.shape[0]

    #     if subset is None:
    #         idx = jnp.arange(N)
    #     else:
    #         idx = jnp.asarray(subset, dtype=jnp.int32).reshape(-1)
    #         if (idx < 0).any() or (idx >= N).any():
    #             raise IndexError(f"subset indices must be in [0, {N-1}]")
            
    #     trajectories = vmap(lambda params: jx.integrate(self.cell, params = params))(
    #         tree_map(lambda x: x[idx], jaxley_params_history)
    #     )
    #     B, rec_dim , T = trajectories.shape
    #     trajectories = trajectories.reshape(B, rec_dim//self.ncomps, self.ncomps, T)
    #     return trajectories    
    