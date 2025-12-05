import jax.numpy as jnp
from jax import vmap
from typing import Dict, List, Optional, Union
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from jaxley.io.graph import to_graph
from jaxley.utils.cell_utils import params_to_pstate

def compute_comp_xyz(cell):
    
    '''
    Returns (Ncomps, 3) array of xyz positions of all the compartments
    '''
    cell.compute_xyz()
    cell.compute_compartment_centers()
    xs = cell.nodes["x"].to_numpy()
    ys = cell.nodes["y"].to_numpy()
    zs = cell.nodes["z"].to_numpy()

    comp_xyz = jnp.stack([xs,ys,zs],axis=1)
    
    return comp_xyz

def distance(grid, cell_positions):
    
    '''
    grid: (Npoints, 3) array
    cell_positions: (Ncomps, 3) array
    Returns (Npoints, Ncomps) array of pairwise distances in um
    '''
    
    def _distance(grid_xyz, cell_xyz):
        
        return jnp.linalg.norm(grid_xyz - cell_xyz)
    
    return vmap(vmap(_distance, in_axes=(None, 0)),in_axes=(0, None))(grid, cell_positions)

def get_surface_areas(lengths, radii):
    
    '''
    Returns (Ncomps,) array of the surface areas in um2 of all the compartments
    '''
    surf_areas = 2*jnp.pi*radii*lengths
    return surf_areas

def rotation_matrix_x(theta_x):
    return jnp.array([
        [1, 0, 0],
        [0, jnp.cos(theta_x), -jnp.sin(theta_x)],
        [0, jnp.sin(theta_x), jnp.cos(theta_x)]
    ])

def rotation_matrix_y(theta_y):
    return jnp.array([
        [jnp.cos(theta_y), 0, jnp.sin(theta_y)],
        [0, 1, 0],
        [-jnp.sin(theta_y), 0, jnp.cos(theta_y)]
    ])

def rotation_matrix_z(theta_z):
    return jnp.array([
        [jnp.cos(theta_z), -jnp.sin(theta_z), 0],
        [jnp.sin(theta_z), jnp.cos(theta_z), 0],
        [0, 0, 1]
    ])

def rotate_translate_vector(vector, theta_x, theta_y, theta_z, translation):
    R_x = rotation_matrix_x(theta_x)
    R_y = rotation_matrix_y(theta_y)
    R_z = rotation_matrix_z(theta_z)
    
    R = R_z @ R_y @ R_x 
    return R @ vector + translation  

def rotate_translate_positions(positions, theta_x, theta_y, theta_z, translation):
    new_positions = vmap(rotate_translate_vector, in_axes=(0,  None, None, None, None))(positions, theta_x, theta_y, theta_z, translation)
    return new_positions

def branch_cell_pos_recons(extrem_pos: jnp.ndarray, ncomps_per_branch: jnp.ndarray) -> jnp.ndarray:
    """
    Reconstruct 3D compartment positions from branch endpoints.

    Args:
        extrem_pos: array of shape (2 * n_branches, 3),
            endpoints for each branch in consecutive pairs.
        ncomps_per_branch: array of shape (n_branches,),
            number of compartments per branch.

    Returns:
        cell_pos: array of shape (sum(ncomps_per_branch), 3),
            all compartment positions concatenated.
    """
    n_branches = extrem_pos.shape[0] // 2

    cell_pos_list = []
    for i in range(n_branches):
        start, end = extrem_pos[2 * i], extrem_pos[2 * i + 1]
        ncomps = int(ncomps_per_branch[i])
        pts = jnp.linspace(start, end, ncomps)  # (ncomps, 3)
        cell_pos_list.append(pts)

    return jnp.concatenate(cell_pos_list, axis=0)

def update_dict(keys_list: List[str],
                sizes: List[int],
                values_dict: Optional[Dict[str, jnp.ndarray]],
                float_array: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Update or create a dictionary mapping keys to slices of float_array.

    Args:
        keys_list (List[str]): Keys to update or create.
        sizes (List[int]): Number of floats for each key in keys_list.
        values_dict (Optional[Dict[str, jnp.ndarray]]): Dictionary to update. 
            If None, a new one is created.
        float_array (jnp.ndarray): Flat array of floats.

    Returns:
        Dict[str, jnp.ndarray]: Updated dict with arrays of the right length.
    """
    updated = {} if values_dict is None else values_dict.copy()
    idx = 0
    for key, size in zip(keys_list, sizes):
        updated[key] = float_array[idx: idx + size]
        idx += size
    return updated

def regroup_learned_params(
    trainable_params: Union[Dict[str, List[str]], List[str]],
    learned_params: List[Dict[str, jnp.ndarray]],
    enforce_names: bool = False,
):
    """
    Convert `learned_params` (a list of single-key dicts with arrays of shape (N,1) or (N,))
    into a structure aligned with `trainable_params`.

    - If `trainable_params` is Dict[str, List[str]], returns Dict[str, Dict[str, jnp.ndarray]].
    - If `trainable_params` is List[str], returns Dict[str, jnp.ndarray].

    Assumes order of `learned_params` matches the order of parameters as they appear in `trainable_params`.
    """
    # Count expected params
    if isinstance(trainable_params, dict):
        expected = sum(len(v) for v in trainable_params.values())
    elif isinstance(trainable_params, list):
        expected = len(trainable_params)
    else:
        raise TypeError("trainable_params must be Dict[str, List[str]] or List[str].")

    if expected != len(learned_params):
        raise ValueError(f"Count mismatch: expected {expected} params, got {len(learned_params)}")

    it = iter(learned_params)

    # Case 1: dict -> nested dict
    if isinstance(trainable_params, dict):
        out: Dict[str, Dict[str, jnp.ndarray]] = {}
        for region, params in trainable_params.items():
            region_dict: Dict[str, jnp.ndarray] = {}
            for pname in params:
                d = next(it)
                if len(d) != 1:
                    raise ValueError(f"Each learned_params item must have exactly one key, got {list(d.keys())}")
                (k, arr), = d.items()
                if enforce_names and k != pname:
                    raise ValueError(f"Order/key mismatch: expected {pname}, got {k}")
                region_dict[pname] = jnp.ravel(arr)
            out[region] = region_dict
        return out

    # Case 2: list -> flat dict
    else:
        out: Dict[str, jnp.ndarray] = {}
        for pname in trainable_params:
            d = next(it)
            if len(d) != 1:
                raise ValueError(f"Each learned_params item must have exactly one key, got {list(d.keys())}")
            (k, arr), = d.items()
            if enforce_names and k != pname:
                raise ValueError(f"Order/key mismatch: expected {pname}, got {k}")
            out[pname] = jnp.ravel(arr)
        return out
    
def plot_cond_params(params: dict,
                layout: str = "horizontal",
                true_params: Optional[dict] = None,
                label_params: Optional[dict] = None):
    """
    Plot parameter histories.

    Args:
        params: dict of dicts or flat dict
            - Nested dict: {'soma': {'HH_gK': array(N,), 'HH_gNa': array(N,)}}
              → one figure per group.
            - Flat dict: {'HH_gK': array(N,), 'HH_gNa': array(N,)}
              → one figure.
        layout: "horizontal" or "vertical"
            Controls subplot arrangement for *both* cases. Default = "horizontal".
        true_params: dict of true values
            - Nested case: {'soma': {'HH_gK': float, 'HH_gNa': float}, ...}
            - Flat case: {'HH_gK': float, 'HH_gNa': float, ...}
            If provided, each subplot shows the true value as a dashed black line.
        label_params: dict of labels to replace keys in plots
            - Nested case: {'soma': {'HH_gK': 'g_K', 'HH_gNa': 'g_Na'}, ...}
            - Flat case: {'HH_gK': 'g_K', 'HH_gNa': 'g_Na'}
            If None, use original keys.
    """
    def plot_group(group_name, pdict, horizontal: bool,
                   true_group: Optional[dict],
                   label_group: Optional[dict]):
        keys = list(pdict.keys())
        n_plots = len(keys)
        if n_plots == 0:
            return

        N = len(next(iter(pdict.values())))
        epochs = jnp.arange(N)

        if horizontal:
            fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3), sharey=False)
        else:
            fig, axes = plt.subplots(n_plots, 1, figsize=(6, 2.5 * n_plots), sharex=True)

        if n_plots == 1:
            axes = [axes]

        for ax, k in zip(axes, keys):
            y = jnp.asarray(pdict[k]).reshape(-1)
            ax.plot(epochs, y, label="learned")

            if true_group is not None and k in true_group:
                ax.axhline(true_group[k], linestyle="--", color="black", label="true")

            # pick label: custom if provided, otherwise use key
            label = label_group.get(k, k) if label_group is not None else k

            ax.set_title(label)
            ax.grid(True, linestyle="--", alpha=0.4)
            if not horizontal:
                ax.set_ylabel(label)
            ax.legend()

        axes[-1].set_xlabel("epoch")
        if group_name is not None:
            fig.suptitle(group_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    horizontal = (layout == "horizontal")

    first_val = next(iter(params.values()))
    if isinstance(first_val, dict):
        # nested dict
        for group, pdict in params.items():
            true_group = true_params.get(group, {}) if true_params else None
            label_group = label_params.get(group, {}) if label_params else None
            plot_group(group, pdict, horizontal=horizontal,
                       true_group=true_group, label_group=label_group)
    else:
        # flat dict
        plot_group(None, params, horizontal=horizontal,
                   true_group=true_params, label_group=label_params)

def is_nested(d: dict) -> bool:
    if not d:   # empty dict, decide what you want
        return False
    first_val = next(iter(d.values()))
    return isinstance(first_val, dict)


def build_axial_matrix_from_edge_coeffs(
    edge_df: pd.DataFrame,
    conds_v: Union[jnp.ndarray, jnp.ndarray],     # non-diffusion "v" channel, aligned row-wise with edge_df
    capacitance_uF_cm2: Union[jnp.ndarray, jnp.ndarray],  # (n_comp,) C_m in µF/cm^2
    surface_area_cm2: Union[jnp.ndarray, jnp.ndarray],    # (n_comp,) lateral area 2π a L in cm^2
) -> jnp.ndarray:
    """
    Build A such that I_mem_mA = A @ V_mV.
    Uses:
      - type 0 (comp -> comp): sink i gets c_i * (V_src - V_i)
      - type 1/2 (branchpoint -> comp): sink i gets c_i * (V_b - V_i), with V_b formed from type 3/4 weights
      - type 3/4 (comp -> branchpoint): used only to define the mixing weights alpha for V_b

    Units:
      conds_v (1/ms) = (G/A_sink)/C_m (from your compute_axial_conductances).
      c_i [mA/mV] = C_m[i] [µF/cm^2] * conds_v * A[i] [cm^2] * 1e-3
    """
    # Arrays
    k = jnp.asarray(conds_v)                # (n_edges,)
    Cm = jnp.asarray(capacitance_uF_cm2)    # (n_comp,)
    A  = jnp.asarray(surface_area_cm2)      # (n_comp,)

    # Prepare A matrix
    n_comp = surface_area_cm2.shape[0]
    A_mat = jnp.zeros((n_comp, n_comp), dtype=jnp.float64)

    # Pull columns
    src  = edge_df["source"].to_numpy()
    sink = edge_df["sink"].to_numpy()
    typ  = edge_df["type"].to_numpy()

    # -------- Build branchpoint weights alpha from type 3/4 (comp -> bp) --------
    # Collect raw weights w_{kb} per branchpoint b
    bp_weights = defaultdict(list)   # b -> [(k, w_kb), ...]
    for row, (s, t, tp) in enumerate(zip(src, sink, typ)):
        if tp in (3, 4):
            # comp -> branchpoint (as specified)
            k_comp = int(s)
            b_id   = int(t)
            assert k_comp < n_comp, "type 3/4 must have source = compartment"
            w = float(k[row])        # proportional to total conductance G_kb; global scale cancels
            bp_weights[b_id].append((k_comp, w))

    # Normalize to alphas: alpha_{kb} = w_{kb} / sum_m w_{mb}
    alpha_by_bp = {}
    for b, lst in bp_weights.items():
        comps = [c for (c, _) in lst]
        wvec  = jnp.array([w for (_, w) in lst], dtype=jnp.float64)
        Wsum  = wvec.sum()
        if Wsum == 0.0:
            # pathological; fall back to uniform weights
            alpha = {c: 1.0/len(comps) for c in comps}
        else:
            alpha = {c: float(w/Wsum) for c, w in zip(comps, wvec)}
        alpha_by_bp[b] = alpha

    # -------- Fill A from edges --------
    for row, (s, i, tp) in enumerate(zip(src, sink, typ)):
        if tp == 0:
            # comp -> comp (directed row contributes to sink compartment i)
            j = int(s); i = int(i)
            assert j < n_comp and i < n_comp, "type 0 must connect compartments"
            c = float(Cm[i] * k[row] * A[i] * 1e-3)  # mA/mV
            # row i: +c on column j, -c on column i
            A_mat = A_mat.at[i, j].add(c)
            A_mat = A_mat.at[i, i].add(-c)

        elif tp in (1, 2):
            # branchpoint -> comp: source=b, sink=i (comp ODE row)
            b = int(s); i = int(i)
            assert i < n_comp, "sink of type 1/2 must be a compartment"
            c = float(Cm[i] * k[row] * A[i] * 1e-3)  # mA/mV

            # Weighted sum V_b = sum_k alpha_{kb} V_k; distribute +c across those k
            alpha = alpha_by_bp.get(b, None)
            if alpha is None:
                # If no 3/4 edges were provided for this bp, treat as degree-1: V_b ≈ V_i -> zero contribution
                # (equivalently alpha_i=1 makes +c on col i and -c on col i cancel)
                continue

            # +c * alpha_k on each neighbor column k, and -c on the sink column i
            for k_comp, a in alpha.items():
                A_mat = A_mat.at[i, k_comp].add(c * a)
            A_mat = A_mat.at[i, i].add(-c)

        else:
            # type 3/4 are already used for alpha; no direct membrane term
            continue

    return A_mat

def build_axial_matrix(cell):
    params = cell.get_parameters() #axial resistivities are not trained here
    pstate = params_to_pstate(params, cell.indices_set_by_trainables)
    all_params = cell.get_all_parameters(pstate, voltage_solver="jaxley.stone")
    conds_v = all_params['axial_conductances']['v']
    edge_df = cell._comp_edges
    capacitance_uF_cm2 = cell.nodes['capacitance'].to_numpy()
    radii_CM = cell.nodes["radius"].to_numpy()*1e-4
    lengths_CM = cell.nodes["length"].to_numpy()*1e-4
    surface_area_cm2 = get_surface_areas(lengths_CM, radii_CM)
    A = build_axial_matrix_from_edge_coeffs(edge_df, conds_v, capacitance_uF_cm2, surface_area_cm2)
    return A

def transmembrane_current(
    A_mat: jnp.ndarray,
    V_mV: Union[jnp.ndarray, jnp.ndarray],               # (n_comp, T) or (n_comp,)
    Iinj_nA: Optional[Union[jnp.ndarray, jnp.ndarray]] = None,  # optional (n_comp, T) or single (T,) at one index
    inj_index: Optional[int] = None
) -> jnp.ndarray:
    
    """
    Compute I_mem_mA = A @ V_mV, optionally adding injected intracellular current (nA -> mA).

    - If Iinj_nA is (T,) and inj_index is provided, it is added at that compartment.
    - If Iinj_nA is (n_comp, T), it is added directly (nA -> mA).
    Returns shape (n_comp, T) or (n_comp,) if V was 1D.
    """
    V = jnp.asarray(V_mV)
    squeezed = False
    if V.ndim == 1:
        V = V[:, None]
        squeezed = True

    I_mem_mA = A_mat @ V  # (n_comp, T), mA

    if Iinj_nA is not None:
        inj = jnp.asarray(Iinj_nA)
        assert inj_index is not None, "Provide inj_index."
        I_mem_mA = I_mem_mA.at[inj_index].add(inj * 1e-6)  # nA -> mA

    return I_mem_mA[:, 0] if squeezed else I_mem_mA

def build_eap_M(
    A_mat: jnp.ndarray,                 # (n_comp, n_comp), maps mV -> mA
    distances_CM: jnp.ndarray,          # (n_elec, n_comp), > 0
    extracellular_resistivity: float,   # ρ [Ω·cm]
) -> jnp.ndarray:
    """
    Return M such that, absent injected current,  eap_mV = M @ voltages_mV.
    """
    scaling = extracellular_resistivity / (4.0 * jnp.pi)          # Ω·cm
    inv_r   = 1.0 / jnp.asarray(distances_CM)                     # 1/cm
    M       = scaling * (inv_r @ A_mat)                           # (n_elec, n_comp)
    return M


def compute_eap(
    voltages: jnp.ndarray,                 # (n_comp,) or (n_comp, T)
    M: jnp.ndarray,                        # (n_elec, n_comp)
    current=None,                          # None, scalar (), (T,), (n_comp,), or (n_comp, T)  [nA]
    idx_stimulus: int = None,
    distances_CM: jnp.ndarray = None,      # needed iff current is provided
    extracellular_resistivity: float = None,
) -> jnp.ndarray:
    V = jnp.asarray(voltages)
    eap = M @ V  # (n_elec,) or (n_elec, T)

    if current is None:
        return eap

    assert distances_CM is not None and extracellular_resistivity is not None, \
        "distances_CM and extracellular_resistivity are required when current is provided."

    n_elec, n_comp = distances_CM.shape
    J_nA = jnp.asarray(current)

    # Build J_mA with correct shape
    if J_nA.ndim == 0:
        # scalar at a single compartment
        assert idx_stimulus is not None, "Provide idx_stimulus for scalar current."
        J_mA = jnp.zeros((n_comp,), dtype=V.dtype)
        J_mA = J_mA.at[idx_stimulus].set(J_nA * 1e-6)  # nA -> mA

    elif J_nA.ndim == 1:
        if J_nA.shape[0] == n_comp:
            # per-compartment current at a single time
            J_mA = J_nA * 1e-6  # (n_comp,)
        else:
            # time series at a single compartment
            assert idx_stimulus is not None, "Provide idx_stimulus for (T,) current."
            T = J_nA.shape[0]
            J_mA = jnp.zeros((n_comp, T), dtype=V.dtype)
            J_mA = J_mA.at[idx_stimulus, :].set(J_nA * 1e-6)

    elif J_nA.ndim == 2:
        # per-compartment time series
        assert J_nA.shape[0] == n_comp, "current must have shape (n_comp, T) if 2D."
        J_mA = J_nA * 1e-6
    else:
        raise ValueError("current must be None, scalar, (T,), (n_comp,), or (n_comp, T).")

    scaling = extracellular_resistivity / (4.0 * jnp.pi)
    inv_r   = 1.0 / jnp.asarray(distances_CM)    # (n_elec, n_comp)
    b       = scaling * (inv_r @ J_mA)           # (n_elec,) or (n_elec, T)

    return eap + b
