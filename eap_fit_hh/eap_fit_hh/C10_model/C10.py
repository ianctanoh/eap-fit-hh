import jaxley as jx
import pandas as pd
import jax.numpy as jnp
from pathlib import Path
from tempfile import NamedTemporaryFile
from eap_fit_hh.C10_model.channels import HHc

from eap_fit_hh.utils import rotate_translate_positions

def extract_structure_and_positions_from_swc(swc_file):
    df = pd.read_csv(
        swc_file,
        delim_whitespace=True,
        comment='#',
        names=["id", "type", "x", "y", "z", "radius", "parent"]
    )
    df["id"]     = df["id"].astype(int)
    df["type"]   = df["type"].astype(int)
    df["parent"] = df["parent"].astype(int)
    df["radius"] = df["radius"].astype(float)
    structure_df = df[["id", "type", "radius", "parent"]]
    positions_array = df[["x", "y", "z"]].to_numpy()

    return structure_df, positions_array

def generate_custom_swc_from_positions(positions, structure_df, filename: str):
    """
    Generate an SWC file using custom (x, y, z) positions and a structure dataframe.

    Args:
        positions (ndarray): shape (N, 3), array of x, y, z coordinates.
        structure_df (pd.DataFrame): columns ["id", "type", "radius", "parent"],
                                     with dtypes int, int, float, int.
        filename (str): output SWC filename.

    Raises:
        ValueError: if shape mismatch between positions and structure_df.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"`positions` must have shape (N, 3), got {positions.shape}")
    if len(structure_df) != positions.shape[0]:
        raise ValueError(
            f"Row count mismatch: positions has {positions.shape[0]} rows, "
            f"structure_df has {len(structure_df)} rows."
        )

    with open(filename, "w") as f:
        f.write("# SWC format: n type x y z radius parent\n\n")
        for (nid, ntype, radius, parent), (x, y, z) in zip(
            structure_df[["id", "type", "radius", "parent"]].itertuples(index=False, name=None),
            positions
        ):
            f.write(f"{nid} {ntype} {x:.6f} {y:.6f} {z:.6f} {radius:.6f} {parent}\n")

class C10_model:

    def __init__(self):
        here = Path(__file__).resolve().parent
        self.swc_path = here / "base_pyramidal_C10.swc"
        self.structure_df, self.base_positions = extract_structure_and_positions_from_swc(self.swc_path)
        # Instantiate cell in base position
        self.base_cell = jx.read_swc(self.swc_path, ncomp=1)
        self._set_ncomp(self.base_cell)
        self._set_lengths(self.base_cell)
        self._assign_groups(self.base_cell)
        self._insert_channels(self.base_cell)
        self._set_group_params(self.base_cell)
        self._set_globals(self.base_cell)
        self.base_cell.init_states()

    # ---- public API ----
    def build(self,
              theta_x = jnp.pi/20, theta_y = jnp.pi/20, theta_z = jnp.pi/20,
              translation = jnp.array([-5., 20., -10.])):
        """Return a transformed cell given (theta_x, theta_y, theta_z, translation)."""
        pos = rotate_translate_positions(self.base_positions, theta_x, theta_y, theta_z, translation)
        cell = self._make_cell(pos)
        cell.init_states()
        self.cell = cell
        return cell

    # ---- internals ----
    def _make_cell(self, positions):
        with NamedTemporaryFile(suffix=".swc") as tmp:
            generate_custom_swc_from_positions(positions, self.structure_df, tmp.name)
            cell = jx.read_swc(tmp.name, ncomp=1)

        self._set_ncomp(cell)
        self._set_lengths(cell)
        self._assign_groups(cell)
        self._insert_channels(cell)
        self._set_group_params(cell)
        self._set_globals(cell)
        return cell

    @staticmethod
    def _set_ncomp(cell):
        cell.branch(2).set_ncomp(10)
        cell.branch(3).set_ncomp(7)
        cell.branch(4).set_ncomp(10)
        cell.branch(5).set_ncomp(13)
        cell.branch(6).set_ncomp(9)
        cell.branch(7).set_ncomp(9)

    @staticmethod
    def _set_lengths(cell):
        ###Axon 
        cell.branch(3).comp([0,1,2,3,4,5,6]).set("length",150/3.)

        ### Oriprox/Oridist
        cell.branch(2).comp([0,1,2]).set("length",100/3.)
        cell.branch(4).comp([0,1,2]).set("length",100/3.)

        cell.branch(2).comp([3,4,5,6,7,8,9]).set("length",200/7.)
        cell.branch(4).comp([3,4,5,6,7,8,9]).set("length",200/7.)

        ### Rad
        cell.branch(5).comp([0,1,2]).set("length",100/3.)
        cell.branch(5).comp([3,4,5]).set("length",100/3.)
        cell.branch(5).comp([6,7,8,9,10,11,12]).set("length",200/7.)

        ### LM
        cell.branch(6).comp([0,1,2]).set("length",100/3.)
        cell.branch(6).comp([3,4,5]).set("length",100/3.)
        cell.branch(6).comp([6,7,8]).set("length",50/3.)

        cell.branch(7).comp([0,1,2]).set("length",100/3.)
        cell.branch(7).comp([3,4,5]).set("length",100/3.)
        cell.branch(7).comp([6,7,8]).set("length",50/3.)

    @staticmethod
    def _assign_groups(cell):
        #Soma
        cell.branch(0).add_to_group("soma")
        cell.branch(1).add_to_group("soma")
        #Axon
        cell.branch(3).add_to_group("axon")
        #OriProx
        cell.branch(2).comp([0,1,2]).add_to_group("oriprox")
        cell.branch(4).comp([0,1,2]).add_to_group("oriprox")
        #Oridist
        cell.branch(2).comp([3,4,5,6,7,8,9]).add_to_group("oridist")
        cell.branch(4).comp([3,4,5,6,7,8,9]).add_to_group("oridist")
        #RadProx
        cell.branch(5).comp([0,1,2]).add_to_group("radprox")
        #RadMed
        cell.branch(5).comp([3,4,5]).add_to_group("radmed")
        #RadDist
        cell.branch(5).comp([6,7,8,9,10,11,12]).add_to_group("raddist")
        #LM
        cell.branch(6).add_to_group("lm")
        cell.branch(7).add_to_group("lm")

    @staticmethod
    def _insert_channels(cell):
        cell.soma.insert(HHc(location="soma",na_att=0.8))
        cell.axon.insert(HHc(location="axon",na_att=1.))
        cell.oriprox.insert(HHc(location="dendrite",na_att=1.))
        cell.oridist.insert(HHc(location="dendrite",na_att=1.))
        cell.radprox.insert(HHc(location="dendrite",na_att=0.5))
        cell.radmed.insert(HHc(location="dendrite",na_att=0.5))
        cell.raddist.insert(HHc(location="dendrite",na_att=0.5))
        cell.lm.insert(HHc(location="dendrite",na_att=0.5))

    @staticmethod
    def _set_group_params(cell):
        ### soma
        cell.soma.set("HHc_gNa",0.007)
        cell.soma.set("HHc_gK",0.007/5.)
        cell.soma.set("HHc_gLeak",0.0002)

        ### axon
        cell.axon.set("HHc_gNa",0.1)
        cell.axon.set("HHc_gK",0.02)
        cell.axon.set("HHc_gLeak",0.000005)


        ### oriprox
        cell.oriprox.set("HHc_gNa", 0.007)
        cell.oriprox.set("HHc_gK",  0.000868)
        cell.oriprox.set("HHc_gLeak", 0.000005)


        ### oridist
        cell.oridist.set("HHc_gNa", 0.007)
        cell.oridist.set("HHc_gK",  0.000868)
        cell.oridist.set("HHc_gLeak", 0.000005)


        ### radprox
        cell.radprox.set("HHc_gNa", 0.007)
        cell.radprox.set("HHc_gK",  0.000868)
        cell.radprox.set("HHc_gLeak", 0.000005)

        ### radmed
        cell.radmed.set("HHc_gNa", 0.007)
        cell.radmed.set("HHc_gK",  0.000868)
        cell.radmed.set("HHc_gLeak", 0.000005)


        ### raddist
        cell.raddist.set("HHc_gNa", 0.007)
        cell.raddist.set("HHc_gK",  0.000868)
        cell.raddist.set('HHc_gLeak', 0.000005)

        ### lm
        cell.lm.set("HHc_gNa", 0.007)
        cell.lm.set("HHc_gK",  0.000868)
        cell.lm.set("HHc_gLeak",  0.000005)

    @staticmethod
    def _set_globals(cell):
        cell.set('axial_resistivity', 150.)
        cell.set('HHc_eLeak', -123.)
        cell.set('HHc_eK',-80.)
        cell.set('HHc_eNa', 50.)