import jaxley as jx
from eap_fit_hh.RGC_model.channels import HHf
from pathlib import Path

def RGC():
    here = Path(__file__).resolve().parent
    file = here / "RGC_morph_4.swc"
    cell = jx.read_swc(file, ncomp=1)
    cell.insert(HHf())
    ### Assign conductances
    ### Soma
    cell.soma.set('HHf_gLeak',0.0001)
    cell.soma.set('HHf_gNa',0.06)
    cell.soma.set('HHf_gK',0.035)
    ### Dendrite
    cell.basal.set('HHf_gLeak',0.0001)
    cell.basal.set('HHf_gNa',0.06)
    cell.basal.set('HHf_gK',0.035)
    ### general parameters
    cell.set("HHf_eNa",60.6)
    cell.set("HHf_eK",-101.34)
    cell.set("HHf_eLeak",-67.1469)
    cell.set('axial_resistivity', 143.2)
    cell.set("v",-65.02069782)
    cell.init_states()

    return cell