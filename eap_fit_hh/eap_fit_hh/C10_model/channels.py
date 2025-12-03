import jax.numpy as jnp
from typing import Dict, Optional
from jax.lax import select

from jaxley.channels import Channel
from jaxley.solver_gate import solve_inf_gate_exponential
from jaxley.pumps import Pump

# --- Helper Functions ---
R = 8.315  # J/degC
F = 96480.0  # Coulombs
T = 273.16  # K

def Q(temp_c):
    return F/(R * (T + temp_c))

def T_adj(temp_c):
    return 2.3 ** ((temp_c - 23.0) / 10.0)

def KTF(temp_c):
    return (25.0 / 293.15) * (temp_c + 273.15)

def efun(z):
        return jnp.where(jnp.abs(z) < 1e-4, 1.0 - z / 2.0, z / (jnp.exp(z) - 1.0))

class HHc(Channel):
    def __init__(self,
                name: Optional[str] = None,
                na_att: float = 1.0,
                temp_c: float = 34.0,
                location: str = "dend"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = self._name
        self.channel_params = {
            f"{prefix}_gNa": 0.0,
            f"{prefix}_gK": 0.0,
            f"{prefix}_gLeak": 0.0,
            f"{prefix}_eNa": 60.0,
            f"{prefix}_eK": -77.0,
            f"{prefix}_eLeak": -70.0,
        }
        self.channel_states = {
            f"{prefix}_m": 0.0,
            f"{prefix}_h": 1.0,
            f"{prefix}_s": 1.0,
            f"{prefix}_n": 0.0,
        }
        self.current_name = "i_HHc"
        self.na_att = na_att
        self.temp_c = temp_c
        self.location = location

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        prefix = self._name
        m, h, s = states[f"{prefix}_m"], states[f"{prefix}_h"], states[f"{prefix}_s"]
        n = states[f"{prefix}_n"]

        soma_or_axon = self.location in ["soma", "axon"]

        # --- Gating dynamics (Na) ---
        if soma_or_axon:
            m_inf = 1.0 / (1.0 + jnp.exp((v + 44.0) / -3.0))
            h_inf = 1.0 / (1.0 + jnp.exp((v + 49.0) / 3.5))
            tau_h = 1.0
            tau_n = 3.5
            n_inf = 1.0 / (1.0 + jnp.exp((v + 46.3) / -3.0))
        else:
            m_inf = 1.0 / (1.0 + jnp.exp((v + 40.0) / -3.0))
            h_inf = 1.0 / (1.0 + jnp.exp((v + 45.0) / 3.0))
            tau_h = 0.5
            tau_n = 2.2
            n_inf = 1.0 / (1.0 + jnp.exp((v + 42.0) / -2.0))

        tau_m = 0.05

        # --- Sodium attenuation gate "s" (same for both locations) ---
        vhalfr = -60.0
        vvs = 2.0
        taumin = 3.0
        ar2 = self.na_att
        celsius = self.temp_c

        s_inf = (1 + ar2 * jnp.exp((v - vhalfr) / vvs)) / (1 + jnp.exp((v - vhalfr) / vvs))

        zetar = 12.0
        gmr = 0.2
        a0r = 0.0003
        b0r = 0.0003

        q_val = Q(celsius)
        alpr = jnp.exp(1e-3 * zetar * (v - vhalfr) * q_val)
        betr = jnp.exp(1e-3 * zetar * gmr * (v - vhalfr) * q_val)
        tau_s = betr / (a0r + b0r * alpr)
        tau_s = jnp.maximum(tau_s, taumin)

        # --- Update states ---
        m = solve_inf_gate_exponential(m, dt, m_inf, tau_m)
        h = solve_inf_gate_exponential(h, dt, h_inf, tau_h)
        s = solve_inf_gate_exponential(s, dt, s_inf, tau_s)
        n = solve_inf_gate_exponential(n, dt, n_inf, tau_n)

        return {f"{prefix}_m": m, f"{prefix}_h": h, f"{prefix}_s": s, f"{prefix}_n": n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        prefix = self._name
        gNa, gK, gL = params[f"{prefix}_gNa"], params[f"{prefix}_gK"], params[f"{prefix}_gLeak"]
        eNa, eK, eL = params[f"{prefix}_eNa"], params[f"{prefix}_eK"], params[f"{prefix}_eLeak"]
        m, h, s = states[f"{prefix}_m"], states[f"{prefix}_h"], states[f"{prefix}_s"]
        n = states[f"{prefix}_n"]

        i_Na = gNa * m**2 * h * s * (v - eNa)
        i_K = gK * n**2 * (v - eK)
        i_L = gL * (v - eL)

        return i_Na + i_K + i_L

    def init_state(self, states, v, params, delta_t):
        
        prefix = self._name
        soma_or_axon = self.location in ["soma", "axon"]

        if soma_or_axon:
            m_inf = 1.0 / (1.0 + jnp.exp((v + 44.0) / -3.0))
            h_inf = 1.0 / (1.0 + jnp.exp((v + 49.0) / 3.5))
            n_inf = 1.0 / (1.0 + jnp.exp((v + 46.3) / -3.0))
        else:
            m_inf = 1.0 / (1.0 + jnp.exp((v + 40.0) / -3.0))
            h_inf = 1.0 / (1.0 + jnp.exp((v + 45.0) / 3.0))
            n_inf = 1.0 / (1.0 + jnp.exp((v + 42.0) / -2.0))

        vhalfr = -60.0
        vvs = 2.0
        s_inf = (1 + self.na_att * jnp.exp((v - vhalfr) / vvs)) / (1 + jnp.exp((v - vhalfr) / vvs))

        return {f"{prefix}_m": m_inf, f"{prefix}_h": h_inf, f"{prefix}_s": s_inf, f"{prefix}_n": n_inf}


class KA(Channel):
    def __init__(self, name: Optional[str] = None, location: str = "proximal"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gKA": 0.007 if location == "proximal" else 0.018,  # mho/cm²
            "eK": -77.0  # K+ reversal potential
        }
        self.channel_states = {
            "KA_n": 0.0,
            "KA_l": 1.0,
        }
        self.current_name = "i_KA"
        self.location = location

        # Shared inactivation gate parameters
        self.vhalfl = -56.0
        self.tau_l_params = {
            "offset": -20,
            "base": 5.0,
            "scale": 2.6 / 10.0,
        }

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        n, l = states["KA_n"], states["KA_l"]

        # Disjoin dynamics based on location
        if self.location == "proximal":
            a_n = -0.01 * (v + 21.3) / (jnp.exp((v + 21.3) / -35.0) - 1.0)
            b_n =  0.01 * (v + 21.3) / (jnp.exp((v + 21.3) / 35.0) - 1.0)
        else:  # distal
            a_n = -0.01 * (v + 34.4) / (jnp.exp((v + 34.4) / -21.0) - 1.0)
            b_n =  0.01 * (v + 34.4) / (jnp.exp((v + 34.4) / 21.0) - 1.0)

        n_inf = a_n / (a_n + b_n)
        tau_n = 0.2  # same in both versions
        n = solve_inf_gate_exponential(n, dt, n_inf, tau_n)

        # Inactivation gate is the same in both versions
        a_l = -0.01 * (v + 58.0) / (jnp.exp((v + 58.0) / 8.2) - 1.0)
        b_l =  0.01 * (v + 58.0) / (jnp.exp((v + 58.0) / -8.2) - 1.0)
        l_inf = a_l / (a_l + b_l)

        tau_l = jnp.where(v > -20, 5.0 + 2.6 * (v + 20.0) / 10.0, 5.0)
        l = solve_inf_gate_exponential(l, dt, l_inf, tau_l)

        return {"KA_n": n, "KA_l": l}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        gKA = params["gKA"]
        eK = params["eK"]
        n, l = states["KA_n"], states["KA_l"]
        gka = gKA * n**4 * l
        return gka * (v - eK)

    def init_state(self, states, v, params, delta_t):
        if self.location == "proximal":
            a_n = -0.01 * (v + 21.3) / (jnp.exp((v + 21.3) / -35.0) - 1.0)
            b_n =  0.01 * (v + 21.3) / (jnp.exp((v + 21.3) / 35.0) - 1.0)
        else:
            a_n = -0.01 * (v + 34.4) / (jnp.exp((v + 34.4) / -21.0) - 1.0)
            b_n =  0.01 * (v + 34.4) / (jnp.exp((v + 34.4) / 21.0) - 1.0)

        n_inf = a_n / (a_n + b_n)

        a_l = -0.01 * (v + 58.0) / (jnp.exp((v + 58.0) / 8.2) - 1.0)
        b_l =  0.01 * (v + 58.0) / (jnp.exp((v + 58.0) / -8.2) - 1.0)
        l_inf = a_l / (a_l + b_l)

        return {"KA_n": n_inf, "KA_l": l_inf}

    
class h(Channel):
    def __init__(self, name: Optional[str] = None, temp_c: float = 34.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gh": 1e-4,     # default 0.0001 mho/cm²
            "eh": -30.0,       # reversal potential (adjust as needed)
            "vhalfl_h": -81.
        }
        self.channel_states = {
            "h_l": 0.0,
        }
        self.current_name = "i_h"
        self.temp_c = temp_c

        # Fixed parameters from MOD file
        self.vhalft = -75.0
        self.a0t = 0.011
        self.zetat = 2.2
        self.gmt = 0.4
        self.q10 = 4.5
        self.qtl = 1.0
        self.kl = -8.0

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        l = states["h_l"]
        vhalfl = params["vhalfl_h"]
        qt = self.q10 ** ((self.temp_c - 33.0) / 10.0)
        a = jnp.exp(0.0378 * self.zetat * (v - self.vhalft))
        linf = 1.0 / (1.0 + jnp.exp(-(v - vhalfl) / self.kl))
        taul = jnp.exp(0.0378 * self.zetat * self.gmt * (v - self.vhalft)) / (self.qtl * qt * self.a0t * (1 + a))

        l = solve_inf_gate_exponential(l, dt, linf, taul)
        return {"h_l": l}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        ghdbar = params["gh"]
        ehd = params["eh"]
        l = states["h_l"]
        ghd = ghdbar * l
        return ghd * (v - ehd)

    def init_state(self, states, v, params, delta_t):
        vhalfl = params["vhalfl_h"]
        linf = 1.0 / (1.0 + jnp.exp(-(v - vhalfl) / self.kl))
        return {"Ih_l": linf}
    

class KM(Channel):
    def __init__(self, name: Optional[str] = None, temp_c: float = 34.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gKM": 10.0,     # pS/um² (0.03 mho/cm²)
            "eK": -77.0       # reversal potential for K+
        }
        self.channel_states = {
            "KM_n": 0.0,
        }
        self.current_name = "i_KM"
        self.temp_c = temp_c

        # MOD file constants
        self.Ra = 0.001
        self.Rb = 0.001
        self.tha = -30.0
        self.qa = 9.0
        self.temp = 23.0
        self.q10 = 2.3

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        n = states["KM_n"]
        Ra, Rb, tha, qa = self.Ra, self.Rb, self.tha, self.qa

        a = Ra * (v - tha) / (1 - jnp.exp(-(v - tha) / qa))
        b = -Rb * (v - tha) / (1 - jnp.exp((v - tha) / qa))

        ntau = 1.0 / (a + b)
        ninf = a * ntau

        tadj = self.q10 ** ((self.temp_c - self.temp) / 10.0)
        n = solve_inf_gate_exponential(n, dt, ninf, ntau / tadj)

        return {"KM_n": n}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        gbar = params["gKM"]
        eK = params["eK"]
        n = states["KM_n"]

        gk = gbar * n  # pS/um²
        return 1e-4 * gk * (v - eK)  # convert to mA/cm²

    def init_state(self, states, v, params, delta_t):
        Ra, Rb, tha, qa = self.Ra, self.Rb, self.tha, self.qa

        a = Ra * (v - tha) / (1 - jnp.exp(-(v - tha) / qa))
        b = -Rb * (v - tha) / (1 - jnp.exp((v - tha) / qa))
        ntau = 1.0 / (a + b)
        ninf = a * ntau

        return {"KM_n": ninf}

# High-voltage activated R-type calcium current (ICaR)
class CaR(Channel):
    def __init__(self, name: Optional[str] = None, temp_c: float = 34.0, location: str = "dend"):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gCaR": 0.0,    # mho/cm²
            "eCa": 140.0    # mV
        }
        self.channel_states = {
            "CaR_m": 0.0,
            "CaR_h": 1.0,
        }
        self.current_name = "i_Ca"
        self.temp_c = temp_c
        self.location = location

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        m, h = states["CaR_m"], states["CaR_h"]

        if self.location == "soma":
            m_inf = 1.0 / (1.0 + jnp.exp((v + 60.0) / -3.0))
            h_inf = 1.0 / (1.0 + jnp.exp((v + 62.0) / 1.0))
            tau_m = 100.0
            tau_h = 5.0
        else:  # distal
            m_inf = 1.0 / (1.0 + jnp.exp((v + 48.5) / -3.0))
            h_inf = 1.0 / (1.0 + jnp.exp((v + 53.0) / 1.0))
            tau_m = 50.0
            tau_h = 5.0

        m = solve_inf_gate_exponential(m, dt, m_inf, tau_m)
        h = solve_inf_gate_exponential(h, dt, h_inf, tau_h)

        return {"CaR_m": m, "CaR_h": h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        gCaR = params["gCaR"]
        eCa = params["eCa"]
        m = states["CaR_m"]
        h = states["CaR_h"]
        return gCaR * m**3 * h * (v - eCa)

    def init_state(self, states, v, params, delta_t):
        if self.location == "soma":
            m_inf = 1.0 / (1.0 + jnp.exp((v + 60.0) / -3.0))
            h_inf = 1.0 / (1.0 + jnp.exp((v + 62.0) / 1.0))
        else:  # distal
            m_inf = 1.0 / (1.0 + jnp.exp((v + 48.5) / -3.0))
            h_inf = 1.0 / (1.0 + jnp.exp((v + 53.0) / 1.0))

        return {"CaR_m": m_inf, "CaR_h": h_inf}
    

class CaPump(Pump):
    """Calcium dynamics tracking inside calcium concentration."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.channel_params = {
            f"{self._name}_gamma": 1/18.,  # Fraction of free calcium (not buffered).
            f"{self._name}_decay": 7*200.,  # Buffering time constant in ms.
            f"{self._name}_depth": 0.1,  # Depth of shell in um.
            f"{self._name}_minCai": 1e-4,  # Minimum intracell. ca concentration in mM.
        }
        self.channel_states = {"i_Ca": 1e-8, "Cai": 1e-4}
        self.ion_name = "Cai"
        self.current_name = "i_CaPump"

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        """Update states if necessary (but this pump has no states to update)."""
        return {"Cai": states["Cai"], "i_Ca": states["i_Ca"]}

    def compute_current(self, states, modified_state, params):
        """Return change of calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = states["i_Ca"]
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485  # Coulombs per mole.

        # Calculate the contribution of calcium currents to cai change.
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)
        drive_channel = select(
            drive_channel <= 0, jnp.zeros_like(drive_channel), drive_channel
        )
        state_decay = (modified_state - minCai) / decay
        diff = drive_channel - state_decay
        return -diff

    def init_state(self, states, v, params, delta_t):
        """Initialize states of channel."""
        return {}
    
class CaL(Channel):
    def __init__(self, name: Optional[str] = None, temp_c: float = 34.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gCaL": 0.0,     # mho/cm²
            "cao": 2.0,      # mM (external calcium)
            "tfa": 5.0,      # time factor scaling
            "ki": 0.001,     # mM (for h2())
        }
        self.channel_states = {
            "CaL_m": 0.0,
            "Cai": 5e-5,     # mM (internal calcium concentration)
        }
        self.current_name = "i_Ca"
        self.temp_c = temp_c

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        m = states["CaL_m"]
        a = self.alpm(v)
        b = self.betm(v)
        taum = 1.0 / (params["tfa"] * (a + b))
        minf = a / (a + b)

        m = solve_inf_gate_exponential(m, dt, minf, taum)
        return {"CaL_m": m}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        gCaL = params["gCaL"]
        cao = params["cao"]
        ki = params["ki"]
        m = states["CaL_m"]
        cai = states["Cai"]

        gcal = gCaL * m * self.h2(cai, ki)
        return gcal * self.ghk(v, cai, cao)

    def init_state(self, states, v, params, delta_t):
        a = self.alpm(v)
        b = self.betm(v)
        minf = a / (a + b)
        return {"CaL_m": minf}

    def h2(self, cai, ki):
        return ki / (ki + cai)

    def ghk(self, v, ci, co):
        f = KTF(self.temp_c) / 2
        nu = v / f
        return -f * (1.0 - (ci / co) * jnp.exp(nu)) * efun(nu)

    def alpm(self, v):
        return 0.055 * (-27.01 - v) / (jnp.exp((-27.01 - v) / 3.8) - 1.0)

    def betm(self, v):
        return 0.94 * jnp.exp((-63.01 - v) / 17.0)
    
class CaLH(Channel):
    def __init__(self, name: Optional[str] = None, temp_c: float = 34.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gCaLH": 0.0,   # mho/cm²
            "eCa": 140.0    # mV
        }
        self.channel_states = {
            "CaLH_m": 0.0,
            "CaLH_h": 1.0,
        }
        self.current_name = "i_Ca"
        self.temp_c = temp_c

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        m = states["CaLH_m"]
        h = states["CaLH_h"]

        m_inf = 1.0 / (1.0 + jnp.exp((v + 37.0) / -1.0))
        h_inf = 1.0 / (1.0 + jnp.exp((v + 41.0) / 0.5))

        tau_m = 3.6    # ms
        tau_h = 29.0   # ms

        m = solve_inf_gate_exponential(m, dt, m_inf, tau_m)
        h = solve_inf_gate_exponential(h, dt, h_inf, tau_h)

        return {"CaLH_m": m, "CaLH_h": h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        gCaLH = params["gCaLH"]
        eCa = params["eCa"]
        m = states["CaLH_m"]
        h = states["CaLH_h"]

        return gCaLH * m**3 * h * (v - eCa)

    def init_state(self, states, v, params, delta_t):
        m_inf = 1.0 / (1.0 + jnp.exp((v + 37.0) / -1.0))
        h_inf = 1.0 / (1.0 + jnp.exp((v + 41.0) / 0.5))
        return {"CaLH_m": m_inf, "CaLH_h": h_inf}
    
class CaT(Channel):
    def __init__(self, name: Optional[str] = None, temp_c: float = 22.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gCaT": 0.0,       # mho/cm²
            "cao": 2.0,        # mM
            "ki": 0.001,       # mM
            "tfa": 1.0,        # activation time constant scaling
            "tfi": 0.68,       # inactivation time constant scaling
        }
        self.channel_states = {
            "CaT_m": 0.0,
            "CaT_h": 1.0,
            "Cai": 5e-5,       # mM
        }
        self.current_name = "i_Ca"
        self.temp_c = temp_c
        self.tBase = 23.5     # original reference temperature for Q10

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        m = states["CaT_m"]
        h = states["CaT_h"]

        a_m = self.alpm(v)
        b_m = self.betm(v)
        taum = 1.0 / (params["tfa"] * (a_m + b_m))
        minf = a_m / (a_m + b_m)

        a_h = self.alph(v)
        b_h = self.beth(v)
        tauh = 1.0 / (params["tfi"] * (a_h + b_h))
        hinf = a_h / (a_h + b_h)

        m = solve_inf_gate_exponential(m, dt, minf, taum)
        h = solve_inf_gate_exponential(h, dt, hinf, tauh)

        return {"CaT_m": m, "CaT_h": h}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        gCaT = params["gCaT"]
        cao = params["cao"]
        ki = params["ki"]
        cai = states["Cai"]
        m = states["CaT_m"]
        h = states["CaT_h"]

        gcat = gCaT * m**2 * h * self.h2(cai, ki)
        return gcat * self.ghk(v, cai, cao)

    def init_state(self, states, v, params, delta_t):
        a_m = self.alpm(v)
        b_m = self.betm(v)
        minf = a_m / (a_m + b_m)

        a_h = self.alph(v)
        b_h = self.beth(v)
        hinf = a_h / (a_h + b_h)

        return {"CaT_m": minf, "CaT_h": hinf}

    def h2(self, cai, ki):
        return ki / (ki + cai)

    def ghk(self, v, ci, co):
        f = KTF(self.temp_c) / 2
        nu = v / f
        return -f * (1.0 - (ci / co) * jnp.exp(nu)) * efun(nu)

    def alpm(self, v):
        return 0.1967 * (-v + 19.88) / (jnp.exp((-v + 19.88) / 10.0) - 1.0)

    def betm(self, v):
        return 0.046 * jnp.exp(-v / 22.73)

    def alph(self, v):
        return 1.6e-4 * jnp.exp(-(v + 57.0) / 19.0)

    def beth(self, v):
        return 1.0 / (jnp.exp((-v + 15.0) / 10.0) + 1.0)
                    
class KCa(Channel):
    def __init__(self, name: Optional[str] = None, temp_c: float = 36.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gKCa": 0.01,      # mho/cm²
            "eK": -80.0,       # mV
            "beta": 0.03,      # 1/ms
            "cac": 0.025,      # mM
            "taumin": 0.5,     # ms
        }
        self.channel_states = {
            "IKCa_m": 0.0,
            "Cai": 2.4e-5,     # mM (intracellular calcium)
        }
        self.current_name = "i_KCa"
        self.temp_c = temp_c

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        m = states["IKCa_m"]
        cai = states["Cai"]

        car = (cai / params["cac"]) ** 2
        m_inf = car / (1.0 + car)

        tadj = 3.0 ** ((self.temp_c - 22.0) / 10.0)
        tau_m = 1.0 / (params["beta"] * (1.0 + car) * tadj)
        tau_m = jnp.maximum(tau_m, params["taumin"])

        m = solve_inf_gate_exponential(m, dt, m_inf, tau_m)
        return {"IKCa_m": m}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        gbar = params["gKCa"]
        ek = params["eK"]
        m = states["IKCa_m"]
        gk = gbar * m**3
        return gk * (v - ek)

    def init_state(self, states, v, params, delta_t):
        cai = states["Cai"]
        car = (cai / params["cac"]) ** 2
        m_inf = car / (1.0 + car)
        return {"IKCa_m": m_inf}
                    
class MyKCa(Channel):
    def __init__(self, name: Optional[str] = None, temp_c: float = 20.0):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        self.channel_params = {
            "gmKCa": 0.01,  # mho/cm²
            "eK": -80.0,    # mV
            "abar": 0.48,   # /ms
            "bbar": 0.28,   # /ms
            "d1": 0.84,
            "d2": 1.0,
            "k1": 0.18,     # mM
            "k2": 0.011,    # mM
        }
        self.channel_states = {
            "o": 0.0,
            "Cai": 1e-3,  # mM
        }
        self.current_name = "i_mKCa"
        self.temp_c = temp_c

        # Constants
        self.FARADAY = 96485.0      # C/mol
        self.R = 8.313424           # J/(mol·K)

    def update_states(
        self,
        states: Dict[str, jnp.ndarray],
        dt,
        v,
        params: Dict[str, jnp.ndarray],
    ):
        o = states["o"]
        cai = states["Cai"]

        a = self.alp(v, cai, params)
        b = self.bet(v, cai, params)
        tau = 1.0 / (a + b)
        oinf = a * tau

        o = solve_inf_gate_exponential(o, dt, oinf, tau)
        return {"o": o}

    def compute_current(
        self, states: Dict[str, jnp.ndarray], v, params: Dict[str, jnp.ndarray]
    ):
        gkbar = params["gmKCa"]
        ek = params["eK"]
        o = states["o"]
        return gkbar * o * (v - ek)

    def init_state(self, states, v, params, delta_t):
        cai = states["Cai"]
        a = self.alp(v, cai, params)
        b = self.bet(v, cai, params)
        tau = 1.0 / (a + b)
        oinf = a * tau
        return {"o": oinf}

    def alp(self, v, ca, p):
        return p["abar"] / (1.0 + self.exp1(p["k1"], p["d1"], v) / ca)

    def bet(self, v, ca, p):
        return p["bbar"] / (1.0 + ca / self.exp1(p["k2"], p["d2"], v))

    def exp1(self, k, d, v):
        T = self.temp_c + 273.15  # K
        return k * jnp.exp(-2 * d * self.FARADAY * v / (self.R * T))