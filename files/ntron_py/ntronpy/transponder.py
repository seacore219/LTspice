"""
The superconducting transponder ("SNN_node"), transcribed 1:1 from
`SNN_node.lib` in the spiking_transponder_circuit repo.

Topology (LTspice net names kept verbatim so you can diff against the .lib)
--------------------------------------------------------------------------

    v1 --[5ns line]-- N009 --[R2 10k]-- N007 --[Lb_in 5n]-- bias1
    v2 --[5ns line]-- N010 --[R4 10k]-- N008 --[L2 5n]---- OUT

    IN --[R9 20]-- N005 --[L6 5n]-- N006 = gate(U2)
    U2: g=N006  d=bias1  s=0                (input nTron)

    bias1 --[L3 5n]-- N003 --[R5 R_sh]-- 0        (input shunt)
    bias1 --[R6 2*R_loop]-- N002 --[Lb1 ~148n]-- N004 = gate(U1)
    U1: g=N004  d=OUT    s=0                (output nTron)

    OUT --[L4 5n]-- N001 --[R7 R_sh]-- 0          (output shunt / readout)

How it spikes
-------------
1. A current pulse at IN drives U2's gate past Isw_g.  U2's channel Ic
   collapses (the `A1*exp(...)` gain term) and a hotspot appears across
   its drain-source channel.
2. The bias current Ib1 = v1/10k, which was flowing harmlessly through
   the superconducting U2, is now forced elsewhere: partly into the 3 ohm
   shunt, partly through R6 into the big loop inductor Lb1.
3. Lb1 (~148 nH) holds that current.  With R6 = 2*R_loop in series the
   loop decays as exp(-t R/L) -- **tau_m = L_loop / R_loop**.  This is
   the membrane potential.  It is an L/R leak, not an RC leak, which is
   exactly the point made on slide 7 of the kickoff deck.
4. The loop current *is* U1's gate current.  When it crosses Isw_g(U1)
   the output nTron fires, dumping the second bias current Ib2 = v2/10k
   into the output shunt -> a voltage spike on OUT.
5. U1's own gate hotspot damps the loop current -> reset + refractory.

Tunable parameters are exactly the two bias voltages v1, v2 (i.e. two
bias currents), with R and L "fixed", per the Aug-14 to-do list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import Circuit, DC
from .ntron import NTronParams

# --- .param block from SNN_node.lib (S1 = 1 => S = 1) ----------------------
S1 = 1.0
S = S1**2 / (0.6 + 0.4 * S1)

DEFAULTS = dict(
    width_g=20e-9 / S,
    width_s=200e-9 / S,
    width_d=200e-9 / S,
    width_c=250e-9 / S,
    sq_g=5 / S,
    sq_d=400 / S,
    sq_s=80 / S,
    sq_c=10 / S,
    thickness=19e-9,
    sheetRes=130.0,
    Tc=8.5,
    Tsub=4.3,
    Jc=40e9,
    A1=0.4,
)
R_SH = 3.0 / S          # .param R_sh={3/S}
R_LOOP = 2.0 / S        # .param R_loop={2/S}
SQ_S2 = 80.0
SQ_TOT_NT2 = 400.0 / S + 80.0
L_LOOP_NOMINAL = 150e-9  # .param L1={150n}
TLINE_DELAY = 5e-9       # every tline in the schematic is Td=5n


def _lb1_value(L1: float = L_LOOP_NOMINAL, sq_g: float = 5 / S,
               sq_s2: float = SQ_S2, sheetRes: float = 130.0,
               Tc: float = 8.5) -> float:
    """Lb1 = L1 - (sq_g+sq_s2)*1.38p*(sheetRes/Tc): the discrete inductor is
    trimmed by the kinetic inductance already present in U1's gate+source."""
    return L1 - (sq_g + sq_s2) * 1.38e-12 * (sheetRes / Tc)


@dataclass
class TransponderConfig:
    """Everything you'd want to sweep for one node."""

    v1: float = 1.3          # bias on the input nTron's drain  (-> Ib1 = v1/10k)
    v2: float = 1.7          # bias on the output nTron's drain (-> Ib2 = v2/10k)
    L_loop: float = L_LOOP_NOMINAL
    R_loop: float = R_LOOP
    R_sh: float = R_SH
    R_in: float = 20.0       # R9
    R_bias: float = 10e3     # R2 / R4
    tline_delay: float = TLINE_DELAY
    loop_mult: float = 2.0   # R2/R10 = {2.0*R_loop}; C's R30 = {R_loop}

    @property
    def Ib1(self) -> float:
        return self.v1 / self.R_bias

    @property
    def Ib2(self) -> float:
        return self.v2 / self.R_bias

    @property
    def tau_m(self) -> float:
        """L/R membrane time constant of the integrating loop."""
        return self.L_loop / (self.loop_mult * self.R_loop)


def ntron_params(role: str, **over) -> NTronParams:
    """
    role='input'  -> U2  (sq_g=10, sq_d=400, sq_s=80)
    role='output' -> U1  (sq_g=5,  sq_d=sq_tot_nt2-sq_s2=400, sq_s=80)
    """
    kw = dict(DEFAULTS)
    if role == "input":
        kw.update(sq_g=10.0, sq_d=400.0 / S, sq_s=80.0 / S, sq_c=10.0 / S)
    elif role == "output":
        kw.update(sq_g=5.0 / S, sq_d=SQ_TOT_NT2 - SQ_S2, sq_s=SQ_S2, sq_c=10.0 / S)
    else:
        raise ValueError(role)
    kw.update(over)
    return NTronParams(**kw)


def add_transponder(cir: Circuit, tag: str, cfg: TransponderConfig,
                    in_node: str | None = None) -> dict:
    """
    Instantiate one transponder into `cir`.

    Returns a dict of the interesting net / device names, prefixed by `tag`:
        IN, OUT, bias1, loop (the Lb1 inductor), u_in, u_out
    """
    n = lambda s: f"{tag}_{s}"
    IN = in_node if in_node is not None else n("IN")

    p_in = ntron_params("input")
    p_out = ntron_params("output")

    # --- bias branches (transmission line collapsed to a pure delay) ---
    cir.V(n("N009"), DC(cfg.v1), name=n("Vb1"))   # tline is a short at the .op
    cir.R(n("N009"), n("N007"), cfg.R_bias, name=n("R2"))
    cir.L(n("N007"), n("bias1"), 5e-9, name=n("Lb_in"))

    cir.V(n("N010"), DC(cfg.v2), name=n("Vb2"))
    cir.R(n("N010"), n("N008"), cfg.R_bias, name=n("R4"))
    cir.L(n("N008"), n("OUT"), 5e-9, name=n("L2"))

    # --- input path into U2's gate ---
    # N005 is the fan-in bus: every incoming axon lands here through its own
    # R9 = 20 ohm, so currents from several presynaptic transponders sum.
    cir.L(n("N005"), n("N006"), 5e-9, name=n("L6"))
    if in_node is not None:
        cir.R(in_node, n("N005"), cfg.R_in, name=n("R9"))

    # --- U2: input nTron.  drain sits on the bias1 node ---
    u_in = cir.ntron(n("U2"), g=n("N006"), d=n("bias1"), s="0", params=p_in)

    # --- shunt on bias1 ---
    cir.L(n("bias1"), n("N003"), 5e-9, name=n("L3"))
    cir.R(n("N003"), "0", cfg.R_sh, name=n("R5"))

    # --- the integrating loop: R6 + Lb1 into U1's gate ---
    cir.R(n("bias1"), n("N002"), cfg.loop_mult * cfg.R_loop, name=n("R6"))
    loop = cir.L(n("N002"), n("N004"),
                 _lb1_value(cfg.L_loop), name=n("Lb1"))

    # --- U1: output nTron.  drain sits on OUT ---
    u_out = cir.ntron(n("U1"), g=n("N004"), d=n("OUT"), s="0", params=p_out)

    # --- output shunt / readout ---
    cir.L(n("OUT"), n("N001"), 5e-9, name=n("L4"))
    cir.R(n("N001"), "0", cfg.R_sh, name=n("R7"))

    return dict(tag=tag, IN=IN, OUT=n("OUT"), bias1=n("bias1"),
                fanin=n("N005"), gate_in=n("N006"), gate_out=n("N004"),
                loop=n("Lb1"), u_in=n("U2"), u_out=n("U1"), cfg=cfg)


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------
def connect(cir: Circuit, src: dict, dst: dict, delay: float,
            R_in: float | None = None, gain: float = 1.0):
    """
    An axon from `src`'s OUT to `dst`'s fan-in bus, with propagation delay.

    Physically this is a NbN kinetic-inductance transmission line; the delay
    is set by the geometric path length, which is exactly the "weight" of a
    polychronous network -- fixed once the chip is laid out, and therefore
    the thing the *bias currents* have to be trained against.
    """
    R_in = dst["cfg"].R_in if R_in is None else R_in
    node = f"axon_{src['tag']}_{dst['tag']}"
    cir.link(src["OUT"], node, delay, gain=gain)
    cir.R(node, dst["fanin"], R_in, name=f"R9_{src['tag']}_{dst['tag']}")
    return node


def stimulate(cir: Circuit, dst: dict, times, amp: float = 0.25,
              R: float = 10e3, trise: float = 2.9e-9, ton: float = 0.1e-9,
              tag: str = "stim"):
    """
    External SNSPD-like spike(s) straight into the input nTron's gate,
    matching the pulse generators in SNN_2input.asc:
        PULSE(0 {0.25/S} t0 2.9n 2.9n 0.1n ...) -> 10k -> 5n -> gate
    0.25 V through 10k is ~25 uA, comfortably above Isw_g = 15.2 uA.
    """
    from .circuit import Pulse
    for i, ts in enumerate(np.atleast_1d(times)):
        nm = f"{tag}_{dst['tag']}_{i}"
        cir.V(nm, Pulse(0, amp, float(ts), trise, trise, ton, 1e9, 1), name=nm)
        cir.R(nm, nm + "a", R, name="R" + nm)
        cir.L(nm + "a", dst["gate_in"], 5e-9, name="L" + nm)
