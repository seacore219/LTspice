"""
`SNN_2input` built *literally* from the LTspice expanded netlist.

Element and node names are LTspice's, verbatim, so a probe means the same
thing in both tools:  `I(Lb1)` here is `I(Lb1)` there,  `I(R28)` is
`I(R28)`, `V(bias3)` is `V(bias3)`.  Sign conventions match too --
`I(Rxx)` is positive from the first node to the second as written below.

Reading the netlist fixed two things my visual read of the schematic got
wrong:

* **The pulser nTrons are `sq_g = 10` (input type), not `sq_g = 5`.**
  u1 and u4 have `Rnorm_g = 1300` in the expanded netlist, same as the
  transponder input nTrons u3/u6/u8; only the three *output* nTrons
  u2/u5/u7 have `Rnorm_g = 650`.  A 2x gate inductance and a 2x normal
  resistance is not a detail -- it changes the gate's switching dynamics.

* **A and B drive C from the shunt junction, not from the drain node.**
  `R32 N028 N017` and `R33 N033 N032`, where `N017`/`N032` sit *between*
  the 5 nH output inductor and the 3 ohm shunt resistor:

      bias4 --[L2 5n]-- N017 --[R1 3]-- 0        <- R32 taps N017
      bias2 --[L8 5n]-- N032 --[R15 3]-- 0       <- R33 taps N032

  The pulsers, by contrast, tap their *drain* nodes directly (`R28` and
  `R16` on `bias11`, `R29` and `R23` on `pulser2`).  So the pulser->
  transponder hop sees the raw hotspot voltage while the transponder->
  transponder hop sees it low-passed by 5 nH into 3 ohm.  Two different
  couplings in one circuit; worth knowing which one you are designing.

Everything else -- device parameters, loop values, the 148.206 nH Lb's,
R30 = 2 vs R2/R10 = 4, the bias assignments -- matched what I already had.
"""

from __future__ import annotations

import re

from .circuit import Circuit, DC, Delayed, Pulse
from .ntron import NTronParams

# ---------------------------------------------------------------------------
# The passive netlist, copied verbatim (values normalised to SI floats).
# ---------------------------------------------------------------------------
NETLIST = """
R1  N017 0      3
R2  bias3 N018  4
R3  N020 0      3
R4  N046 0      50
R5  N046 N047   10000
R6  N029 0      3
R7  N016 0      50
R8  N016 N023   10000
R9  N040 0      3
R10 bias1 N039  4
R11 N035 0      50
R12 N035 N036   10000
R13 N052 0      50
R14 N049 0      3
R15 N032 0      3
R16 N042 bias11 20
R17 N052 N050   10000
R18 N053 0      50
R19 N053 N051   10000
R20 N038 0      3
R21 N015 0      50
R22 N015 N022   10000
R23 N043 pulser2 20
R24 N004 0      50
R25 N004 N006   10000
R26 N003 0      50
R27 N003 N005   10000
R28 N019 bias11 20
R29 N025 pulser2 20
R30 bias5 N027  2
R31 N026 0      3
R32 N028 N017   20
R33 N033 N032   20
R34 N010 0      50
R35 N010 N014   10000
R36 N009 0      50
R37 N009 N013   10000
L1  bias3 N020  5e-9
L2  bias4 N017  5e-9
L3  N047 N048   5e-9
L4  bias5 N029  5e-9
L5  N023 pulser2 5e-9
L6  bias1 N040  5e-9
L7  bias11 N049 5e-9
L8  bias2 N032  5e-9
L9  N036 N037   5e-9
L10 N050 bias1  5e-9
L11 N051 bias2  5e-9
L12 pulser2 N038 5e-9
L13 N042 N044   5e-9
L14 N022 bias11 5e-9
L15 N006 bias4  5e-9
L16 N005 bias3  5e-9
L17 N043 N044   5e-9
L18 N019 N024   5e-9
L19 N025 N024   5e-9
L20 bias6 N026  5e-9
L21 N028 N031   5e-9
L22 N033 N031   5e-9
L23 N014 bias6  5e-9
L24 N013 bias5  5e-9
Lb1 N018 N021   1.48206e-7
Lb2 N039 N041   1.48206e-7
Lb3 N027 N030   1.48206e-7
"""

# transmission lines: name, source-side node, far-side node, delay
TLINES = [("T1", "N012", "N016"), ("T2", "N011", "N015"),
          ("T3", "N045", "N046"), ("T4", "N034", "N035"),
          ("T5", "N054", "N052"), ("T6", "N055", "N053"),
          ("T7", "N008", "N010"), ("T8", "N002", "N004"),
          ("T9", "N001", "N003"), ("T10", "N007", "N009")]
TD = 5e-9

# DC sources: name, node, value, and which bias net it ultimately feeds
DC_SOURCES = {
    "V2": ("N012", 1.3, "pulser2"),   # pulser2 drain bias
    "V3": ("N011", 1.3, "bias11"),    # pulser1 drain bias
    "V8": ("N001", 1.5, "bias3"),     # A input nTron
    "V5": ("N002", 1.6, "bias4"),     # A output nTron
    "V6": ("N054", 1.3, "bias1"),     # B input nTron
    "V7": ("N055", 1.7, "bias2"),     # B output nTron
    "V10": ("N007", 1.4, "bias5"),    # C input nTron
    "V9": ("N008", 1.4, "bias6"),     # C output nTron
}

# pulse sources: name, node, t_delay  (V4 first, V1 15.5 ns later)
PULSES = {"V4": ("N034", 1.0e-9), "V1": ("N045", 16.5e-9)}

# nTrons: name -> (gate, drain, source, sq_g)
NTRONS = {
    "u1": ("N048", "bias11", "0", 10),    # pulser 1
    "u4": ("N037", "pulser2", "0", 10),   # pulser 2
    "u3": ("N024", "bias3", "0", 10),     # A input
    "u2": ("N021", "bias4", "0", 5),      # A output
    "u6": ("N044", "bias1", "0", 10),     # B input
    "u5": ("N041", "bias2", "0", 5),      # B output
    "u8": ("N031", "bias5", "0", 10),     # C input
    "u7": ("N030", "bias6", "0", 5),      # C output
}

# human-readable roles, for the GUI
ROLE = {
    "u1": "pulser1", "u4": "pulser2",
    "u3": "A in", "u2": "A out",
    "u6": "B in", "u5": "B out",
    "u8": "C in", "u7": "C out",
}

# which slider drives which DC source
BIAS_SLIDERS = [
    ("V3", "pulser1 bias (bias11)"), ("V2", "pulser2 bias (pulser2)"),
    ("V8", "A  v1 in  (bias3)"), ("V5", "A  v2 out (bias4)"),
    ("V6", "B  v1 in  (bias1)"), ("V7", "B  v2 out (bias2)"),
    ("V10", "C  v1 in  (bias5)"), ("V9", "C  v2 out (bias6)"),
]


def _params(sq_g: int) -> NTronParams:
    return NTronParams(width_g=20e-9, width_s=200e-9, width_d=200e-9,
                       width_c=250e-9, sq_g=float(sq_g), sq_d=400.0,
                       sq_s=80.0, sq_c=10.0, thickness=19e-9, sheetRes=130.0,
                       Tc=8.5, Tsub=4.3, Jc=40e9, A1=0.4)


def build(dt_detect: float | None = None, biases: dict | None = None):
    """
    Build the circuit exactly as netlisted.

    dt_detect  overrides the 15.5 ns separation between the two detector
               hits (V4 fixed at 1 ns, V1 moved).
    biases     {source_name: volts}, e.g. {"V8": 1.55}
    """
    cir = Circuit("SNN_2input")

    for line in NETLIST.strip().splitlines():
        name, a, b, val = line.split()
        if name.startswith("R"):
            cir.R(a, b, float(val), name=name)
        else:
            cir.L(a, b, float(val), name=name)

    bv = {k: v[1] for k, v in DC_SOURCES.items()}
    if biases:
        bv.update(biases)

    src_at = {}
    for nm, (node, _v, _net) in DC_SOURCES.items():
        src_at[node] = DC(bv[nm])
    t4 = PULSES["V4"][1]
    t1 = t4 + (15.5e-9 if dt_detect is None else dt_detect)
    for nm, t0 in (("V4", t4), ("V1", t1)):
        src_at[PULSES[nm][0]] = Pulse(0, 0.25, t0, 2.9e-9, 2.9e-9,
                                      0.1e-9, 1e9, 1)

    # Each tline is driven by an ideal source and terminated in 50 ohm = Z0,
    # so it is a pure delay: put the delayed waveform on the far-side node.
    # DC sources are the exception -- a lossless line is a short at the
    # operating point, so a bias that is "on" at t=0 must stay on. Wrapping
    # it in the delay instead switches every bias on at t = Td, which throws
    # a >100 uA startup transient through the shunts and shifts the whole
    # trace by 5 ns.
    for nm, near, far in TLINES:
        src = src_at[near]
        cir.V(far, src if isinstance(src, DC) else Delayed(src, TD),
              name=f"{nm}_far")

    for nm, (g, d, s, sq_g) in NTRONS.items():
        cir.ntron(nm, g=g, d=d, s=s, params=_params(sq_g))

    return cir, dict(bias=bv, t_pulse=(t4, t1))


# ---------------------------------------------------------------------------
# Signal sets matching the LTspice screenshots
# ---------------------------------------------------------------------------
PRESETS = {
    "fan-in currents": ["I(R28)", "I(R29)", "I(R16)", "I(R23)"],
    "input shunts": ["I(L1)", "I(L6)", "I(R3)", "I(R9)"],
    "loop currents (EPSPs)": ["I(Lb1)", "I(Lb2)", "I(Lb3)"],
    "output shunts": ["I(R1)", "I(R15)", "I(R31)"],
    "A->C, B->C hops": ["I(R32)", "I(R33)", "I(L21)", "I(L22)"],
    "C stage": ["I(L4)", "I(R6)", "I(Lb3)", "I(R31)"],
    "bias node voltages": ["V(bias3)", "V(bias4)", "V(bias1)",
                           "V(bias2)", "V(bias5)", "V(bias6)"],
    "screenshot 3": ["I(L1)", "I(L6)", "I(R3)", "I(R9)"],
    "screenshot 4": ["I(L1)", "I(L6)", "I(R3)", "I(R9)", "I(Lb1)", "I(Lb2)",
                     "I(R15)", "I(R1)", "I(R32)", "I(R33)"],
    "screenshot 5": ["I(L1)", "I(L6)", "I(R3)", "I(R9)", "I(Lb1)", "I(Lb2)",
                     "I(R15)", "I(R1)", "I(R32)", "I(R33)", "I(L21)",
                     "I(L22)", "I(L4)", "I(R6)"],
}


def all_signals():
    """Every LTspice-named probe this build can produce."""
    sigs = []
    for line in NETLIST.strip().splitlines():
        sigs.append(f"I({line.split()[0]})")
    for nm in NTRONS:
        sigs += [f"I({nm}.g)", f"I({nm}.d)", f"R({nm}.d)", f"R({nm}.g)"]
    sigs += [f"V({n})" for n in
             ["bias1", "bias2", "bias3", "bias4", "bias5", "bias6",
              "bias11", "pulser2"]]
    return sigs
