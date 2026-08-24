"""
The real `SNN_2input.asc` topology, read off the printed schematic.

                 bias11 (1.3)                bias3/4 (1.5/1.6)
                     |                              |
  V1 pulse --[T3]--> U1  --> pulser1 --+--[20,5n]--> [ A ] --+
   (16.5 ns)                           |                     |   bias5/6 (1.4/1.4)
                                       X  (both pulsers      +--[20,5n]--> [ C ] --> out
  V4 pulse --[T4]--> U4  --> pulser2 --+   feed both          |
   (1.0 ns)             |                  transponders)     |
                 bias12 (1.3)                bias1/2 (1.3/1.7)
                                              |
                                          [ B ] ------------+

Six corrections relative to my first pass:

1. **The pulse generators do not drive the transponders directly.**  Two
   single-nTron *pulser* stages (U1, U4) sit in between, biased at 1.3 V.
   A pulser is just the output half of a transponder: gate driven by the
   voltage pulse, drain biased through 10k, drain node shunted by 5n+R_sh
   and tapped as the output.  These are the SNSPD front ends.

2. **Both pulsers feed both mid transponders.**  A sees pulser1 *and*
   pulser2 (R28/R29); B sees the same pair (R16/R23).  So A and B are two
   coincidence detectors watching the *same* spike pair with *different*
   biases -- an opponent-channel code for dt, not two independent relays.
   This kills the "relay front end" workaround I used before; it was never
   what the schematic does.

3. **Fan-in has a per-axon inductor.**  Each incoming line is its own
   20 ohm *and* its own 5 nH, joining only at the gate node.  I previously
   had both axons share one 5 nH after a common bus, which under-isolates
   the branches.

4. **C's loop resistor is R30 = {R_loop}, not {2.0*R_loop}.**  So C
   integrates with tau_m = 148.2 nH / 2 ohm = 74 ns, twice A and B's 37 ns.
   C is deliberately the slow, wide-window node.

5. **Bias values are all different**: A = (1.5, 1.6), B = (1.3, 1.7),
   C = (1.4, 1.4), pulsers = 1.3.

6. **There are no axon delay lines between transponders** -- they are
   direct wires.  Every `tline` in the schematic is on a bias feed or a
   pulse input.  Inter-node delay here is device latency only; the
   variable-delay interconnects of M2.1 are still to be added.
"""

from __future__ import annotations

import numpy as np

from .circuit import Circuit, DC, Pulse
from .transponder import (TransponderConfig, add_transponder, ntron_params,
                          R_SH, TLINE_DELAY)

# schematic bias values
BIAS = {
    "pulser1": 1.3,   # bias11 (V3)
    "pulser2": 1.3,   # bias12 (V2)
    "A": (1.5, 1.6),  # bias3 (V8), bias4 (V5)
    "B": (1.3, 1.7),  # bias1 (V6), bias2 (V7)
    "C": (1.4, 1.4),  # bias5 (V10), bias6 (V9)
}
T_PULSE = {"pulser2": 1.0e-9, "pulser1": 16.5e-9}   # V4 at 1n, V1 at 16.5n


def add_pulser(cir: Circuit, tag: str, t_spike: float, vbias: float = 1.3,
               amp: float = 0.25, R_sh: float = R_SH, R_bias: float = 10e3):
    """
    One SNSPD front end: a single nTron whose drain node is the output.

        Vpulse --[tline 5n]-- 50R -- 10k -- 5n -- gate
        Vbias  --[tline 5n]-- 50R -- 10k -- 5n -- drain(= OUT)
        OUT    --[5n]--[R_sh]-- gnd
    """
    n = lambda s: f"{tag}_{s}"
    OUT = n("out")

    cir.V(n("vp"), Pulse(0, amp, t_spike, 2.9e-9, 2.9e-9, 0.1e-9, 1e9, 1,
                         extra_delay=TLINE_DELAY), name=n("Vp"))
    cir.R(n("vp"), "0", 50.0, name=n("Rterm"))
    cir.R(n("vp"), n("g1"), R_bias, name=n("Rg"))
    cir.L(n("g1"), n("gate"), 5e-9, name=n("Lg"))

    cir.V(n("vb"), DC(vbias), name=n("Vb"))
    cir.R(n("vb"), n("b1"), R_bias, name=n("Rb"))
    cir.L(n("b1"), OUT, 5e-9, name=n("Lb"))

    cir.L(OUT, n("sh"), 5e-9, name=n("Lsh"))
    cir.R(n("sh"), "0", R_sh, name=n("Rsh"))

    cir.ntron(n("U"), g=n("gate"), d=OUT, s="0", params=ntron_params("output"))
    return dict(tag=tag, OUT=OUT, gate=n("gate"), u=n("U"))


def fan_in(cir: Circuit, dst: dict, sources, R_in: float = 20.0,
           L_in: float = 5e-9, delays=None):
    """
    Wire a list of source nodes into `dst`'s input nTron gate, each through
    its own R_in + L_in (as R28/L18 and R29/L19 do in the schematic).

    `delays` optionally inserts a propagation delay per axon -- not present
    in SNN_2input, but this is where the M2.1 variable-delay interconnects
    go.
    """
    for k, src in enumerate(sources):
        node = src
        if delays is not None and delays[k] > 0:
            node = f"{dst['tag']}_ax{k}"
            cir.link(src, node, delays[k])
        mid = f"{dst['tag']}_fan{k}"
        cir.R(node, mid, R_in, name=f"{dst['tag']}_Rin{k}")
        cir.L(mid, dst["gate_in"], L_in, name=f"{dst['tag']}_Lin{k}")


def build_snn_2input(dt_detect: float | None = None, biases: dict | None = None,
                     delays_AC: float = 0.0, delays_BC: float = 0.0,
                     loop_mult_C: float = 1.0):
    """
    Full 8-nTron network.  `dt_detect` overrides the schematic's 15.5 ns
    separation between the two detector hits (V4 at 1 ns, V1 at 16.5 ns).
    """
    b = dict(BIAS)
    if biases:
        b.update(biases)

    t2 = T_PULSE["pulser2"]
    t1 = t2 + (15.5e-9 if dt_detect is None else dt_detect)

    cir = Circuit("SNN_2input")
    p2 = add_pulser(cir, "P2", t2, b["pulser2"])
    p1 = add_pulser(cir, "P1", t1, b["pulser1"])

    tps = {}
    for tag in ("A", "B", "C"):
        v1, v2 = b[tag]
        cfg = TransponderConfig(v1=v1, v2=v2)
        if tag == "C":
            cfg.loop_mult = loop_mult_C          # R30 = {R_loop}
        tps[tag] = add_transponder(cir, tag, cfg)

    fan_in(cir, tps["A"], [p1["OUT"], p2["OUT"]])
    fan_in(cir, tps["B"], [p1["OUT"], p2["OUT"]])
    fan_in(cir, tps["C"], [tps["A"]["OUT"], tps["B"]["OUT"]],
           delays=[delays_AC, delays_BC] if (delays_AC or delays_BC) else None)

    return cir, dict(P1=p1, P2=p2, **tps), (t2, t1)


def net_probes():
    p = ["P1_out", "P2_out", "I(P1_U.g)", "I(P2_U.g)", "R(P1_U.d)", "R(P2_U.d)"]
    for k in "ABC":
        p += [f"I({k}_Lb1)", f"{k}_OUT", f"R({k}_U1.d)", f"R({k}_U2.d)",
              f"I({k}_U2.g)"]
    return p


def edges(t, sig, thresh=1.0):
    on = np.asarray(sig) > thresh
    return np.asarray(t)[np.flatnonzero(on[1:] & ~on[:-1]) + 1]
