"""
The three-transponder motif of `SNN_2input.asc`:

    SNSPD1 --> [A] --d_AC--\
                            >--> [C] --> readout
    SNSPD2 --> [B] --d_BC--/

A and B are the front-end transponders sitting at the two detector planes;
C is the coincidence/readout node.  Because each transponder is itself an
L/R integrator with tau_m ~ 37 ns, C fires only when A's and B's spikes
land inside its integration window -- and the arrival times at C are
(detector hit time) + (transponder latency) + (axon delay).

That is the whole time-of-flight discriminator in three nodes: sweep the
detector-hit separation dt and watch C's firing probability / latency.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from ntronpy import Circuit, TransponderConfig, add_transponder
from ntronpy.transponder import connect, stimulate


# A and B are *relays*: v1 = 1.85 puts one detector hit over threshold on its
# own (see scan_bias.py -- below 1.80 a single spike is sub-threshold, above
# 1.90 the channel self-switches).  C is left in *coincidence* mode at the
# stock v1 = 1.3, so it fires only if A and B land inside its L/R window.
RELAY = (1.85, 1.7)
COINC = (1.30, 1.7)


def build(dt_detect, d_AC=5e-9, d_BC=5e-9, biases=None, t0=10e-9):
    """dt_detect: arrival-time difference between the two detector hits."""
    biases = biases or {"A": RELAY, "B": RELAY, "C": COINC}
    cir = Circuit("net3")
    tps = {k: add_transponder(cir, k, TransponderConfig(v1=v[0], v2=v[1]))
           for k, v in biases.items()}
    stimulate(cir, tps["A"], [t0])
    stimulate(cir, tps["B"], [t0 + dt_detect])
    connect(cir, tps["A"], tps["C"], d_AC)
    connect(cir, tps["B"], tps["C"], d_BC)
    return cir, tps


def probes():
    p = []
    for k in "ABC":
        p += [f"I({k}_Lb1)", f"{k}_OUT", f"R({k}_U1.d)", f"R({k}_U2.d)"]
    return p


def spike_times(t, R_chan, thresh=1.0):
    """Rising edges of the output nTron's channel hotspot = output spikes."""
    on = R_chan > thresh
    idx = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    return t[idx]


def main():
    tstop = 200e-9
    print(f"{'dt_det':>8} {'A':>6} {'B':>6} {'C fires':>8} {'C latency':>11} "
          f"{'peak I_loop(C)':>15}")
    print("-" * 62)
    rows = []
    for dt_detect in [0, 5e-9, 10e-9, 15e-9, 20e-9, 25e-9, 30e-9, 40e-9, 60e-9]:
        t0 = time.time()
        cir, tps = build(dt_detect)
        r = cir.transient(tstop, record=probes(), record_dt=20e-12)
        t = r["t"]
        sA = spike_times(t, r["R(A_U1.d)"])
        sB = spike_times(t, r["R(B_U1.d)"])
        sC = spike_times(t, r["R(C_U1.d)"])
        lat = (sC[0] - 10e-9) * 1e9 if len(sC) else np.nan
        rows.append((dt_detect, len(sA), len(sB), len(sC), lat,
                     r["I(C_Lb1)"].max()))
        print(f"{dt_detect*1e9:7.1f}n {len(sA):6d} {len(sB):6d} {len(sC):8d} "
              f"{lat:10.2f}n {r['I(C_Lb1)'].max()*1e6:14.2f}uA"
              f"   [{time.time()-t0:.0f}s]", flush=True)

    np.save("/home/claude/ntron_py/figures/net3_scan.npy", np.array(rows))

    # keep one full trace for plotting / LIF fitting
    cir, tps = build(10e-9)
    r = cir.transient(tstop, record=probes(), record_dt=20e-12)
    np.savez("/home/claude/ntron_py/figures/net3_trace.npz",
             **{k: r[k] for k in ["t"] + probes()})
    print("\nsaved net3_trace.npz")


if __name__ == "__main__":
    main()
