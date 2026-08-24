"""Joint fit of the LIF kernel across 1- and 2-spike sub-threshold traces."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from scipy.optimize import least_squares
from ntronpy import Circuit, TransponderConfig, add_transponder, lif_trace
from ntronpy.transponder import stimulate
from run_network3 import spike_times

def run(spikes, tstop=220e-9):
    cfg = TransponderConfig(v1=1.3, v2=1.7)
    cir = Circuit(); tp = add_transponder(cir, "A", cfg)
    stimulate(cir, tp, list(spikes))
    r = cir.transient(tstop, record=["I(A_Lb1)","R(A_U1.d)","R(A_U2.d)"], record_dt=20e-12)
    return r, spike_times(r["t"], r["R(A_U2.d)"]), r["R(A_U1.d)"].max() > 1

# sub-threshold training set only: 1 spike, and pairs too far apart to fire
sets = [(10e-9,), (10e-9, 50e-9), (10e-9, 90e-9)]
data = []
for sp in sets:
    r, sw, fired = run(sp)
    print(f"  {len(sp)} spike(s) {[round(x*1e9,1) for x in sp]}: nTron switched at "
          f"{[round(x*1e9,2) for x in sw]} ns, peak {r['I(A_Lb1)'].max()*1e6:.2f} uA, fired={fired}")
    data.append((r["t"], r["I(A_Lb1)"], sw))

def resid(x):
    tau_m, tau_s, w = np.exp(x[0]), np.exp(x[1]), x[2]
    out = []
    for t, I, sw in data:
        m = t >= sw[0]
        out.append(lif_trace(t[m], sw, w, tau_m, tau_s) - I[m])
    return np.concatenate(out)

sol = least_squares(resid, [np.log(40e-9), np.log(3e-9), 10.7e-6], method="trf")
tau_m, tau_s, w = np.exp(sol.x[0]), np.exp(sol.x[1]), sol.x[2]
print(f"\ntau_m = {tau_m*1e9:6.2f} ns   tau_s = {tau_s*1e9:5.2f} ns   w = {w*1e6:6.2f} uA")

print("\n=== held-out check (never used in the fit) ===")
for sp in [(10e-9, 20e-9), (10e-9, 30e-9), (10e-9, 70e-9)]:
    r, sw, fired = run(sp)
    pred = lif_trace(r["t"], sw, w, tau_m, tau_s)
    circ = r["I(A_Lb1)"].max()
    tag = " (clamped by threshold)" if fired else ""
    print(f"  dt={((sp[1]-sp[0])*1e9):5.1f}ns  circuit={circ*1e6:6.2f}uA  "
          f"LIF={pred.max()*1e6:6.2f}uA  err={100*(pred.max()-circ)/circ:+6.2f}%  "
          f"fires: circuit={fired} LIF={pred.max()>=15.2e-6}{tag}", flush=True)

np.savez("/home/claude/ntron_py/figures/lif_joint.npz", tau_m=tau_m, tau_s=tau_s, w=w,
         t=data[0][0], I=data[0][1], sw=data[0][2],
         fit=lif_trace(data[0][0], data[0][2], w, tau_m, tau_s))
