"""Conductance-based LIF reduction: fit on sub-threshold traces, validate held-out."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from scipy.optimize import least_squares
from ntronpy import Circuit, TransponderConfig, add_transponder
from ntronpy.lif import cond_lif_trace, lif_trace
from ntronpy.transponder import stimulate
from run_network3 import spike_times

CACHE = {}
def run(spikes, tstop=220e-9, v1=1.3, v2=1.7):
    key = (spikes, v1, v2)
    if key in CACHE: return CACHE[key]
    cfg = TransponderConfig(v1=v1, v2=v2)
    cir = Circuit(); tp = add_transponder(cir, "A", cfg)
    stimulate(cir, tp, list(spikes))
    r = cir.transient(tstop, record=["I(A_Lb1)","R(A_U1.d)","R(A_U2.d)"], record_dt=20e-12)
    out = (r["t"], r["I(A_Lb1)"], spike_times(r["t"], r["R(A_U2.d)"]), r["R(A_U1.d)"].max() > 1)
    CACHE[key] = out; return out

train = [(10e-9,), (10e-9, 50e-9), (10e-9, 90e-9)]
data = [run(s)[:3] for s in train]

def resid(x):
    a, tau_m, tau_s, E = np.exp(x)
    out = []
    for t, I, sw in data:
        m = t >= sw[0] - 1e-9
        out.append((cond_lif_trace(t[m], sw, a, tau_m, tau_s, E) - I[m]) * 1e6)
    return np.concatenate(out)

x0 = np.log([2e7, 45e-9, 2e-9, 130e-6])
sol = least_squares(resid, x0, method="trf")
a, tau_m, tau_s, E = np.exp(sol.x)
print(f"a     = {a:.3e} 1/s")
print(f"tau_m = {tau_m*1e9:6.2f} ns   (L_loop/R_loop with U2's channel in the return path ~ 39.6 ns)")
print(f"tau_s = {tau_s*1e9:6.2f} ns")
print(f"E     = {E*1e6:6.1f} uA   <-- compare Ib1 = v1/10k = 130.0 uA")
print(f"train rms = {np.sqrt(np.mean(resid(sol.x)**2)):.3f} uA")

print("\n=== held-out sub-threshold summation ===")
print(f"{'dt':>7} {'circuit':>9} {'cond-LIF':>9} {'err':>8} {'current-LIF':>12} {'err':>8}")
w_cur, tm_cur, ts_cur = 10.51e-6, 50.77e-9, 1.90e-9   # from the current-based joint fit
for dt in [20e-9, 30e-9, 60e-9, 70e-9, 110e-9]:
    t, I, sw, fired = run((10e-9, 10e-9+dt))
    if fired: continue
    pc = cond_lif_trace(t, sw, a, tau_m, tau_s, E).max()
    pl = lif_trace(t, sw, w_cur, tm_cur, ts_cur).max()
    c = I.max()
    print(f"{dt*1e9:6.1f}n {c*1e6:8.2f}u {pc*1e6:8.2f}u {100*(pc-c)/c:+7.2f}% "
          f"{pl*1e6:11.2f}u {100*(pl-c)/c:+7.2f}%", flush=True)

np.savez("/home/claude/ntron_py/figures/lif_cond.npz", a=a, tau_m=tau_m, tau_s=tau_s, E=E,
         t=data[0][0], I=data[0][1], sw=data[0][2],
         fit=cond_lif_trace(data[0][0], data[0][2], a, tau_m, tau_s, E))
