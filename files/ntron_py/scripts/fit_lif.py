"""Collapse the electrothermal transponder onto a LIF unit and fit its kernel."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from ntronpy import Circuit, TransponderConfig, add_transponder, fit_lif_to_trace, lif_trace
from ntronpy.transponder import stimulate

def epsp(v1=1.3, v2=1.7, spikes=(10e-9,), tstop=200e-9):
    cfg = TransponderConfig(v1=v1, v2=v2)
    cir = Circuit(); tp = add_transponder(cir, "A", cfg)
    stimulate(cir, tp, list(spikes))
    return cir.transient(tstop, record=["I(A_Lb1)","R(A_U1.d)","R(A_U2.d)"], record_dt=20e-12), cfg

# --- single sub-threshold EPSP: this is what the kernel must reproduce ---
r, cfg = epsp()
t, I = r["t"], r["I(A_Lb1)"]
# the input nTron switches ~2.9 ns after the pulse starts (gate ramp); use the
# measured switching instant as the true presynaptic spike time
from run_network3 import spike_times
t_sw = spike_times(t, r["R(A_U2.d)"])
print("input nTron switched at %.2f ns" % (t_sw[0]*1e9))
fit = fit_lif_to_trace(t, I, t_sw[:1], tau_m0=cfg.tau_m, tau_s0=4e-9,
                       t_window=(t_sw[0], 200e-9))
print("\n=== single-EPSP fit ===")
print("tau_m = %6.2f ns   (L/R prediction = %5.2f ns)" % (fit["tau_m"]*1e9, cfg.tau_m*1e9))
print("tau_s = %6.2f ns" % (fit["tau_s"]*1e9))
print("w     = %6.2f uA  (peak loop current per input spike)" % (fit["w"]*1e6))
print("rms   = %6.3f uA  (%.2f%% of peak)" % (fit["rms"]*1e6, 100*fit["rel_rms"]))
np.savez("/home/claude/ntron_py/figures/lif_fit.npz", t=t, I=I, fit=fit["fit"],
         tau_m=fit["tau_m"], tau_s=fit["tau_s"], w=fit["w"], t_sw=t_sw[:1])

# --- predict the two-spike response with NO refitting ---
print("\n=== two-spike prediction (parameters frozen) ===")
for dt_in in [10e-9, 20e-9, 40e-9, 80e-9]:
    r2, _ = epsp(spikes=(10e-9, 10e-9+dt_in))
    sw = spike_times(r2["t"], r2["R(A_U2.d)"])
    pred = lif_trace(r2["t"], sw, fit["w"], fit["tau_m"], fit["tau_s"])
    fired = r2["R(A_U1.d)"].max() > 1
    print("dt=%5.1fns  circuit peak=%6.2fuA  LIF peak=%6.2fuA  "
          "err=%+5.2f%%  circuit fires=%-5s  LIF fires=%s"
          % (dt_in*1e9, r2["I(A_Lb1)"].max()*1e6, pred.max()*1e6,
             100*(pred.max()-r2["I(A_Lb1)"].max())/r2["I(A_Lb1)"].max(),
             fired, pred.max() >= 15.2e-6), flush=True)
