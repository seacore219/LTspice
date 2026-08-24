"""Single 3-transponder run at one detector separation -> full traces."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from run_network3 import build, probes, spike_times

dt_detect = float(sys.argv[1]) * 1e-9
tstop = float(sys.argv[2]) * 1e-9 if len(sys.argv) > 2 else 160e-9
t0 = time.time()
cir, tps = build(dt_detect)
r = cir.transient(tstop, record=probes(), record_dt=20e-12)
t = r["t"]
res = {}
for k in "ABC":
    s = spike_times(t, r[f"R({k}_U1.d)"])
    res[k] = s
    print(f"{k}: {len(s)} spike(s) at {np.round(s*1e9,2)} ns  "
          f"peak I_loop={r[f'I({k}_Lb1)'].max()*1e6:6.2f} uA  "
          f"Vout={r[f'{k}_OUT'].max()*1e3:6.3f} mV", flush=True)
np.savez(f"/home/claude/ntron_py/figures/net3_dt{sys.argv[1]}.npz",
         **{k: r[k] for k in ["t"]+probes()},
         **{f"spk_{k}": res[k] for k in "ABC"})
print(f"[{time.time()-t0:.0f}s, {r['n_steps']} steps]")
