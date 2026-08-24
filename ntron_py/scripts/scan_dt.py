"""TOF discrimination curve: sweep detector-hit separation, record C's response."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from run_network3 import build, probes, spike_times

dts = np.array([float(x) for x in sys.argv[1].split(",")]) * 1e-9
rows = []
for dt in dts:
    t0 = time.time()
    cir, tps = build(dt)
    r = cir.transient(160e-9, record=probes(), record_dt=20e-12)
    t = r["t"]
    sA, sB, sC = (spike_times(t, r[f"R({k}_U1.d)"]) for k in "ABC")
    lat = (sC[0] - 10e-9) if len(sC) else np.nan
    rows.append([dt, len(sA), len(sB), len(sC), lat, r["I(C_Lb1)"].max()])
    print(f"dt={dt*1e9:6.1f}ns  A={len(sA)} B={len(sB)} C={len(sC)}  "
          f"C_lat={lat*1e9 if len(sC) else float('nan'):7.2f}ns  "
          f"peakC={r['I(C_Lb1)'].max()*1e6:6.2f}uA  [{time.time()-t0:.0f}s]", flush=True)
np.save(f"/home/claude/ntron_py/figures/scan_dt_{sys.argv[2]}.npy", np.array(rows))
