"""Why does C prefer a non-zero dt?  Count the input nTron's switching events."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from run_network3 import build, spike_times

for dt in [float(x)*1e-9 for x in sys.argv[1].split(",")]:
    t0=time.time(); cir, tps = build(dt)
    pr = ["I(C_Lb1)","R(C_U2.d)","R(C_U1.d)","I(C_U2.g)","C_N005"]
    r = cir.transient(160e-9, record=pr, record_dt=20e-12)
    nsw = len(spike_times(r["t"], r["R(C_U2.d)"]))
    print(f"dt={dt*1e9:5.1f}ns  C_U2 switch events={nsw}  "
          f"peak I_gate(C_U2)={r['I(C_U2.g)'].max()*1e6:6.2f}uA  "
          f"peak V_fanin={r['C_N005'].max()*1e3:6.3f}mV  "
          f"peak I_loop(C)={r['I(C_Lb1)'].max()*1e6:6.2f}uA  "
          f"C fires={r['R(C_U1.d)'].max()>1}  [{time.time()-t0:.0f}s]", flush=True)
