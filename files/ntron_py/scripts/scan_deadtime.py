"""Input-nTron dead time: how close can two input spikes be and still count as two?"""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from ntronpy import Circuit, TransponderConfig, add_transponder
from ntronpy.transponder import stimulate
from run_network3 import spike_times

rows=[]
for dt in [float(x)*1e-9 for x in sys.argv[1].split(",")]:
    t0=time.time()
    cir = Circuit(); tp = add_transponder(cir, "A", TransponderConfig(v1=1.3, v2=1.7))
    stimulate(cir, tp, [10e-9, 10e-9+dt])
    r = cir.transient(150e-9, record=["I(A_Lb1)","R(A_U2.d)","R(A_U1.d)"], record_dt=20e-12)
    n = len(spike_times(r["t"], r["R(A_U2.d)"]))
    rows.append([dt, n, r["I(A_Lb1)"].max(), float(r["R(A_U1.d)"].max()>1)])
    print(f"dt={dt*1e9:5.1f}ns  U2 events={n}  peak={r['I(A_Lb1)'].max()*1e6:6.2f}uA  "
          f"fires={r['R(A_U1.d)'].max()>1}  [{time.time()-t0:.0f}s]", flush=True)
np.save(f"/home/claude/ntron_py/figures/deadtime_{sys.argv[2]}.npy", np.array(rows))
