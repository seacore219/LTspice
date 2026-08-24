"""Two input spikes into one transponder: EPSP summation + coincidence firing."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from ntronpy import Circuit, Pulse, TransponderConfig, add_transponder

def run(dt_in, v1=1.3, v2=1.7, tstop=140e-9, t0=10e-9):
    cfg = TransponderConfig(v1=v1, v2=v2)
    cir = Circuit(); tp = add_transponder(cir, "A", cfg)
    for i, ts in enumerate([t0, t0+dt_in]):
        p = Pulse(0, 0.25, ts, 2.9e-9, 2.9e-9, 0.1e-9, 1e9, 1)
        cir.V(f"s{i}", p); cir.R(f"s{i}", f"sa{i}", 10e3)
        cir.L(f"sa{i}", tp["gate_in"], 5e-9)
    pr = ["I(A_Lb1)","A_OUT","R(A_U2.d)","R(A_U1.d)","R(A_U1.g)","I(A_U1.d)"]
    r = cir.transient(tstop, record=pr, record_dt=20e-12)
    return r

for dt_in in [4e-9, 10e-9, 20e-9, 40e-9, 80e-9]:
    t0=time.time(); r = run(dt_in)
    fired = r["R(A_U1.d)"].max() > 1.0
    print("dt=%5.1f ns  peak I_loop=%6.2f uA  U1 chan R=%7.1f  V_OUT=%7.3f mV  fired=%s  (%.0fs)"
          % (dt_in*1e9, r["I(A_Lb1)"].max()*1e6, r["R(A_U1.d)"].max(),
             r["A_OUT"].max()*1e3, fired, time.time()-t0))
