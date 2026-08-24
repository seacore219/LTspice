"""How far can the two bias currents push a single-spike response? (M1.1/M1.2)"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from ntronpy import Circuit, TransponderConfig, add_transponder
from ntronpy.transponder import stimulate

def probe(v1, v2, n_spikes=1, dt_in=10e-9, tstop=120e-9):
    cfg = TransponderConfig(v1=v1, v2=v2)
    cir = Circuit(); tp = add_transponder(cir, "A", cfg)
    stimulate(cir, tp, [10e-9 + i*dt_in for i in range(n_spikes)])
    r = cir.transient(tstop, record=["I(A_Lb1)","R(A_U1.d)","A_OUT","R(A_U2.d)"], record_dt=20e-12)
    # spontaneous switching = channel hot before the stimulus arrives
    pre = r["t"] < 9e-9
    unstable = (r["R(A_U2.d)"][pre].max() > 1) or (r["R(A_U1.d)"][pre].max() > 1)
    return dict(peak=r["I(A_Lb1)"].max(), fired=r["R(A_U1.d)"].max() > 1,
                vout=r["A_OUT"].max(), unstable=unstable)

print("Isw_c (channel critical current) = 190.0 uA -> v1 must stay below ~1.90 V\n")
print(f"{'v1':>5} {'Ib1':>8} {'1 spike peak':>13} {'fires?':>7} {'stable?':>8}")
print("-"*48)
for v1 in [1.3, 1.5, 1.7, 1.8, 1.85, 1.9, 2.0]:
    r = probe(v1, 1.7)
    print(f"{v1:5.2f} {v1/10e3*1e6:7.1f}u {r['peak']*1e6:12.2f}u "
          f"{str(r['fired']):>7} {str(not r['unstable']):>8}", flush=True)

print(f"\n{'v2':>5} {'Ib2':>8} {'2-spike Vout':>13} {'fires?':>7} {'stable?':>8}")
print("-"*48)
for v2 in [1.4, 1.6, 1.7, 1.8, 1.85, 1.9]:
    r = probe(1.3, v2, n_spikes=2)
    print(f"{v2:5.2f} {v2/10e3*1e6:7.1f}u {r['vout']*1e3:11.3f}mV "
          f"{str(r['fired']):>7} {str(not r['unstable']):>8}", flush=True)
