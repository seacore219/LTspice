"""Does the missing S02 switch on the source-segment hotspot explain the latch pattern?"""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from ntronpy.snn2_netlist import build
pr = ['I(R3)','I(R9)','I(R6)','I(R1)','I(R15)','I(R31)','I(Lb1)','I(Lb2)','I(Lb3)']
for flag in (True, False):
    cir, meta = build()
    for x in cir.ntrons:
        x.dev.reset_segment_hotspots = flag
    t0=time.time(); r = cir.transient(90e-9, record=pr, record_dt=20e-12)
    print(f"reset_segment_hotspots={flag}  [{time.time()-t0:.0f}s]")
    for k in pr:
        print(f"   {k:<9s} max {r[k].max()*1e6:8.2f}  final {r[k][-1]*1e6:8.2f} uA"
              f"  {'<-- LATCHED' if abs(r[k][-1])>20e-6 else ''}")
    print(flush=True)
