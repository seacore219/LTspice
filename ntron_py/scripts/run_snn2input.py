"""Run the corrected 8-nTron SNN_2input network over a range of detector dt."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np
from ntronpy import build_snn_2input, net_probes, edges

dts = [float(x)*1e-9 for x in sys.argv[1].split(",")]
tag = sys.argv[2]; rows=[]
for dt in dts:
    t0=time.time()
    cir, nodes, ts = build_snn_2input(dt_detect=dt)
    r = cir.transient(200e-9, record=net_probes(), record_dt=20e-12)
    t = r["t"]
    sp = {k: edges(t, r[f"R({k}_U1.d)" if k in "ABC" else f"R({k}_U.d)"])
          for k in ["P1","P2","A","B","C"]}
    pk = {k: r[f"I({k}_Lb1)"].max() for k in "ABC"}
    ev = {k: len(edges(t, r[f"R({k}_U2.d)"])) for k in "ABC"}
    rows.append([dt, len(sp["P1"]), len(sp["P2"]), len(sp["A"]), len(sp["B"]),
                 len(sp["C"]), pk["A"], pk["B"], pk["C"], ev["A"], ev["B"], ev["C"]])
    print(f"dt={dt*1e9:6.1f}ns | P1={len(sp['P1'])} P2={len(sp['P2'])} | "
          f"A={len(sp['A'])}({ev['A']}ev,{pk['A']*1e6:5.2f}uA) "
          f"B={len(sp['B'])}({ev['B']}ev,{pk['B']*1e6:5.2f}uA) "
          f"C={len(sp['C'])}({ev['C']}ev,{pk['C']*1e6:5.2f}uA) [{time.time()-t0:.0f}s]", flush=True)
    if abs(dt-15.5e-9) < 1e-12:
        np.savez("/home/claude/ntron_py/figures/snn2input_trace.npz",
                 **{k: r[k] for k in ["t"]+net_probes()})
np.save(f"/home/claude/ntron_py/figures/snn2_{tag}.npy", np.array(rows))
