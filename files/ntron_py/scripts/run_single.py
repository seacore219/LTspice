"""Single transponder: does it spike, and what does the loop current look like?"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ntronpy import Circuit, Pulse, TransponderConfig, add_transponder

cfg = TransponderConfig(v1=1.3, v2=1.7)
cir = Circuit("single")
tp = add_transponder(cir, "A", cfg)

# external stimulus exactly as in SNN_2input.asc:
# PULSE(0 0.25 1n 2.9n 2.9n 0.1n 10n 1) -> 5 ns tline -> 10k -> 5n -> gate
stim = Pulse(0, 0.25, 1e-9, 2.9e-9, 2.9e-9, 0.1e-9, 10e-9, 1, extra_delay=5e-9)
cir.V("stim", stim, name="Vin")
cir.R("stim", "stimA", 10e3, name="Rstim")
cir.L("stimA", tp["gate_in"], 5e-9, name="Lstim")

probes = ["I(A_Lb1)", "I(A_U1.g)", "I(A_U2.d)", "A_OUT", "A_bias1",
          "R(A_U2.d)", "R(A_U1.d)", "R(A_U1.g)"]
t0=time.time()
r = cir.transient(120e-9, record=probes, record_dt=10e-12, progress=True)
print(f"{r['n_steps']} steps in {time.time()-t0:.1f} s")
t = r["t"]*1e9
print("peak loop current  : %.2f uA  (threshold Isw_g = 15.20 uA)" % (r["I(A_Lb1)"].max()*1e6))
print("peak U1 gate current: %.2f uA" % (r["I(A_U1.g)"].max()*1e6))
print("peak OUT voltage   : %.3f mV" % (r["A_OUT"].max()*1e3))
print("peak R_hs(U2 chan) : %.1f ohm" % r["R(A_U2.d)"].max())
print("peak R_hs(U1 chan) : %.1f ohm" % r["R(A_U1.d)"].max())
print("peak R_hs(U1 gate) : %.1f ohm" % r["R(A_U1.g)"].max())
np.savez("/home/claude/ntron_py/figures/single.npz", **{k:r[k] for k in ["t"]+probes})
