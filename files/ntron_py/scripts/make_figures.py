import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F="/home/claude/ntron_py/figures/"
plt.rcParams.update({"font.size":8,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
fig, ax = plt.subplots(2,2, figsize=(9.5,6.4))

# --- (a) single EPSP + LIF fit ---
d=np.load(F+"lif_cond.npz"); a0=ax[0,0]
t=d["t"]*1e9
a0.plot(t, d["I"]*1e6, lw=1.6, label="electrothermal circuit")
a0.plot(t, d["fit"]*1e6, "--", lw=1.4, label="reduced LIF kernel")
a0.axhline(15.2, color="crimson", ls=":", lw=1, label="threshold $I_{sw,g}$ = 15.2 µA")
a0.axvline(d["sw"][0]*1e9, color="0.5", lw=.8)
a0.set(xlabel="time (ns)", ylabel="loop current (µA)",
       title="(a) single EPSP: one input spike is sub-threshold")
a0.legend(fontsize=7, loc="upper right")

# --- (b) single-transponder coincidence window ---
a1=ax[0,1]
dt=np.array([4,10,20,40,60,70,80,110.]); pk=np.array([15.20,15.17,15.20,14.34,12.75,12.20,11.77,10.96])
fired=np.array([1,1,1,0,0,0,0,0],bool)
a1.plot(dt,pk,"o-",lw=1.4,ms=4,color="0.3")
a1.plot(dt[fired],pk[fired],"o",ms=7,color="crimson",label="fires")
a1.axhline(15.2,color="crimson",ls=":",lw=1)
a1.axvline(37.5,color="steelblue",ls="--",lw=1,label=r"$\tau_m=L/R$ = 37.5 ns")
a1.set(xlabel="input spike separation Δt (ns)", ylabel="peak loop current (µA)",
       title="(b) one transponder: coincidence window set by L/R")
a1.legend(fontsize=7)

# --- (c) 3-transponder TOF response ---
a2=ax[1,0]
D=np.array([0,2,5,8,10,12,15,18,22,26,30,35,45.])
P=np.array([9.22,9.10,14.12,15.03,15.20,15.20,15.20,15.13,14.73,14.28,13.84,13.32,12.43])
Fr=np.array([0,0,0,0,1,1,1,0,0,0,0,0,0],bool)
a2.plot(D,P,"o-",lw=1.4,ms=4,color="0.3")
a2.plot(D[Fr],P[Fr],"o",ms=8,color="crimson",label="C fires")
a2.axhline(15.2,color="crimson",ls=":",lw=1)
a2.axvspan(9,17,color="crimson",alpha=.10)
a2.annotate("1 input event\n(spikes merge)",(1,9.6),fontsize=7,color="0.35")
a2.annotate("2 input events",(24,14.9),fontsize=7,color="0.35")
a2.set(xlabel="detector hit separation Δt (ns)", ylabel="peak loop current in C (µA)",
       title="(c) 3-transponder network: band-pass TOF response")
a2.legend(fontsize=7, loc="lower right")

# --- (d) bias design space ---
a3=ax[1,1]
v1=np.array([1.30,1.50,1.70,1.80,1.85,1.90,2.00]); pk1=np.array([10.30,11.92,13.62,15.18,15.19,15.19,15.20])
a3.plot(v1,pk1,"o-",lw=1.4,ms=4,color="0.3")
a3.axhline(15.2,color="crimson",ls=":",lw=1,label="threshold")
a3.axvspan(1.30,1.79,color="steelblue",alpha=.12)
a3.axvspan(1.80,1.90,color="seagreen",alpha=.15)
a3.axvspan(1.90,2.05,color="crimson",alpha=.15)
a3.annotate("coincidence",(1.40,10.6),fontsize=7,color="steelblue")
a3.annotate("relay",(1.81,11.8),fontsize=7,color="seagreen",rotation=90)
a3.annotate("unstable\n($I_{b1}>I_{sw,c}$)",(1.91,10.4),fontsize=7,color="crimson")
a3.set(xlim=(1.28,2.03), xlabel="input bias $v_1$ (V)   [$I_{b1}=v_1/10\\,k\\Omega$]",
       ylabel="peak loop current, 1 spike (µA)",
       title="(d) bias design space: sensitivity vs stability")
a3.legend(fontsize=7, loc="lower right")

fig.tight_layout(); fig.savefig(F+"summary.png", dpi=170)
print("wrote", F+"summary.png")
