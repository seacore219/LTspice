import sys, os; sys.path.insert(0,'..'); import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
dt=np.array([0,5,10,15.5,20,30,45.])
pk={"A":[10.82,15.20,15.19,15.19,15.20,15.19,14.89],
    "B":[ 9.11,14.05,15.20,15.20,15.02,13.92,12.53],
    "C":[ 0.00,11.24,15.20,15.20,11.24,11.24, 0.00]}
fire={"A":[0,1,1,1,1,1,0],"B":[0,0,1,1,0,0,0],"C":[0,0,1,1,0,0,0]}
col={"A":"#1f77b4","B":"#d62728","C":"#2ca02c"}
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
for k in "ABC":
    y=np.array(pk[k]); f=np.array(fire[k],bool)
    ax[0].plot(dt,y,"o-",color=col[k],lw=1.4,ms=4,label=f"{k}")
    ax[0].plot(dt[f],y[f],"o",color=col[k],ms=9,mfc="none",mew=1.8)
ax[0].axhline(15.2,color="crimson",ls=":",lw=1)
ax[0].axvline(15.5,color="0.5",ls="--",lw=1)
ax[0].annotate("schematic\noperating point\nΔt = 15.5 ns",(15.9,3),fontsize=7,color="0.4")
ax[0].set(xlabel="detector separation Δt (ns)",ylabel="peak loop current (µA)",
          title="corrected 8-nTron network: peak EPSP\n(open rings = node fires)")
ax[0].legend(fontsize=7)
for i,k in enumerate("ABC"):
    f=np.array(fire[k],bool)
    ax[1].scatter(dt[f],[i]*f.sum(),s=90,color=col[k],marker="s")
    ax[1].scatter(dt[~f],[i]*(~f).sum(),s=40,facecolor="none",edgecolor="0.7")
ax[1].axvline(15.5,color="0.5",ls="--",lw=1)
ax[1].set(yticks=[0,1,2],yticklabels=["A (1.5/1.6)","B (1.3/1.7)","C (1.4/1.4)"],
          xlabel="detector separation Δt (ns)", ylim=(-.6,2.6),
          title="firing map: A is the broad channel,\nB the narrow one, C = A ∧ B")
fig.tight_layout(); fig.savefig("/home/claude/ntron_py/figures/snn2_response.png",dpi=170)
print("ok")
