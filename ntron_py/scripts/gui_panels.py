#!/usr/bin/env python3
"""
Interactive explorer for the SNN_2input 8-nTron network.

    python3 scripts/gui.py

Four stacked panels, top to bottom:

  1. PULSERS      gate current arriving at each SNSPD front-end nTron,
                  against its switching current -- "the spike arrives"
  2. GATES        gate current at each transponder's *input* nTron, against
                  Isw_g -- "the gate opens", and how many times it opens
  3. EPSPs        loop current in A, B, C against the firing threshold.
                  This is the panel that matters: watch two EPSPs sum when
                  the hits are close and fail to sum when they are not.
  4. OUTPUTS      output-node voltages -- "the nTron fires downstream"

Sliders set the detector separation and all eight bias voltages live.
Press Run to solve the full electrothermal circuit (10-150 s depending on
how much switching happens).  Hold freezes the current traces as a grey
ghost so you can compare two bias settings directly.

To match LTspice: in LTspice do File > Export data as text on the same
signals, then press "Load LTspice" and pick the file.  Any column whose
name matches one being plotted is overlaid as a dashed line.

Nothing here is real-time -- the electrothermal solve genuinely takes
seconds.  Sliders are instant; the solve is not.  The progress bar in the
title updates as it goes and Run doubles as Abort while a solve is live.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

from ntronpy import build_snn_2input, net_probes, edges, BIAS
from ntronpy.transponder import ntron_params

ISW_G = ntron_params("output").Isw_g          # 15.2 uA -- gate switching current
ISW_C = ntron_params("output").Isw_c          # 190 uA -- channel critical current

COL = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c",
       "P1": "#7f4fc9", "P2": "#e08214"}


# ---------------------------------------------------------------------------
# LTspice "Export data as text" reader
# ---------------------------------------------------------------------------
def load_ltspice(path):
    """
    Tolerant reader for LTspice text exports (tab- or space-separated,
    first column time).  Returns {column_name: array} plus 'time'.
    """
    with open(path, "r", errors="ignore") as f:
        header = f.readline().strip()
    names = [c.strip() for c in header.replace(",", "\t").split("\t") if c.strip()]
    data = np.genfromtxt(path, skip_header=1, delimiter=None, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    out = {}
    for i, nm in enumerate(names[: data.shape[1]]):
        out[nm] = data[:, i]
    if names:
        out["time"] = data[:, 0]
    return out


def ltspice_key_matches(name, probe):
    """Loose match between an LTspice column name and one of our probes."""
    a = name.lower().replace(" ", "")
    b = probe.lower().replace(" ", "")
    return a == b or a.strip("v()") == b.strip("v()") or b in a


# ---------------------------------------------------------------------------
class Explorer:
    def __init__(self):
        self.fig = plt.figure(figsize=(15.0, 9.0))
        self.fig.canvas.manager.set_window_title("SNN_2input explorer")
        gs = self.fig.add_gridspec(4, 2, width_ratios=[3.15, 1.0],
                                   hspace=0.30, wspace=0.18,
                                   left=0.065, right=0.985, top=0.93, bottom=0.06)
        self.ax = [self.fig.add_subplot(gs[i, 0]) for i in range(4)]
        self.ctrl = self.fig.add_subplot(gs[:, 1]); self.ctrl.axis("off")

        self.ghost = None
        self.lt = None
        self.result = None
        self.worker = None
        self.abort = False
        self._build_controls()
        self._style_axes()
        self.fig.suptitle("press Run to solve", fontsize=11)

    # -- controls ---------------------------------------------------------
    def _build_controls(self):
        specs = [
            ("dt", "detector Δt (ns)", 0.0, 60.0, 15.5),
            ("P1", "pulser1 bias (V)", 1.0, 1.85, BIAS["pulser1"]),
            ("P2", "pulser2 bias (V)", 1.0, 1.85, BIAS["pulser2"]),
            ("Av1", "A  v1 in-bias (V)", 1.0, 1.90, BIAS["A"][0]),
            ("Av2", "A  v2 out-bias (V)", 1.0, 1.90, BIAS["A"][1]),
            ("Bv1", "B  v1 in-bias (V)", 1.0, 1.90, BIAS["B"][0]),
            ("Bv2", "B  v2 out-bias (V)", 1.0, 1.90, BIAS["B"][1]),
            ("Cv1", "C  v1 in-bias (V)", 1.0, 1.90, BIAS["C"][0]),
            ("Cv2", "C  v2 out-bias (V)", 1.0, 1.90, BIAS["C"][1]),
            ("tstop", "sim length (ns)", 60.0, 400.0, 100.0),
        ]
        self.sl = {}
        x0, w, y0, dy = 0.795, 0.165, 0.885, 0.049
        for i, (key, lab, lo, hi, init) in enumerate(specs):
            ax = self.fig.add_axes([x0, y0 - i * dy, w, 0.020])
            s = Slider(ax, "", lo, hi, valinit=init,
                       valfmt="%.2f", color="#9ecae1")
            ax.set_title(f"{lab} = {init:.2f}", fontsize=8, pad=3, loc="left")
            s.on_changed(lambda v, k=key, l=lab: self._label(k, l, v))
            self.sl[key] = s

        yb = y0 - len(specs) * dy - 0.035
        self.b_run = Button(self.fig.add_axes([x0, yb, 0.078, 0.033]), "Run",
                            color="#c7e9c0")
        self.b_run.on_clicked(self.on_run)
        self.b_hold = Button(self.fig.add_axes([x0 + 0.087, yb, 0.078, 0.033]),
                             "Hold", color="#e5e5e5")
        self.b_hold.on_clicked(self.on_hold)

        self.b_lt = Button(self.fig.add_axes([x0, yb - 0.045, 0.078, 0.033]),
                           "Load LTspice", color="#fde0dd")
        self.b_lt.on_clicked(self.on_load_lt)
        self.b_clear = Button(self.fig.add_axes([x0 + 0.087, yb - 0.045, 0.078, 0.033]),
                              "Clear", color="#e5e5e5")
        self.b_clear.on_clicked(self.on_clear)

        self.chk = CheckButtons(self.fig.add_axes([x0, yb - 0.135, w, 0.075]),
                                ["log-y on EPSP", "mark switch events"],
                                [False, True])
        self.chk.on_clicked(lambda _l: self.redraw())

        self.info = self.fig.text(x0, yb - 0.16, "", fontsize=8,
                                  va="top", family="monospace")

    def _label(self, key, lab, v):
        self.sl[key].ax.set_title(f"{lab} = {v:.2f}", fontsize=8, pad=3, loc="left")

    def _style_axes(self):
        titles = ["1  PULSERS — gate current into the SNSPD front-end nTrons",
                  "2  GATES — input-nTron gate current at A, B, C",
                  "3  EPSPs — loop current (the membrane variable)",
                  "4  OUTPUTS — output-node voltage"]
        ylabs = ["gate current (µA)", "gate current (µA)",
                 "loop current (µA)", "output (mV)"]
        for a, t, yl in zip(self.ax, titles, ylabs):
            a.set_title(t, fontsize=9, loc="left")
            a.set_ylabel(yl, fontsize=8)
            a.grid(alpha=0.25)
            a.tick_params(labelsize=8)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        self.ax[-1].set_xlabel("time (ns)", fontsize=9)

    # -- solving ----------------------------------------------------------
    def _params(self):
        g = lambda k: float(self.sl[k].val)
        return dict(dt_detect=g("dt") * 1e-9,
                    biases={"pulser1": g("P1"), "pulser2": g("P2"),
                            "A": (g("Av1"), g("Av2")),
                            "B": (g("Bv1"), g("Bv2")),
                            "C": (g("Cv1"), g("Cv2"))},
                    tstop=g("tstop") * 1e-9)

    def on_run(self, _evt):
        if self.worker and self.worker.is_alive():
            self.abort = True
            return
        p = self._params()
        self.abort = False
        self.b_run.label.set_text("Abort")
        self._pending = None

        def job():
            cir, nodes, ts = build_snn_2input(dt_detect=p["dt_detect"],
                                              biases=p["biases"])
            def cb(frac):
                self._frac = frac
                return not self.abort
            r = cir.transient(p["tstop"], record=net_probes(), record_dt=20e-12,
                              on_progress=cb)
            r["_spikes"] = ts
            self._pending = r

        self._frac = 0.0
        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()
        self.timer = self.fig.canvas.new_timer(interval=250)
        self.timer.add_callback(self._poll)
        self.timer.start()

    def _poll(self):
        if self.worker.is_alive():
            self.fig.suptitle(f"solving…  {100*self._frac:5.1f}%   "
                              f"(press Abort to stop)", fontsize=11)
            self.fig.canvas.draw_idle()
            return
        self.timer.stop()
        self.b_run.label.set_text("Run")
        if self._pending is not None:
            self.result = self._pending
            self.redraw()
        else:
            self.fig.suptitle("aborted", fontsize=11)
            self.fig.canvas.draw_idle()

    def on_hold(self, _evt):
        if self.result is not None:
            self.ghost = self.result
            self.redraw()

    def on_clear(self, _evt):
        self.ghost = None; self.lt = None; self.redraw()

    def on_load_lt(self, _evt):
        path = os.environ.get("LTSPICE_TXT", "")
        if not path:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk(); root.withdraw()
                path = filedialog.askopenfilename(
                    title="LTspice exported text",
                    filetypes=[("text", "*.txt *.csv *.tsv"), ("all", "*")])
                root.destroy()
            except Exception:
                print("No file dialog available. Set LTSPICE_TXT=/path/file.txt "
                      "and press Load LTspice again.")
                return
        if path and os.path.exists(path):
            self.lt = load_ltspice(path)
            print(f"loaded {len(self.lt)-1} columns from {path}")
            self.redraw()

    # -- drawing ----------------------------------------------------------
    def redraw(self):
        r = self.result
        if r is None:
            return
        logy, marks = self.chk.get_status()
        for a in self.ax:
            a.clear()
        self._style_axes()
        t = r["t"] * 1e9

        def ghost(key, ax, scale):
            if self.ghost is not None and key in self.ghost:
                ax.plot(self.ghost["t"] * 1e9, self.ghost[key] * scale,
                        color="0.75", lw=1.0, zorder=1)

        # 1 - pulsers
        for k in ("P1", "P2"):
            ghost(f"I({k}_U.g)", self.ax[0], 1e6)
            self.ax[0].plot(t, r[f"I({k}_U.g)"] * 1e6, color=COL[k], lw=1.3,
                            label=f"{k} gate")
        self.ax[0].axhline(ISW_G * 1e6, color="k", ls=":", lw=1,
                           label=f"$I_{{sw,g}}$ = {ISW_G*1e6:.1f} µA")
        self.ax[0].legend(fontsize=7, ncol=3, loc="upper right")

        # 2 - transponder input gates
        for k in "ABC":
            ghost(f"I({k}_U2.g)", self.ax[1], 1e6)
            self.ax[1].plot(t, r[f"I({k}_U2.g)"] * 1e6, color=COL[k], lw=1.3,
                            label=f"{k} in-gate")
            if marks:
                for te in edges(r["t"], r[f"R({k}_U2.d)"]):
                    self.ax[1].axvline(te * 1e9, color=COL[k], lw=0.8,
                                       alpha=0.35, ls="--")
        self.ax[1].axhline(ISW_G * 1e6, color="k", ls=":", lw=1)
        self.ax[1].legend(fontsize=7, ncol=3, loc="upper right")

        # 3 - EPSPs  (the important one)
        for k in "ABC":
            ghost(f"I({k}_Lb1)", self.ax[2], 1e6)
            self.ax[2].plot(t, r[f"I({k}_Lb1)"] * 1e6, color=COL[k], lw=1.7,
                            label=f"{k} loop")
        self.ax[2].axhline(ISW_G * 1e6, color="crimson", ls="--", lw=1.2,
                           label="threshold")
        if logy:
            self.ax[2].set_yscale("log")
        self.ax[2].legend(fontsize=7, ncol=4, loc="upper right")

        # 4 - outputs
        for k in ("P1", "P2"):
            self.ax[3].plot(t, r[f"{k}_out"] * 1e3, color=COL[k], lw=1.0,
                            alpha=0.65, label=f"{k} out")
        for k in "ABC":
            ghost(f"{k}_OUT", self.ax[3], 1e3)
            self.ax[3].plot(t, r[f"{k}_OUT"] * 1e3, color=COL[k], lw=1.4,
                            label=f"{k} out")
        self.ax[3].legend(fontsize=7, ncol=5, loc="upper right")

        # LTspice overlay
        if self.lt is not None:
            tl = self.lt["time"] * 1e9
            for name, arr in self.lt.items():
                if name == "time":
                    continue
                for ax, probes, sc in ((self.ax[2], [f"I({k}_Lb1)" for k in "ABC"], 1e6),
                                       (self.ax[3], [f"{k}_OUT" for k in "ABC"], 1e3)):
                    for pr in probes:
                        if ltspice_key_matches(name, pr):
                            ax.plot(tl, arr * sc, "k--", lw=1.0, alpha=0.8,
                                    label=f"LTspice {name}")
            for ax in (self.ax[2], self.ax[3]):
                ax.legend(fontsize=7, ncol=4, loc="upper right")

        # auto-zoom x to where anything actually happens
        last = 0.0
        for k in "ABC":
            for probe in (f"R({k}_U1.d)", f"R({k}_U2.d)"):
                e_ = edges(r["t"], r[probe])
                if len(e_):
                    last = max(last, e_[-1] * 1e9)
        hi = min(t[-1], max(60.0, last + 45.0))
        for a in self.ax:
            a.set_xlim(0, hi)

        # readout
        lines = []
        for k in "ABC":
            nev = len(edges(r["t"], r[f"R({k}_U2.d)"]))
            nsp = len(edges(r["t"], r[f"R({k}_U1.d)"]))
            pk = r[f"I({k}_Lb1)"].max() * 1e6
            lines.append(f"{k}: {nev} input event(s)  peak {pk:5.2f} µA  "
                         f"{'FIRES' if nsp else '  --  '}")
        dt = self.sl['dt'].val
        self.info.set_text(f"Δt = {dt:.2f} ns\n" + "\n".join(lines))
        self.fig.suptitle(
            f"Δt = {dt:.2f} ns   |   " +
            "   ".join(f"{k}:{'fire' if len(edges(r['t'], r[f'R({k}_U1.d)'])) else 'no'}"
                       for k in "ABC"), fontsize=11)
        self.fig.canvas.draw_idle()


def main():
    if matplotlib.get_backend().lower() in ("agg",):
        print("Matplotlib is using the non-interactive 'Agg' backend, so the "
              "GUI cannot open.\nInstall a GUI backend, e.g.  pip install "
              "PyQt5   (then rerun), or use scripts/sweep_dt.py instead.")
        return
    Explorer()
    plt.show()


if __name__ == "__main__":
    main()
