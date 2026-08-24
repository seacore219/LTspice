#!/usr/bin/env python3
"""
LTspice-style waveform viewer for the SNN_2input network.

    python3 scripts/gui.py

Signals carry LTspice's own element names -- I(Lb1), I(R28), V(bias3) --
and the plot header lists them as coloured labels across the top, the way
the LTspice waveform window does, so a trace here and a trace there are
the same object. Each curve is also annotated at its peak.

  Preset       loads the exact signal sets from the LTspice screenshots
               ("screenshot 3", "fan-in currents", "loop currents", ...)
  Sliders      detector dt and all eight bias sources (V2/V3/V5/V6/V7/V8/
               V9/V10), each labelled with the bias net it feeds
  Run          solve the full electrothermal circuit; doubles as Abort
  Hold         freeze current traces as grey ghosts to compare biases
  LTspice      overlay a "File > Export data as text" dump; columns whose
               names match a plotted signal are drawn dashed in white

The solve takes ~30-150 s: sub-picosecond steps are unavoidable when a
650 ohm hotspot sits across 105 pH of kinetic inductance. Sliders are
instant; the solve is not.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from ntronpy.snn2_netlist import (build, PRESETS, BIAS_SLIDERS, DC_SOURCES,
                                  ROLE, all_signals)

PALETTE = ["#00ff00", "#5070ff", "#ff3030", "#00d0d0", "#ff50ff", "#e8e8e8",
           "#80ff80", "#70a0ff", "#ffa000", "#ff90d0", "#ff8080", "#ffff00",
           "#40c040", "#40a0ff", "#c0c0c0", "#ff6060"]
ISW_G = 15.2e-6


def load_ltspice(path):
    """Tolerant reader for LTspice 'Export data as text' output."""
    with open(path, "r", errors="ignore") as f:
        header = f.readline().strip()
    names = [c.strip() for c in header.replace(",", "\t").split("\t") if c.strip()]
    data = np.genfromtxt(path, skip_header=1, delimiter=None, dtype=float)
    if data.ndim == 1:
        data = data[None, :]
    out = {nm: data[:, i] for i, nm in enumerate(names[: data.shape[1]])}
    out["time"] = data[:, 0]
    return out


def norm(name):
    return name.lower().replace(" ", "")


class Viewer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16.4, 8.6), facecolor="#f4f4f4")
        try:
            self.fig.canvas.manager.set_window_title("SNN_2input - LTspice-named viewer")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.050, 0.09, 0.655, 0.73])
        self.ax.set_facecolor("#101010")
        self.hdr = self.fig.add_axes([0.050, 0.835, 0.655, 0.085])
        self.hdr.axis("off")

        self.preset = "screenshot 3"
        self.signals = list(PRESETS[self.preset])
        self.result = None
        self.ghost = None
        self.lt = None
        self.worker = None
        self.abort = False
        self._frac = 0.0
        self._pending = None
        self._build_controls()
        self._style()
        self.hdr.text(0.0, 0.5, "press Run", fontsize=10, va="center")

    def _build_controls(self):
        x0, w = 0.735, 0.205
        self.sl = {}
        y = 0.905
        ax = self.fig.add_axes([x0, y, w, 0.017])
        s = Slider(ax, "", 0.0, 60.0, valinit=15.5, valfmt="%.2f", color="#9ecae1")
        ax.set_title("detector dt (ns) = 15.50", fontsize=8, loc="left", pad=3)
        s.on_changed(lambda v: ax.set_title(f"detector dt (ns) = {v:.2f}",
                                            fontsize=8, loc="left", pad=3))
        self.sl["dt"] = s

        for i, (nm, lab) in enumerate(BIAS_SLIDERS):
            yy = y - 0.043 * (i + 1)
            a = self.fig.add_axes([x0, yy, w, 0.017])
            init = DC_SOURCES[nm][1]
            sl = Slider(a, "", 1.0, 1.90, valinit=init, valfmt="%.2f",
                        color="#c7e9c0")
            a.set_title(f"{nm}  {lab} = {init:.2f}", fontsize=8, loc="left", pad=3)
            sl.on_changed(lambda v, a=a, nm=nm, lab=lab:
                          a.set_title(f"{nm}  {lab} = {v:.2f}", fontsize=8,
                                      loc="left", pad=3))
            self.sl[nm] = sl

        yb = y - 0.043 * (len(BIAS_SLIDERS) + 1) - 0.025
        self.b_run = Button(self.fig.add_axes([x0, yb, 0.052, 0.030]), "Run",
                            color="#c7e9c0")
        self.b_run.on_clicked(self.on_run)
        self.b_hold = Button(self.fig.add_axes([x0 + 0.058, yb, 0.052, 0.030]),
                             "Hold", color="#e5e5e5")
        self.b_hold.on_clicked(self.on_hold)
        self.b_lt = Button(self.fig.add_axes([x0 + 0.116, yb, 0.052, 0.030]),
                           "LTspice", color="#fde0dd")
        self.b_lt.on_clicked(self.on_load_lt)
        self.b_clr = Button(self.fig.add_axes([x0 + 0.174, yb, 0.051, 0.030]),
                            "Clear", color="#e5e5e5")
        self.b_clr.on_clicked(self.on_clear)

        keys = list(PRESETS)
        rax = self.fig.add_axes([x0, yb - 0.325, w, 0.29])
        rax.set_title("signal preset", fontsize=8, loc="left", pad=6)
        self.radio = RadioButtons(rax, keys, active=keys.index(self.preset))
        for lbl in self.radio.labels:
            lbl.set_fontsize(7.5)
        self.radio.on_clicked(self.on_preset)
        self.info = self.fig.text(x0, yb - 0.345, "", fontsize=7.5, va="top",
                                  family="monospace")

    def on_preset(self, label):
        self.preset = label
        self.signals = list(PRESETS[label])
        self.redraw()

    def on_hold(self, _e):
        self.ghost = self.result
        self.redraw()

    def on_clear(self, _e):
        self.ghost = None
        self.lt = None
        self.redraw()

    def _style(self):
        self.ax.grid(color="#303030", lw=0.6)
        self.ax.tick_params(colors="#404040", labelsize=8)
        self.ax.set_xlabel("time (ns)", fontsize=9)
        for sp in self.ax.spines.values():
            sp.set_color("#808080")

    def on_run(self, _evt):
        if self.worker and self.worker.is_alive():
            self.abort = True
            return
        dt = float(self.sl["dt"].val) * 1e-9
        biases = {nm: float(self.sl[nm].val) for nm, _ in BIAS_SLIDERS}
        self.abort = False
        self._pending = None
        self.b_run.label.set_text("Abort")
        probes = sorted(set(all_signals()))

        def job():
            cir, meta = build(dt_detect=dt, biases=biases)
            def cb(f):
                self._frac = f
                return not self.abort
            r = cir.transient(120e-9, record=probes, record_dt=20e-12,
                              on_progress=cb)
            r["_meta"] = meta
            self._pending = r

        self._frac = 0.0
        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()
        self.timer = self.fig.canvas.new_timer(interval=250)
        self.timer.add_callback(self._poll)
        self.timer.start()

    def _poll(self):
        if self.worker.is_alive():
            self.hdr.clear(); self.hdr.axis("off")
            self.hdr.text(0.0, 0.5, f"solving...  {100*self._frac:5.1f}%   "
                                    "(Run -> Abort)", fontsize=10, va="center")
            self.fig.canvas.draw_idle()
            return
        self.timer.stop()
        self.b_run.label.set_text("Run")
        if self._pending is not None:
            self.result = self._pending
            self.redraw()

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
                print("No file dialog. Set LTSPICE_TXT=/path/file.txt and retry.")
                return
        if path and os.path.exists(path):
            self.lt = load_ltspice(path)
            print(f"loaded {len(self.lt)-1} columns from {path}")
            self.redraw()

    def redraw(self):
        r = self.result
        self.ax.clear(); self.ax.set_facecolor("#101010"); self._style()
        self.hdr.clear(); self.hdr.axis("off")
        if r is None:
            self.hdr.text(0.0, 0.5, "press Run", fontsize=10, va="center")
            self.fig.canvas.draw_idle()
            return

        t = r["t"] * 1e9
        avail = [s for s in self.signals if s in r]
        is_v = bool(avail) and all(s.startswith("V(") for s in avail)
        scale, unit = (1e3, "mV") if is_v else (1e6, "uA")

        ncol = max(1, min(7, len(avail)))
        for i, s in enumerate(avail):
            c = PALETTE[i % len(PALETTE)]
            col, row = i % ncol, i // ncol
            self.hdr.text(col / ncol, 0.80 - 0.36 * row, s, color=c, fontsize=9,
                          family="monospace", weight="bold",
                          transform=self.hdr.transAxes)

        for i, s in enumerate(avail):
            c = PALETTE[i % len(PALETTE)]
            if self.ghost is not None and s in self.ghost:
                self.ax.plot(self.ghost["t"] * 1e9, self.ghost[s] * scale,
                             color="#585858", lw=0.9, zorder=1)
            y = r[s] * scale
            self.ax.plot(t, y, color=c, lw=1.2, zorder=3)
            k = int(np.argmax(np.abs(y)))
            self.ax.annotate(s, (t[k], y[k]), color=c, fontsize=7.5,
                             family="monospace", xytext=(4, 4),
                             textcoords="offset points", zorder=4)

        if not is_v:
            self.ax.axhline(ISW_G * 1e6, color="#ff5050", ls=":", lw=1)
            self.ax.annotate("Isw_g = 15.2 uA", (t[-1], ISW_G * 1e6),
                             color="#ff5050", fontsize=7.5, ha="right",
                             va="bottom")
        self.ax.set_ylabel(unit, fontsize=9)

        if self.lt is not None:
            tl = self.lt["time"] * 1e9
            for name, arr in self.lt.items():
                if name == "time":
                    continue
                for s in avail:
                    if norm(name) == norm(s):
                        self.ax.plot(tl, arr * scale, "w--", lw=1.0, alpha=0.85,
                                     zorder=5)
                        kk = int(np.argmax(np.abs(arr)))
                        self.ax.annotate(f"LTspice {name}",
                                         (tl[kk], arr[kk] * scale), color="w",
                                         fontsize=7, xytext=(4, -11),
                                         textcoords="offset points", zorder=5)

        lines = []
        for u in ("u1", "u4", "u3", "u2", "u6", "u5", "u8", "u7"):
            key = f"R({u}.d)"
            if key not in r:
                continue
            sig = np.asarray(r[key]) > 1.0
            n = int(np.sum(sig[1:] & ~sig[:-1]))
            latched = sig[-1]
            for other in (f"R({u}.g)",):
                if other in r:
                    latched = latched or (np.asarray(r[other])[-1] > 1.0)
            lines.append(f"{u} {ROLE[u]:<8s} {n} switch"
                         + ("  LATCHED" if latched else ""))
        self.info.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()


def main():
    if matplotlib.get_backend().lower() == "agg":
        print("Matplotlib is on the non-interactive 'Agg' backend, so no window "
              "can open.\nTry:  pip install PyQt5   then rerun.")
        return
    Viewer()
    plt.show()


if __name__ == "__main__":
    main()
