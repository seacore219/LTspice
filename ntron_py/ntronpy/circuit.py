"""
A minimal SPICE-like transient engine, specialised for the superconducting
transponder circuits.

Why not just call ngspice?  Because the point of this exercise (per the
Aug-14 1:1: *"simulate arbitrary networks with delays etc (NOT in SPICE),
at the circuit level"*) is to get the transponder into a Python object we
can differentiate, sweep, batch, and eventually collapse into a LIF unit.

Method
------
Backward Euler with companion models.  Every inductive branch

    L(I) dI/dt = V_a - V_b - R(t) I

is discretised as

    I_new * (L/h + R) = (L/h) I_old + (V_a - V_b)

i.e. a conductance g_eq = 1/(L/h + R) in parallel with a current source
I_eq = g_eq (L/h) I_old.  This is unconditionally stable, which matters a
lot here: a gate hotspot of 650 ohm across 105 pH of kinetic inductance is
an L/R time constant of 0.16 ps, so an explicit integrator would need
femtosecond steps.

Nonlinearities (kinetic inductance L(I), hotspot resistance R(r)) are
evaluated at the start of each step -- semi-implicit, no Newton loop.
Accuracy is controlled by the adaptive step size instead: sub-picosecond
while any hotspot is alive or a branch is near its critical current,
~100 ps while the network is quiescent.

Transmission lines are modelled as pure delays.  Every `tline` in these
schematics is driven by an ideal source and terminated in 50 ohm == Z0,
so the reflection coefficient is < 0.3% and a one-pass delay is exact to
within that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .ntron import NTron, NTronParams

GND = "0"


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
@dataclass
class Pulse:
    """LTspice PULSE(v1 v2 tdelay trise tfall ton period ncycles)."""

    v1: float
    v2: float
    tdelay: float
    trise: float
    tfall: float
    ton: float
    period: float
    ncycles: int = 1
    extra_delay: float = 0.0     # e.g. a 5 ns transmission line in front

    def __call__(self, t: float) -> float:
        t = t - self.extra_delay - self.tdelay
        if t < 0:
            return self.v1
        n = int(t // self.period) if self.period > 0 else 0
        if self.ncycles and n >= self.ncycles:
            return self.v1
        tt = t - n * self.period
        if tt < self.trise:
            return self.v1 + (self.v2 - self.v1) * tt / self.trise
        tt -= self.trise
        if tt < self.ton:
            return self.v2
        tt -= self.ton
        if tt < self.tfall:
            return self.v2 + (self.v1 - self.v2) * tt / self.tfall
        return self.v1

    def dc(self) -> float:
        return self.v1

    def event_times(self, tstop: float) -> list[float]:
        """Times where the waveform has a slope discontinuity."""
        out: list[float] = []
        base = self.extra_delay + self.tdelay
        n = self.ncycles if self.ncycles else int(tstop / max(self.period, 1e-12)) + 1
        for k in range(n):
            t0 = base + k * self.period
            for dt in (0.0, self.trise, self.trise + self.ton,
                       self.trise + self.ton + self.tfall):
                if t0 + dt <= tstop:
                    out.append(t0 + dt)
        return out


@dataclass
class DC:
    """Constant source, optionally behind a transmission-line delay."""

    value: float
    extra_delay: float = 0.0

    def dc(self) -> float:
        return self.value

    def __call__(self, t: float) -> float:
        return self.value if t >= self.extra_delay else 0.0

    def event_times(self, tstop: float) -> list[float]:
        return [self.extra_delay] if 0 < self.extra_delay <= tstop else []


@dataclass
class Delayed:
    """Wrap any callable behind a pure delay (a matched transmission line)."""

    src: Callable[[float], float]
    delay: float

    def dc(self) -> float:
        return getattr(self.src, "dc", lambda: 0.0)()

    def __call__(self, t: float) -> float:
        return self.src(t - self.delay)

    def event_times(self, tstop: float) -> list[float]:
        base = getattr(self.src, "event_times", lambda _: [])(tstop)
        return [t + self.delay for t in base if t + self.delay <= tstop]


class DelayLink:
    """
    An axon: reproduces V(src_node, t - delay) at another node.

    This is the "variable-delay interconnect" of milestone M2.1.  It is an
    idealisation of a NbN kinetic-inductance line -- ideal buffering, no
    loading of the driver, no dispersion -- which is the right abstraction
    while the delay itself is the design variable ("weight associated with
    the link is the time delay").  Swap in a real lossy line model later if
    the fan-out budget turns out to matter.
    """

    def __init__(self, src_node: str, delay: float, gain: float = 1.0,
                 capacity: int = 1 << 21):
        self.src_node = src_node
        self.delay = delay
        self.gain = gain
        self._t = np.zeros(capacity)
        self._v = np.zeros(capacity)
        self._n = 1                # sample 0 is (t=0, v=0)
        self._read = 0

    def push(self, t: float, v: float) -> None:
        if self._n >= self._t.size:          # grow
            self._t = np.concatenate([self._t, np.zeros(self._t.size)])
            self._v = np.concatenate([self._v, np.zeros(self._v.size)])
        self._t[self._n] = t
        self._v[self._n] = v
        self._n += 1

    def __call__(self, t: float) -> float:
        tq = t - self.delay
        if tq <= 0.0:
            return 0.0
        # queries are monotonically increasing -> walk a read pointer
        i = self._read
        n = self._n
        while i + 1 < n and self._t[i + 1] <= tq:
            i += 1
        self._read = i
        if i + 1 >= n:
            return self.gain * self._v[n - 1]
        t0, t1 = self._t[i], self._t[i + 1]
        if t1 <= t0:
            return self.gain * self._v[i]
        w = (tq - t0) / (t1 - t0)
        return self.gain * ((1 - w) * self._v[i] + w * self._v[i + 1])

    def dc(self) -> float:
        return 0.0

    def event_times(self, tstop: float) -> list[float]:
        return []


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------
@dataclass
class Resistor:
    a: str
    b: str
    R: float
    name: str = ""


@dataclass
class Inductor:
    a: str
    b: str
    L: float
    name: str = ""
    Rser: float = 0.0
    I: float = 0.0          # state


@dataclass
class VSource:
    node: str
    fn: Callable[[float], float]
    name: str = ""


@dataclass
class NTronElement:
    g: str
    d: str
    s: str
    dev: NTron
    name: str = ""

    @property
    def center(self) -> str:
        return f"__{self.dev.name}_center"


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------
class Circuit:
    """
    Vectorised backward-Euler engine.

    Every inductive branch (plain inductors *and* the three nTron terminal
    branches) goes into one branch list with an incidence matrix `Inc`
    (B x N).  Then the whole companion stamp is two matrix products:

        A_nn = A_resistive + Inc.T @ diag(g) @ Inc
        z_n  = z_resistive - Inc.T @ i_eq
        I_new = g * (Inc @ v) + i_eq

    which keeps the per-step Python work to just the device physics.
    """

    def __init__(self, name: str = "circuit"):
        self.name = name
        self.resistors: list[Resistor] = []
        self.inductors: list[Inductor] = []
        self.vsources: list[VSource] = []
        self.ntrons: list[NTronElement] = []
        self.delay_links: list[DelayLink] = []
        self._nodes: dict[str, int] = {}

    # -- construction -----------------------------------------------------
    def R(self, a, b, value, name=""):
        e = Resistor(a, b, value, name); self.resistors.append(e); return e

    def L(self, a, b, value, name="", Rser=0.0):
        e = Inductor(a, b, value, name, Rser); self.inductors.append(e); return e

    def V(self, node, fn, name=""):
        if not callable(fn):
            fn = DC(float(fn))
        e = VSource(node, fn, name); self.vsources.append(e); return e

    def link(self, src_node, dst_node, delay, gain=1.0, name=""):
        """Connect src_node -> dst_node through a pure propagation delay."""
        dl = DelayLink(src_node, delay, gain)
        self.delay_links.append(dl)
        self.V(dst_node, dl, name=name or f"axon_{src_node}_{dst_node}")
        return dl

    def ntron(self, name, g, d, s, params, **kw):
        dev = NTron(name, params, **kw)
        e = NTronElement(g, d, s, dev, name); self.ntrons.append(e); return e

    # -- assembly ---------------------------------------------------------
    def _idx(self, node):
        return -1 if node == GND else self._nodes[node]

    def _build(self):
        names = []
        for e in self.resistors + self.inductors:
            names += [e.a, e.b]
        for v in self.vsources:
            names.append(v.node)
        for x in self.ntrons:
            names += [x.g, x.d, x.s, x.center]
        for dl in self.delay_links:
            names.append(dl.src_node)
        seen, ordered = {GND}, []
        for n in names:
            if n not in seen:
                seen.add(n); ordered.append(n)
        self._nodes = {n: i for i, n in enumerate(ordered)}
        N = self.n_nodes = len(ordered)
        M = self.n_v = len(self.vsources)

        # ---- branch list: inductors first, then nTron g/d/s ----
        pairs = [(self._idx(e.a), self._idx(e.b)) for e in self.inductors]
        self.n_ind = len(pairs)
        self._nt = []
        for x in self.ntrons:
            ic = self._idx(x.center)
            base = len(pairs)
            pairs += [(self._idx(x.g), ic), (self._idx(x.d), ic),
                      (self._idx(x.s), ic)]
            self._nt.append((x.dev, base))
        B = self.n_branch = len(pairs)

        Inc = np.zeros((B, N))
        for k, (ia, ib) in enumerate(pairs):
            if ia >= 0:
                Inc[k, ia] += 1.0
            if ib >= 0:
                Inc[k, ib] -= 1.0
        self.Inc = Inc

        self.L_vec = np.zeros(B)
        self.R_vec = np.zeros(B)
        self.I_vec = np.zeros(B)
        for k, e in enumerate(self.inductors):
            self.L_vec[k] = e.L
            self.R_vec[k] = e.Rser

        # ---- constant resistive part ----
        self.A_base = np.zeros((N + M, N + M))
        for e in self.resistors:
            ia, ib, g = self._idx(e.a), self._idx(e.b), 1.0 / e.R
            if ia >= 0:
                self.A_base[ia, ia] += g
            if ib >= 0:
                self.A_base[ib, ib] += g
            if ia >= 0 and ib >= 0:
                self.A_base[ia, ib] -= g
                self.A_base[ib, ia] -= g
        for k, v in enumerate(self.vsources):
            iv, row = self._idx(v.node), N + k
            if iv >= 0:
                self.A_base[row, iv] += 1.0
                self.A_base[iv, row] += 1.0
        self._v_rows = [(v, N + k) for k, v in enumerate(self.vsources)]
        self._dl_idx = [(dl, self._idx(dl.src_node)) for dl in self.delay_links]
        self._A = np.empty((N + M, N + M))
        self._z = np.zeros(N + M)

    # -- device <-> vector plumbing ---------------------------------------
    def _pull_devices(self):
        for dev, b in self._nt:
            Lg, Ld, Ls = dev.branch_L()
            Rg, Rd, Rs = dev.branch_R()
            self.L_vec[b:b + 3] = (Lg, Ld, Ls)
            self.R_vec[b:b + 3] = (Rg, Rd, Rs)
            self.I_vec[b:b + 3] = (dev.I_g, dev.I_d, dev.I_s)

    def _push_devices(self):
        for dev, b in self._nt:
            dev.I_g, dev.I_d, dev.I_s = self.I_vec[b:b + 3]
        for k, e in enumerate(self.inductors):
            e.I = self.I_vec[k]

    # -- DC operating point -----------------------------------------------
    def operating_point(self, R_eps=1e-6):
        """
        LTspice's implicit `.op`: inductors are shorts, sources at DC.

        This matters enormously here.  The nTrons are biased at ~0.9 of
        their channel critical current; starting every inductor at zero
        current instead leaves the devices far below their operating point
        for tens of nanoseconds and the input nTron's channel never latches.
        """
        N, M = self.n_nodes, self.n_v
        g = 1.0 / (R_eps + np.concatenate([
            np.array([e.Rser for e in self.inductors]),
            np.zeros(self.n_branch - self.n_ind)]))
        A = self.A_base.copy()
        A[:N, :N] += self.Inc.T @ (g[:, None] * self.Inc)
        z = np.zeros(N + M)
        for v, row in self._v_rows:
            z[row] = getattr(v.fn, "dc", lambda: v.fn(0.0))()
        sol = np.linalg.solve(A, z)
        self.I_vec = g * (self.Inc @ sol[:N])
        self._push_devices()
        return sol

    # -- transient --------------------------------------------------------
    def transient(self, tstop, h_fast=1e-12, h_slow=100e-12, record=(),
                  record_dt=20e-12, gate_margin=0.5, channel_margin=0.985,
                  uic=False, progress=False, on_progress=None):
        self._build()
        N, M = self.n_nodes, self.n_v
        self._pull_devices()
        v = np.zeros(N + M) if uic else self.operating_point()

        events = sorted({e for vs in self.vsources
                         for e in getattr(vs.fn, "event_times",
                                          lambda _: [])(tstop)})
        ev_i = 0
        t, next_rec, n_steps = 0.0, 0.0, 0
        out_t, out = [], {k: [] for k in record}
        ind_by_name = {e.name: k for k, e in enumerate(self.inductors) if e.name}
        nt_by_name = {x.dev.name: x.dev for x in self.ntrons}
        # LTspice convention: I(Rxx) flows from the first node to the second
        res_by_name = {e.name: (self._idx(e.a), self._idx(e.b), 1.0 / e.R)
                       for e in self.resistors if e.name}

        def sample():
            out_t.append(t)
            for k in record:
                if k.startswith("I(") and k.endswith(")"):
                    ref = k[2:-1]
                    if "." in ref:
                        dn, br = ref.split(".")
                        d = nt_by_name[dn]
                        out[k].append({"g": d.I_g, "d": d.I_d, "s": d.I_s}[br])
                    elif ref in ind_by_name:
                        out[k].append(self.I_vec[ind_by_name[ref]])
                    elif ref in res_by_name:
                        ia, ib, g = res_by_name[ref]
                        va = 0.0 if ia < 0 else v[ia]
                        vb = 0.0 if ib < 0 else v[ib]
                        out[k].append((va - vb) * g)
                    else:
                        raise KeyError(f"unknown probe {k}")
                elif k.startswith("R(") and k.endswith(")"):
                    dn, br = k[2:-1].split(".")
                    Rg, Rd, Rs = nt_by_name[dn].branch_R()
                    out[k].append({"g": Rg, "d": Rd, "s": Rs}[br])
                else:
                    nm = k[2:-1] if (k.startswith("V(") and k.endswith(")")) else k
                    i = self._idx(nm)
                    out[k].append(0.0 if i < 0 else v[i])

        sample(); next_rec += record_dt
        Inc, IncT = self.Inc, self.Inc.T

        while t < tstop:
            # ---- device state -> latches, hotspot growth ----
            active = hot = False
            for x in self.ntrons:
                d = x.dev
                d.update_latches()
                if d.normal_g or d.normal_d or d.normal_s:
                    active = True
                p = d.p
                Ic = p.Isw_c * d.channel_ic_factor()
                if (abs(d.I_g) > gate_margin * p.Isw_g
                        or abs(d.I_d) > channel_margin * Ic
                        or abs(d.I_s) > channel_margin * Ic):
                    hot = True

            h = h_fast if (active or hot) else h_slow
            while ev_i < len(events) and events[ev_i] <= t + 1e-18:
                ev_i += 1
            if ev_i < len(events):
                h = min(h, max(events[ev_i] - t, h_fast))
            h = min(h, tstop - t)
            if h <= 0:
                break

            for x in self.ntrons:
                x.dev.step_hotspots(h)

            # ---- companion stamp (vectorised) ----
            self._pull_devices()
            gl = self.L_vec / h
            g = 1.0 / (gl + self.R_vec)
            ieq = g * gl * self.I_vec

            A = self._A
            np.copyto(A, self.A_base)
            A[:N, :N] += IncT @ (g[:, None] * Inc)
            z = self._z
            z[:] = 0.0
            z[:N] = -(IncT @ ieq)
            tn = t + h
            for vs, row in self._v_rows:
                z[row] = vs.fn(tn)

            v = np.linalg.solve(A, z)
            self.I_vec = g * (Inc @ v[:N]) + ieq
            self._push_devices()

            for dl, i in self._dl_idx:
                dl.push(tn, 0.0 if i < 0 else v[i])

            t = tn
            n_steps += 1
            if t >= next_rec:
                sample(); next_rec += record_dt
            if n_steps % 20000 == 0:
                if progress:
                    print(f"    t = {t*1e9:7.2f} ns  ({n_steps} steps)", flush=True)
                if on_progress is not None and on_progress(t / tstop) is False:
                    break   # caller asked us to abort

        sample()
        res = {"t": np.asarray(out_t), "n_steps": n_steps}
        for k in record:
            res[k] = np.asarray(out[k])
        return res
