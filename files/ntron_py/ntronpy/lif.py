"""
The reduced transponder: a leaky integrate-and-fire unit whose "membrane
potential" is the current in the loop inductor.

Derivation
----------
While the output nTron U1 is superconducting, the integrating loop is

    L_loop dI/dt = V_bias1(t) - R_loop I

with L_loop = Lb1 + L_gate(U1) ~ 148 nH and R_loop = 2*R_loop_param.
So the leak is

    tau_m = L_loop / R_loop            (an L/R leak, not RC)

Each presynaptic spike switches the *input* nTron U2 for a hotspot dwell
time T_hs.  During that window U2's channel resistance R_hs(t) forces the
bias current Ib1 out of U2 and part of it into the loop.  Rather than
carry R_hs(t) around, collapse the whole injection into an exponentially
decaying synaptic drive with time constant tau_s ~ T_hs:

    L_loop dI/dt = -R_loop I + V_syn(t)
    tau_s dV_syn/dt = -V_syn + L_loop * sum_k w_k delta(t - t_k - d_k)

which gives the standard double-exponential ("beta") kernel.  For a single
input spike at t = 0, starting from rest:

    I(t) = w * (tau_m / (tau_m - tau_s)) * (exp(-t/tau_m) - exp(-t/tau_s))

Fire when I >= I_th, where I_th = Isw_g(U1) = Jc * w_g * thickness, then
hold I at I_reset for a refractory time t_ref (physically: U1's gate
hotspot is alive and actively damping the loop).

What the two bias currents do
-----------------------------
  Ib1 (v1) scales the synaptic weight w -- how much charge each input
      spike dumps into the loop.  It is the gain knob.
  Ib2 (v2) sets the output drive and, through the channel Ic suppression,
      the effective threshold and the refractory period.  It is the
      excitability knob.

Both appear here as explicit parameters so they stay trainable, which is
the whole point: on hardware you train the biases, not the wires.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Closed-form trace
# ---------------------------------------------------------------------------
def epsp_kernel(t: np.ndarray, tau_m: float, tau_s: float,
                normalise: str = "peak") -> np.ndarray:
    """
    Double-exponential EPSP kernel, causal.

        k(t) = (exp(-t/tau_m) - exp(-t/tau_s)) * Theta(t)

    normalise='peak' scales so max(k) == 1 (so `w` is directly the peak
    loop-current contribution in amps -- convenient for setting thresholds).
    normalise='none' leaves the raw difference of exponentials.
    """
    t = np.asarray(t, dtype=float)
    k = np.where(t >= 0, np.exp(-np.clip(t, 0, None) / tau_m)
                 - np.exp(-np.clip(t, 0, None) / tau_s), 0.0)
    if normalise == "peak":
        if abs(tau_m - tau_s) < 1e-18:
            return k
        t_pk = (tau_m * tau_s) / (tau_m - tau_s) * np.log(tau_m / tau_s)
        k_pk = np.exp(-t_pk / tau_m) - np.exp(-t_pk / tau_s)
        if k_pk > 0:
            k = k / k_pk
    return k


def lif_trace(t: np.ndarray, spike_times, w, tau_m: float, tau_s: float,
              I0: float = 0.0) -> np.ndarray:
    """
    Sub-threshold loop-current trace for a list of input spikes.

    This is the *equation to model the trace* -- superposition of kernels,
    valid until the first output spike (after which the reset makes it
    piecewise).  Use LIFNeuron for the full spiking dynamics.
    """
    t = np.asarray(t, dtype=float)
    spike_times = np.atleast_1d(spike_times)
    w = np.broadcast_to(np.atleast_1d(w), spike_times.shape)
    I = I0 * np.exp(-np.clip(t, 0, None) / tau_m)
    for ts, wk in zip(spike_times, w):
        I = I + wk * epsp_kernel(t - ts, tau_m, tau_s)
    return I


# ---------------------------------------------------------------------------
# Spiking unit
# ---------------------------------------------------------------------------
@dataclass
class LIFParams:
    """All in SI units; the state variable is a current, in amps."""

    tau_m: float = 37e-9      # L_loop / R_loop
    tau_s: float = 6e-9       # synaptic / hotspot dwell time
    I_th: float = 15.2e-6     # Isw_g of the output nTron
    I_reset: float = 0.0
    t_ref: float = 8e-9       # gate-hotspot recovery
    w: float = 12e-6          # peak loop current per unit input spike

    # provenance -- what the circuit values were when this was fitted
    L_loop: float = 148.2e-9
    R_loop: float = 4.0
    Ib1: float = 130e-6
    Ib2: float = 170e-6

    @classmethod
    def from_circuit(cls, cfg, I_th: float, **over) -> "LIFParams":
        L = cfg.L_loop
        R = 2.0 * cfg.R_loop
        kw = dict(tau_m=L / R, I_th=I_th, L_loop=L, R_loop=R,
                  Ib1=cfg.Ib1, Ib2=cfg.Ib2)
        kw.update(over)
        return cls(**kw)


class LIFNeuron:
    """Event-free fixed-step LIF; vectorises trivially over a population."""

    def __init__(self, p: LIFParams):
        self.p = p
        self.reset()

    def reset(self) -> None:
        self.I = 0.0        # loop current  (membrane)
        self.g = 0.0        # synaptic drive
        self.last_spike = -np.inf

    def step(self, t: float, dt: float, drive: float = 0.0) -> bool:
        """`drive` is the summed weight of input spikes arriving in this step."""
        p = self.p
        self.g += drive
        self.g *= np.exp(-dt / p.tau_s)
        if t - self.last_spike < p.t_ref:
            self.I = p.I_reset
            return False
        # exact update of  dI/dt = -I/tau_m + g/tau_s  over dt with g decaying
        a = np.exp(-dt / p.tau_m)
        self.I = self.I * a + self.g * dt / p.tau_s * a
        if self.I >= p.I_th:
            self.I = p.I_reset
            self.last_spike = t
            return True
        return False


def simulate_lif_network(params: list[LIFParams], adjacency, delays,
                         inputs, tstop: float, dt: float = 20e-12,
                         input_weight: float = 1.0):
    """
    N-transponder LIF network with variable-delay interconnects.

    adjacency[i][j] : weight of the link j -> i (amps of peak loop current)
    delays[i][j]    : propagation delay of link j -> i (seconds)
    inputs          : list, per neuron, of external spike times

    Returns (t, I_trace[N, T], spikes[N] -> list of times).
    """
    N = len(params)
    A = np.asarray(adjacency, float)
    D = np.asarray(delays, float)
    n_steps = int(round(tstop / dt))
    t_grid = np.arange(n_steps + 1) * dt

    neurons = [LIFNeuron(p) for p in params]
    Itr = np.zeros((N, n_steps + 1))
    spikes: list[list[float]] = [[] for _ in range(N)]

    # scheduled arrivals: index -> per-neuron accumulated drive
    pending = np.zeros((N, n_steps + 2))
    for i, times in enumerate(inputs):
        for ts in np.atleast_1d(times):
            k = int(round(ts / dt))
            if 0 <= k <= n_steps:
                pending[i, k] += input_weight

    for k in range(n_steps + 1):
        t = t_grid[k]
        for i, nrn in enumerate(neurons):
            fired = nrn.step(t, dt, pending[i, k])
            Itr[i, k] = nrn.I
            if fired:
                spikes[i].append(t)
                for j in range(N):
                    if A[j, i] != 0.0:
                        kk = k + int(round(D[j, i] / dt))
                        if kk <= n_steps:
                            pending[j, kk] += A[j, i]
    return t_grid, Itr, spikes


# ---------------------------------------------------------------------------
# Fitting the reduction to the full circuit
# ---------------------------------------------------------------------------
def fit_lif_to_trace(t: np.ndarray, I_loop: np.ndarray, spike_times,
                     tau_m0: float = 37e-9, tau_s0: float = 6e-9,
                     w0: float | None = None, t_window=None) -> dict:
    """
    Least-squares fit of the double-exponential kernel to a loop-current
    trace measured on the full electrothermal circuit.

    Returns dict with tau_m, tau_s, w, and the fitted curve.
    """
    t = np.asarray(t, float)
    I = np.asarray(I_loop, float)
    spike_times = np.atleast_1d(np.asarray(spike_times, float))

    m = np.ones_like(t, bool) if t_window is None else (
        (t >= t_window[0]) & (t <= t_window[1]))
    tf, If = t[m], I[m]
    if w0 is None:
        w0 = max(np.max(np.abs(If)), 1e-9)

    def resid(x):
        tau_m, tau_s, w = np.exp(x[0]), np.exp(x[1]), x[2]
        return lif_trace(tf, spike_times, w, tau_m, tau_s) - If

    x0 = np.array([np.log(tau_m0), np.log(tau_s0), w0])
    sol = least_squares(resid, x0, method="trf",
                        bounds=([np.log(1e-10), np.log(1e-11), -1e-2],
                                [np.log(1e-5), np.log(1e-6), 1e-2]))
    tau_m, tau_s, w = np.exp(sol.x[0]), np.exp(sol.x[1]), sol.x[2]
    if tau_s > tau_m:                       # keep the labels meaningful
        tau_m, tau_s = tau_s, tau_m
    fit = lif_trace(t, spike_times, w, tau_m, tau_s)
    rms = float(np.sqrt(np.mean((fit[m] - If) ** 2)))
    return dict(tau_m=tau_m, tau_s=tau_s, w=w, fit=fit, rms=rms,
                rel_rms=rms / max(np.max(np.abs(If)), 1e-15), sol=sol)


# ---------------------------------------------------------------------------
# Conductance-based reduction
# ---------------------------------------------------------------------------
def cond_lif_trace(t, spike_times, a, tau_m, tau_s, E, I0=0.0):
    """
    Conductance-based LIF, integrated on the supplied time grid:

        dI/dt = -I/tau_m + g(t) * (E - I)
        g(t)  = a * sum_k exp(-(t - t_k)/tau_s) * Theta(t - t_k)

    The current-based kernel over-predicts summation by ~9%: the second
    input spike delivers less charge than the first because the loop is
    already carrying current, so there is less voltage left across it.
    That is exactly a driving-force (reversal-potential) effect, and the
    "reversal current" E turns out to be the input bias current Ib1 --
    the current the input nTron has available to divert in the first place.

    So the two hardware knobs land on standard neuron parameters:
        Ib1 (v1) -> reversal current E, i.e. synaptic driving force
        Ib2 (v2) -> output drive and, via channel-Ic suppression, threshold
    """
    t = np.asarray(t, float)
    st = np.atleast_1d(np.asarray(spike_times, float))
    g = np.zeros_like(t)
    for ts in st:
        d = t - ts
        g += np.where(d >= 0, a * np.exp(-np.clip(d, 0, None) / tau_s), 0.0)
    I = np.empty_like(t)
    I[0] = I0
    for k in range(1, t.size):
        h = t[k] - t[k - 1]
        gm = 0.5 * (g[k] + g[k - 1])
        lam = 1.0 / tau_m + gm                 # total decay rate
        inf = gm * E / lam                     # instantaneous fixed point
        e = np.exp(-lam * h)
        I[k] = inf + (I[k - 1] - inf) * e      # exponential Euler
    return I
