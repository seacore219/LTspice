"""
ALL-IN-ONE FILE, HEAVILY COMMENTED VERSION.

Every line below has a comment explaining what it does in plain English.
The actual code is 100% identical to the working version -- comments don't
change how Python runs anything, they're just notes for a human reader.

How to read a comment: anything after a # symbol is a comment. Python
ignores everything from the # to the end of that line -- it's purely for
humans, the computer skips right over it.

Run it the same way as before:
    python3 transponder_elements_and_tests_COMMENTED.py
"""

# "import" means "go get a toolkit that's already written, and let me use it."
# numpy is a toolkit for doing math on lists of numbers efficiently.
# "as np" means "but let me refer to it by the short nickname np from now on."
import numpy as np

# From inside the scipy toolkit (a science/math toolkit), grab just one tool
# called solve_ivp. This is the actual engine that will simulate how our
# circuit changes over time, step by step.
from scipy.integrate import solve_ivp

# Grab the plotting part of the matplotlib toolkit, nicknamed plt.
# This is what will draw the graphs at the end.
import matplotlib.pyplot as plt


# =============================================================================
# PART A -- CIRCUIT ELEMENT CLASSES
#
# A "class" is a blueprint for creating something. Think of it like a
# cookie cutter: the class is the cutter shape, and every time we use it we
# get a new cookie (called an "object" or "instance") shaped like that
# cutter. Below, "CircuitElement" is a generic blueprint, and "Resistor",
# "Inductor", etc. are more specific blueprints built on top of it.
# =============================================================================

# "class CircuitElement:" begins the definition of our generic blueprint.
# Nothing happens yet -- we're just writing down the recipe.
class CircuitElement:
    """
    Base class. Not a translation of anything in LTspice -- pure Python
    scaffolding so every element can be treated the same way: ask it for
    n_states (how many ODE state variables it owns), call derivatives() for
    its rate of change, call output() for its current signal.
    """
    # (the quoted text right above is called a "docstring" -- a comment
    # written in a special way so tools can show it as documentation)

    # Every component built from this blueprint starts out assuming it has
    # 0 "states" -- meaning 0 pieces of memory it needs to keep track of
    # over time. Specific components below will override this number.
    n_states = 0

    # This is the "constructor" -- the function that runs automatically the
    # very first time we create a new component from this blueprint.
    # "self" always refers to "this particular component we're creating."
    # "name" is a piece of information we have to hand it when we make one.
    def __init__(self, name):
        # Take the name we were given and store it inside this component,
        # so we (or other code) can look it up later as component.name
        self.name = name

    # This function will later calculate "how fast are this component's
    # internal values changing right now" -- the actual physics math.
    # t = the current moment in time.
    # local_state = this component's own current internal values.
    # inputs = whatever signals are flowing into this component right now.
    def derivatives(self, t, local_state, inputs):
        """Return d(local_state)/dt as an array of length n_states."""
        # The base blueprint doesn't know any real physics, so by default
        # it just says "nothing is changing" -- a list of zeros, one zero
        # for each state this component has.
        return np.zeros(self.n_states)

    # This function will later report "what signal is this component
    # currently putting out" -- e.g. a resistor reports a voltage, an
    # inductor reports a current.
    def output(self, t, local_state, inputs):
        """Return this element's output signal(s) given its state and inputs."""
        # "raise NotImplementedError" is Python's way of saying "whoever
        # actually uses a real component (not this generic base one) MUST
        # write their own version of this function -- this base version is
        # just a placeholder that refuses to run on its own."
        raise NotImplementedError


# Now we build a MORE SPECIFIC blueprint, "Resistor," using the generic one
# above as its starting point. "(CircuitElement)" means "Resistor is a type
# of CircuitElement, and starts out with everything CircuitElement has."
class Resistor(CircuitElement):
    """
    Translates: any plain resistor in the .asc files (R1, R5, R_sh, R_loop,
    all of them). Stateless -- just Ohm's law, no dynamics to get wrong.
    """

    # A resistor reacts instantly -- it has no memory of the past, so it
    # needs 0 states, same as the base default. Written here again just to
    # be explicit and easy to find.
    n_states = 0

    # The constructor for a resistor needs a name (handled by the parent
    # blueprint) AND a resistance value in ohms, specific to a resistor.
    def __init__(self, name, resistance):
        # "super().__init__(name)" means "go run the parent blueprint's
        # constructor first, using this same name" -- so self.name gets set
        # exactly like it does for any CircuitElement.
        super().__init__(name)
        # Now store this resistor's own specific value: its resistance.
        self.R = resistance

    # Ohm's law, rearranged to solve for current: if you know the voltage
    # across a resistor, current = voltage / resistance.
    def current(self, voltage):
        # Divide the given voltage by this resistor's own resistance value,
        # and hand back the answer.
        return voltage / self.R

    # Ohm's law the other way around: if you know the current THROUGH a
    # resistor, the voltage across it = current * resistance.
    def voltage(self, current):
        # Multiply the given current by this resistor's resistance value.
        return current * self.R


# Another specific blueprint: Inductor, also built from CircuitElement.
class Inductor(CircuitElement):
    """
    Translates: L1-L24, every plain (non-nTron) inductor in the .asc files.
    Linear: V = L*dI/dt. State = its own current.
    """

    # Unlike a resistor, an inductor DOES have memory: its own current is a
    # value that persists and changes gradually over time. So it needs
    # exactly 1 state variable to describe it.
    n_states = 1

    # Constructor: needs a name, an inductance value (in Henries), and
    # optionally a starting current (i0), which defaults to 0 if not given.
    def __init__(self, name, inductance, i0=0.0):
        # Run the parent constructor first, same as before.
        super().__init__(name)
        # Store this inductor's inductance value.
        self.L = inductance
        # Store the starting current (usually 0 unless told otherwise).
        self.i0 = i0

    # Calculate how fast this inductor's current is changing right now.
    def derivatives(self, t, local_state, inputs):
        # Pull the "voltage" value out of whatever inputs we were handed --
        # this is the voltage currently being applied across the inductor.
        voltage = inputs["voltage"]
        # The physics equation for an inductor: dI/dt = V / L
        # (how fast current changes = voltage divided by inductance).
        # We wrap the single number in np.array([...]) because the rest of
        # the code expects a list of derivatives, even if there's just one.
        return np.array([voltage / self.L])

    # Report this inductor's current output signal: simply its own current.
    def output(self, t, local_state, inputs):
        # local_state[0] is the first (and only) stored value for this
        # component -- its current. Just hand that back directly.
        return local_state[0]


# A composite (not fundamental) blueprint: LeakyLoop, combining the effect
# of a resistor and inductor together into one convenient package.
class LeakyLoop(CircuitElement):
    """
    NOT one of the core primitives -- a composite convenience class for the
    L-parallel-R_sh pattern that recurs constantly (L3/R5, L4/R7, etc).
    Provably just Resistor + Inductor + KCL (see the validation test below,
    which rebuilds this from scratch independently and confirms an exact
    match). Kept only because writing the same 3-line KCL derivation at
    every loop in the network gets repetitive.

    State = loop current I_L.  dI_L/dt = (i_drive - I_L) / tau, tau = L/R_sh
    """

    # Like an inductor, this loop has memory -- its own circulating current
    # -- so it needs exactly 1 state variable too.
    n_states = 1

    # Constructor: needs a name, the loop's inductance (L), and the shunt
    # resistor's value (R_sh). i0 is again an optional starting current.
    def __init__(self, name, L, R_sh, i0=0.0):
        # Run the parent constructor.
        super().__init__(name)
        # Store the inductance value.
        self.L = L
        # Store the shunt resistor value.
        self.R_sh = R_sh
        # Pre-calculate the "time constant" tau = L / R_sh. This single
        # number controls how fast this loop's current rises and decays --
        # bigger tau means slower, smaller tau means faster.
        self.tau = L / R_sh
        # Store the starting current.
        self.i0 = i0

    # Calculate how fast this loop's stored current is changing right now.
    def derivatives(self, t, local_state, inputs):
        # Pull out how much current is currently being driven INTO this
        # loop from whatever's upstream (e.g. an nTron that just fired).
        i_drive = inputs["i_drive"]
        # Pull out this loop's own current stored value.
        I_L = local_state[0]
        # The leaky-integrator equation: the loop's current moves toward
        # whatever's being driven in (i_drive), but "leaks" away over time
        # according to tau. This one line is the entire "memory that fades"
        # behavior we've discussed throughout this whole project.
        return np.array([(i_drive - I_L) / self.tau])

    # Report this loop's output: its own current.
    def output(self, t, local_state, inputs):
        # Same pattern as Inductor -- just hand back the stored value.
        return local_state[0]


# Another blueprint: TransmissionLine, which represents a delay.
class TransmissionLine(CircuitElement):
    """
    Translates: every T1-T10 in the .asc files, all identically Td=5n Z0=50.
    Pure delay: output(t) = input(t - Td). Not expressible as an ODE state,
    so instead of tracking state it just looks up the input's own value at
    a past time via a supplied history function.
    """

    # A transmission line doesn't need to remember anything internally --
    # it just reports "whatever the input looked like a few nanoseconds
    # ago." So it needs 0 states, same as a resistor.
    n_states = 0

    # Constructor: needs a name, a delay time (Td), and optionally a
    # characteristic impedance (Z0), which defaults to 50 ohms if not given.
    def __init__(self, name, Td, Z0=50.0):
        # Run the parent constructor.
        super().__init__(name)
        # Store the delay time.
        self.Td = Td
        # Store the impedance value.
        self.Z0 = Z0

    # Report this line's output at time t: whatever the input was Td
    # seconds earlier.
    def output(self, t, local_state, inputs):
        # Pull out the "history function" we were given -- this is a
        # separate piece of code (handed to us from outside) that knows
        # what the input signal looked like at any past moment.
        history_fn = inputs["history"]
        # If we're already past the delay time, ask the history function
        # for the value from Td seconds ago, and return that.
        # If we haven't even reached the delay time yet, nothing has had
        # time to arrive, so just return 0.
        return history_fn(t - self.Td) if t >= self.Td else 0.0


# A blueprint for a signal source -- not a real circuit component, just
# something to generate a test signal so we have something to feed in.
class PulseSource(CircuitElement):
    """
    Not a physical element -- a source. Reproduces LTspice's
    PULSE(V1 V2 Td Tr Tf Pw Per Ncycles) syntax parameter-for-parameter, so
    a call straight from a .asc file pastes in unchanged.
    """

    # A pulse source's output at any given time can be calculated directly
    # from its settings -- it doesn't need to remember anything either.
    n_states = 0

    # Constructor: needs a name, plus every parameter from LTspice's PULSE
    # syntax -- starting voltage (V1), peak voltage (V2), delay before it
    # starts (Td), rise time (Tr), fall time (Tf), how long it holds at
    # peak (Pw), how often it repeats (Per), and how many times (Ncycles,
    # defaulting to 1 if not given).
    def __init__(self, name, V1, V2, Td, Tr, Tf, Pw, Per, Ncycles=1):
        # Run the parent constructor.
        super().__init__(name)
        # Store the starting and peak voltages.
        self.V1, self.V2 = V1, V2
        # Store all the timing parameters together in one line.
        self.Td, self.Tr, self.Tf, self.Pw, self.Per = Td, Tr, Tf, Pw, Per
        # Store how many times this pulse repeats.
        self.Ncycles = Ncycles

    # Calculate exactly what voltage this source is putting out at any
    # given moment t -- this is the actual shape of the pulse.
    def value(self, t):
        # Before the delay time has even passed, nothing has started yet,
        # so the output is just the resting voltage V1.
        if t < self.Td:
            return self.V1

        # If this pulse is only supposed to fire once (the usual case
        # here):
        if self.Ncycles == 1:
            # Figure out how much time has passed SINCE the pulse started
            # (i.e. since Td).
            tp = t - self.Td
            # If we're past the entire rise+hold+fall duration, the pulse
            # is over -- back to resting voltage V1.
            if tp > self.Tr + self.Pw + self.Tf:
                return self.V1
        else:
            # If it repeats, figure out where we are within the CURRENT
            # repetition using the % (remainder/modulo) operator -- this
            # wraps time around every "Per" seconds.
            tp = (t - self.Td) % self.Per

        # Now figure out which phase of the pulse we're in, based on tp
        # (time since this pulse cycle began):

        # Phase 1: still rising from V1 toward V2.
        if tp < self.Tr:
            # Linearly interpolate between V1 and V2 based on how far
            # through the rise time we are (a fraction from 0 to 1).
            return self.V1 + (self.V2 - self.V1) * (tp / self.Tr)
        # Phase 2: holding steady at the peak voltage V2.
        elif tp < self.Tr + self.Pw:
            return self.V2
        # Phase 3: falling back down from V2 toward V1.
        elif tp < self.Tr + self.Pw + self.Tf:
            # Same linear interpolation idea, but going down instead of up.
            return self.V2 - (self.V2 - self.V1) * ((tp - self.Tr - self.Pw) / self.Tf)
        # Phase 4: pulse is completely finished, resting at V1 again.
        else:
            return self.V1

    # The standard "output" function every component has -- for a pulse
    # source, this is just whatever value(t) calculates.
    def output(self, t, local_state, inputs):
        return self.value(t)


# The most complex blueprint: NTron, our simplified stand-in for the real
# superconducting switch.
class NTron(CircuitElement):
    """
    STAGE 1 PLACEHOLDER -- a simplified stand-in for ntron_r (ntron_2.lib),
    not yet the full electrothermal model. What's real: Isw_g = Jc*width_g
    *thickness is the exact ntron_2.lib formula (~15.2uA with the repo's
    default parameters). What's simplified: the real device has three
    separate branches (gate/source/drain) each with their own hotspot-
    growth ODE; this collapses all of that into one relaxation toward 0 or
    1. The hysteresis (Isw_g to turn on, a LOWER Ihold_g to turn back off)
    isn't arbitrary -- it mirrors the real device's Ihs = sqrt(2/psi)*Isw
    being genuinely lower than the nucleation threshold, and fixes a real
    bug: without it, this placeholder flickered on/off within femtoseconds
    instead of staying on long enough for anything downstream to respond.

    State = switching_state s, in [0, 1].
        ds/dt = (target - s) / tau_switch
        target = 1 if (off AND |gate_current| > Isw_g)
                    or (on  AND |gate_current| > Ihold_g)
                 else 0
    """

    # An nTron has one piece of memory: how "switched on" it currently is,
    # a single number between 0 (fully off) and 1 (fully on).
    n_states = 1

    # Constructor: needs a name, plus the physical parameters that
    # determine its switching threshold (Jc, width_g, thickness, and an
    # optional constriction factor C), plus two behavior-tuning knobs:
    # tau_switch (how fast it switches) and hold_fraction (how much lower
    # the "turn back off" threshold is compared to the "turn on" one).
    def __init__(self, name, Jc, width_g, thickness, C=1.0,
                 tau_switch=0.15e-9, hold_fraction=0.3):
        # Run the parent constructor.
        super().__init__(name)
        # Calculate the critical (turn-on) current using the real formula:
        # critical current density times width times thickness (times an
        # optional correction factor C).
        self.Isw_g = Jc * width_g * thickness * C
        # Calculate the lower "stay on" threshold: some fraction (by
        # default 30%) of the turn-on threshold.
        self.Ihold_g = hold_fraction * self.Isw_g
        # Store how fast the switching itself happens.
        self.tau_switch = tau_switch

    # Calculate how fast this nTron's switching state is changing right now.
    def derivatives(self, t, local_state, inputs):
        # Pull out this nTron's current switching state (0 to 1).
        s = local_state[0]
        # Pull out the current currently arriving at this nTron's gate,
        # and take its absolute value (so it doesn't matter if it's
        # flowing in a positive or negative direction).
        gate_current = abs(inputs["gate_current"])
        # Check whether this nTron currently counts as "on" (using 0.5 as
        # the halfway dividing line).
        currently_on = s > 0.5
        # Decide WHICH threshold applies right now: if it's already on, use
        # the lower "stay on" threshold; if it's off, use the higher
        # "turn on" threshold. This is the hysteresis behavior.
        threshold = self.Ihold_g if currently_on else self.Isw_g
        # Decide where this nTron is "trying" to go: fully on (1) if the
        # gate current beats whichever threshold applies, otherwise fully
        # off (0).
        target = 1.0 if gate_current > threshold else 0.0
        # The actual equation: move the switching state toward the target,
        # at a speed controlled by tau_switch (faster tau_switch = faster
        # response). This is the same "leaky" style of equation as
        # LeakyLoop above, just applied to a 0-to-1 switching state instead
        # of a current.
        return np.array([(target - s) / self.tau_switch])

    # A small helper function to read out the switching state by name,
    # rather than remembering "it's local_state[0]" everywhere else.
    def switching_state(self, local_state):
        return local_state[0]


# =============================================================================
# PART B -- VALIDATION TESTS. Each element checked in isolation against a
# known analytical solution (or, for LeakyLoop, an independent from-scratch
# derivation) -- never against another piece of this same file.
# =============================================================================

# A small helper function: given a text label and a True/False condition,
# print either "PASS" or a loud "FAIL" flag, plus the label, so results are
# easy to scan.
def check(label, condition):
    print(f"  [{'PASS' if condition else 'FAIL  <-- CHECK THIS'}] {label}")


# The main function that runs every test, one after another.
def run_all_tests():
    # Print a header line for the first test section.
    print("=" * 70)
    print("1. RESISTOR -- Ohm's law")
    print("=" * 70)
    # Create one actual Resistor object to test, with a resistance of 50 ohms.
    R = Resistor("R_test", resistance=50.0)
    # Try a few different voltage values, each paired with the current we
    # KNOW (by hand-calculation) should come out.
    for V, I_expected in [(1.0, 0.02), (5.0, 0.1), (0.25, 0.005)]:
        # Ask our Resistor object to actually compute the current.
        I_computed = R.current(V)
        # Check whether the computed answer matches the expected answer
        # (np.isclose allows for tiny floating-point rounding differences).
        check(f"V={V}V -> I={I_computed:.5f}A (expected {I_expected:.5f}A)",
              np.isclose(I_computed, I_expected))

    # Blank line and header for the next test section.
    print()
    print("=" * 70)
    print("2. INDUCTOR -- constant-V current ramp vs analytical I(t)=Vt/L")
    print("=" * 70)
    # Set up test values: a 5 nanohenry inductor, driven with a constant
    # 0.1 volt.
    L_val, V_applied = 5e-9, 0.1
    # Create the actual Inductor object we're testing.
    L_elem = Inductor("L_test", inductance=L_val)

    # Define a small function representing "the physics equation to solve"
    # -- this is what solve_ivp will call over and over as it simulates
    # forward in time.
    def rhs_inductor(t, y):
        # Ask our Inductor object for its derivative, given that a
        # constant voltage V_applied is being applied to it.
        return L_elem.derivatives(t, y, {"voltage": V_applied})

    # Simulate from time 0 to 2 nanoseconds.
    t_span = (0, 2e-9)
    # Actually run the simulation: solve_ivp numerically integrates the
    # rhs_inductor equation forward in time, starting from current=0.0,
    # and reports the answer at 200 evenly-spaced points across t_span.
    sol_L = solve_ivp(rhs_inductor, t_span, [0.0], max_step=1e-12,
                       t_eval=np.linspace(*t_span, 200))
    # Pull out the simulated (numerical) current values over time.
    I_numerical = sol_L.y[0]
    # Calculate what the current SHOULD be at each of those same time
    # points, using the known, hand-derived formula I(t) = V*t/L.
    I_analytical = V_applied * sol_L.t / L_val
    # Compare the two: find the biggest relative difference anywhere
    # between the simulated answer and the known-correct answer.
    max_err = np.max(np.abs(I_numerical - I_analytical) / np.max(I_analytical))
    # Check that this biggest difference is extremely small (less than
    # 0.01%) -- confirming the Inductor class behaves correctly.
    check(f"max relative error vs analytical ramp = {max_err:.2e}", max_err < 1e-4)

    print()
    print("=" * 70)
    print("3. TRANSMISSION LINE -- output(t) must equal input(t - Td) exactly")
    print("=" * 70)
    # Set up a 5 nanosecond delay for this test.
    Td = 5e-9
    # Create the actual TransmissionLine object being tested.
    TL = TransmissionLine("T_test", Td=Td)

    # Define a simple test signal: a sine wave, but only starting at t=0
    # (before that, it's just 0).
    def test_input(t):
        return np.sin(2 * np.pi * 1e9 * t) if t >= 0 else 0.0

    # Try several different probe times, and at each one, confirm the
    # transmission line's output equals the input signal from Td seconds
    # earlier.
    for t_probe in [6e-9, 7.3e-9, 10e-9, 15e-9]:
        # Ask the TransmissionLine what it's outputting at this probe time.
        out = TL.output(t_probe, None, {"history": test_input})
        # Independently calculate what the input looked like Td seconds
        # before this probe time.
        expected = test_input(t_probe - Td)
        # Confirm they match.
        check(f"t={t_probe*1e9:.1f}ns: output={out:.4f}, expected={expected:.4f}",
              np.isclose(out, expected))
    # Also confirm that before the delay time has even passed, the output
    # is exactly zero (nothing has had time to arrive yet).
    check("exactly 0 before t=Td",
          TL.output(2e-9, None, {"history": test_input}) == 0.0)

    print()
    print("=" * 70)
    print("4. NTRON -- threshold value, ON-transition shape, retrap hysteresis")
    print("=" * 70)
    # Set up the real physical parameters from the SPICE file's default
    # values: critical current density, gate width, and film thickness.
    Jc, width_g, thickness = 40e9, 20e-9, 19e-9
    # Create the actual NTron object being tested, using those parameters.
    nt = NTron("nt_test", Jc=Jc, width_g=width_g, thickness=thickness,
               tau_switch=0.15e-9, hold_fraction=0.3)
    # Print out the calculated threshold values so we can eyeball them.
    print(f"  Isw_g   = {nt.Isw_g*1e6:.3f} uA  (hand-derived value: 15.2 uA)")
    print(f"  Ihold_g = {nt.Ihold_g*1e6:.3f} uA  (0.3 x Isw_g, the retrap floor)")
    # Confirm the calculated turn-on threshold matches our hand-derived
    # 15.2 microamp value from earlier in this project.
    check("Isw_g matches hand-derived value", np.isclose(nt.Isw_g, 15.2e-6, rtol=1e-3))

    # Test 4a: does the nTron correctly stay OFF when given a gate current
    # that's below both thresholds the whole time?
    def rhs_off(t, y):
        # Always supply a small 5 microamp gate current -- below threshold.
        return nt.derivatives(t, y, {"gate_current": 5e-6})
    # Simulate this for 5 nanoseconds.
    sol_off = solve_ivp(rhs_off, (0, 5e-9), [0.0], max_step=1e-12)
    # Confirm the switching state stayed near 0 (essentially off) the
    # whole time.
    check(f"stays OFF under constant 5uA: final s={sol_off.y[0][-1]:.4f}",
          sol_off.y[0][-1] < 0.01)

    # Test 4b: when given a gate current well above threshold, does the
    # ON-transition follow the exact mathematical curve we expect?
    def rhs_on(t, y):
        # Supply a strong 60 microamp gate current -- well above threshold.
        return nt.derivatives(t, y, {"gate_current": 60e-6})
    # Set up 300 evenly spaced time points across 1 nanosecond.
    t_eval = np.linspace(0, 1e-9, 300)
    # Run the simulation.
    sol_on = solve_ivp(rhs_on, (0, 1e-9), [0.0], max_step=1e-13, t_eval=t_eval)
    # Pull out the simulated switching-state values over time.
    s_numerical = sol_on.y[0]
    # Calculate the KNOWN, textbook-correct shape for this kind of
    # "relax toward a target" equation: 1 minus a decaying exponential.
    s_analytical = 1 - np.exp(-t_eval / nt.tau_switch)
    # Find the single biggest difference anywhere between our simulation
    # and the known-correct curve.
    max_err_switch = np.max(np.abs(s_numerical - s_analytical))
    # Confirm that biggest difference is tiny.
    check(f"ON-transition matches analytical 1-exp(-t/tau), max err = {max_err_switch:.2e}",
          max_err_switch < 1e-3)

    # Calculate a current level exactly halfway between the two
    # thresholds -- this is the critical test case for hysteresis.
    mid_current = (nt.Ihold_g + nt.Isw_g) / 2

    # Test 4c: trigger the nTron on, THEN drop the gate current down to
    # that halfway level, and confirm it STAYS on (doesn't accidentally
    # turn back off).
    def rhs_hysteresis(t, y):
        # For the first 0.6 nanoseconds, use a strong triggering current.
        # After that, drop down to the halfway "mid_current" level.
        gc = 60e-6 if t < 0.6e-9 else mid_current
        return nt.derivatives(t, y, {"gate_current": gc})
    # Run this simulation for 3 nanoseconds.
    sol_hyst = solve_ivp(rhs_hysteresis, (0, 3e-9), [0.0], max_step=1e-13)
    # Confirm that by the end, the switching state is still close to 1
    # (still on) -- proving the hysteresis behavior works.
    check(f"stays ON at mid-level current ({mid_current*1e6:.1f}uA): "
          f"final s={sol_hyst.y[0][-1]:.4f}", sol_hyst.y[0][-1] > 0.9)

    # Test 4d: trigger the nTron on, THEN drop the gate current all the
    # way below even the lower threshold, and confirm it DOES turn off.
    def rhs_retrap(t, y):
        # Same trigger, but this time drop to a very low 2 microamp level
        # -- below the "stay on" threshold.
        gc = 60e-6 if t < 0.6e-9 else 2e-6
        return nt.derivatives(t, y, {"gate_current": gc})
    # Run this simulation.
    sol_retrap = solve_ivp(rhs_retrap, (0, 3e-9), [0.0], max_step=1e-13)
    # Confirm that by the end, the switching state has dropped back near 0.
    check(f"turns back OFF below Ihold_g: final s={sol_retrap.y[0][-1]:.4f}",
          sol_retrap.y[0][-1] < 0.05)

    print()
    print("=" * 70)
    print("5. LEAKYLOOP vs a from-scratch Inductor+Resistor+KCL composition")
    print("=" * 70)
    # Set up test values: a 5 nanohenry inductor, 3 ohm resistor, and a
    # constant 10 microamp drive current.
    L_val2, R_val2, i_drive_const = 5e-9, 3.0, 10e-6
    # Create the actual LeakyLoop object being tested.
    loop = LeakyLoop("loop_test", L=L_val2, R_sh=R_val2)

    # Define the equation using LeakyLoop's own built-in formula.
    def rhs_loop(t, y):
        return loop.derivatives(t, y, {"i_drive": i_drive_const})

    # Now build the SAME physical situation completely independently, using
    # only a raw Inductor object and a raw Resistor object, wired together
    # by hand using basic circuit-law reasoning (KCL = "Kirchhoff's Current
    # Law," the rule that current in must equal current out at any point).
    L_elem2 = Inductor("L_manual", inductance=L_val2)
    R_elem2 = Resistor("R_manual", resistance=R_val2)

    # Define the equation for this hand-built version.
    def rhs_manual(t, y):
        # Read out the inductor's current so far.
        I_L = y[0]
        # By Kirchhoff's Current Law: whatever current isn't going through
        # the inductor must be going through the resistor instead.
        I_R = i_drive_const - I_L
        # Use the Resistor object itself to calculate the voltage that
        # current produces (V = I * R).
        V = R_elem2.voltage(I_R)
        # Use the Inductor object itself to calculate how fast its current
        # is changing, given that voltage.
        return [L_elem2.derivatives(t, [I_L], {"voltage": V})[0]]

    # Set up 200 evenly spaced time points across 10 nanoseconds.
    t_eval2 = np.linspace(0, 10e-9, 200)
    # Run BOTH simulations -- the LeakyLoop shortcut version, and the
    # hand-built manual version -- across the exact same time points.
    sol_loop = solve_ivp(rhs_loop, (0, 10e-9), [0.0], t_eval=t_eval2, max_step=1e-12)
    sol_manual = solve_ivp(rhs_manual, (0, 10e-9), [0.0], t_eval=t_eval2, max_step=1e-12)
    # Find the single biggest difference anywhere between the two results.
    max_diff = np.max(np.abs(sol_loop.y[0] - sol_manual.y[0]))
    # Confirm that difference is essentially nothing (just floating-point
    # rounding noise) -- proving LeakyLoop is truly just Resistor +
    # Inductor + basic circuit law, nothing extra hidden inside it.
    check(f"LeakyLoop vs manual composition, max diff = {max_diff:.3e} A",
          max_diff < 1e-9)

    # Hand back everything the plotting function will need, bundled into
    # a dictionary (a labeled collection of results).
    return {
        "inductor": (sol_L, I_analytical),
        "tline": (TL, test_input),
        "ntron_on": sol_on,
        "leakyloop": (sol_loop, sol_manual),
    }


# A function that takes the test results and draws four graphs from them.
def make_plots(results):
    # Unpack the inductor test results back out of the dictionary.
    sol_L, I_analytical = results["inductor"]
    # Unpack the transmission line test results.
    TL, test_input = results["tline"]
    # Unpack the nTron ON-transition results.
    sol_on = results["ntron_on"]
    # Unpack the leaky loop comparison results.
    sol_loop, sol_manual = results["leakyloop"]

    # Create a figure containing a 2x2 grid of individual plots (axes),
    # sized 11 by 7 inches overall.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # --- Top-left plot: Inductor ---
    # Plot the simulated current (converted to milliamps) over time (in
    # nanoseconds), as a solid line.
    axes[0, 0].plot(sol_L.t * 1e9, sol_L.y[0] * 1e3, label="numerical", lw=2)
    # Plot the analytically-known correct answer as a dashed line on top.
    axes[0, 0].plot(sol_L.t * 1e9, I_analytical * 1e3, "--", label="analytical")
    # Give this subplot a title.
    axes[0, 0].set_title("Inductor: constant-V current ramp")
    # Label the horizontal and vertical axes, and show the legend box.
    axes[0, 0].set_xlabel("ns"); axes[0, 0].set_ylabel("mA"); axes[0, 0].legend()

    # --- Top-right plot: TransmissionLine ---
    # Create 400 evenly-spaced time points across 20 nanoseconds, purely
    # for plotting a smooth curve.
    ts = np.linspace(0, 20e-9, 400)
    # Plot the original input signal.
    axes[0, 1].plot(ts * 1e9, [test_input(t) for t in ts], label="input(t)")
    # Plot the transmission line's delayed output signal.
    axes[0, 1].plot(ts * 1e9, [TL.output(t, None, {"history": test_input}) for t in ts],
                     label="output(t) = input(t-Td)")
    axes[0, 1].set_title("TransmissionLine: pure delay")
    axes[0, 1].set_xlabel("ns"); axes[0, 1].legend()

    # --- Bottom-left plot: NTron switching ---
    # Recalculate the analytical curve here too, purely for plotting.
    s_analytical = 1 - np.exp(-sol_on.t / 0.15e-9)
    # Plot the simulated switching state.
    axes[1, 0].plot(sol_on.t * 1e9, sol_on.y[0], label="numerical")
    # Plot the known-correct curve on top as a dashed line.
    axes[1, 0].plot(sol_on.t * 1e9, s_analytical, "--", label="analytical 1-exp(-t/tau)")
    axes[1, 0].set_title("NTron: ON-transition shape")
    axes[1, 0].set_xlabel("ns"); axes[1, 0].set_ylabel("switching state"); axes[1, 0].legend()

    # --- Bottom-right plot: LeakyLoop vs manual ---
    # Plot the LeakyLoop shortcut's result as a thick, semi-transparent line.
    axes[1, 1].plot(sol_loop.t * 1e9, sol_loop.y[0] * 1e6, label="LeakyLoop", lw=3, alpha=0.6)
    # Plot the manually-composed result as a dashed line on top -- if the
    # two classes truly match, this dashed line should sit exactly on top
    # of the thick line underneath it.
    axes[1, 1].plot(sol_manual.t * 1e9, sol_manual.y[0] * 1e6, "--", label="manual Inductor+Resistor")
    axes[1, 1].set_title("LeakyLoop vs manual composition")
    axes[1, 1].set_xlabel("ns"); axes[1, 1].set_ylabel("uA"); axes[1, 1].legend()

    # Automatically adjust spacing so subplot titles/labels don't overlap.
    plt.tight_layout()
    # Save the entire figure to a PNG image file, at a decent resolution.
    plt.savefig("element_validation.png", dpi=140)
    # Print a confirmation message so we know it worked.
    print("\nSaved plots to element_validation.png")


# This special check means "only run the code below if this file is being
# run directly (not if it's being imported by some other file)." It's the
# standard way Python scripts mark their own starting point.
if __name__ == "__main__":
    # Run every validation test, and capture the results.
    results = run_all_tests()
    # Use those results to draw and save the four graphs.
    make_plots(results)