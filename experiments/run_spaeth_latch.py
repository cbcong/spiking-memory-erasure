"""Reconstruct Spaeth et al. S1 cells 3, 4 and 8 and exercise the nominal latch."""
import argparse
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import scipy
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "literature/spaeth_2020_s1.ipynb"
NOTEBOOK_HASH = "9301afd8b60a16bd8d4ce549532327cc9de9b2709f01ffb447a862b6f24280fb"
VR = np.array([-60., -60., -56.])
VP = np.array([35., 35., 20.])
TAU = np.array([5., 5., 20.])
C = 100.
VT = np.array([-40., -40., -42.])
K = np.array([.7, .7, 1.])
B = np.array([-2., -2., 8.])
VN = np.array([0., 0., -70.])
RESET_V = np.array([-50., -50., -53.])
JUMP_U = np.array([100., 100., 20.])
PULSE_MS = 2.
PULSE_PA = 5000.


def resting():
    return np.r_[VR, np.zeros(9)]


def jump(state, cell):
    state[cell] = RESET_V[cell]
    state[3+cell] += JUMP_U[cell]
    state[9+cell] += 1.


def threshold(cell):
    def event(t, state):
        return state[cell] - VP[cell]
    event.terminal = True
    event.direction = 1
    return event


EVENTS = tuple(threshold(cell) for cell in range(3))


def evolve(state, start, stop, conductance, events, drive=None, tol=1e-8,
           pulse_pa=PULSE_PA, isolated=False):
    g = np.array([[0.,20.,10.],[20.,0.,10.],[conductance,conductance,0.]])
    if isolated:
        g.fill(0)
    def rhs(t, s):
        v,u,x,y = s.reshape(4,3)
        current = g @ (x*VN) - (g @ x)*v
        if drive is not None:
            current[drive] += pulse_pa
        return np.concatenate(((K*(v-VR)*(v-VT)-u+current)/C,
                               .03*(B*(v-VR)-u), y/TAU, -(x+2*y)/TAU))
    t = start
    while t < stop:
        sol = solve_ivp(rhs, (t, stop), state, method="DOP853", rtol=tol,
                        atol=tol/100, max_step=2., events=EVENTS)
        if not sol.success:
            raise RuntimeError(sol.message)
        t = float(sol.t[-1])
        state = sol.y[:, -1].copy()
        if sol.status == 1:
            hits = set(i for i, times in enumerate(sol.t_events) if len(times))
            hits.update(np.flatnonzero(state[:3] >= VP-1e-9))
            for i in sorted(hits):
                jump(state, i)
                events.append((t, int(i)))
    return state


def count(events, lo, hi, cells=(0,1,2)):
    return sum(lo <= t < hi and i in cells for t,i in events)


def figure_archive():
    payload = NOTEBOOK.read_bytes()
    if sha256(payload).hexdigest() != NOTEBOOK_HASH:
        raise ValueError("Unexpected publisher notebook")
    img = json.loads(payload)["cells"][8]["outputs"][0]["data"]["image/png"]
    pixels = np.asarray(Image.open(BytesIO(base64.b64decode(img))))
    # One horizontal cut near the spike tops in the upper panel, above the
    # subthreshold traces. Ignore the imposed E1 spike at time zero (off plot).
    peaks = []
    for rgb in [(31,119,180),(255,127,14),(44,160,44)]:
        cols = np.flatnonzero((pixels[200,:,:3] == rgb).all(axis=1))
        runs = np.split(cols, np.flatnonzero(np.diff(cols)>1)+1)
        peaks.append([int(np.median(run)) for run in runs if len(run)])
    return peaks


def figure(tol):
    state = resting()
    jump(state,0)
    events = []
    end = evolve(state,0.,1000.,7.0747,events,tol=tol)
    return dict(initial_imposed_E1_excluded=True, events_ms=events,
                spikes_through_150=[count(events,0.,150.,(i,)) for i in range(3)],
                spikes_after_150=count(events,150.,1000.),
                final_max_abs_voltage_deviation_from_rest_mV=float(np.max(abs(end[:3]-VR))))


def trial(index, reset=True, tol=1e-8, amplitudes=None, direct=False):
    # New task interface: rectangular current to E1 for writes, to I for
    # resets. S1 drives Figure 2 by a prescribed E1 jump instead.
    events = []
    end_hold = 110.+3*index
    rewrite = end_hold+150.
    finish = rewrite+110.
    stages = [(0.,PULSE_MS,0),(PULSE_MS,end_hold,None),
              (end_hold,end_hold+PULSE_MS,2 if reset else None),
              (end_hold+PULSE_MS,rewrite,None),(rewrite,rewrite+PULSE_MS,0),
              (rewrite+PULSE_MS,finish,None)]
    state = resting()
    for t0,t1,drive in stages:
        current = PULSE_PA if amplitudes is None or drive is None else amplitudes[drive]
        if drive is not None and direct:
            jump(state,drive)
            events.append((t0,drive))
            drive=None
        state=evolve(state,t0,t1,5.,events,drive,tol,current)
    exc=(0,1)
    pre=count(events,end_hold-50.,end_hold,exc)
    quiet=count(events,rewrite-50.,rewrite,exc)
    after=count(events,finish-50.,finish,exc)
    signatures=[[count(events,a,b,(i,)) for i in range(3)]
                for a,b in [(0.,end_hold),(end_hold,rewrite),(rewrite,finish)]]
    late=[t-end_hold for t,i in events if i in exc and end_hold<=t<rewrite]
    return dict(id=index,hold_ms=end_hold,reset_present=reset,window_exc_spikes=[pre,quiet,after],
                success=pre>0 and quiet==0 and after>0,phase_spikes_by_neuron=signatures,
                reset_to_last_exc_spike_ms=max(late,default=0.),events_ms=events)


def calibrate():
    amplitudes = {}
    records = []
    for cell in (0,2):
        for magnitude in range(250,5001,250):
            events=[]
            state=evolve(resting(),0.,PULSE_MS,5.,events,cell,pulse_pa=magnitude,isolated=True)
            evolve(state,PULSE_MS,50.,5.,events,isolated=True)
            records.append(dict(cell=cell,pA=magnitude,events_ms=events))
            if len(events)==1 and events[0][0] <= 10.:
                amplitudes[cell]=magnitude
                break
        else:
            raise ValueError(f"no one-spike command for cell {cell}")
    return amplitudes, records


def run():
    pixels=figure_archive()
    fig=figure(1e-8)
    strict=figure(1e-10)
    amplitudes, calibration=calibrate()
    def arm(direct):
        rows=[trial(i,amplitudes=amplitudes,direct=direct) for i in range(16)]
        control=[trial(i,False,amplitudes=amplitudes,direct=direct) for i in range(16)]
        strict_rows=[trial(i,tol=1e-10,amplitudes=amplitudes,direct=direct) for i in (0,5,10,15)]
        return dict(rows=rows,sham_rows=control,strict_rows=strict_rows,
                    summary=dict(n=len(rows),passed=sum(r["success"] for r in rows),
                                 hold_passed=sum(r["window_exc_spikes"][0]>0 for r in rows),
                                 reset_passed=sum(r["window_exc_spikes"][1]==0 for r in rows),
                                 rewrite_passed=sum(r["window_exc_spikes"][2]>0 for r in rows),
                                 sham_reset_quiet=sum(r["window_exc_spikes"][1]==0 for r in control),
                                 tolerance_checks_agree=all(r["phase_spikes_by_neuron"]==rows[r["id"]]["phase_spikes_by_neuron"] for r in strict_rows)))
    current=arm(False)
    exact=arm(True)
    return dict(author_notebook_url="https://doi.org/10.1371/journal.pone.0240267.s001",
                source_sha256=NOTEBOOK_HASH, implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                scipy=scipy.__version__,solver="SciPy DOP853; author used Julia Tsit5",
                design=dict(native_current_pA=amplitudes,pulse_ms=PULSE_MS,
                            reset_observation_ms=150.,window_ms=50.,hold_ms="110+3*i",
                            task="active E1 or E2 in final hold window, quiet after reset, active after rewrite",
                            units="source-native ms and pA; rectangular inputs are experiment-defined"),
                fig2_author_png_voltage_peak_pixel_x=pixels,figure2=fig,figure2_strict=strict,
                figure2_count_matches_png=fig["spikes_through_150"]==list(map(len,pixels)),
                calibration_rule="smallest 250-pA grid amplitude eliciting exactly one spike in 50 ms, with latency <=10 ms on isolated cell at rest",
                calibration=calibration,calibrated_current=current,imposed_event=exact)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    path=parser.parse_args().output
    if path.exists():
        parser.error("output exists")
    result=run()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as stream:
        json.dump(result,stream,indent=2)
    print(result["design"],result["calibrated_current"]["summary"],result["imposed_event"]["summary"],result["figure2_count_matches_png"])
