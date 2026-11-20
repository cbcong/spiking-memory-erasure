"""Synchronous full-state box reachability, partitioned by spike pattern."""

from itertools import product

from .model import trunc_div


def start_boxes(spec):
    """Cover every legal post-update state, keeping reset and spike correlated."""
    p = spec.params
    size = 2 * spec.module_count
    silent = (-p.qv, p.threshold - 1, 0, p.qi)
    fired = (0, 0, 0, p.qi)
    return {
        spikes: tuple(fired if spike else silent for spike in spikes)
        for spikes in product((0, 1), repeat=size)
    }


def box_step(spec, boxes, amplitude):
    """Map each box under the exact spike-conditioned affine integer update."""
    p = spec.params
    vden = 1 << p.alpha_shift
    iden = 1 << p.beta_shift
    successors = {}
    for spikes, bounds in boxes.items():
        choices = []
        for cell, (v_lo, v_hi, i_lo, i_hi) in enumerate(bounds):
            side = cell % 2
            excitation = sum(w * spikes[2 * source + side]
                             for source, w in enumerate(spec.weights[cell // 2]))
            offset = excitation - p.inhibition * spikes[cell ^ 1] - amplitude
            recurrent = p.weight * spikes[cell]
            i_lo = min(p.qi, (p.beta_num * i_lo) // iden + recurrent)
            i_hi = min(p.qi, (p.beta_num * i_hi) // iden + recurrent)
            dec_lo = trunc_div(p.alpha_num * v_lo, vden) + offset
            dec_hi = trunc_div(p.alpha_num * v_hi, vden) + offset
            branches = []
            if dec_lo + i_lo < p.threshold:
                silent_i_hi = min(i_hi, p.threshold - 1 - dec_lo)
                new_lo = max(-p.qv, dec_lo + i_lo)
                new_hi = max(-p.qv, min(p.threshold - 1, dec_hi + silent_i_hi))
                branches.append((0, (new_lo, new_hi, i_lo, silent_i_hi)))
            if dec_hi + i_hi >= p.threshold:
                fired_i_lo = max(i_lo, p.threshold - dec_hi)
                branches.append((1, (0, 0, fired_i_lo, i_hi)))
            choices.append(branches)
        for combination in product(*choices):
            pattern, next_box = zip(*combination)
            previous = successors.get(pattern)
            if previous is not None:
                next_box = tuple((min(a[0], b[0]), max(a[1], b[1]),
                                  min(a[2], b[2]), max(a[3], b[3]))
                                 for a, b in zip(previous, next_box))
            successors[pattern] = next_box
    return successors


def box_erasure(spec, deadline):
    """Return (E,D,H) with the lowest amplitude proved by this abstraction."""
    if type(deadline) is not int or deadline < 0:
        raise ValueError("deadline must be a nonnegative integer")
    p = spec.params
    initial = start_boxes(spec)
    zero_pattern = (0,) * (2 * spec.module_count)
    zero = {zero_pattern: ((0, 0, 0, 0),) * len(zero_pattern)}

    # At this amplitude every held update saturates voltage to -qv.
    # Higher amplitudes therefore produce the same trajectories.
    amplitude_max = (p.alpha_num * p.qv // (1 << p.alpha_shift)
                     + p.qi + max(map(sum, spec.weights)) + p.qv)
    for amplitude in range(amplitude_max + 1):
        pressure = initial
        best = None
        for duration in range(deadline + 1):
            released = pressure
            recovery = 0
            remaining = deadline - duration if best is None else best[0] - duration
            while released != zero and recovery < remaining:
                successor = box_step(spec, released, 0)
                if successor == released:
                    break
                released = successor
                recovery += 1
            if released == zero:
                candidate = duration + recovery, duration, recovery
                best = candidate if best is None else min(best, candidate)
            if duration == deadline or (best is not None and duration >= best[0]):
                break
            pressure = box_step(spec, pressure, amplitude)
        if best is not None:
            return amplitude, best[1], best[2]
    return None
