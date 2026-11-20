"""Delayed selection among four encoded SHD utterances, not original SHD."""
from dataclasses import dataclass

from .delayed_query import QueryReady, advance


def encode_classes(classes):
    if not isinstance(classes, tuple) or len(classes) != 4:
        raise ValueError("task stores exactly four class estimates")
    if any(type(value) is not int or not 0 <= value < 20 for value in classes):
        raise ValueError("classes must be integer values 0..19")
    return tuple((value >> bit) & 1 for value in classes for bit in range(5))


@dataclass(frozen=True)
class ClassReadout:
    slot: int
    prediction: int | None
    counts: tuple[tuple[int, int], ...]
    next_states: tuple
    end_tick: int

    @property
    def phase_valid(self):
        return all(pair in ((1, 0), (0, 1)) for pair in self.counts)


def query_class_after_delay(ready: QueryReady, slot: int, *, reference=False, audit=None):
    if not isinstance(ready, QueryReady) or ready.spec.module_count != 20:
        raise ValueError("a class query requires a 20-module QueryReady")
    if type(slot) is not int or not 0 <= slot < 4:
        raise ValueError("query slot must be 0..3")
    states = ready.states
    counts = [[0, 0] for _ in range(5)]
    for _ in range(2):
        states = advance(ready.spec, states, 1, reference=reference, audit=audit)
        for bit in range(5):
            pulse = states[5 * slot + bit].spike
            for cell in (0, 1):
                counts[bit][cell] += pulse[cell]
    # Content uses the same permissive unique-winner rule for both policies.
    # A noncanonical count is a separate phase error, not automatically a bit error.
    bits = [0 if pair[0] > pair[1] else 1 if pair[1] > pair[0] else None for pair in counts]
    value = None if None in bits else sum(bit << i for i, bit in enumerate(bits))
    if value is not None and value >= 20:
        value = None
    return ClassReadout(slot, value, tuple(tuple(pair) for pair in counts),
                        states, ready.ready_tick + 2)
