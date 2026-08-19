"""Proves the M1 acceptance criteria without a GPU, camera, or weights."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.state_machine import ViolationStateMachine

VTYPE = "NO-Hardhat"
TID = 7
STEP = 1 / 15

DEFAULTS = dict(
    persist_seconds=2.0,
    clear_seconds=1.5,
    track_lost_seconds=1.0,
    cooldown_seconds=120.0,
    max_incidents_per_minute=10,
)


def machine(**overrides) -> ViolationStateMachine:
    try:
        from config import ViolationConfig

        cfg = ViolationConfig(**{**DEFAULTS, **overrides})
    except ModuleNotFoundError:
        from types import SimpleNamespace

        cfg = SimpleNamespace(**{**DEFAULTS, **overrides})
    return ViolationStateMachine(cfg)


def run(m: ViolationStateMachine, seconds: float, start: float, violating: bool, seen: bool = True):
    """Feed frames at 15fps; returns (events, end_time)."""
    events, now = [], start
    for _ in range(int(seconds / STEP)):
        now += STEP
        active = {TID: {VTYPE: 0.91}} if violating else {}
        events += m.update(now, active, {TID} if seen else set())
    return events, now


def test_under_two_seconds_does_not_fire():
    m = machine()
    events, _ = run(m, 1.8, 0.0, violating=True)
    assert events == [], f"expected no event under persist_seconds, got {events}"


def test_walk_past_fires_exactly_once():
    m = machine()
    events, _ = run(m, 3.0, 0.0, violating=True)
    assert len(events) == 1, f"expected exactly 1 event, got {len(events)}"
    assert events[0].violation_type == VTYPE
    assert events[0].track_id == TID
    assert events[0].duration_seconds >= 2.0


def test_standing_still_sixty_seconds_fires_once_total():
    """Acceptance: standing non-compliant for 60s must not spam — the 120s cooldown holds."""
    m = machine()
    events, _ = run(m, 60.0, 0.0, violating=True)
    assert len(events) == 1, f"cooldown breached: {len(events)} events in 60s"
    assert m.stats.events_suppressed_cooldown > 0


def test_refires_after_cooldown_expires():
    m = machine(cooldown_seconds=10.0)
    first, t = run(m, 3.0, 0.0, violating=True)
    second, _ = run(m, 12.0, t, violating=True)
    assert len(first) == 1 and len(second) == 1, f"{len(first)} then {len(second)}"


def test_clean_for_clear_seconds_resets_to_idle():
    m = machine()
    _, t = run(m, 1.0, 0.0, violating=True)          # ARMING, not yet fired
    _, t = run(m, 2.0, t, violating=False)           # clean past clear_seconds -> IDLE
    events, _ = run(m, 1.8, t, violating=True)       # under persist again
    assert events == [], "timer did not reset after the clear window"


def test_track_lost_while_arming_discards_timer():
    m = machine()
    _, t = run(m, 1.5, 0.0, violating=True)
    _, t = run(m, 1.5, t, violating=False, seen=False)
    assert m.stats.arming_discarded_track_lost == 1
    events, _ = run(m, 1.8, t, violating=True)
    assert events == [], "re-appearing track should restart the 2s timer"


def test_circuit_breaker_caps_emissions():
    m = machine(cooldown_seconds=0.0, max_incidents_per_minute=3)
    events, now = [], 0.0
    for tid in range(20):
        for _ in range(int(3.0 / STEP)):
            now += STEP
            events += m.update(now, {tid: {VTYPE: 0.9}}, {tid})
    assert len(events) == 3, f"breaker allowed {len(events)} events, cap was 3"
    assert m.stats.events_suppressed_breaker > 0


def test_elapsed_reports_hold_time():
    m = machine()
    _, t = run(m, 1.0, 0.0, violating=True)
    assert 0.8 <= m.elapsed(t, TID, VTYPE) <= 1.1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
