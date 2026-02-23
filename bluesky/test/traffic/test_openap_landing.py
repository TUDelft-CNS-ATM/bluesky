import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_traffic_state(traffic_):
    traffic_.reset()
    yield
    traffic_.reset()


def get_openap_perf(traffic_):
    perf = traffic_.perf
    if perf._selected().__name__ != "OpenAP":
        pytest.skip("OpenAP performance model is not selected")
    return perf


def test_touchdown_sets_speed_target_to_zero(traffic_):
    traffic_.cre("LDG1", "A320", 52.0, 4.0, 180.0, 10.0, 70.0)
    idx = traffic_.id2idx("LDG1")
    perf = get_openap_perf(traffic_)

    traffic_.alt[idx] = 0.0
    traffic_.selspd[idx] = 70.0
    traffic_.swvnavspd[idx] = True
    perf.post_flight[idx] = True
    perf.pf_flag[idx] = True

    intent_v_tas = np.array([70.0])
    intent_vs = np.array([0.0])
    intent_h = np.array([0.0])
    ax = np.array([0.0])

    allow_v_tas, allow_vs, _ = perf.limits(intent_v_tas, intent_vs, intent_h, ax)

    assert allow_v_tas[idx] == pytest.approx(0.0)
    assert allow_vs[idx] == pytest.approx(0.0)
    assert traffic_.selspd[idx] == pytest.approx(0.0)
    assert not traffic_.swvnavspd[idx]
    assert not perf.pf_flag[idx]


def test_touchdown_stop_can_be_overridden_after_impulse(traffic_):
    traffic_.cre("LDG2", "A320", 52.0, 4.0, 180.0, 10.0, 70.0)
    idx = traffic_.id2idx("LDG2")
    perf = get_openap_perf(traffic_)

    traffic_.alt[idx] = 0.0
    perf.post_flight[idx] = True
    perf.pf_flag[idx] = True

    perf.limits(np.array([70.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))

    traffic_.selspd[idx] = 10.0
    allow_v_tas, _, _ = perf.limits(
        np.array([10.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
    )

    assert allow_v_tas[idx] > 0.0
    assert traffic_.selspd[idx] == pytest.approx(10.0)

