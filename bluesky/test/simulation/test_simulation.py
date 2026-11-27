''' Tests for BlueSky simulation control. '''
import pytest
import bluesky as bs
from bluesky.core import varexplorer as ve


@pytest.fixture(scope='module')
def sim(tmp_path_factory):
    if bs.sim is None:
        workdir = tmp_path_factory.mktemp('bs_workdir')
        bs.init(detached=True, workdir=str(workdir))
    bs.sim.reset()
    bs.sim.op()
    yield bs.sim
    bs.sim.reset()


def test_simdt_assignment_raises_attribute_error(sim):
    initial_dt = sim.simdt
    with pytest.raises(AttributeError):
        sim.simdt = 5.0
    assert sim.simdt == initial_dt


def test_simdt_tracks_clock_steps_and_reset(sim):
    sim.reset()
    sim.op()
    t0 = sim.simt
    bs.stack.stack('DT 5')
    sim.step()
    assert sim.simdt == pytest.approx(5.0)
    assert sim.simt - t0 == pytest.approx(5.0)

    sim.step(dt_increment=1.0)
    assert sim.simdt == pytest.approx(6.0)
    assert sim.simt - t0 == pytest.approx(11.0)

    sim.step()
    assert sim.simdt == pytest.approx(5.0)
    assert sim.simt - t0 == pytest.approx(16.0)

    sim.reset()
    assert sim.simdt == pytest.approx(bs.settings.simdt)


def test_simdt_exposed_in_varexplorer(sim):
    ve.init()
    var = ve.findvar('simdt')
    assert var is not None
    assert var.get() == pytest.approx(sim.simdt)
    ok, sim_info = ve.lsvar('sim')
    assert ok
    assert 'simdt' in sim_info
