''' Regression tests for adding waypoints after passing the last waypoint. '''
import pytest
import bluesky as bs


@pytest.fixture(scope='module')
def sim_env(tmp_path_factory):
    if bs.sim is None:
        workdir = tmp_path_factory.mktemp('bs_workdir')
        bs.init(detached=True, workdir=str(workdir))
    yield
    bs.sim.reset()


def _reset_with_aircraft(acid='KL100'):
    bs.sim.reset()
    bs.sim.op()
    bs.stack.stack('DT 1')
    assert bs.traf.cre(acid, 'A320', 52.0, 4.0, 90.0, 3000.0, 130.0)
    return bs.traf.id2idx(acid)


def _step_until_lnav_off(acidx, max_steps=120):
    for _ in range(max_steps):
        bs.sim.step()
        if not bs.traf.swlnav[acidx]:
            return True
    return False


def test_addwpt_after_passing_last_waypoint_flyby(sim_env):
    acidx = _reset_with_aircraft('KL100')
    route = bs.traf.ap.route[acidx]

    bs.stack.stack('ADDWPT KL100, 52.0, 4.01')
    bs.stack.stack('ADDWPT KL100, 52.0, 4.02')
    bs.sim.step()
    assert route.nwp == 2
    assert bs.traf.swlnav[acidx]

    assert _step_until_lnav_off(acidx)
    assert route.iactwp == 1
    for _ in range(15):
        bs.sim.step()

    bs.stack.stack('ADDWPT KL100, 52.0, 4.10')
    bs.stack.simstack.process()

    assert route.nwp == 3
    assert route.iactwp == 2
    assert bs.traf.swlnav[acidx]
    assert bs.traf.actwp.swlastwp[acidx]
    assert bs.traf.actwp.lon[acidx] == pytest.approx(4.10)


def test_addwpt_after_passing_last_waypoint_flyover_stays_forward(sim_env):
    acidx = _reset_with_aircraft('KL104')
    route = bs.traf.ap.route[acidx]

    bs.stack.stack('ADDWPT KL104, FLYOVER')
    bs.stack.stack('ADDWPT KL104, 52.0, 4.01')
    bs.stack.stack('ADDWPT KL104, 52.0, 4.02')
    bs.sim.step()

    assert _step_until_lnav_off(acidx)
    for _ in range(15):
        bs.sim.step()

    bs.stack.stack('ADDWPT KL104, 52.0, 4.10')
    bs.sim.step()

    assert route.nwp == 3
    assert route.iactwp == 2
    assert bs.traf.swlnav[acidx]
    assert bs.traf.actwp.swlastwp[acidx]
    assert bs.traf.actwp.lon[acidx] == pytest.approx(4.10)
    assert bs.traf.ap.trk[acidx] == pytest.approx(90.0, abs=1.0)


def test_addwpt_after_passing_single_waypoint(sim_env):
    acidx = _reset_with_aircraft('KL101')
    route = bs.traf.ap.route[acidx]

    bs.stack.stack('ADDWPT KL101, 52.0, 4.01')
    bs.sim.step()
    assert route.nwp == 1
    assert bs.traf.swlnav[acidx]

    assert _step_until_lnav_off(acidx)
    assert route.iactwp == 0
    for _ in range(15):
        bs.sim.step()

    bs.stack.stack('ADDWPT KL101, 52.0, 4.10')
    bs.stack.simstack.process()

    assert route.nwp == 2
    assert route.iactwp == 1
    assert bs.traf.swlnav[acidx]
    assert bs.traf.actwp.swlastwp[acidx]
    assert bs.traf.actwp.lon[acidx] == pytest.approx(4.10)


def test_addwpt_before_passing_last_waypoint_keeps_active(sim_env):
    acidx = _reset_with_aircraft('KL102')
    route = bs.traf.ap.route[acidx]

    bs.stack.stack('ADDWPT KL102, 52.0, 4.05')
    bs.stack.stack('ADDWPT KL102, 52.0, 4.15')
    bs.sim.step()
    route.direct(acidx, route.wpname[1])
    assert route.iactwp == 1
    assert bs.traf.swlnav[acidx]

    bs.stack.stack('ADDWPT KL102, 52.0, 4.25')
    bs.sim.step()

    assert route.nwp == 3
    assert route.iactwp == 1
    assert not bs.traf.actwp.swlastwp[acidx]
    assert bs.traf.actwp.lon[acidx] == pytest.approx(4.15)


def test_addwpt_after_passing_dest(sim_env):
    acidx = _reset_with_aircraft('KL103')
    route = bs.traf.ap.route[acidx]

    bs.stack.stack('ADDWPT KL103, 52.0, 4.01')
    bs.stack.stack('DEST KL103, 52.0, 4.02')
    bs.sim.step()
    assert route.nwp == 2

    assert _step_until_lnav_off(acidx)
    assert route.iactwp == 1
    for _ in range(15):
        bs.sim.step()

    bs.stack.stack('ADDWPT KL103, 52.0, 4.10')
    bs.stack.simstack.process()

    assert route.nwp == 3
    assert route.iactwp == 1
    assert bs.traf.swlnav[acidx]
    assert not bs.traf.actwp.swlastwp[acidx]
    assert bs.traf.actwp.lon[acidx] == pytest.approx(4.10)


def test_orig_after_passing_last_waypoint_keeps_end_of_route_state(sim_env):
    acidx = _reset_with_aircraft('KL105')
    route = bs.traf.ap.route[acidx]

    bs.stack.stack('ADDWPT KL105, 52.0, 4.01')
    bs.sim.step()

    assert _step_until_lnav_off(acidx)
    for _ in range(15):
        bs.sim.step()

    bs.stack.stack('ORIG KL105, 52.0, 3.90')
    bs.stack.simstack.process()
    assert not bs.traf.swlnav[acidx]
    assert route.flag_lastwp_passed

    bs.stack.stack('ADDWPT KL105, 52.0, 4.10')
    bs.stack.simstack.process()
    assert route.nwp == 3
    assert route.iactwp == 2
    assert bs.traf.swlnav[acidx]
    assert bs.traf.actwp.lon[acidx] == pytest.approx(4.10)


def test_delwpt_tail_then_pass_last_and_addwpt(sim_env):
    acidx = _reset_with_aircraft('KL106')
    route = bs.traf.ap.route[acidx]

    bs.stack.stack('ADDWPT KL106, 52.0, 4.02')
    bs.stack.stack('ADDWPT KL106, 52.0, 4.08')
    bs.sim.step()
    assert route.nwp == 2
    assert not bs.traf.actwp.swlastwp[acidx]

    bs.stack.stack(f'DELWPT KL106, {route.wpname[1]}')
    bs.stack.simstack.process()
    assert route.nwp == 1
    assert bs.traf.actwp.swlastwp[acidx]

    assert _step_until_lnav_off(acidx)
    for _ in range(15):
        bs.sim.step()

    bs.stack.stack('ADDWPT KL106, 52.0, 4.12')
    bs.stack.simstack.process()
    assert route.nwp == 2
    assert route.iactwp == 1
    assert bs.traf.swlnav[acidx]
    assert bs.traf.actwp.lon[acidx] == pytest.approx(4.12)

