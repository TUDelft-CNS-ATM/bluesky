"""Regression test: SAVEIC must not write numpy scalar reprs into scenario files.

Since numpy 2, repr() of a numpy scalar (e.g. np.float64) includes the type
name, so lines like "SPD TEST1,np.float64(200.0)" end up in the saved
scenario file, which the stack parser then fails to read back.
"""
import re

import pytest

import bluesky


@pytest.fixture(scope='module')
def sim():
    bluesky.settings.is_sim = True
    bluesky.init()
    yield bluesky


def test_saveic_no_numpy_scalar_repr(sim, tmp_path):
    bs = sim
    bs.stack.stack('CRE TEST1,A320,52.0,4.5,45,5000,250')
    bs.stack.stack('HDG TEST1,90')
    bs.stack.stack('SPD TEST1,200')
    bs.sim.step()

    scnfile = tmp_path / 'test_saveic'
    bs.stack.stack(f'SAVEIC {scnfile}')
    bs.sim.step()
    bs.stack.stack('SAVEIC CLOSE')
    bs.sim.step()

    content = scnfile.with_suffix('.scn').read_text()

    # The HDG and SPD select commands must actually have been recorded,
    # otherwise this test would trivially pass without exercising the bug.
    assert re.search(r'^00:00:00\.00>HDG TEST1,', content, re.MULTILINE)
    assert re.search(r'^00:00:00\.00>SPD TEST1,', content, re.MULTILINE)

    # Every recorded numeric argument must be a plain number, parseable by
    # the stack parser on file reload.
    assert 'np.float64(' not in content
    for value in re.findall(r'(?<=,)(-?[\d.]+)', content):
        float(value)
