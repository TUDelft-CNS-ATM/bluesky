"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Tests traffic module
"""

from bluesky.tools.aero import casormach


def test_traffic_create_missingarg_fail(traffic_):
    """
    Test create command without arguments.
    Expect error message.
    """
    ok, _msg = traffic_.create()
    assert not ok


def validate_create(traffic_, result,
                    len_, acid, actype, aclat, aclon, achdg, acalt, casmach):
    """
    Generic test for validating successful creation
    of a new aircraft.
        - traffic_:
            The current traffic object directory
        - result:
            The result of the create command (True | False)
        - len_:
            The number of expected traffic objects
        - acid:
            The callsign of the created aircraft
        - actype:
            The type of the created aircraft
        - aclat/aclon:
            The position of the created aircraft
        - achdg:
            Current heading of the created aircraft
        - acalt:
            Current altitude of the created aircraft
        - casmach:
            Current speed of the created aircraft
            Based on casmach and altitude you can calculate:
                - TAS (True airspeed in knots)
                - CAS (Calibrated airspeed in knots)
                - M (Mach)
    """

    assert isinstance(result, bool)
    assert result

    validate_lengths(traffic_, len_)

    assert traffic_.id[-1] == acid
    assert traffic_.type[-1] == actype

    assert traffic_.lat[-1] == aclat
    assert traffic_.lon[-1] == aclon
    assert traffic_.alt[-1] == acalt

    assert traffic_.hdg[-1] == achdg
    assert traffic_.trk[-1] == achdg

    tas, cas, m = casormach(casmach, acalt)
    assert traffic_.tas[-1] == tas
    assert traffic_.cas[-1] == cas
    assert traffic_.M[-1] == m

    # should complete assertions, in due course


def validate_lengths(traffic_, len_):
    """
    Generic function to test number of
    traffic objects is as expected and traffic objects have
    all expected attributes.
    """
    assert traffic_.ntraf == len_
    assert len(traffic_.id) == len(traffic_.type) == len(traffic_.lat) == \
        len(traffic_.lon) == len(traffic_.lon) == len(traffic_.alt) == \
        len(traffic_.hdg) == len(traffic_.trk) == len(traffic_.tas) == \
        len(traffic_.cas) == len(traffic_.M) == len_  # etc

    # check that one of the children
    assert len(traffic_.children) == 8
    child = traffic_.children[0]
    akey = child.ArrVars[0]
    assert len(child.Vars[akey]) == len_


def test_traffic_create(traffic_):
    """
    Test creation of new aircraft by successively creating
    three new aircraft.

    First two uses default callsign (KL204+) and aircraft type (B744).

    Expects new aircraft to be added correctly.
    """
    ntraf = traffic_.ntraf

    result = traffic_.create(
        aclat=0.0, aclon=50.0, achdg=270, acalt=2000, casmach=200)
    validate_create(
        traffic_, result,
        ntraf + 1, 'KL204', 'B744', 0.0, 50.0, 270, 2000, 200)

    result = traffic_.create(
        aclat=-10.0, aclon=60.0, achdg=180, acalt=1000, casmach=100)
    validate_create(
        traffic_, result,
        ntraf + 2, 'KL205', 'B744', -10.0, 60.0, 180, 1000, 100)

    result = traffic_.create(
        'BA1', 'A320', 10.0, 55.0, 90, 3000, 300)
    validate_create(
        traffic_, result,
        ntraf + 3, 'BA1', 'A320', 10.0, 55.0, 90, 3000, 300)


def test_traffic_delete(traffic_):
    """
    Test deletion of existing aircraft using index.

    Still BA1 must be the last traffic object, number of traffic objects
    must decrease by one.
    """
    ntraf = traffic_.ntraf
    result = traffic_.delete(1)

    validate_create(
        traffic_, result,
        ntraf - 1, 'BA1', 'A320', 10.0, 55.0, 90, 3000, 300)


def test_traffic_reset(traffic_):
    """
    Test reset command.

    Expects zero aircraft after reset.
    """
    traffic_.reset()
    validate_lengths(traffic_, 0)


# test remaining traffic functions
