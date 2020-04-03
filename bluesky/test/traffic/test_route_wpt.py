"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Tests route module, wpt functionality
"""
from . import assert_fl


def print_route(route):
    """
    Prints all components of a waypoint.
    E.g. its name, lat/long, etc.
    """
    print(route.wpname)
    print(route.wplat)
    print(route.wplon)
    print(route.wpalt)
    print(route.wpspd)
    print(route.wptype)
    print(route.wpflyby)


def test_route_get_available_name(route_):
    """
    Test get_available_name function.

    Expects function to correctly append
    names already in use.
    """
    data = ['aname', 'aname01', 'bname', 'cname']
    name = route_.Route.get_available_name(data, 'aname')
    assert name == 'aname02'

    name = route_.Route.get_available_name(data, 'bname')
    assert name == 'bname01'

    name = route_.Route.get_available_name(data, 'dname')
    assert name == 'dname'

    data = ['aname']
    name = route_.Route.get_available_name(data, 'aname', 3)
    assert name == 'aname001'


def test_addwpt_data(route_):
    """
    Tests the insertion/modification of waypoints.

    The addwpt_data function expects:
        - Overwrites (existing waypoint):
            True | False
        - Waypoint index:
            Integer
        - Waypoint name:
            String
        - Waypoint latitude:
            String
        - Waypoint longitude:
            String
        - Waypoint type:
            0 - Lat/long waypoint
            1 - VOR/NAV waypoint
            2 - Origin airport
            3 - Destination airport
            4 - Calculated waypoint
            5 - Runway
        - Altitude constraint:
            Integer
        - Speed constraint:
            Integer
        - Flyby (aircraft flies over or not):
            True | False
    """
    # Init route
    route = route_.Route()

    # All init values must be empty lists
    assert route.wpname == route.wplat == route.wplon == route.wpalt == \
        route.wpspd == route.wptype == route.wpflyby == []

    # Add new waypoint and check if added correctly
    route.addwpt_data(False, 0, 'FOO', 10., -10., 0, 1000., 100.)
    assert route.wpname == ['FOO']
    assert route.wplat == [10.]
    assert route.wplon == [-10.]
    assert route.wpalt == [1000.]
    assert route.wpspd == [100.]
    assert route.wptype == [0]
    assert route.wpflyby == [False]

    # Add another waypoint, see if both this and previous are in list
    route.addwpt_data(False, 0, 'BAZ', 20., -20., 0, 2000., 200.)
    assert route.wpname == ['BAZ', 'FOO']
    assert route.wplat == [20., 10.]
    assert route.wplon == [-20., -10.]
    assert route.wpalt == [2000., 1000.]
    assert route.wpspd == [200., 100.]
    assert route.wptype == [0, 0]
    assert route.wpflyby == [True, False]

    # This waypoint must overwrite FOO (which was previously at index 1)
    route.addwpt_data(True, 1, 'BAR', 30., -30., 0, 3000., 300.)
    assert route.wpname == ['BAZ', 'BAR']
    assert route.wplat == [20., 30.]
    assert route.wplon == [-20., -30.]
    assert route.wpalt == [2000., 3000.]
    assert route.wpspd == [200., 300.]
    assert route.wptype == [0, 0]
    assert route.wpflyby == [True, True]


def test_add_wp_orig(traffic_, route_):
    """
    Tests addition of origin waypoint.

    Expects new origin waypoints to be added
    correctly to list of waypoints (route).
    """

    # Create test aircraft and get its index
    traffic_.create(
        'BA111', 'A320', 0., 0., 90, 1000., 100.)
    acidx = traffic_.id2idx('BA111')

    # Init route of new aircraft
    route = route_.Route()
    traffic_.ap.route[acidx] = route

    # Add some dummy data
    #
    route.addwpt_data(False, 0, 'BAZ', 20., -20., 0, 2000., 200.)
    route.addwpt_data(False, 1, 'BAR', 30., -30., 0, 3000., 300.)
    route.iactwp = 1
    route.nwp = 2

    #
    # Insert new orig as one does not exist already
    #
    idx = route.addwpt(
        acidx, 'BAR', route.orig, 52.29, 4.74, 50, 100, '', '')
    assert idx == 0
    assert route.wptype[idx] == route.orig
    assert route.nwp == 3
    assert route.wpname[idx] == 'BAR01'
    assert route.iactwp == 2

    #
    # Overwrite existing orig
    #
    idx = route.addwpt(
        acidx, 'BAR', route.orig, 52.29, 4.74, 50, 100)
    assert idx == 0
    assert route.wptype[idx] == route.orig
    assert route.nwp == 3
    assert route.wpname[idx] == 'BAR02'
    assert route.iactwp == 2


def test_add_wp_dest(traffic_, route_):
    """
    Tests addition of destination waypoint.

    Expects new destination waypoints to be added
    correctly to list of waypoints (route).
    """
    acidx = traffic_.id2idx('BA111')
    route = traffic_.ap.route[acidx]

    #
    # Insert dest
    #
    idx = route.addwpt(
        acidx, 'BAR', route.dest, 52.29, 4.74, 50, 100, '', '')

    assert idx == 3
    assert route.wptype[idx] == route.dest
    assert route.nwp == 4
    assert route.wpname[idx] == 'BAR01'
    assert route.iactwp == 2

    #
    # Overwrite existing orig/dest
    #
    idx = route.addwpt(
        acidx, 'BAR', route.dest, 52.29, 4.74, 50, 100)
    assert idx == 3
    assert route.wptype[idx] == route.dest
    assert route.nwp == 4
    assert route.wpname[idx] == 'BAR03'
    assert route.iactwp == 2


def test_add_wp_normal(traffic_, route_):
    """
    Tests insertion of en-route waypoints
    (i.e. not dest or origin).
    """
    traffic_.create(
        'BA222', 'A320', 0., 0., 90, 1000., 100.)
    idx = traffic_.id2idx('BA222')
    route = route_.Route()

    idx = route.addwpt(
        idx, 'BAZ', route.wplatlon, 10., -10., 1000., 100., '', '')

    assert idx == 0
    assert route.nwp == 1

    assert route.wplat == [10.]
    assert route.wplon == [-10.]
    assert route.wpalt == [1000.]
    assert route.wpspd == [100.]
    assert route.wpname == ['BAZ']
    assert route.wptype == [0]

    idx = route.addwpt(    # don't worry too much re numbers here
        idx, 'BAZ', route.runway, 20., -20., 2000., 200., 'BAZ', '')
    assert idx == 1

    assert route.wplat == [10., 20.]
    assert route.wplon == [-10., -20.]
    assert route.wpalt == [1000., 2000.]
    assert route.wpspd == [100., 200.]
    assert route.wpname == ['BAZ', 'BAZ01']
    assert route.wptype == [0, 5]

    idx = route.addwpt(  # don't worry too much re numbers here
        idx, 'FANBAN', route.wpnav, 30., -30., 3000., 300., '', 'BAZ')
    assert idx == -1

    idx = route.addwpt(  # don't worry too much re numbers here
        idx, 'LIFFY', route.wpnav, 30., -30., 3000., 300., '', 'BAZ')
    assert idx == 0

    assert_fl(route.wplat[0], 53.48)
    assert_fl(route.wplon[0], -5.5)
    assert route.wpalt == [3000., 1000., 2000.]
    assert route.wpspd == [300., 100., 200.]
    assert route.wpname == ['LIFFY', 'BAZ', 'BAZ01']
    assert route.wptype == [1, 0, 5]

    idx = route.addwpt(  # don't worry too much re numbers here
        idx, 'EGKK', route.wpnav, 0., 0., 50., 50., '', '')
    assert idx == 3

    assert_fl(route.wplat[3], 51.14)
    assert_fl(route.wplon[3], -0.19)
    assert route.wpalt == [3000., 1000., 2000., 50.]
    assert route.wpspd == [300., 100., 200., 50.]
    assert route.wpname == ['LIFFY', 'BAZ', 'BAZ01', 'EGKK']
    assert route.wptype == [1, 0, 5, 1]


def test_addwpt_stack_setflyby(traffic_, route_):
    """
    Tests Fly-By and Fly-over modes of
    addwpt.

    Expects only one or the other to be
    set at a time.
    """
    route = route_.Route()
    route.swflyby = False

    succ = route.addwptStack(0, 'FLY-BY')
    assert route.swflyby
    assert succ
    route.swflyby = False

    succ = route.addwptStack(0, 'FLYBY')
    assert route.swflyby
    assert succ
    route.swflyby = False

    succ = route.addwptStack(0, 'FLY-OVER')
    assert not route.swflyby
    assert succ
    route.swflyby = False

    succ = route.addwptStack(0, 'FLYOVER')
    assert not route.swflyby
    assert succ
    route.swflyby = False


def test_addwpt_stack_takeoffwp(traffic_, route_):
    """
    Test takeoff argument of addwpt command.

    Expect success with takeoff waypoint added to route.

    Created aircraft is persisted for subsequent tests.
    """
    traffic_.create(
        'BA222', 'A320', 0., 0., 90, 2000., 200.)
    idx = traffic_.id2idx('BA222')
    route = route_.Route()

    args = ['TAKEOFF', 'EGKK', 'RW08R']

    succ = route.addwptStack(idx, *args)
    assert succ
    assert route.nwp == 1

    assert_fl(route.wplat[0], 51.15)
    assert_fl(route.wplon[0], -0.15)
    assert route.wpalt == [-999.]
    assert route.wpspd == [-999.]
    assert route.wpname == ['T/O-BA222']
    assert route.wptype == [0]

    traffic_.ap.route[idx] = route


def test_addwpt_stack(traffic_):
    """
    Test addition of multiple waypoints with altitude/speed constraints.

    Expect waypoints to be inserted into route with correct constraints
    and in correct order.
    """
    idx = traffic_.id2idx('BA222')
    route = traffic_.ap.route[idx]

    args = ['LIFFY', 3000., 300.]

    succ = route.addwptStack(idx, *args)
    assert succ

    assert_fl(route.wplat[1], 53.48)
    assert_fl(route.wplon[1], -5.5)
    assert route.wpalt[1] == 3000.
    assert route.wpspd[1] == 300.
    assert route.wpname[1] == 'LIFFY'
    assert route.wptype[1] == 1

    args = ['EGKK', 50., 50.]

    succ = route.addwptStack(idx, *args)
    assert succ

    assert int(route.wplat[2]) == 51
    assert int(route.wplon[2]) == 0
    assert route.wpalt[2] == 50.
    assert route.wpspd[2] == 50.
    assert route.wpname[2] == 'EGKK'
    assert route.wptype[2] == 1


def test_after_addwpt_stack(traffic_):
    """
    Test after command - i.e. adding a waypoint
    after an already specified one.

    Expects waypoint to ba added correctly.
    Since the waypoint is not a pre-defined one (must be added as lat/lng),
    the waypoint is named after the callsign.
    """
    idx = traffic_.id2idx('BA222')
    route = traffic_.ap.route[idx]

    args = ['LIFFY', '', '10.,-10.', 1000.0, 100.]

    succ = route.afteraddwptStack(idx, *args)
    assert succ
    assert route.nwp == 4

    assert route.wplat[2] == 10.
    assert route.wplon[2] == -10.
    assert route.wpalt[2] == 1000.
    assert route.wpspd[2] == 100.
    assert route.wpname[2] == 'BA222'
    assert route.wptype[2] == 0


def test_before_addwpt_stack(traffic_):
    """
    Test inserting a waypoint before an already
    specified waypoint.

    Expect waypoint to be inserted correctly before LIFFY.
    """
    idx = traffic_.id2idx('BA222')
    route = traffic_.ap.route[idx]

    args = ['LIFFY', '', '20.,-20.', 2000.0, 200.]

    succ = route.beforeaddwptStack(idx, *args)
    assert succ
    assert route.nwp == 5

    assert route.wplat[1] == 20.
    assert route.wplon[1] == -20.
    assert route.wpalt[1] == 2000.
    assert route.wpspd[1] == 200.
    assert route.wpname[1] == 'BA222001'
    assert route.wptype[1] == 0
