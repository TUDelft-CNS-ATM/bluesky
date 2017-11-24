"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Tests position module
"""

from bluesky.tools.position import islatlon, Position


def test_route_is_lat():
    lat = 'N51\'08\'52.0"'
    assert islatlon(lat)

    lat = 'N51\'08\'52.0",W001\'0\'0.0"'
    assert islatlon(lat)

    lat = 'P51\'08\'52.0"'
    assert not islatlon(lat)

    lon = 'W001\'0\'0.0"'
    assert islatlon(lon, False)


def test_route_latlon(traffic_):
    latlon = Position('N51\'0\'0",W001\'0\'0.0"', 'foo', 'bar')
    assert latlon.lat == 51.0
    assert latlon.lon == -1.0
    assert latlon.type == Position.latlon
    assert not latlon.error


def test_route_apt(traffic_):
    airport = Position('EGKK', 'foo', 'bar')

    assert airport.type == Position.apt
    assert airport.lat == 51.148
    assert airport.lon == -0.19
    assert not airport.error


def test_route_nav(traffic_):
    nav = Position('LIFFY', 51.148, -0.19)

    assert nav.type == Position.nav
    assert nav.lat == 53.48
    assert nav.lon == -5.5
    assert not nav.error


def test_route_rway(traffic_):
    runway = Position('EHAM/RW06', 'foo', 'bar')

    assert runway.type == Position.runway
    assert runway.lat == 52.28906612851886
    assert runway.lon == 4.737262710501425
    assert not runway.error


def test_route_ac_latlon(traffic_):
    traffic_.create(
        'BA2', 'A320', 10.0, 55.0, 90, 3000, 300)
    ac = Position('BA2', 'foo', 'bar')

    assert ac.type == Position.latlon
    assert ac.lat == 10.0
    assert ac.lon == 55.0
    assert not ac.error


def test_route_dir_pan_left(traffic_):
    pan = Position('LEFT', 1.0, 2.0)
    assert pan.type == Position.dir
    assert pan.lat == 1.0
    assert pan.lon == 2.0
    assert not pan.error

    pan = Position('RIGHT', 1.0, 2.0)
    assert pan.type == Position.dir
    assert not pan.error

    #  File Bug for clash between 'PAN' dir 'ABOVE' and 'wp' nav 'ABOVE'
    #pan = Position('ABOVE', 1.0, 2.0)
    #assert pan.type == Position.dir
    #assert not pan.error

    pan = Position('DOWN', 1.0, 2.0)
    assert pan.type == Position.dir
    assert not pan.error

    pan = Position('DOWNDOWN', 1.0, 2.0)
    assert pan.error
