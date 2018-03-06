"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Tests TCP interface.
"""
import socket
from .. import TCP_HOST, TCP_PORT, wait_for, sock_receive, sock_send


def test_create_fail(sock):
    """
    Tests that sending a spurious `CRE` instruction over tcp interface,
    yields some kind of syntax error, without being rigidly prescriptive
    regarding what is returned.
    """
    sock_, _bluesky = sock

    sock_send(sock_, "CRE")
    data = sock_receive(sock_)
    assert data == 'CRE acid,type,lat,lon,hdg,alt,spd'


def test_create_success(sock):
    """
    Tests that creating an aircraft properly yields an 'ok.' response.
    """
    print(sock)
    sock_, bluesky = sock

    sock_send(sock_, "CRE KL204, B744, 52, 4, 180, 2000, 220")
    data = sock_receive(sock_)
    assert data == bluesky.BS_OK


def test_pos(sock):
    """
    Tests that we get meaningful information on the aircraft just created.
    """
    sock_, _bluesky = sock

    sock_send(sock_, "POS KL204")
    data = sock_receive(sock_)

    assert data.startswith("Info on KL204")


def test_disconnect(sock):
    """
    Tests that when disconnecting a socket, it is removed from bluesky's
    records of active socket connections.
    """
    sock_, bluesky = sock

    def conns(num_conns):
        """
        Tests for appropriate number of connections
        as determined by `num_conns`
        """
        sock_send(sock_, bluesky.CMD_TCP_CONNS)
        data = sock_receive(sock_)
        return num_conns == int(data)

    # Open second socket
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.connect((TCP_HOST, TCP_PORT))
    wait_for(lambda: conns(2), 2, 2)
    sock2.close()
    wait_for(lambda: conns(1), 2, 2)
