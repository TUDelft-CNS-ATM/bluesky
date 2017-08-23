"""
Tests TCP interface.
- Andrew Farrell, SPARKL.
"""
import os
import sys
import socket
import signal
import psutil
import pytest

BLUESKY = "BlueSky_qtgl.py"
TCP_HOST = "127.0.0.1"
TCP_PORT = 8888
BUFFER_SIZE = 1024

SYNTAX_ERROR = "Syntax error"


@pytest.fixture(scope="module")
def sock(pytestconfig):
    """
    Suite-level setup and teardown function, for those test functions
    naming `sock` in their parameter lists.
    """
    rootdir = str(pytestconfig.rootdir)
    sys.path.append(rootdir)
    from BlueSky_qtgl import start, gui_prestart, gui_exec, stop
    import bluesky
    import bluesky.test

    tcp, manager = start()
    gui_prestart()
    newpid = os.fork()
    if newpid != 0:
        sock_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_.connect((TCP_HOST, TCP_PORT))
        yield sock_, bluesky, bluesky.test
        sock_.close()
        stop(tcp, manager)
        parent_pid = os.getpid()
        parent = psutil.Process(parent_pid)
        child = parent.children(recursive=False)[0]
        child.send_signal(signal.SIGKILL)
    else:
        gui_exec()


def test_create_fail(sock):
    """
    Tests that sending a spurious `CRE` instruction over tcp interface,
    yields some kind of syntax error, without being rigidly prescriptive
    regarding what is returned.
    """
    sock_, _blue, blue_test = sock
    sock_.send("CRE\n")
    data = sock_.recv(BUFFER_SIZE).strip().split("\n")
    blue_test.printrecv(data)
    assert len(data) == 3
    assert SYNTAX_ERROR in data[0]


def test_create_success(sock):
    """
    Tests that creating an aircraft properly yields an 'ok.' response.
    """
    sock_, blue, blue_test = sock
    sock_.send("CRE KL204, B744,52,4,180,2000,220\n")
    data = sock_.recv(BUFFER_SIZE).strip().split("\n")
    blue_test.printrecv(data)
    assert len(data) == 1
    assert data[0] == blue.MSG_OK


def test_pos(sock):
    """
    Tests that we get meaningful information on the aircraft just created.
    """
    sock_, _blue, blue_test = sock
    sock_.send("POS KL204\n")
    data = sock_.recv(BUFFER_SIZE).strip().split("\n")
    blue_test.printrecv(data)
    assert data[0].startswith("Info on KL204")
    assert len(data) > 5


def test_disconnect(sock):
    """
    Tests that when disconnecting a socket, it is removed from bluesky's
    records of active socket connections.
    """
    sock_, blue, blue_test = sock

    def conns(num_conns):
        """
        Tests for appropriate number of connections
        as determined by `num_conns`
        """
        sock_.send(blue.CMD_TCP_CONNS + "\n")
        data = sock_.recv(BUFFER_SIZE).strip().split("\n")
        blue_test.printrecv(data, 5)
        return num_conns == int(data[0])

    # Open second socket
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.connect((TCP_HOST, TCP_PORT))
    blue_test.wait_for(lambda: conns(2), 2, 2)
    sock2.close()
    blue_test.wait_for(lambda: conns(1), 2, 2)
