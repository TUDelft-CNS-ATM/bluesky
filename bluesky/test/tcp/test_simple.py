"""
Tests TCP interface, validating:

- Andrew Farrell, SPARKL Ltd.
"""
import os, sys
import socket
import signal, psutil
import pytest

BLUESKY = "BlueSky_qtgl.py"
TCP_HOST = "127.0.0.1"
TCP_PORT = 8888
BUFFER_SIZE = 1024

SYNTAX_ERROR = "Syntax error"


@pytest.fixture(scope="module")
def sock(pytestconfig):
    rootdir = str(pytestconfig.rootdir)
    sys.path.append(rootdir)
    from BlueSky_qtgl import start, gui_prestart, gui_exec, stop
    import bluesky as bs
    import bluesky.test as bs_test

    tcp, manager = start()
    gui_prestart()
    newpid = os.fork()
    if newpid != 0:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_HOST, TCP_PORT))
        yield s, bs, bs_test
        s.close()
        stop(tcp, manager)
        parent_pid = os.getpid()
        parent = psutil.Process(parent_pid)
        child = parent.children(recursive=False)[0]
        child.send_signal(signal.SIGKILL)
    else:
        gui_exec()


def test_create_fail(sock):
    sock_, bs, bs_test = sock
    sock_.send("CRE\n")
    data = sock_.recv(BUFFER_SIZE).strip().split("\n")
    bs_test.printrecv(data)
    assert len(data) == 3
    assert SYNTAX_ERROR in data[0]


def test_create_success(sock):
    sock_, bs, bs_test = sock
    sock_.send("CRE KL204, B744,52,4,180,2000,220\n")
    data = sock_.recv(BUFFER_SIZE).strip().split("\n")
    bs_test.printrecv(data)
    assert len(data) == 1
    assert data[0] == bs.MSG_OK


def test_pos(sock):
    sock_, bs, bs_test = sock
    sock_.send("POS KL204\n")
    data = sock_.recv(BUFFER_SIZE).strip().split("\n")
    bs_test.printrecv(data)
    assert data[0].startswith("Info on KL204")
    assert len(data) > 5


def test_disconnect(sock):
    sock_, bs, bs_test = sock

    def conns(num_conns):
        sock_.send(bs.CMD_TCP_CONNS + "\n")
        data = sock_.recv(BUFFER_SIZE).strip().split("\n")
        bs_test.printrecv(data)
        return num_conns == int(data[0])

    # Open second socket
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.connect((TCP_HOST, TCP_PORT))
    bs_test.wait_for(lambda: conns(2), 2, 2)
    sock2.close()
    bs_test.wait_for(lambda: conns(1), 2, 2)
