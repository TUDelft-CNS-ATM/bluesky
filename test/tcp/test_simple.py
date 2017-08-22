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


@pytest.fixture(scope="module")
def sock(pytestconfig):
    rootdir = str(pytestconfig.rootdir)
    sys.path.append(rootdir)
    from BlueSky_qtgl import start, gui_prestart, gui_exec, stop

    telnet_in, manager = start()
    gui_prestart()
    newpid = os.fork()
    if newpid != 0:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((TCP_HOST, TCP_PORT))
        yield s
        s.close()
        stop(telnet_in, manager)
        parent_pid = os.getpid()
        parent = psutil.Process(parent_pid)
        child = parent.children(recursive=False)[0]
        child.send_signal(signal.SIGKILL)
    else:
        gui_exec()


def test_simple(sock):
    sock.send("CRE")
    data = sock.recv(BUFFER_SIZE).split("\n")
    print "Data received: {}".format(data)
    assert data[0] == "Syntax error: Too few arguments"
    assert data[1] == "CRE"
    assert data[2] == "CRE acid,type,lat,lon,hdg,alt,spd"

