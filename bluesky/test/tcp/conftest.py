"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Common test fixtures.

NOTE - The test suite is written in Python3 only.
It tests BlueSky running in either Python2 or 3.
"""
from __future__ import print_function
import socket
import os
import subprocess
import signal
import psutil
import bluesky
import pytest
from .. import sock_connect, wait_for, TCP_HOST, TCP_PORT, BLUESKY


@pytest.fixture(scope="session")
def sock(pytestconfig):
    """
    Suite-level setup and teardown function, for those test functions
    naming `sock` in their parameter lists.
    """
    newpid = os.fork()
    if newpid != 0:
        sock_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        wait_for(
            lambda: sock_connect(sock_, TCP_HOST, TCP_PORT), -1, 5)
        yield sock_, bluesky
        sock_.close()
        parent_pid = os.getpid()
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.send_signal(signal.SIGKILL)
            except psutil.NoSuchProcess:
                pass
    else:
        subprocess.call([os.environ['PYEXEC'], BLUESKY])
