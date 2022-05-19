"""
Copyright (c) 2017 SPARKL Limited. All Rights Reserved.
For inclusion with BlueSky upstream code:
https://github.com/ProfHoekstra/bluesky/, distributed under
GNU General Public License v3.

Author <ahfarrell@sparkl.com> Andrew Farrell
Common test functionality.

NOTE - The test suite is written in Python3 only.
It tests BlueSky running in either Python2 or 3.
"""
from __future__ import print_function
import inspect
import time
from bluesky.tools.network import as_bytes


BLUESKY = "BlueSky_qtgl.py"
BUFFER_SIZE = 1024
TCP_HOST = "127.0.0.1"
TCP_PORT = 8888


def sock_connect(socket_, host, port):
    """
    Attempts a socket connection, and returns success boolean.

    Args:
        socket_: the socket, created with 'socket' method
        host:: the host
        port: the port

    Returns:
        whether socket is connected
    """
    try:
        socket_.connect((host, port))
        return True
    except ConnectionRefusedError:
        return False


def sock_send(socket_, msg):
    """
        Sends data across socket.
    """
    socket_.send(
        as_bytes(msg + "\n"))


def sock_receive(socket_):
    """
        Gets data from socket.
    """
    data = bytes(socket_.recv(BUFFER_SIZE)).decode('utf8').rstrip()
    printrecv(data)
    return data


def funname(stackpos):
    """
    Returns the function name from the current call stack, unwinding `stackpos`
    entries. This function is at `stackpos` 0, its caller is at 1, etc.

    Args:
        stackpos: Call stack position.

    Returns:
        Stack frame function name.
    """
    return inspect.stack()[stackpos][3]


def funname_message(message):
    return funname(2) + ":" + str(message)


def printrecv(data, stackpos=2):
    """
    Prints the data received by test from bluesky.
    Also prints the calling test function, for context.
    Args:
        data: Received data.
    """
    print(f"-- {funname(stackpos)} --: Data received: {data}")


def wait_for(test, iters, period):
    """
    Performs up to `iters` iterations of a `test`, stopping when `test` is True.
    Employs an exponentially increasing wait time between test iterations.
    Args:
        test: The test to perform
        iters: The number of iterations of `test`
        period: The initial wait period in seconds between test iterations.

    Returns:

    """
    if iter == 0:
        raise BlueSkyTestException()

    time.sleep(period)  # front-load a sleep

    success = test()
    if not success:
        wait_for(test, iters - 1, 2 * period)


class BlueSkyTestException(Exception):

    """
    Simple base exception class for bluesky tests.
    """
    pass
