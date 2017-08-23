"""
Common test functionality.
- Andrew Farrell, SPARKL.
"""
from __future__ import print_function
import inspect
import time


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


def printrecv(data, stackpos = 2):
    """
    Prints the data received by test from bluesky.
    Also prints the calling test function, for context.
    Args:
        data: Received data.
    """
    print("-- {} --: Data received: {}".format(funname(stackpos), data))


def wait_for(test, iters, period):
    """
    Performs up to `iters` iterations of a `test`, stopping when `test` is True.
    Employs an exponentially increasing wait between test iterations.
    Args:
        test: The test to perform
        iters: The number of iterations of `test`
        period: The initial wait period in seconds between test iterations.

    Returns:

    """
    if iter == 0:
        raise BlueSkyTestException()

    success = test()
    if not success:
        time.sleep(period)
        wait_for(test, iters - 1, 2 * period)


class BlueSkyTestException(Exception):
    """
    Simple base exception class for bluesky tests.
    """
    pass
