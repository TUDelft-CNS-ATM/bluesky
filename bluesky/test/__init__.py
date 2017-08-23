from __future__ import print_function
import inspect, time


def funname(stackpos):
    return inspect.stack()[stackpos][3]


def printrecv(data):
    print("\n{}: Data received: {}".format(funname(2), data))


def wait_for(test, reps, period):
    if iter == 0:
        raise BlueSkyTestException()

    success = test()
    if not success:
        time.sleep(period)
        wait_for(test, reps-1, 2 * period)


class BlueSkyTestException(Exception):
    def __init__(self):
        super(BlueSkyTestException, self).__init__()
