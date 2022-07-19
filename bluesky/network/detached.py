""" Node encapsulates the sim process, and manages process I/O. """
import os
import bluesky as bs
from bluesky.core.walltime import Timer


class Node:
    def __init__(self, *args):
        self.node_id = b'\x00' + os.urandom(4)
        self.host_id = b''
        self.running = True

    def update(self):
        ''' Update timers. '''
        # Process timers
        Timer.update_timers()

    def stop(self):
        ''' Stack stop/quit command. '''
        # On a stack quit command, detached simulation just quits.
        self.quit()

    def quit(self):
        ''' Quit the simulation process. '''
        self.running = False

    def run(self):
        ''' Start the main loop of this node. '''
        while self.running:
            # Perform a simulation step
            self.update()
            bs.sim.step()

    def connect(self):
        ''' Connect node to the BlueSky server. This does nothing in detached mode. '''
        pass

    def addnodes(self, count=1):
        pass

    def send_event(self, eventname, data=None, target=None):
        pass

    def send_stream(self, name, data):
        pass
