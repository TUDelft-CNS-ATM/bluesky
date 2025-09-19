""" Node encapsulates the sim process, and manages process I/O. """
import os
import bluesky as bs
from bluesky import stack
from bluesky.core.base import Base
from bluesky.core.walltime import Timer


class Node(Base):
    def __init__(self, *args):
        super().__init__()
        self.node_id = b'\x00' + os.urandom(4)
        self.server_id = b''
        self.act_id = None

    @stack.command(name='QUIT', annotations='', aliases=('CLOSE', 'END', 'EXIT', 'Q', 'STOP'))
    def quit(self):
        ''' Quit the simulation process. '''
        bs.sim.quit()

    def update(self):
        pass

    def connect(self):
        ''' Connect node to the BlueSky server. This does nothing in detached mode. '''
        pass

    def addnodes(self, count=1):
        pass

    def send(self, topic, data='', to_group=b''):
        pass

    def subscribe(self, topic, from_id=b'', to_group=b''):
        pass

    def close(self):
        pass