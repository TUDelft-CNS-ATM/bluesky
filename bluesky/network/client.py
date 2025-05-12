''' BlueSky Client class. '''
from collections import defaultdict

from bluesky.core.signal import Signal
from bluesky.stack.clientstack import process
from bluesky.network.node import Node
from bluesky.network.common import genid, GROUPID_CLIENT, GROUPID_SIM, GROUPID_DEFAULT

class Client(Node):
    def __init__(self, group_id=GROUPID_CLIENT):
        super().__init__(group_id)
        self.acttopics = defaultdict(set)
        self.discovery = None

        # Signals
        self.actnode_changed = Signal('actnode-changed')
        self.node_added.connect(self.actnode)

    def update(self):
        ''' Client periodic update function.

            Periodically call this function to allow client to receive and process data.
        '''
        super().update()

        # Process any waiting stacked commands
        process()
    
    def actnode(self, newact=None):
        ''' Set the new active node, or return the current active node. '''
        if newact:
            if newact not in self.nodes:
                print('Error selecting active node (unknown node)')
                return None
            if self.act_id is None:
                # This is the first time an active node is selected
                # We can now unsubscribe this method from the node-added signal
                self.node_added.disconnect(self.actnode)

            # Unsubscribe from previous node, subscribe to new one.
            if newact != self.act_id:
                for topic, groupset in self.acttopics.items():
                    for to_group in groupset:
                        if self.act_id:
                            self._unsubscribe(topic, self.act_id, to_group)
                        self._subscribe(topic, newact, to_group)
                self.act_id = newact
                self.actnode_changed.emit(newact)

        return self.act_id

    def _subscribe(self, topic, from_group=GROUPID_DEFAULT, to_group='', actonly=False):
        if from_group == GROUPID_DEFAULT:
            from_group = GROUPID_SIM
            if actonly:
                self.acttopics[topic].add(to_group)
                if self.act_id is not None:
                    from_group = self.act_id
                else:
                    return
        return super()._subscribe(topic, from_group, to_group)
    
    def _unsubscribe(self, topic, from_group=GROUPID_DEFAULT, to_group=''):
        if from_group == GROUPID_DEFAULT:
            from_group = GROUPID_SIM
            if topic in self.acttopics:
                self.acttopics[topic].discard(to_group)
                if self.act_id is not None:
                    from_group = self.act_id
                else:
                    return
        return super()._unsubscribe(topic, from_group, to_group)

    def addnodes(self, count=1, *node_ids, server_id=None):
        ''' Tell the specified server to add 'count' nodes. 
        
            If provided, create these nodes with the specified node ids.

            If no server_id is specified, the corresponding server of the
            currently-active node is targeted.
        '''
        self.send('ADDNODES', dict(count=count, node_ids=node_ids), server_id or genid(self.act_id[:-1], seqidx=0))
