''' ActData provides automatic storage of data that needs to be individually
    updated and accessed for each individual remote node, and easy access of data of the active node.
'''
from types import SimpleNamespace
from numbers import Number
from copy import copy
from collections import defaultdict

import bluesky as bs


# Keep a dict of default attribute values of mutable type.
# These always need to be stored per remote node.
defaults = dict()

# Keep a dict of remote state storage namespaces
remotes = defaultdict(lambda: SimpleNamespace(**{k:copy(v) for k, v in defaults.items()}))


def get(remote_id=None):
    return remotes[remote_id or bs.net.act_id]


class ActData:
    def __init__(self, default):
        self.default = default
        self.name = ''

    def __set_name__(self, owner, name):
        self.name = name
        # If underlying datatype is mutable, always immediately
        # store per remote node
        if not isinstance(self.default, (str, tuple, Number, frozenset, bytes)):
            defaults[name] = self.default

            # In case remote data already exists, update stores
            for remote in remotes.values():
                setattr(remote, name, copy(self.default))

    def __get__(self, obj, objtype=None):
        ''' Return the actual value for the currently active node. '''
        if not bs.net.act_id:
            return self.default
        return getattr(remotes[bs.net.act_id], self.name, self.default)
        # TODO: What is the active (client) node on the sim-side? Is this always within a currently processed stack command? -> stack.sender_id

    def __set__(self, obj, value):
        if not bs.net.act_id:
            self.default = value
        else:
            setattr(remotes[bs.net.act_id], self.name, value)
