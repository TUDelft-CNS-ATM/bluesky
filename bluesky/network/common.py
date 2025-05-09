import socket
import binascii
from os import urandom
from typing import Union
from enum import Enum, auto


# Message headers (first byte): (un)subscribe
MSG_SUBSCRIBE = 1
MSG_UNSUBSCRIBE = 0
# Message headers (second byte): group identifiers
GROUPID_DEFAULT = 0
GROUPID_CLIENT = ord('C')
GROUPID_SIM = ord('S')
GROUPID_NOGROUP = ord('N')
# Connection identifier string length
IDLEN = 5

class ActionType(Enum):
    ''' Shared state action types. 
    
        An incoming shared state update can be of the following types:
        Append: An item is appended to the state
        Extend: Two or more items are appended to the state
        Delete: One or more items are deleted from the state
        Update: One or more items within the state are updated
        Replace: The full state object is replaced
        Reset: The entire object is reset to its (empty) default
        ActChange: A new active remote is selected
    '''
    Append = b'A'
    Extend = b'E'
    Delete = b'D'
    Update = b'U'
    Replace = b'R'
    Reset = b'X'
    ActChange = b'C'
    NoAction = b''

    @classmethod
    def isaction(cls, data):
        ''' Returns True if passed data is an ActionType '''
        return any([data == a.value for a in cls])


class MessageType(Enum):
    ''' BlueSky network message type indicator. '''
    Unknown = auto()
    Regular = auto()
    SharedState = auto()


def genid(groupid: Union[str, bytes, int]='', idlen=IDLEN, seqidx=1):
    ''' Generate a binary identifier string 
    
        The identifier string consists of a group identifier of idlen-1 bytes,
        and ends with a sequence number that is indicated with curidx.

        Arguments:
        - groupid: The group identifier of the generated id can be part of a
          larger group. A group id passed to the function is extended with random
          bytes up to a length of idlen-1. Valid values in each position are all
          possible byte values except the wildcard character '*', which is reserved
          as wildcard padding.

        - idlen: The length in bytes of the generated identifier string

        - seqidx: The value of the sequence index part of this identifier.
    '''
    groupid = asbytestr(groupid)
    if len(groupid) >= idlen:
        return groupid
    elif len(groupid) < idlen - 1:
        groupid += urandom(idlen - 1 - len(groupid)).replace(b'*', b'_')
    return groupid + seqidx2id(seqidx)


def seqid2idx(seqid):
    ''' Transform a bytestring sequence id to a numeric sequence index.
        Returns -1 if this is not a valid sequence id.
    '''
    ret = (seqid if isinstance(seqid, int) else ord(seqid)) - 128
    return max(-1, ret)


def seqidx2id(seqidx):
    ''' Transform a numeric sequence index to a bytestring sequence id '''
    return chr(128 + seqidx).encode('charmap')


def asbytestr(val:int|str|bytes) -> bytes:
    return chr(val).encode('charmap') if isinstance(val, int) else \
              val.encode('charmap') if isinstance(val, str) else \
              val

def bin2hex(bstr):
    return binascii.hexlify(bstr).decode()


def hex2bin(hstr):
    return binascii.unhexlify(hstr)


def get_ownip():
    ''' Try to determine the IP address of this machine. '''
    try:
        local_addrs = socket.gethostbyname_ex(socket.gethostname())[-1]

        for addr in local_addrs:
            if not addr.startswith('127'):
                return addr
    except:
        pass
    return '127.0.0.1'
