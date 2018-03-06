import numpy as np

def encode_ndarray(o):
    '''Msgpack encoder for numpy arrays.'''
    if isinstance(o, np.ndarray):
        return {b'numpy': True,
                b'type': o.dtype.str,
                b'shape': o.shape,
                b'data': o.tobytes()}
    return o

def decode_ndarray(o):
    '''Msgpack decoder for numpy arrays.'''
    if o.get(b'numpy'):
        return np.fromstring(o[b'data'], dtype=np.dtype(o[b'type'])).reshape(o[b'shape'])
    return o
