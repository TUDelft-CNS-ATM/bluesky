import socket

def get_ownip():
    try:
        local_addrs = socket.gethostbyname_ex(socket.gethostname())[-1]

        for addr in local_addrs:
            if not addr.startswith('127'):
                return addr
    except:
        pass
    return '127.0.0.1'
