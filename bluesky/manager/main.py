import os
import zmq


class Manager(object):
    def __init__(self):
        self.host_id = b'\x00' + os.urandom(4)
        self.clients = dict()
        self.workers = dict()

        # Prepare ZMQ context
        self.ctx = zmq.Context()

        # Create connection points for clients
        self.fe_event  = self.ctx.socket(zmq.ROUTER)
        self.fe_stream = self.ctx.socket(zmq.XPUB)
        self.fe_event.bind('tcp://*:9000')
        self.fe_stream.bind('tcp://*:9001')


        # Create connection points for sim workers
        self.be_event  = self.ctx.socket(zmq.ROUTER)
        self.be_stream = self.ctx.socket(zmq.XSUB)

        self.be_event.bind('tcp://*:10000')
        self.be_stream.bind('tcp://*:10001')

        # Create poller for both event connection points and the stream reader
        self.poller = zmq.Poller()
        self.poller.register(self.fe_event, zmq.POLLIN)
        self.poller.register(self.be_event, zmq.POLLIN)
        self.poller.register(self.be_stream, zmq.POLLIN)
        self.poller.register(self.fe_stream, zmq.POLLIN)

    def forward_event(self, src, dest, srcnames, destnames):
        msg = src.recv_multipart()
        if msg[-1] == 'REGISTER':
            # This is a registration message for a new connection
            connid = self.host_id + chr(100 + len(srcnames))
            srcnames[msg[0]] = connid
            # Send reply with connection ID
            src.send_multipart([msg[0], connid])

        else:
            # This is a message that should be forwarded
            # Swap sender and target so that msg is sent to target
            sender = srcnames.get(msg[0]) or 'unknown'
            target = msg[1]
            msg[:2] = target, sender
            if target == '*':
                # This is a send-to-all message
                for name in destnames:
                    msg[0] = name
                    dest.send_multipart(msg)
            else:
                dest.send_multipart(msg)

    def run(self):
        while True:
            try:
                events = dict(self.poller.poll(None))
            except zmq.ZMQError:
                break  # interrupted
            except KeyboardInterrupt:
                print("Interrupt received, stopping")
                break
            # The socket with incoming data
            for sock in events:
                if sock == self.fe_event:
                    self.forward_event(self.fe_event, self.be_event, self.clients, self.workers)
                elif sock == self.be_event:
                    self.forward_event(self.be_event, self.fe_event, self.workers, self.clients)
                elif sock == self.be_stream:
                    self.fe_stream.send_multipart(sock.recv_multipart())
                elif sock == self.fe_stream:
                    self.be_stream.send_multipart(sock.recv_multipart())

if __name__ == '__main__':
    mgr = Manager()
    mgr.run()
