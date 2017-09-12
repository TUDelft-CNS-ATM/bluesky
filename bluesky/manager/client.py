import zmq


def run():
    ctx = zmq.Context()

    event_io = ctx.socket(zmq.DEALER)
    stream_in = ctx.socket(zmq.SUB)

    event_io.connect('tcp://localhost:9000')
    event_io.send('REGISTER')
    msg = event_io.recv()
    print('Hostid =', msg[:4], 'myid =', ord(msg[-1]) - 100)
    stream_in.setsockopt(zmq.SUBSCRIBE, b'')
    stream_in.connect('tcp://localhost:9001')
    # event_io.send('target', flags=zmq.SNDMORE)
    # event_io.send_pyobj(['iets anders', 'teststring'])

    poller = zmq.Poller()
    poller.register(event_io, zmq.POLLIN)
    poller.register(stream_in, zmq.POLLIN)

    while True:
        try:
            events = dict(poller.poll(None))
        except zmq.ZMQError:
            break  # interrupted
        except KeyboardInterrupt:
            print("Interrupt received, stopping")
            break
        if event_io in events:
            msg = event_io.recv_multipart()
            print('Event from worker', ord(msg[0][-1]) - 100, ':', msg[-1])
        if stream_in in events:
            msg = stream_in.recv_multipart()
            print(msg)

if __name__ == '__main__':
    run()
