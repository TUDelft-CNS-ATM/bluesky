from thread import Thread


def MainLoop(gui, sim):
    # Create a simulation thread, and start it at the highest priority
    simthread = Thread(sim)
    simthread.start(Thread.HighestPriority)

    # Set sim as an event target for the gui, so that the gui can send events to the sim object
    gui.setSimEventTarget(sim.screenio)

    # Start the gui
    gui.start()

    # Stopping simulation thread
    print 'Stopping Threads'
    sim.stop()
    simthread.quit()
    simthread.wait()
