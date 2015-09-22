from thread import Thread


def MainLoop(gui, sim):
    # =============================================================================
    # Connect signals between gui and sim
    # =============================================================================
    # Periodic simulation statistics from sim thread to gui (actual update frequency, dt, simtime)
    sim.screenio.signal_siminfo.connect(gui.callback_siminfo)
    # Periodic communication of aircraft states to gui for visualization of traffic
    sim.screenio.signal_update_aircraft.connect(gui.callback_update_aircraft)
    # Non-periodic signal to open a file dialog from the stack
    sim.screenio.signal_show_filedialog.connect(gui.show_file_dialog)
    # Non-periodic signal to display stack text in the gui text box
    sim.screenio.signal_display_text.connect(gui.callback_stack_output)
    # Non-periodic signal to alter radarscreen pan/zoom from the stack
    sim.screenio.signal_panzoom.connect(gui.callback_panzoom)
    # Non-periodic signal to send user-inputs from gui to stack
    gui.signal_command.connect(sim.screenio.callback_userinput)

    simthread = Thread(sim)
    simthread.start(Thread.HighestPriority)
    gui.start()

    # Stopping simulation thread
    print 'Stopping Threads'
    sim.stop()
    simthread.quit()
    simthread.wait()
