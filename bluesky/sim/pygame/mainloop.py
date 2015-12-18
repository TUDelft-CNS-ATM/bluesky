def MainLoop(gui, sim):    
    sim.start()

    # Main loop for tmx object
    while not sim.mode == sim.end:
        sim.update(gui.scr)  # Update sim
        gui.update(sim)      # Update GUI

        # Restart traffic simulation:
        if sim.mode == sim.init:
            sim.reset()
            gui.reset()

    # After the simulation is done, close the gui
    gui.close()
