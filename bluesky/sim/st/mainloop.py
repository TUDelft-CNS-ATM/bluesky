class MainLoop:
    gui = []
    sim = []

    @staticmethod
    def start():
        MainLoop.sim.start()

        # Main loop for tmx object
        while not MainLoop.sim.mode == MainLoop.sim.end:
            MainLoop.sim.update(MainLoop.gui.scr)  # Update sim
            MainLoop.gui.update(MainLoop.sim)      # Update GUI

            # Restart traffic simulation:
            if MainLoop.sim.mode == MainLoop.sim.init:
                MainLoop.sim.reset()
                MainLoop.gui.reset()
