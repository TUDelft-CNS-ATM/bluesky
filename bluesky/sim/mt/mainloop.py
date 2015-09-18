from thread import Thread


class MainLoop:
    gui = []
    sim = []

    @staticmethod
    def start():
        # =============================================================================
        # Connect signals between gui and sim
        # =============================================================================
        # Periodic simulation statistics from sim thread to gui (actual update frequency, dt, simtime)
        MainLoop.sim.screenio.signal_siminfo.connect(MainLoop.gui.callback_siminfo)
        # Periodic communication of aircraft states to gui for visualization of traffic
        MainLoop.sim.screenio.signal_update_aircraft.connect(MainLoop.gui.callback_update_aircraft)
        # Non-periodic signal to open a file dialog from the stack
        MainLoop.sim.screenio.signal_show_filedialog.connect(MainLoop.gui.show_file_dialog)
        # Non-periodic signal to display stack text in the gui text box
        MainLoop.sim.screenio.signal_display_text.connect(MainLoop.gui.callback_stack_output)
        # Non-periodic signal to alter radarscreen pan/zoom from the stack
        MainLoop.sim.screenio.signal_panzoom.connect(MainLoop.gui.callback_panzoom)
        # Non-periodic signal to send user-inputs from gui to stack
        MainLoop.gui.signal_command.connect(MainLoop.sim.screenio.callback_userinput)

        simthread = Thread(MainLoop.sim)
        simthread.start(Thread.HighestPriority)
        MainLoop.gui.start()
