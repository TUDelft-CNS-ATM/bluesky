import pygame as pg
from ... import stack
from math import *
from ..radarclick import radarclick


class Keyboard:
    """ 
    Keyboard class definition : keyboard & mouse input processing class

    Methods:
        Keyboard(tmx)                      :  constructor

        update()                           : Check for input & process it
        radarclick(pos,command,scr,traf,navdb)   : Process click on radar window to
                                                   to insert data in edit window

    Created by  : Jacco M. Hoekstra (TU Delft)
    """
    def __init__(self):

        self.dragedit = False  # Edit window initially not connected to mouse
        self.dragpotmenu = False
        self.dragmenu = False
        self.dragdx = 0
        self.dragdy = 0
        self.lastcmd  = ""     # previously typed command line
        self.firstx = True
        return
    
    def update(self, sim, cmd, scr, traf):

        # First time: IC window in pygame version        
        if self.firstx:
            stack.stack("IC")
            self.firstx = False
        
        # Get events
        for event in pg.event.get():
            if event.type==pg.QUIT:
                sim.stop()

            elif event.type==pg.KEYDOWN:

                # Alphanumeric key
                if event.key>31 and event.key<127:
                    scr.editwin.insert(str(event.unicode).upper())

                elif event.key==13: # ENTER
                    cmdline = scr.editwin.getline()
                    scr.editwin.enter()
                    if len(cmdline)>0:
                        self.lastcmd = cmdline
                        stack.stack(cmdline)

                elif event.key==8:   # BACKSPACE
                    scr.editwin.backspace()

                elif event.key==27: # ESCAPE
                    sim.stop()

                elif event.key ==pg.K_F3: # F3: Insert last command
                    scr.editwin.insert(self.lastcmd.strip().upper()+" ")
                   
                # Display keys
                elif event.key == 269: # Num lock minus
                    stack.stack("ZOOM OUT")
                elif event.key == 270: # Num lock pluus
                    stack.stack("ZOOM IN")

                elif event.key == 273: # Cursor up
                    scr.pan("ABOVE")
                elif event.key == 274: # Cursor down
                    scr.pan("DOWN")
                elif event.key == 275: # Num lock down
                    scr.pan("RIGHT")
                elif event.key == 276: # Num lock right
                    scr.pan("LEFT")

                elif event.key ==pg.K_F11: # F11: Toggle full screen
                    
                    scr.swfullscreen = not scr.swfullscreen 
                    scr.fullscreen(scr.swfullscreen)

                else: #TEST OPTION
                    pass
                
                # scr.editwin.insert(str(event.key))
                # TBD            
                # scr.insedit(chr(ikey))
                
            # End of keys selection


            # Mouse events:
            #    MOUSEMOTION      pos, rel, buttons
            #    MOUSEBUTTONUP    pos, button
            #    MOUSEBUTTONDOWN  pos, button

            # Mouse button 1 release: enter value in edit line
            elif event.type==pg.MOUSEBUTTONUP:

                # Reselase edit window if necessary
                if event.button==1:
                    if self.dragedit:
                        scr.editwin.winx = event.pos[0]-self.dragdx
                        scr.editwin.winy = event.pos[1]-self.dragdy
                        scr.editwin.winy = scr.editwin.fontedit.linedy *       \
                              (scr.editwin.winy/scr.editwin.fontedit.linedy)
                        scr.editwin.rect = pg.Rect(scr.editwin.winx,scr.editwin.winy, \
                             scr.editwin.bmpdx,scr.editwin.bmpdy)
                        scr.redrawedit = True
                    
                    # Menu button click
                    elif scr.menu.rect.collidepoint(event.pos) and \
                         not self.dragmenu:
                        cmdtxt = scr.menu.getcmd(event.pos)
                        if cmdtxt != "":
                            stack.stack(cmdtxt)

                    # In all other cases process as radar click
                    elif self.dragmenu:
                        self.dragmenu    = False
                        self.dragpotmenu = False
                        
                    else:
                        if not scr.swnavdisp:
                            # Interpret current edit line
                            cmdline = scr.editwin.getline()
                            lat, lon = scr.xy2ll(event.pos[0], event.pos[1])
                            tostack, todisplay = radarclick(cmdline, lat, lon, traf, traf.navdb)
                            if len(todisplay) > 0:
                                if todisplay[0] == '\n':
                                    scr.editwin.enter()
                                scr.editwin.insert(todisplay.strip('\n'))
                                if todisplay[-1] == '\n':
                                    scr.editwin.enter()
                            if len(tostack) > 0:
                                stack.stack(tostack)

                # Make sure edit and menu window are released
                self.dragedit    = False
                self.dragmenu    = False
                self.dragpotmenu = False

            # Mouse button down: lock onto edit window if insied edit window     
            elif event.type==pg.MOUSEBUTTONDOWN:

                self.dragmenu = False
                self.dragpotmenu = False

                if event.button==1:
                    if scr.editwin.rect.collidepoint(event.pos):
                        self.dragedit = True
                        self.dragdx = event.pos[0]-scr.editwin.winx
                        self.dragdy = event.pos[1]-scr.editwin.winy
                        scr.redrawedit = True

                    elif scr.menu.rect.collidepoint(event.pos):
                        self.dragpotmenu = True
                        self.dragmenu    = False
                        scr.redrawedit   = True
                        self.dragdx = event.pos[0] - scr.menu.x
                        self.dragdy = event.pos[1] - scr.menu.y
                elif event.button==4:
                    scr.zoom(0.5*sqrt(2.0))
                elif event.button==5:
                    scr.zoom(sqrt(2.0))
                
            # Mouse motion: drag edit/menu window with mouse, if necessary
            # Check also for mouse button 1                    
            elif event.type==pg.MOUSEMOTION and \
                (self.dragedit or self.dragpotmenu or self.dragmenu):
                if self.dragedit:
                    pressed = pg.mouse.get_pressed()[0]
                    if not pressed:
                        self.dragedit = False
                    else:
                        scr.editwin.winx = event.pos[0]-self.dragdx
                        scr.editwin.winy = event.pos[1]-self.dragdy
                        scr.editwin.rect = pg.Rect(scr.editwin.winx,scr.editwin.winy, \
                                 scr.editwin.bmpdx,scr.editwin.bmpdy)
                        scr.redrawedit = True             
     
                elif self.dragpotmenu:
                    pressed = pg.mouse.get_pressed()[0]
                    if  not pressed:
                        self.dragpotmenu = False
                        self.dragmenu = False
                    else:
                        mx,my = pg.mouse.get_pos()
                        outside = not scr.menu.rect.collidepoint((mx,my))
                        if outside and pressed:
                            self.dragmenu = True
                            self.dragpotmenu = False
                            
                elif self.dragmenu:
    
                    mx,my = event.pos
                    scr.menu.x = mx - self.dragdx
                    scr.menu.y = my - self.dragdy
                    scr.menu.update() # Update rectangle
                    scr.redrawedit = True             
                    pressed = pg.mouse.get_pressed()[0]
                    if  not pressed:
                        self.dragpotmenu = False
                        self.dragmenu = False

        #----- End of Update -----
        return
