import pygame as pg
from .fastfont import Fastfont

black = (0, 0, 0)
white    = (255,255,255)
darkgrey = (25, 25, 48)
grey = (84, 84, 114)
darkblue = (25, 25, 64, 100)
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 255)
red = (255, 0, 0)
lightgreyblue = (130, 150, 190)  # waypoint symbol color
lightgreygreen = (149, 215, 179)  # grid color
lightcyan = (0, 255, 255)  # FIR boundaries
amber = (255,163,71)  # Conflicting aircraft
magenta = (255,0,255) # Used for route

class Console:
    """ 
    Console (aka EditWin) class definition : Edit window & console class

    Methods:
        echo(msg)         : Print a message
        insert(message)   : insert characters in current edit line
        backspace()       : process backspace
        getline()         : return current edit line
        enter()           : enter, end of edit line
        scroll()          : scroll up one line
        update()          : redraw update bitmap of edit window

    Created by  : Jacco M. Hoekstra (TU Delft)
    """

    def __init__(self,win,nch,nlin,winx,winy):
        # Was Helvetica,14
        self.fontedit = Fastfont(win,'Courier New',14,white,False,False) # name, size, bold,italic

        # Edit window: 6 line of 64 chars
        self.content = []
        self.nch = nch    # number of chars per line
        self.nlin = nlin   # number of lines in windows
        self.winx = winx   # x-coordinate in pixels of left side
        self.winy = winy - self.nlin*self.fontedit.linedy   # y-coordinate in pixels of top
        self.msg = []        # Messages in edit window
        
        for i in range(self.nlin):
            line= self.nch*[' ']
            self.content.append(line)
        self.content0 = self.content
        
        self.xcursor = 0
        self.xedit = 0
        # self.printeditwin('Testing 1,2,3')
        self.bmpdy = self.nlin*self.fontedit.linedy
        self.bmpdx = int(self.nch*self.fontedit.linedy*10/17) + 2 # Guess max aspect ratio
        self.bmp = pg.Surface([self.bmpdx,self.bmpdy],
                                   pg.SRCALPHA, 32)
                                   
        self.bmp.fill(darkblue)
        self.rect = pg.Rect(self.winx,self.winy,
                             self.bmpdx,self.bmpdy)
        self.redraw = True
     
        return


    def echo(self,msg): 
        """print a message to console window"""
        if self.xedit==self.xcursor:
            self.insert(msg)
            j = int(self.xcursor/self.nch)      
            self.xcursor = (j+1)*self.nch
            self.xedit = self.xcursor
    
            # Check for End of window
            if self.xedit >= (self.nch-1)*(self.nlin-1):
                del self.content[0]
                self.content.append(self.nch*[' '])
                self.xcursor = j*self.nch
                self.xedit = self.xcursor
        else:
            self.msg.append(msg)   # buffer
        return


    def insert(self,message):
        i = self.xcursor%self.nch
        j = int(self.xcursor/self.nch)
        for ich in range(len(message)):
            self.content[j][i]=message[ich]
            i = i+1
            # Check for end-of line
            if i>=self.nch:
                i = 0
                j = j+1
                # Check for end-of edit window
                if j>=self.nlin:
                    self.scroll()
                    j = j-1
        self.xcursor = j*self.nch+i
        self.redraw = True
        return

    def backspace(self):
        if self.xcursor>self.xedit:
            self.xcursor = self.xcursor-1
        self.redraw = True
        i = self.xcursor%self.nch
        j = int(self.xcursor/self.nch)
        self.content[j][i]=" "
        return

    def getline(self): # enter was pressed ro we need current command line
        line = ""
        for idx in range(self.xedit,self.xcursor):        
            i = idx%self.nch
            j = int(idx/self.nch)
            line = line+self.content[j][i]
        return line

    def enter(self):           
        j = int(self.xcursor/self.nch)        
        self.xcursor = (j+1)*self.nch
        self.xedit = self.xcursor

        # End of window
        if self.xedit >= (self.nch-1)*(self.nlin-1):
            del self.content[0]
            self.content.append(self.nch*[' '])
            self.xcursor = j*self.nch
            self.xedit = self.xcursor

        # Print buffered messages
        self.redraw = True
        while len(self.msg)>0:
            self.echo(self.msg[0])  # No endless recursion becasue xedit==xcursor
            del self.msg[0]
        return

    def scroll(self):
        """Scroll window"""
        del self.content[0]
        self.content.append(self.nch*[' '])
        self.xcursor = self.xcursor-self.nch
        self.xedit = self.xedit-self.nch

    def update(self):
        """Update: Draw a new frame"""
        # Draw edit window
        if self.redraw:
            self.bmp.fill(darkgrey)
            
            for j in range(self.nlin):
                for i in range(self.nch):
                    if True or self.content[j][i] != self.content0[j][i]:
                        x = i*int(self.fontedit.linedy*10/17) + 1
                        y = j*self.fontedit.linedy+int(self.fontedit.linedy/6)
                        self.fontedit.printat(self.bmp,
                                              x,y,
                                              self.content[j][i])
                        self.content0[j][i]=self.content[j][i]
            # Draw cursor
            i = self.xcursor%self.nch
            j = int(self.xcursor/self.nch)
            x = i*int(self.fontedit.linedy*10/17)
            y = j*self.fontedit.linedy+int(self.fontedit.linedy/6)
            self.fontedit.printat(self.bmp,x,y,"_")
            self.bmp.set_alpha(127)    
            self.redraw = False

        return
