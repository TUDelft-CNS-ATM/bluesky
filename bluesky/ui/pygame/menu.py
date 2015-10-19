import os
import pygame as pg
from PIL import Image

white = (255, 255, 255)

class Menu:
    """ 
    Menu class definition : Menu button window

    # To adapt menu content: 
    #
    # 1. Edit screen in powerpoint file in ./devtools/menu
    # 2. Save all slide as jpg
    # 3. In the resulting subfolder run rename.bat, 
    #    which renames and copies files to right location
    # 4. Add commands in menu.dat 
    #

    Methods:
        Menu()         :  constructor

    Created by  : Jacco M. Hoekstra (TU Delft)
    """
    def __init__(self,win,x,y):
        self.win   = win    # Screen to draw window on    
        self.x     = x      # x-position
        self.y     = y      # y-position
        self.ipage = 0      # which menu
        self.npages = 4     # Number of pages

        # Read, crop and scale powerpoint bitmaps
        self.bmps  = []     # list with menu bitmpas 
        
        for i in range(self.npages):
            imgpath = "data/graphics/menu/menu"+str(i+1)+".jpg"
            
            im = Image.open(imgpath)
            
            image = im.crop((200, 194, 200+222, 194+330)) 
            width, height = image.size # Get dimensions
            mode = image.mode
            size = image.size
            data = image.tobytes() 
            
            surface = pg.transform.smoothscale(pg.image.frombuffer(data, size, mode),
                                         (width/2,height/2))
            
            rect = surface.get_rect()
            pg.draw.rect(surface, white, rect,1)
            self.bmps.append(surface)

        # Get dimensions
        self.dx,self.dy = surface.get_size()
        self.rect = pg.Rect(self.x,self.y,self.dx,self.dy)

        # Read data
        f = open("data/graphics/menu.dat")
        self.cmd = []
        ipage = 0
        page = []
        for line in f.readlines():
            
            if len(line.strip())==0:
                continue
            elif line.lower().strip()[:5]=="start": # starting page number
                self.ipage = int(line.split("=")[1])
            elif line[0].strip()==">":
                ipage = ipage+1
                if len(page)<20:
                    page=page+(20-len(page))*[""]
                self.cmd.append(page)
                page = []
            else:
                page.append(line[:-1])
        f.close()

        # Clean up
        del self.cmd[0]  # Remove first empty page due to >Traffic line
        self.ipage = max(0,min(self.npages,self.ipage)) # limit ipage
        
        return

    def update(self):

        self.rect = pg.Rect(self.x,self.y,self.dx,self.dy)
        
        return self.bmps[self.ipage]

    def getcmd(self,mpos):
        ix  = max(0,min(1,int((mpos[0]-self.x)/54)))
        idx = max(0,int((mpos[1]-self.y)/15))
        if ix == 0:
            self.ipage = min(self.npages-1,int(idx/2))
            cmdtxt =""
        else:
            cmdtxt = self.cmd[self.ipage][min(20,idx)]
        
        return cmdtxt
