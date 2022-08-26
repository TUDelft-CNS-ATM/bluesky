import pygame as pg

class Fastfont:
    """ 
    Faster font for printing graphically, renders only at creation

    Methods:
        CFastfont(screen,name,size,color,bold,italic) :  constructor, renders font
        
        printat(screen,x,y,text): print text at x,y, at screen window (blit)
        setpos(x,y)   : not used anymore?

    Members: see create

    Created by  : Jacco M. Hoekstra
    """

    def __init__(self,screen,name,size,color,bold,italic):
        self.swposx = -1  # Default x = left side
        self.swposy = -1  # Default y = top

        pfont = pg.font.SysFont(name,size,bold,italic)

        # Render font

        # Convert chars 32-126 to map of standard height bitmaps:
        self.chmaps = []
        self.chw = []
        self.linedy = pfont.get_linesize()
        
        for ich in range(32,126):
            ch = pfont.render(chr(ich),False,color)
            ch = ch.convert_alpha(screen)
            self.chmaps.append(ch)
            self.chw.append(ch.get_width())
        del pfont
        return      

    def printat(self,screen,x,y,text):

        w = 0
        for ch in text:
            ich = ord(ch)
            if ich>=32 and ich<=126:
                w = w + self.chw[ich-32]

        txtimg = pg.Surface((w,self.linedy)) # Standard height bitmap
        txtimg = txtimg.convert_alpha(screen)
        
        ix = 0
        for ch in text:
            ich = ord(ch)
            if ich>=32 and ich<=126:
                w = w + self.chw[ich-32]
                dest = self.chmaps[ich-32].get_rect()
                dest.top = 0
                dest.left = ix
                ix = ix+self.chw[ich-32]
                # chimg = self.chmaps[ich-32].convert_alpha(screen)

                txtimg.blit(self.chmaps[ich-32],dest)# Removed pg.BLEND_ADD which broke it on Windows machine

        dest = txtimg.get_rect()

        # Set position
        if self.swposx <0:
            dest.left = x
        elif self.swposx>0:
            dest.right = x
        else:
            dest.centerx = x

        if self.swposy <0:
            dest.top = y
        elif self.swposy>0:
            dest.bottom = y
        else:
            dest.centery = y

        # Paste it onto the screen
        screen.blit(txtimg,dest)# Removed pg.BLEND_ADD which broke it on Windows machine

        return

    def setpos(self,swposx,swposy):
        self.swposx = swposx
        self.swposy = swposy
        return
