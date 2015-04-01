""" 
Standard Dialog modules, using Tkinter

Created by  : Jacco M. Hoekstra
"""

from Tkinter import *
import tkFileDialog
import os
import sys


def fileopen():
    """returns filename fo scenatriofile selected"""

    # load Tk crash on Mac
    # temporary work around mac without loading and file
    if sys.platform == 'darwin':
        return ''

    dirpath = imgpath = os.path.dirname(__file__) + "/../../data/scenario"
    os.chdir(dirpath)

    master = Tk()
    master.withdraw() #hiding tkinter window
    master.focus_set()

    file_path = tkFileDialog.askopenfilename(title="Open scenario file", \
        filetypes=[("Scenario files",".scn"),("All files",".*")])

    # Close Tk, return to working directory    
    master.quit()
    os.chdir('..')
    return file_path