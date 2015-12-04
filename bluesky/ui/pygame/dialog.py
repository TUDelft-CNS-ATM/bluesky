""" 
Standard Dialog modules, using Tkinter

Created by  : Jacco M. Hoekstra
"""

from Tkinter import *
import tkFileDialog
import os


def fileopen():
    """returns filename of scenariofile selected"""
    cdir = os.getcwd()

    # load Tk crash on Mac
    # temporary work around mac without loading and file
    if sys.platform == 'darwin':
       return ''

    dirpath = imgpath = "scenario"
    os.chdir(dirpath)

    master = Tk()
    master.withdraw()  # hiding tkinter window
    master.focus_set()
    
    file_path = tkFileDialog.askopenfilename(title="Open scenario file", 
        filetypes=[("Scenario files",".scn"),("All files",".*")])

    # Close Tk, return to working directory
    master.quit()
    os.chdir(cdir)
#    print file_path
    return file_path
