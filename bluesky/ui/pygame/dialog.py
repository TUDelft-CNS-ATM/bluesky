""" 
Standard Dialog modules, using Tkinter

Created by  : Jacco M. Hoekstra
"""

from tkinter import *
import tkinter.filedialog
import os
from bluesky import settings

def fileopen():
    """returns filename of scenariofile selected"""
    cdir = os.getcwd()
    # load Tk crash on Mac
    # temporary work around mac without loading and file
    if sys.platform == 'darwin':
       return ''

    dirpath = settings.scenario_path 
    os.chdir(dirpath)

    master = Tk()
    master.withdraw()  # hiding tkinter window
    master.focus_set()
    
    file_path = tkinter.filedialog.askopenfilename(title="Open scenario file", 
        filetypes=[("Scenario files",".scn"),("All files",".*")],\
        initialdir=".")

    # Close Tk, return to working directory
    master.quit()
    os.chdir(cdir)
    if type(file_path)==str or type(file_path)==str:
        fpath = str(file_path)
    else:
        print("Warning: Unexpected type from FileDialog:",file_path)
        print(type(file_path))
        print("No file selected.")
        fpath = ""
#    print file_path,fpath
    return fpath
