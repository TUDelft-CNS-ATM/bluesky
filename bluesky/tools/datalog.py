""" 
Datalog class definition : Data logging class

Methods:
    Datalog(filename)          :  constructor

    write(txt)         : add a line to the datalogging buffer
    save()             : save data to file
   
Created by  : Jacco M. Hoekstra (TU Delft)
Date        : October 2013

Modifation  :
By          :
Date        :

"""
import os
from misc import tim2txt
from time import strftime,gmtime

#-----------------------------------------------------------------

class Datalog():
    def __init__(self):
        # Create a buffer and save filename
        self.fname = os.path.dirname(__file__) + "/../../data/output/" \
            + strftime("%Y-%m-%d-%H-%M-%S-BlueSky.tmx", gmtime())
        
        self.buffer=[]
        return
    
    def write(self,t,txt):
        # Add text to buffer with timestamp t
        self.buffer.append(tim2txt(t)+" > "+txt+chr(13)+chr(10))
        return

    def save(self):
        # Write buffer to file 
        f = open(self.fname,"w")
        f.writelines(self.buffer)
        f.close()
        return

