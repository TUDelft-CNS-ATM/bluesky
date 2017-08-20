#!/usr/bin/env python
import tkinter
from tkinter import *

from FileFrame import FileFrame


class ScencarioCreator(tkinter.Tk):
    def __init__(self, parent):
        tkinter.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()


    def initialize(self):
        # Display the GUI
        self.grid()

        # Create booleans to keep track of FileFrame creations
        self.firstRun = True
        self.frameCreated = False

        # Number of files to load
        # Set the standard to 1
        self.numberOfFiles = IntVar()
        self.numberOfFiles.set(1)

        loadText = StringVar()
        loadText.set("How many files do you want to load?")
        loadLabel = Label(self.parent, textvariable=loadText)
        loadLabel.grid(column=0, row=0)

        # Drop-down menu
        optionList = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        options = OptionMenu(self.parent, self.numberOfFiles, *optionList)
        options.grid(column=1, row=0)

        # Create submit button
        button = Button(self.parent, text="Submit", command=(lambda: self.changeNumberOfFiles()))
        button.grid(column=1, row=1)


    # This method changes the number of files which are going to be selected
    def changeNumberOfFiles(self):
        print(self.numberOfFiles.get())
        i = 0
        if self.firstRun == True:

            self.fileFrame = FileFrame(self.parent, self.numberOfFiles)
            self.firstRun = False

        else:
            # delete all frames
            if self.frameCreated == False:
                self.fileFrame.destroy()
                self.frameCreated = True

            else:
                self.newFrame.grid_forget

            # create new frames
            self.newFrame = FileFrame(self.parent, self.numberOfFiles)


if __name__ == '__main__':
    app = ScencarioCreator(None)
    app.title("BlueSky scenario creator")
    app.geometry('400x100+400+400')
    app.mainloop()









