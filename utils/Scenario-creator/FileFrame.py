from tkinter import *
import tkinter.filedialog
import tkinter.messagebox
import DatToScn
import CitationToDatFile  #/Users/yoeritorel1old/School/Afstuderen/BlueSky_yoeri/BlueSky\ Tools/Scenario\ creator/GUI/so6_to_scn.pyc
import so6_to_scn


class FileFrame(Toplevel):
    def __init__(self, parent, filenumber):
        Toplevel.__init__(self, parent, background="white")
        self.parent = parent
        self.filenumber = filenumber
        self.configure(bg="white")

        # Create a list which stores the locations of the temporary .dat files
        self.tempDatFiles = []

        #Define Variables
        self.optionMenuVar = []
        i = 0
        for i in range(0, self.filenumber.get()):
            self.optionMenuVar.append(StringVar(self.parent))
            i += 1

        #Define widgets
        self.optionMenuStatusbar = []
        self.optionMenuSplittedSelectedFile = []
        self.optionMenuLoadbuttons = []

        i = 0
        for i in range(0, self.filenumber.get()):
            # Status bars
            statusbar = Label(self, text="", bd=1)
            self.optionMenuStatusbar.append(statusbar)

            # String for location of files
            splittedSelectedFiles = StringVar()
            self.optionMenuSplittedSelectedFile.append(splittedSelectedFiles)

            '''
            DO NOT DELETE!!!!!!!
            # Load button
            loadfileButton = Button(self, text="Open File",  command=lambda num=i: self.getOpenFileName(i))
            self.optionMenuLoadbuttons.append(loadfileButton)
            i += 1
            '''

        self.BUGloadbuttoncreator()

        # Create widgets
        i = 0
        for i in range(0, self.filenumber.get()):
            # Indicate the type of the source file
            self.typeText = StringVar()
            self.typeText.set("Select file type " + str(i + 1) + ":")
            self.typeLabel = Label(self, textvariable=self.typeText, height=4)
            self.typeLabel.grid(column=0, row=i, sticky=W)

            # Drop-down menu for file type selection
            optionList = ("ADS-B", "Citation", ".dat", ".mat To70", ".s06", ".scn")
            options = OptionMenu(self, self.optionMenuVar[i], *optionList)
            options.grid(column=1, row=i)

            # Select the load path of the file or files (in case of ADS-B)

            self.optionMenuStatusbar[i].grid(column=3, row=i)

            self.optionMenuLoadbuttons[i].grid(column=2, row=i)

            i += 1

        # Create submit button
        submitButton = Button(self, text="Submit", command=(lambda: self.convertFiles()))
        submitButton.grid(column=1, row=i)

    def convertFiles(self):
        print("Converting...")

        # Start loop until the the selected files are converted
        j = 0
        for j in range(0, self.filenumber.get()):
            # Check if all the file fields have been selected
            # If no filename is given, do nothing for this file
            # If all file fields are selected, continue
            createScnFiles = True
            # Else, display warning message
            if not self.optionMenuStatusbar[j].cget("text"):
                checkPopup = tkinter.messagebox.askquestion("Warning!", "There are still unselected"
                                                                  "file fields. Are you sure that you want to "
                                                                  "continue?", icon='warning')
                if checkPopup == 'yes':
                    createScnFiles = True

                else:
                    createScnFiles = False

            if createScnFiles == True:

                sourceFileType = self.optionMenuVar[j].get()
                print(sourceFileType)

                # Check which type of file needs to be converted to a .dat file
                if sourceFileType == "ADS-B":
                    pass
                elif sourceFileType == "Citation":
                    # Get the file location
                    datalocation = self.optionMenuStatusbar[j].cget("text")
                    # Convert files and store the converted file location in tempDatFiles
                    self.tempDatFiles.append(CitationToDatFile.CitationToDatFile(datalocation))

                elif sourceFileType == ".dat":
                    # Get the file location
                    datalocation = self.optionMenuStatusbar[j].cget("text")
                    # Store the file location in tempDatFiles, no conversion to .dat needed
                    self.tempDatFiles.append(datalocation)
                elif sourceFileType == ".mat To70":
                    pass
                elif sourceFileType == ".s06":
                    fileNumber = str(j)
                    # Get the location of the file
                    datalocation = self.optionMenuStatusbar[j].cget("text")

                    # Convert to so6
                    s06converter = so6_to_scn.readFile(datalocation, fileNumber)
                    air_traffic, aircraft = s06converter.openFile()
                    s06converter.setVariables(air_traffic, aircraft)

                    # Convert files and store the converted file location in tempDatFiles
                    self.tempDatFiles.append(s06converter.fileLocation)

                elif sourceFileType == ".scn":
                    pass

                # Merge all the .dat files, into one file

                print(self.tempDatFiles)
                with open('tempData/mergedDatFile.dat', 'w') as outfile:
                    for fname in self.tempDatFiles:
                        with open(str(fname)) as infile:
                            for line in infile:
                                outfile.write(line)

                # Convert the .datfile
                self.dataConverter = DatToScn.DatToScn('tempData/mergedDatFile.dat')

            elif createScnFiles == False:
                # continue to the next file
                pass

            print("Conversion completed")


    # This method returns the filename of the file which got selected
    def getOpenFileName(self, idx):
        idx = idx - 1

        selectedFiles = tkinter.filedialog.askopenfilenames(multiple=True)
        self.optionMenuSplittedSelectedFile[idx].set(self.tk.splitlist(selectedFiles))
        # Manipulate the string in order to get the right output
        tempStr = self.optionMenuSplittedSelectedFile[idx].get()
        tempStr = tempStr.split("'")
        tempStr = tempStr[1]

        self.optionMenuStatusbar[idx].config(text=tempStr)


    def BUGloadbuttoncreator(self):
        print(self.filenumber.get())
        if self.filenumber.get() >= 1:
            loadfileButton = Button(self, text="Open File", command=lambda num=1: self.getOpenFileName(1))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 2:
            loadfileButton = Button(self, text="Open File", command=lambda num=2: self.getOpenFileName(2))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 3:
            loadfileButton = Button(self, text="Open File", command=lambda num=3: self.getOpenFileName(3))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 4:
            loadfileButton = Button(self, text="Open File", command=lambda num=4: self.getOpenFileName(4))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 5:
            loadfileButton = Button(self, text="Open File", command=lambda num=5: self.getOpenFileName(5))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 6:
            loadfileButton = Button(self, text="Open File", command=lambda num=6: self.getOpenFileName(6))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 7:
            loadfileButton = Button(self, text="Open File", command=lambda num=7: self.getOpenFileName(7))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 8:
            loadfileButton = Button(self, text="Open File", command=lambda num=8: self.getOpenFileName(8))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 9:
            loadfileButton = Button(self, text="Open File", command=lambda num=9: self.getOpenFileName(9))
            self.optionMenuLoadbuttons.append(loadfileButton)
        if self.filenumber.get() >= 10:
            loadfileButton = Button(self, text="Open File", command=lambda num=10: self.getOpenFileName(10))
            self.optionMenuLoadbuttons.append(loadfileButton)


    '''

    # Return the file which is selected
    @property
    def returnSplittedSelectedFiles(self):
        return self.splittedSelectedFiles

'''





