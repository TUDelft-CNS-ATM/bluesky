'''
appendAirports.py

This script creates a new airports.dat file that appends all the data from 
the default airports.dat file and the user defined airport files. 

If the variable interactive is True, then the script asks the user which files
he/she would like to append to the default airports.dat file. If the variable 
interactive is False, it appends all user defined files to airports.dat file.

'''
# import necessary packages
import matplotlib.pyplot as plt
import os
import regex as re

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Close all figures
plt.close("all")    

# supress silly warnings
import warnings
warnings.filterwarnings("ignore") 

print "Running appendAirports.py"

#%% Initialize Variables
interactive = False # True-> use prompts to ask user for the files to append

#%% Find all .dat files

# find all the dat files
datFiles = [g for g in os.listdir(".") if g.count("airport")>0 and g.count(".dat")>0]  

# remove the default BlueSky dat files from the datFiles list
datFiles.remove('airports.dat')

#%% Get data from user defined airport .dat files

# list to store data from user airport files
userAirports = []

 
# ask user
if interactive:
    while True:
        
        # Print Menu
        if len(datFiles)>0:
            print
            print
            print "Select the desired airport file to append to default airports.dat file:"
            for i in range(len(datFiles)):
                print i+1,". ",datFiles[i]
            print "-1 .  Quit"
            print
        else:
            break
        
        # Determine file name of choice. -1 to quit loop
        choice = int(input("Your choice:"))    
        if choice==-1:
            break
        else:
            fname = datFiles[choice-1]
            # remove this file from datFiles list so that it is not displayed or 
            # appended again
            datFiles.remove(fname)
    
        # Read the selected file
        f = open(fname,"r")
        lines = f.readlines()
        f.close()
        
        # combine the lines in the current dat file to total userAirports list
        userAirports = userAirports + lines

# import all airport files    
else:
    
    for i in range(len(datFiles)):
        fname = datFiles[i]
        
        # Read the selected file
        f = open(fname,"r")
        lines = f.readlines()
        f.close()
        
        # combine the lines in the current dat file to userAirports list
        userAirports = userAirports + lines               
  
# remove any empty rows in usedAirports.  
userAirports = filter(lambda x: not re.match(r'^\s*$', x), userAirports)       
    
#%% Read in the default airports file 

f = open('airports.dat','r')    
airports = f.readlines()
f.close()    

# remove any empty rows in usedAirports. 
airports = filter(lambda x: not re.match(r'^\s*$', x), airports)  

#%% Combine all dat files and re-write to airports.dat
 
# combine default airports and user airports
airports = airports + userAirports

# Option 1: Remove duplicates in case this script is (accidently) run more than 
# once for same user airport .dat files. The below logic keeps the order of the file same.
#uniqueAirports = []
#for i in airports:
#    if i not in uniqueAirports:
#        uniqueAirports.append(i)
#airports = uniqueAirports   

# Option 2: Remove duplicates using set theory and loop. Very Fast. Order maintained
uniqueAirports = []
knownAirports = set()
for i in airports:
    if i in knownAirports:
        continue
    uniqueAirports.append(i)
    knownAirports.add(i)
airports=uniqueAirports    
 
# Option 3: Remove duplicates without loop. But the order of the list is not
# maintained.
#airports = list(set(airports))

# write airports to file 
print
print "Writing to airports.dat"
f = open('airports.dat','w')
f.writelines(airports)
f.close()

print
print 'Done!'