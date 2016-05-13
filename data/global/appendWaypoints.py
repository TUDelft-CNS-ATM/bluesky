'''
appendWaypoints.py

This script creates a new waypoints.dat file that appends all the data from 
the default waypoints.dat file and the user defined waypoint files. 

If the variable interactive is True, then the script asks the user which files
he/she would like to append to the default waypoints.dat file. If the variable 
interactive is False, it appends all user defined files to waypoints.dat file.

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

print "Running appendWaypoints.py"

#%% Initialize Variables
interactive = False # True-> use prompts to ask user for the files to append

#%% Find all .dat files

# find all the dat files
datFiles = [g for g in os.listdir(".") if g.count("waypoint")>0 and g.count(".dat")>0]  

# remove the default BlueSky dat files from the datFiles list
datFiles.remove('waypoints.dat')


#%% Get data from user defined waypoint .dat files

# list to store data from user waypoint files
userWaypoints = []

 
# ask user
if interactive:
    while True:
        
        # Print Menu
        if len(datFiles)>0:
            print
            print
            print "Select the waypoint file to append to default waypoints.dat file:"
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
        
        # combine the lines in the current dat file to total userWaypoints list
        userWaypoints = userWaypoints + lines

# import all waypoint files    
else:
    
    for i in range(len(datFiles)):
        fname = datFiles[i]
        
        # Read the selected file
        f = open(fname,"r")
        lines = f.readlines()
        f.close()
        
        # combine the lines in the current dat file to userWaypoints list
        userWaypoints = userWaypoints + lines               
  
# remove any empty rows in userWaypoints
userWaypoints = filter(lambda x: not re.match(r'^\s*$', x), userWaypoints)       
userWaypoints = filter(lambda x: not re.match(r'#', x), userWaypoints)   
    
#%% Read in the default waypoints file 

f = open('waypoints.dat','r')    
waypoints = f.readlines()
f.close()    

# remove any empty rows in waypoints
waypoints = filter(lambda x: not re.match(r'^\s*$', x), waypoints)  
#%% Combine all dat files and re-write to waypoints.dat
 
# combine default waypoints and user waypoints
waypoints = waypoints + userWaypoints

# Remove duplicates using set theory and loop. Very Fast. Order maintained
uniqueWaypoints = []
knownWaypoints = set()
for i in waypoints:
    if i in knownWaypoints:
        continue
    uniqueWaypoints.append(i)
    knownWaypoints.add(i)
waypoints=uniqueWaypoints    
 
# write waypoints to file 
print
print "Writing to waypoints.dat"
f = open('waypoints.dat','w')
f.writelines(waypoints)
f.close()

print
print 'Done!'