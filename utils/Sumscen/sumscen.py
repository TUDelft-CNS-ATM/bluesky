from include.timtxt import tim2txt,txt2tim
from include.filedialogs import askfileopen,askfilesave

print("PTool to merge several scenario files and sort them chronologically")

inscen = "dummy"
i = 1
infiles = []
print("Select the .scn files to merge using time on line (Press Cancel when ready):")
while inscen.strip()!="":
    inscen = askfileopen("Enter scenario file "+str(i)+" to read:")
    infiles.append(inscen)

del infiles[-1] # Remove empty string terminator from filenames list
outfname = askfilesave("Enter an output scenario file name:")
if outfname.lower().count(".scn")==0:
    oytfname=outfname+".scn"

if outfname =="":
    print("No output file selected.")
    print("Using default out: summedscen.scn")
    outfname = "summedscen.scn"

# Get lines
print("Reading files:")
lines = []
for fname in infiles:
    if fname.lower().count(".scn")>0:
        f = open(fname)
        print(fname)
    else:
        f = open(fname+".scn")
        print(fname+".scn")

    readlines = f.readlines()
    t = -1.0 # default time for comment lines at start
    for line in readlines:
        # If contains time, update time
        if line.count(">")>0:
            idx = line.index(">")
            t = txt2tim(line[:idx].strip())

        # Store alle lines, incl. comments and empty lines
        lines.append([t,line])
    f.close()

# Write lines
# Bubblesort using only times! So no Python sort, but us our own
# which only changes when necessary using times only
# Keeps correct order commands and logical comments together

ready = False
print("Sorting records")
while not ready:
    ready = True
    for i in range(len(lines)-1):
        if lines[i][0]>lines[i+1][0]:
            lines[i],lines[i+1]=lines[i+1],lines[i]
            ready = False

outfile = open(outfname,"w")
print("Writing",outfname,"...")
for line in lines:
    outfile.write(line[1])

outfile.close()

# The end
print("Ok.")
        
        
        
    
    
