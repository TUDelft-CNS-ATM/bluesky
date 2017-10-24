f = open("airportsTra.dat")
lines = f.readlines()
f.close()

g = open('airportsTra.scn','w')
g.write("# Call this scenario with the CALL or PCALL command to define the waypoints in this file\n\n")
for line in lines:
    fields= line.split(",")
    if len(fields)>=4:
        name = fields[0].strip()
        lat = fields[2].strip()
        lon = fields[3]
        newline = "00:00:00.00>DEFWPT APT"+name+","+lat+","+lon+"\n"
        g.write(newline)

g.close()

f = open("waypointsExp.dat")
lines = f.readlines()
f.close()

g = open('waypointExp.scn','w')
g.write("# Call this scenario with the CALL or PCALL command to define the waypoints in this file\n\n")

for line in lines:
    fields= line.split(",")
    if len(fields)>=4:
        name = fields[0].strip()
        lat = fields[2].strip()
        lon = fields[3]
        newline = "00:00:00.00>DEFWPT WPT"+name+","+lat+","+lon+"\n"
        g.write(newline)

g.close()

