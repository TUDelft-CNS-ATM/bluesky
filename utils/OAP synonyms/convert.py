f = open("synonym.new","r")
lines = f.readlines()
f.close()

syn2oaplst = ['A124-B744', 'A139-EC35','A140-E75L', 'A148-E75L', 'A306-A319',
           'A30B-A319', 'A310-A319', 'A318-A319', 'A319',
           'A320', 'A321', 'A332', 'A333',
           'A342-B777', 'A343-A320', 'A345-A320', 'A346-A320',
           'A388', 'A3ST-B744', 'AN24-E75L', 'AN28-E75L',
           'AN30-E75L', 'AN32-E75L', 'AN38-E75L', 'AT43-E75L',
           'AT45-E75L', 'AT72-E75L', 'AT73-E75L', 'AT75-E75L',
           'ATP-E75L', 'B190-E190', 'B350-E75L', 'B462-E190',
           'B463-E190', 'B703-B748', 'B712-E190', 'B722-E190',
           'B732-B734', 'B733-B734', 'B734', 'B735-B734',
           'B736-B734', 'B737', 'B738', 'B739',
           'B742-B744', 'B743-B744', 'B744', 'B748',
           'B752-B737', 'B753-B737', 'B762-B772', 'B763-B772',
           'B764-B772', 'B772', 'B773-B772', 'B77L-B772',
           'B77W', 'B788', 'BA11-B737', 'BE20-E75L',
           'BE30-E75L', 'BE40-E75L', 'BE58-E75L', 'BE99-E75L',
           'BE9L-E75L', 'C130-E190', 'C160-E75L', 'C172-E175L',
           'C182-E75L', 'C25A-E75L', 'C25B-E75L', 'C25C-E75L',
           'C421-E75L', 'C510-E190', 'C525-E190', 'C550-E190',
           'C55B-E190','C551-E190', 'C560-E190', 'C56X-E190', 'C650-E190',
           'C680-E190', 'C68A-E190','C750-E190', 'CL60-E190', 'CRJ1-E190','CRJ2-E190',
           'CRJ7-E190', 'CRJ9-E190', 'D228-E75L', 'D328-E75L',
           'DA42-E75L', 'DC10-B744', 'DC87-B737', 'DC93-B737',
           'DC94-B737', 'DH8A-B737', 'DH8C-B737', 'DH8D-B737',
           'E120-E190', 'E135-E190', 'E145-E190', 'E170-E75L',
           'E190', 'E50P-E190', 'E55P-E190', 'EA50-E190',
           'F100-E190', 'F27-E75L', 'F28-E190', 'F2TH-E190',
           'F50-E75L', 'F70-E75L', 'F900-E190', 'FA10-E190',
           'FA20-E190', 'FA50-E75L', 'FA7X-E190', 'FGTH-E190',
           'FGTL-E190', 'FGTN-E190', 'GL5T-E190', 'GLEX-E190',
           'GLF5-E190', 'H25A-E190', 'H25B-E190', 'H47-E190', 'IL76-B767',
           'IL86-B767', 'IL96-B767', 'JS32-E190', 'JS41-E190',
           'L101-B744', 'LJ35-E190', 'LJ45-E190', 'LJ60-E190',
           'MD11-B777', 'MD82-B738', 'MD83-B738', 'MU2-E75L','M20F-E75L',
           'P180-E75L', 'P28A-E75L', 'P28U-E75L', 'P46T-E75L',
           'PA27-E75L', 'PA31-E75L', 'PA34-E75L', 'PA44-E75L',
           'PA46-E75L', 'PAY2-E75L', 'PAY3-E75L', 'PC12-E75L',
           'PRM1-E75L', 'RJ1H-E75L', 'RJ85-E75L', 'SB20-E75L',
           'SF34-E75L', 'SH36-EC35', 'SR22-E75L', 'SU95-B767',
           'SW4-E75L', 'T134-B767', 'T154-B767', 'T204-B767',
           'TAMP-E75L',
           'TB20-E75L', 'TB21-E75L', 'TBM7-E75L', 'TBM8-E75L',
           'YK40-E75L', 'YK42-E75L']

# Convert syno2oap to dictionary

outlines = []
syn2oap = {}
oap = []
for s in syn2oaplst:
    if s.count("-")>0:
        syn,oap = s.split("-")
        syn2oap[syn]=oap
        outlines.append(syn+"="+oap)

oaplst = ["A319","A320","A321","A332","A333","A359","A388",
          "B734","B737","B738","B739","B744","B748","B772",
          "B77W","B788","B789","E75L","E190","E195"]


for line in lines:

    # Skip comment lines marked with CC
    if line[:2]=="CD":
        acid = line[5:9].strip()
        manu = line[12:32].strip()
        model = line[32:57].strip()
        syno = line[57:61].replace("_","").strip()
        if syno not in oaplst:
            syno = syn2oap[syno]
        outlines.append(acid+"="+syno+"\t# "+manu+" "+model)

g = open("synonym.dat","w")
outlines.sort()
i = 0
while i<len(outlines)-1:
    if outlines[i][:4]==outlines[i+1][:4]:
        outlines = outlines[:i]+outlines[i+1:]
    else:
        i = i+1
    
for txt in outlines:
    print(txt,file=g)
g.close()
        
