# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 09:19:29 2021

@author: nino_
"""

# add new wind field
        data = self.extract_wind(grb, self.lat0, self.lon0, self.lat1, self.lon1).T
        data = data[np.lexsort((data[:,2], data[:,1], data[:,0]))]  # Sort by lat, lon, alt
        data = np.concatenate((data, np.degrees(np.arctan2(data[:,3], data[:,4])).reshape(-1,1), 
                                np.sqrt(data[:,3]**2 + data[:,4]**2).reshape(-1,1)), axis=1)  # Append direction and speed to data
        data[:,2] = data[:,2]/ft  # input WindSim requires alt in ft
        data[:,6] = data[:,6]/kts # input WindSim requires spd in kts
        splitvals = np.hstack((0, np.where(np.diff(data[:,1], axis=0))[0]+1, len(data))) # Find new lat, lon pair values in data
        
        # Construct flattend winddata input for add wind function
        for i in range(len(splitvals) - 1):
            lat = data[splitvals[i], 0]
            lon = data[splitvals[i], 1]
            winddata = data[splitvals[i]:splitvals[i+1], [2, 5, 6]].flatten()
            # WindSim.add(self, lat, lon, *winddata)
            # super().add(lat, lon, *winddata) # TODO: change if inherited from WindSim
            self.add(lat, lon, *winddata)