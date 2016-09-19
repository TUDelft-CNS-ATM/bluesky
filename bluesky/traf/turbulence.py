import numpy as np
from ..tools.aero import Rearth

class Turbulence:
	def __init__(self,traf):
		self.traf = traf
		self.active = False
		self.SetStandards([0, 0.1, 0.1])

	def SetNoise(self,n):
		self.active = n

	def SetStandards(self,s):
		self.sd = np.array(s) # m/s standard turbulence  (nonnegative)
		# in (horizontal flight direction, horizontal wing direction, vertical)
		self.sd=np.where(self.sd>1e-6,self.sd,1e-6)

	def Woosh(self,dt):
		if not self.active:
			return

		timescale=np.sqrt(dt)
		# Horizontal flight direction
		turbhf=np.random.normal(0,self.sd[0]*timescale,self.traf.ntraf) #[m]

		# Horizontal wing direction
		turbhw=np.random.normal(0,self.sd[1]*timescale,self.traf.ntraf) #[m]

		# Vertical direction
		turbalt=np.random.normal(0,self.sd[2]*timescale,self.traf.ntraf) #[m]

		trkrad=np.radians(self.traf.trk)
		# Lateral, longitudinal direction
		turblat=np.cos(trkrad)*turbhf-np.sin(trkrad)*turbhw #[m]
		turblon=np.sin(trkrad)*turbhf+np.cos(trkrad)*turbhw #[m]

		# Update the aircraft locations
		self.traf.alt = self.traf.alt + turbalt
		self.traf.lat = self.traf.lat + np.degrees(turblat/Rearth)
		self.traf.lon = self.traf.lon + np.degrees(turblon/Rearth/self.traf.coslat)