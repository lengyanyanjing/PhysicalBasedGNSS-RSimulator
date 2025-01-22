#!/usr/win/python 3.6 by DZN

import math
import numpy as np
import utilities as ut


class Grid:
    def __init__(self, Rs, d=1000.0, Ntheta=401, Nphi=401):
        self.R = ut.EarthRadius(Rs)  # the Earth radius at specular point
        self.d = d  # minimum surface resolution (meters)
        self.Ntheta = Ntheta  # number of grid points in the theta
        self.Nphi = Nphi  # number of grid points in the phi
        self.delta_theta = self.d / self.R  # theta increment
        self.delta_phi = self.d / self.R  # phi increment
        self.theta = [(i - math.floor(self.Ntheta / 2)) * self.delta_theta
                      for i in range(0, self.Ntheta)]
        self.phi = [(i - math.floor(self.Nphi / 2)) * self.delta_phi
                    for i in range(0, self.Nphi)]
        

# =============================================================================
#     def plotiosrange(self,dopp,inv,n):  # dopp:时延数组 inv: 等时延线间隔 n:等时延线数量 ddm
#         for i in range(n):
#             delay=inv*(i+1)
#             step=(delay-schip,delay+schip)
#             ind=np.where((step[0]<self.delay)&(self.delay<step))
#             for j in range(ind[0].shape[0]):
# =============================================================================
if __name__ == "__main__":
    g = Grid(np.array([-3560139.6739622, -3226742.9524790, 4180476.6440793]))
