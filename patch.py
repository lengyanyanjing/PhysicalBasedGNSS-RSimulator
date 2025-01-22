#!/usr/win/python 3.6 by DZN
import numpy as np


class Patch:
    def __init__(self, nphi, ntheta):
        self.area = np.zeros([nphi, ntheta])  # the area over the surface
        self.delay = np.zeros([nphi, ntheta])  # delay of surface patch
        self.dopp = np.zeros([nphi, ntheta])  # dopp of surface patch
        self.theta = np.zeros([nphi, ntheta])  # scattering angle
        self.brcs = np.zeros([nphi, ntheta])  # bistatic radar scattering area
        self.power = np.zeros([nphi, ntheta])  # scatering power

    def extremum(self):
        self.mindelay = 0.0
        self.maxdelay = self.delay.max()
        self.mindopp = self.dopp.min()
        self.maxdopp = self.dopp.max()
        self.side_len_dopp = max(abs(self.mindopp), abs(self.maxdopp))
