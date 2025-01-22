#!/usr/win/python 3.6 By DZN
import math
import numpy as np
import const
from geometry import Geometry
from grid import Grid
from patch import Patch
from antenna import Antenna
from physicalfield import Phyfield
import matplotlib.pyplot as plt


# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
###############################################################################
# AF sinc function
def sinc(array_x):
    array_x[array_x == 0.0] = 1.0
    array_y = np.sin(const.PI * array_x) / (const.PI * array_x)
    array_y[array_x == 0.0] = 1.0
    return array_y


###############################################################################
def lamda(array_x):
    return (array_x+1)*((array_x > -1) & (array_x < 0))+(array_x == 0) + \
           (-array_x+1)*((array_x > 0) & (array_x < 1))


###############################################################################
class SDDM:
    def __init__(self, Rr, Vr, Rt, Vt, ti=0.001, nchip=200,
                 offsetchip=5, schip=0.1, nbin=100, sbin=100,
                 nsinc=10, nlamda=5):
        self.geometry = Geometry(Rr, Vr, Rt, Vt)  # init geometry
        self.grid = Grid(self.geometry.srcs_Rs)  # init grid
        self.patch = Patch(self.grid.Nphi, self.grid.Ntheta)  # init patch
        self.atenna = Antenna()  # init antenna
        self.phyfield = Phyfield()  # init phsical field
        self.ti = ti  # coherent time
        self.Nsinc = nsinc  # AF sinc
        self.Nlamda = nlamda  # AF lamda
        self.Nchip = nchip  # ddm size of chip
        self.Nbin = nbin  # ddm size of bin
        self.step_chip = schip  # chip resolution
        self.step_bin = sbin  # bin resolution
        self.chip_time = const.FREQ1_CA_T / const.FREQ1_CA
        self.chip = np.array([-i*schip+schip/2 for i in range(offsetchip, 0, -1)] +
                             [i*schip+schip/2 for i in range(0, nchip-offsetchip)])
        self.bin = [(i - math.floor(nbin / 2)) * sbin + sbin / 2
                    for i in range(0, nbin)]
        self.ddm_power = np.zeros([self.Nchip, self.Nbin])
        self.ddm_area = np.zeros([self.Nchip, self.Nbin])
        self.ddm_afpower = np.zeros([self.Nchip, self.Nbin])
        self.ddm_afarea = np.zeros([self.Nchip, self.Nbin])
        self.ddm_afbrcs = np.zeros([self.Nchip, self.Nbin])
        self.phyfield.Katzberg()

    # calculate the absolute delay/dopp/scattering area/scattering power 
    # over suface
    def SurfaceScatterPower(self):
        for i, phi in enumerate(self.grid.phi):
            for j, theta in enumerate(self.grid.theta):
                self.patch.area[i][j] = self.grid.R**2*math.cos(theta) \
                                     * self.grid.delta_phi*self.grid.delta_theta
                patch_pos = self.geometry.PatchPosSRCS(phi, theta)
                self.patch.delay[i][j] = self.geometry.RelativeDelay(patch_pos)
                self.patch.dopp[i][j] = self.geometry.RelativeDopp(patch_pos)
                self.patch.theta[i][j] = self.geometry.ScatteringAngle(
                    patch_pos)
                self.patch.brcs[i][j] = self.geometry.RCS(patch_pos, self.phyfield.mssx,
                                                          self.phyfield.mssy, self.phyfield.bxy,
                                                          self.patch.theta[i][j])
                self.patch.power[i][j] = self.patch.brcs[i][j] \
                    * self.patch.area[i][j] * self.atenna.rgain \
                    * self.atenna.tgain \
                    * self.geometry.PassLoss(patch_pos)\
                    * self.phyfield.RainAttenuation(self.geometry.ele_r, self.geometry.ele_t)
        self.patch.extremum()

    # finishing mapping from space to DD (low efficient)
    def MappingPower2DD(self):
        for i in range(self.Nchip - 1):
            step1 = self.chip[i] * self.chip_time, self.chip[
                i + 1] * self.chip_time
            delay_ind = (step1[0] < self.patch.delay) & (self.patch.delay <
                                                         step1[1])
            if delay_ind.any():
                for j in range(self.Nbin - 1):
                    step2 = self.bin[j], self.bin[j + 1]
                    dopp = self.patch.dopp[delay_ind]
                    dopp_ind = (step2[0] < dopp) & (dopp < step2[1])
                    area_ind = self.patch.area[delay_ind]
                    brcs_ind = self.patch.brcs[delay_ind]
                    power_ind = self.patch.power[delay_ind]
                    self.ddm_power[i][j] = sum(power_ind[dopp_ind])
                    self.ddm_area[i][j] = sum(area_ind[dopp_ind])
                    self.ddm_afbrcs[i][j] = sum(brcs_ind[dopp_ind])

    def AFsinc(self):
        for i in range(self.Nchip):
            for j in range(self.Nbin):
                start = j - self.Nsinc
                start = start if start > 0 else 0
                end = j + self.Nsinc + 1
                end = end if end < self.Nbin else self.Nbin
                x = (np.array(self.bin[start:end]) - self.bin[j]) * self.ti
                y = sinc(x)
                self.ddm_afpower[i][j] = np.dot(self.ddm_power[i][start:end],
                                                y * y)
                self.ddm_afarea[i][j] = np.dot(self.ddm_area[i][start:end],
                                               y * y)

    def AFlamda(self):
        for i in range(self.Nbin):
            for j in range(self.Nchip):
                start = j - self.Nlamda
                start = start if start > 0 else 0
                end = j + self.Nlamda + 1
                end = end if end < self.Nchip else self.Nchip
                x = np.array(self.chip[start:end]) - self.chip[j]
                y = lamda(x)
                self.ddm_afpower[j][i] = np.dot(y * y,
                                                self.ddm_power[start:end, i])
                self.ddm_afarea[j][i] = np.dot(y * y,
                                               self.ddm_area[start:end, i])

    def AFddm(self):
        self.AFsinc()
        self.AFlamda()

    def plotGlistenZone(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        h = ax.imshow(self.patch.power, cmap='jet')
        fig.colorbar(h)
        plt.title('Glistening Zone')
        ax.set_xlabel('Doppler (kHz)')
        ax.set_ylabel('Delay (chips)')
        fig.show()

    def plotddm(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        h = ax.imshow(
            np.flipud(self.ddm_power),
            cmap='jet',
            aspect='auto',
            interpolation=None)
        fig.colorbar(h)
        plt.title('DDM without AF Effetive')
        ax.set_xlabel('Doppler (kHz)')
        ax.set_ylabel('Delay (chips)')
        fig.show()

    def plotAFddm(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        h = ax.imshow(
            np.flipud(self.ddm_afpower),
            cmap='jet',
            aspect='auto',
            interpolation=None)
        fig.colorbar(h)
        plt.title('L0 Delay Doppler Map (counts)')
        ax.set_xlabel('Doppler (kHz)')
        ax.set_ylabel('Delay (chips)')
        fig.show()

    def plotAFRCS(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        h = ax.imshow(
            np.flipud(self.ddm_afbrcs),
            cmap='jet',
            aspect='auto',
            interpolation=None)
        fig.colorbar(h)
        plt.title('Bistatic Radar cross Section' + r'($\mathrm{m}^{2}$)')
        ax.set_xlabel('Doppler (kHz)')
        ax.set_ylabel('Delay (chips)')
        fig.show()

    def plotEffectiveArea(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        h = ax.imshow(
            np.flipud(self.ddm_afarea),
            cmap='jet',
            aspect='auto',
            interpolation=None)
        cbar = fig.colorbar(h)
        cbar.solids.set_edgecolor("face")
        plt.title('Effective Scattering Area' + r'($\mathrm{m}^{2}$)')
        #        ax.set_xticks(np.arange(self.bin[0],self.bin[-1],self.step_bin))
        #        ax.set_xticklabels(np.arange(self.bin[0],self.bin[-1],self.step_bin))
        ax.set_xlabel('Doppler (kHz)')
        ax.set_ylabel('Delay (chips)')
        ax.axis('normal')
        fig.tight_layout()
        fig.show()


###############################################################################
if __name__ == '__main__':
    # postion and velocity of transimiter in WGS-84 reference frame
    Rt = np.array([-11178791.991294, -13160191.204988, 20341528.127540])
    Vt = np.array([2523.258023, -361.592839, 1163.748104])
    # postion and veloctiry of receiver in WGS-84 reference frame
    Rr = np.array(
        [-4069896.7033860330, -3583236.9637350840, 4527639.2717581640])
    Vr = np.array([-4738.0742342063, -1796.2525689964, -5654.9952013657])
    ddm = SDDM(Rr, Vr, Rt, Vt)

    ddm.SurfaceScatterPower()
    ddm.plotGlistenZone()
    ddm.MappingPower2DD()
    ddm.plotddm()
    ddm.AFddm()
    ddm.plotAFddm()
    ddm.plotEffectiveArea()
