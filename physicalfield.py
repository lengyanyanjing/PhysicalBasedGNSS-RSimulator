#!/usr/win/python 3.6 by DZN
import math
import cmath
import const
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# GPS L-Band frequency rain attenuation
def RainAttenuation(R, h, ele_r, ele_t):  # rain attenuation
    alpha = 24.312 * 1e-5 * R**0.9567
    rgain = math.exp(-alpha * h * (1 / math.sin(ele_r) + 1 / math.sin(ele_t)))
    return 10 * math.log10(rgain)


#    return rgain
###############################################################################
# calculate Fresnel coefficient
def Fresnel(dielectric, incidence):
    temp = math.asin(1 / math.sqrt(dielectric) * math.sin(incidence))
    fn = -1 * math.sin(incidence - temp) / math.sin(incidence + temp)
    return fn


def FresnelCoefficient(theta):
    p = 74.62 + 51.92j
    RVV = (p*math.sin(theta)-cmath.sqrt(p-math.cos(theta)**2)) / \
        (p*math.sin(theta)+cmath.sqrt(p-math.cos(theta)**2))
    RHH = (math.sin(theta)-cmath.sqrt(p-math.cos(theta)**2)) / \
        (math.sin(theta)+cmath.sqrt(p-math.cos(theta)**2))
    return (RVV + RHH) * 0.5


###############################################################################
# sea dielectric
def SeaDielectric(freq):  # unit : Hz
    epsr = 80  # seawater
    sigmac = 4  # seawater conductivity Siemens
    # eletrical properties of sea water
    lamda = const.CLIGHT / freq
    epsrc = epsr - 1j * 60 * lamda * sigmac  # complex reflection coefficient
    murc = 1  # permittivity
    return cmath.sqrt(epsrc / murc)


###############################################################################
# katzberg model calculates mss
def Katzberg(wind_speed):
    if wind_speed < 3.49:
        f = wind_speed
    elif wind_speed < 46:
        f = 6 * math.log(wind_speed) - 4
    else:
        f = (1.855 * 1e-4 * wind_speed + 0.0185) / (3.16 * 1e-3 * 0.45)
#        f=0.411*wind_speed
    sigmap = 0.45 * 3.16 * 1e-3 * f
    sigmav = 0.45 * (0.003 + 1.92 * 1e-3 * f)
    return sigmap + sigmav


###############################################################################
class Phyfield:  # 风速需要转化（MSS计算中使用的是phi0：风速和y轴的夹角）
    def __init__(self, wind_speed=5.0, wind_velocity=0.0, R=10.0):
        self.U10 = wind_speed  # simulate wind speed
        self.V10 = wind_velocity  # simulate wind velocity
        self.min_speed = 3.0  # acceptable minimum wind speed
        self.R = R  # the rain rate
        self.h = 6  # freezing height unit km
        self.epsilon = 74.62 + 51.92j  # complex dielectric constant
        self.mssx = math.sqrt(0.01)  # MSS sigmax
        self.mssy = math.sqrt(0.02)  # MSS sigmay
        self.bxy = 0.0
        self.RGain = 1.0

    def RainAttenuation(self, ele_r, ele_t):
        alpha = 24.312 * 1e-5 * self.R**0.9567
        self.RGain = math.exp(-alpha * self.h *
                              (1 / math.sin(ele_r) + 1 / math.sin(ele_t)))
        #        return 10*math.log10(self.RGain)
        return self.RGain

    def Katzberg(self):  # reference to Katzberg model
        if self.U10 < 3.49:
            f = self.U10
        elif self.U10 < 46:
            f = 6 * math.log(self.U10) - 4
        else:
            f = (1.855 * 1e-4 * self.U10 + 0.0185) / (3.16 * 1e-3 * 0.45)
#            f=0.411*self.U10
        sigmap = 0.45 * 3.16 * 1e-3 * f
        sigmav = 0.45 * (0.003 + 1.92 * 1e-3 * f)
        self.mssx = math.sqrt(
            sigmav * math.cos(self.V10)**2 + sigmap * math.sin(self.V10)**2)
        self.mssy = math.sqrt(
            sigmap * math.cos(self.V10)**2 + sigmav * math.sin(self.V10)**2)
        self.bxy = 1 / (self.mssx * self.mssy) * (
            sigmap + sigmav) * math.cos(self.V10) * math.sin(self.V10)


###############################################################################
if __name__ == '__main__':
    p = Phyfield()
    p.Katzberg()
    ###########################################################################
    wind = np.arange(0.0, 60.0)
    LPmss = np.zeros(wind.shape)
    for i, f in enumerate(wind):
        LPmss[i] = Katzberg(f)
    plt.figure(figsize=(4, 3))
    plt.plot(wind, LPmss, '.-')
    plt.xlim(0, 60)
    plt.ylim(0, 0.1)
    plt.yticks(np.arange(0.0, 0.1 + 0.01, 0.01))
    plt.xlabel('Wind speed (m/s)')
    plt.ylabel('MSS')
    plt.title('Kazberg 2013')
    plt.grid()
    plt.tight_layout()
    plt.show()
    ###########################################################################
    Rain_rate = np.arange(0.0, 70.0, 10.0)
    ele = np.arange(30.0, 90.0, 10.0)
    str_ele = [
        '30 deg.', '40 deg.', '50 deg.', '60 deg.', '70 deg.', '80. deg.'
    ]
    plt.figure(figsize=(4, 3))
    for i, value in enumerate(ele):
        result = []
        for j in Rain_rate:
            result.append(
                RainAttenuation(j, 6, value * const.D2R, value * const.D2R))
        plt.plot(Rain_rate, result, label=str_ele[i])
    plt.xlim(0, 60)
    plt.ylim(-1.4, 0)
    plt.xlabel('Rain Rate (mm/hr)')
    plt.ylabel('Rain Attenuation (dB)')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.show()
    ###########################################################################
