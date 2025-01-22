#!/usr/win/python 3.6 by DZN

import math
import numpy as np
from scipy.linalg import norm
from utilities import EarthRadius, IntersectionAngle, xyz2enu
from physicalfield import FresnelCoefficient, Fresnel
import const


###############################################################################
# elevation
def elevation(xyz_ref, xyz_trf):
    enu_pos = xyz2enu(xyz_ref, xyz_trf)
    return math.asin(enu_pos[2] / norm(enu_pos))


###############################################################################
# calculate the incidence of signal
def Incidence(Rs, Rt):
    st_unit = (Rt - Rs) / norm(Rt - Rs)  # incoming vector
    s_unit = Rs / norm(Rs)  # normal vector
    return abs(math.acos(np.dot(st_unit, s_unit)))  # incidence


###############################################################################
# convert the ecef position to specular reference coodinate system
def ecef2srcs(Rr, Rt, Rs, v):
    srcs_v = np.zeros(3)
    Rr_unit = Rr / norm(Rr)
    Rt_unit = Rt / norm(Rt)
    temp = Rr_unit - Rt_unit
    tr_enu = xyz2enu(Rs, temp / norm(temp))
    v_enu = xyz2enu(Rs, v)
    delta = math.atan2(v_enu[0], v_enu[1]) - math.atan2(tr_enu[0], tr_enu[1])

    srcs_v[0] = norm([v_enu[:2]]) * math.cos(delta)
    srcs_v[1] = norm([v_enu[:2]]) * math.sin(delta)
    srcs_v[2] = v_enu[2]
    return srcs_v


###############################################################################
# calculate the specular point
def RT2S(r_pos, t_pos):
    # reference to E2ES Technical Memo
    r = EarthRadius(r_pos)
    s = s_temp = r_pos * (r / norm(r_pos))

    k = 10000
    corr = 10000.0
    itera = 1
    while (corr > 1.0e-3):
        s_temp = r_pos - s
        unit_vector_S2R = s_temp / norm(s_temp)
        s_temp = t_pos - s
        unit_vector_S2T = s_temp / norm(s_temp)
        mid_vector = unit_vector_S2R + unit_vector_S2T
        s_temp = s + k * mid_vector
        # constain to earth surface
        r = EarthRadius(s_temp)
        s_temp = (s_temp / norm(s_temp)) * r
        corr = norm(s_temp - s)
        # new specular point
        s = s_temp
        # adjust gain
        k = 1e4 if (corr > 10.0) else 1e3
        itera += 1
        if itera > 1e+4:
            break
    # test Snell's Law
    unit_vector_S2R = (r_pos - s) / norm(r_pos - s)
    unit_vector_S2T = (t_pos - s) / norm(t_pos - s)
    unit_vector_S = s / norm(s)
    incidence = math.acos(np.dot(unit_vector_S2T, unit_vector_S))
    reflectation = math.acos(np.dot(unit_vector_S2R, unit_vector_S))
    diff_angle = abs(incidence - reflectation) * const.R2D * 3600
    return s


###############################################################################
# MSS probability density function
def MSSpdf(x, y, mssx, mssy, bxy):
    ind = -1.0 / (2.0 - 2.0 * bxy * bxy) * (
        x * x / mssx**2 - 2.0 * bxy * x * y / mssx / mssy + y * y / mssy**2)
    return 1.0 / (2.0 * const.PI * mssx * mssy * math.sqrt(1.0 - bxy * bxy)
                  ) * math.exp(ind)


###############################################################################
# define geometry class
class Geometry:
    # initial with ecef position and velocity of transmiter and receiver
    def __init__(self, Rr, Vr, Rt, Vt, Freq=const.FREQ1):
        self.ecef_Rr = np.array(Rr)
        self.ecef_Vr = np.array(Vr)
        self.ecef_Rt = np.array(Rt)
        self.ecef_Vt = np.array(Vt)
        self.freq = Freq
        self.ecef_Rs = RT2S(self.ecef_Rr, self.ecef_Rt)  # specular point
        self.ele_r = elevation(self.ecef_Rs, self.ecef_Rr)
        self.ele_t = elevation(self.ecef_Rs, self.ecef_Rt)
        self.incidence = Incidence(self.ecef_Rs,
                                   self.ecef_Rt)  # signal incidence
        self.c1 = IntersectionAngle(self.ecef_Rt, self.ecef_Rs)  # c1 angle
        self.c2 = IntersectionAngle(self.ecef_Rr, self.ecef_Rs)  # c2 angle
        self.srcs_Rs = np.array([0.0, 0.0, EarthRadius(
            self.ecef_Rs)])  # specular referece frame of specular point
        self.srcs_Rr = np.array([norm(self.ecef_Rr)*math.sin(self.c2),
                                0.0, norm(self.ecef_Rr)*math.cos(self.c2)])
        self.srcs_Rt = np.array([-norm(self.ecef_Rt)*math.sin(self.c1),
                                0.0, norm(self.ecef_Rt)*math.cos(self.c1)])
        self.srcs_Vr = ecef2srcs(self.ecef_Rr, self.ecef_Rt, self.ecef_Rs,
                                 self.ecef_Vr)
        self.srcs_Vt = ecef2srcs(self.ecef_Rr, self.ecef_Rt, self.ecef_Rs,
                                 self.ecef_Vt)
        self.srcs_Vs = np.zeros(3)
        self.sp_delay = (norm(self.srcs_Rt-self.srcs_Rs) +
                         norm(self.srcs_Rr-self.srcs_Rs)) / const.CLIGHT

    def Elevation(self, ecef_pos):
        enu_pos = xyz2enu(self.ecef_Rs, ecef_pos)
        return math.asin(enu_pos[2] / norm(enu_pos))

    # the unit vector from specular to receiver in specular reference frame
    def UnitVectorSR(self, Rs):
        return (self.srcs_Rr - Rs) / norm(self.srcs_Rr - Rs)

    # the unit vector from specular to transmiter in specular reference frame
    def UnitVectorST(self, Rs):
        return (self.srcs_Rt - Rs) / norm(self.srcs_Rt - Rs)

    def losVr(self, Rs):  # the line-of-sight velocity to receiver
        return np.dot(-self.UnitVectorSR(Rs), self.srcs_Vr)

    def losVt(self, Rs):  # the line-of-sight velocity to transmiter
        return np.dot(-self.UnitVectorST(Rs), self.srcs_Vt)

    def PassLoss(self, patch_pos):
        return 1/((norm(self.srcs_Rr-patch_pos))**2
                  * (norm(self.srcs_Rt-patch_pos))**2)

    def Delay(self, patch_pos):
        return (norm(self.srcs_Rr-patch_pos) +
                norm(self.srcs_Rt-patch_pos))/const.CLIGHT

    def RelativeDelay(self, patch_pos):
        return self.Delay(patch_pos) - self.sp_delay

    def SpecularDopp(self):
        self.sp_dopp = -self.freq/const.CLIGHT * \
                     (self.losVr(self.srcs_Rs)+self.losVt(self.srcs_Rs))
        return self.sp_dopp

    def RelativeDopp(self, patch_pos):
        Dt = (self.losVt(patch_pos)) * self.freq / const.CLIGHT
        return -self.SpecularDopp()-self.losVr(patch_pos) * \
            (self.freq+Dt)/const.CLIGHT-Dt

    def ScatteringAngle(self, patch_pos):
        return 0.5*math.acos(np.dot(-(self.UnitVectorST(patch_pos)),
                                    self.UnitVectorSR(patch_pos)))

    def ScatteringVector(self, patch_pos):
        return (2*const.PI*self.freq/const.CLIGHT) * \
                (self.UnitVectorST(patch_pos) + self.UnitVectorSR(patch_pos))

    def RCS(self, patch_pos, mssx, mssy, bxy, theta):
        sc_vector = self.ScatteringVector(patch_pos)
        x = -sc_vector[0] / sc_vector[2]
        y = -sc_vector[1] / sc_vector[2]
        #        fn=norm(FresnelCoefficient(theta))
        fn = Fresnel(73, theta)
        return fn * fn * (norm(sc_vector))**4 / (
            sc_vector[2])**4 * MSSpdf(x, y, mssx, mssy, bxy)

    def PatchPosSRCS(
            self, phi, theta
    ):  # the patch position in specular reference coordinate system
        sinp = math.sin(phi)
        cosp = math.cos(phi)
        sint = math.sin(theta)
        cost = math.cos(theta)

        RA = np.mat([[-cosp, 0, sinp], [0, 1, 0], [sinp, 0, cosp]])
        RB = np.mat([[1, 0, 0], [0, cost, -sint], [0, sint, cost]])

        return (self.srcs_Rs * RA * RB).flatten().A[0]


###############################################################################
if __name__ == '__main__':
    # postion and velocity of transimiter in WGS-84 reference frame
    Rt = np.array([-11178791.991294, -13160191.204988, 20341528.127540])
    Vt = np.array([2523.258023, -361.592839, 1163.748104])
    # postion and veloctiry of receiver in WGS-84 reference frame
    Rr = np.array(
        [-4069896.7033860330, -3583236.9637350840, 4527639.2717581640])
    Vr = np.array([-4738.0742342063, -1796.2525689964, -5654.9952013657])
    g = Geometry(Rr, Vr, Rt, Vt)
    #    RT2S(Rr,Rt)
    phi = -0.0157313035111680
    patch_pos = g.PatchPosSRCS(phi, phi)
    print(g.RCS(patch_pos, math.sqrt(0.01), math.sqrt(0.02), 0.0, g.incidence))
    print(ecef2srcs(g.ecef_Rr, g.ecef_Rt, g.ecef_Rs, g.ecef_Vr))
    g.RelativeDopp(patch_pos)
