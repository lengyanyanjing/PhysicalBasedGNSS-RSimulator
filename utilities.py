#!/usr/win/phython 3.6 by DZN

import math
import numpy as np
from scipy.linalg import norm
import scipy.constants as const


###############################################################################
# calculate the intersection of R1 vector and R2 vector
def IntersectionAngle(R1, R2):
    R1_unit = R1 / norm(R1)
    R2_unit = R2 / norm(R2)
    return math.acos(np.dot(R1_unit, R2_unit))


###############################################################################
# coordinates conversion from WGS-84 to Geodetic coordinates
def xyz2llh(xyz_pos):
    llh = np.zeros(3)
    # earth constants
    a = 6378137.0
    #   WGS-84 equatorial radius (m).
    f = 1 / 298.257223563
    #   WGS-84 Flattening.
    e2 = f * (2.0 - f)
    #   second eccentricity

    r2 = np.dot(xyz_pos[:2], xyz_pos[:2])
    z = xyz_pos[2]
    zk = 0.0
    while (abs(z - zk)):
        zk = z
        sinp = z / math.sqrt(r2 + z * z)
        v = a / math.sqrt(1.0 - e2 * sinp * sinp)
        z = xyz_pos[2] + v * e2 * sinp

    llh[0] = math.atan(
        z / math.sqrt(r2)) if r2 > 1e-12 else (const.pi / 2.0
                                               if xyz_pos[2] > 0.0 else
                                               -const.pi / 2.0)
    llh[1] = math.atan2(xyz_pos[1], xyz_pos[0]) if r2 > 1e-12 else 0.0
    llh[2] = math.sqrt(r2 + z * z) - v

    return llh


###############################################################################
# transform ecef to specular referece coodinate system
def xyz2enu(xyz_ref, xyz_trf):  # para: position of specular in ecef
    enu = np.zeros(3)
    E = np.zeros([3, 3])  # tansform matrix
    llh = xyz2llh(
        xyz_ref)  # tansform refrence coodinate to Ellipsoidal coordinates

    sinp = math.sin(llh[0])
    cosp = math.cos(llh[0])
    sinl = math.sin(llh[1])
    cosl = math.cos(llh[1])

    E[0][0] = -sinl
    E[0][1] = -sinp * cosl
    E[0][2] = cosp * cosl
    E[1][0] = cosl
    E[1][1] = -sinp * sinl
    E[1][2] = cosp * sinl
    E[2][0] = 0.0
    E[2][1] = cosp
    E[2][2] = sinp

    enu = np.dot(xyz_trf, E)

    return enu


###############################################################################
# calculate the average earth radius at ECEF point
def EarthRadius(xyz_pos):
    # WGS-84 constants
    a = 6378137.0
    #   WGS-84 equatorial radius (m).
    f = 1 / 298.257223563
    #   WGS-84 Flattening.
    e2 = f * (2.0 - f)
    #  e**2

    distance = norm(xyz_pos)
    theta = math.asin(xyz_pos[2] / distance)
    temp = 1 - e2 * math.cos(theta)**2

    return a * math.sqrt((1 - e2) / temp)


###############################################################################

if __name__ == '__main__':
    # postion and velocity of transimiter in WGS-84 reference frame
    Rt = np.array(
        [-4069896.7033860330, -3583236.9637350840, 4527639.2717581640])
    Vt = np.array([2523.258023, -361.592839, 1163.748104])
    # postion and veloctiry of receiver in WGS-84 reference frame
    Rr = np.array([-11178792.991294, -13160191.204988, 20341528.127540])
    Vr = np.array([-4738.0742342063, -1796.2525689964, -5654.9952013657])
    g = xyz2enu(Rr, Vr)
