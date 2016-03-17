# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:58:03 2016

@author: lei
"""
from __future__ import print_function
import sys

import numpy as np
from scipy.integrate import trapz, cumtrapz
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import rcParams

from FPSDP.GeneralSettings.UnitSystem import cgs
import FPSDP.Diagnostics.ECEI.ECEI2D.Reciprocal as rcp
from FPSDP.Diagnostics.ECEI.ECEI2D.Detector2D import GaussianAntenna
import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp

rcParams['figure.figsize'] = [16, 12]
rcParams['font.size'] = 18

c = cgs['c']
keV = cgs['keV']
e = cgs['e']
me = cgs['m_e']

tp.set_parameter2D(Te_0 = 10*keV, Te_shape='uniform', ne_shape='uniform')
p2d = tp.create_profile2D()
p2d.setup_interps()

omega = 8e11
k = 25.102
# single frequency detector
detector = GaussianAntenna(omega_list=[omega], k_list=[k], power_list=[1], 
                           waist_x=175, waist_y=2, waist_z=2, w_0y=2, 
                           tilt_v=0, tilt_h=0)
                           
ece = rcp.ECE2D(plasma=p2d,detector=detector, polarization='X', max_harmonic=2,
                max_power=2, weakly_relativistic=True, isotropic=True)

X1D = np.linspace(251, 150, 100)
Y1D = np.linspace(-30, 30, 65)
Z1D = np.linspace(-30, 30, 65)

ece.set_coords([Z1D, Y1D, X1D])

#ece.view_point
#Ps = ece.diagnose(debug=True, auto_patch=True)

