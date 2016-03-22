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
import FPSDP.Diagnostics.ECEI.ECEI2D.Reciprocity as rcp
from FPSDP.Diagnostics.ECEI.ECEI2D.Imaging import ECEImagingSystem
from FPSDP.Diagnostics.ECEI.ECEI2D.Detector2D import GaussianAntenna
import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp

rcParams['figure.figsize'] = [16, 12]
rcParams['font.size'] = 18

c = cgs['c']
keV = cgs['keV']
e = cgs['e']
me = cgs['m_e']

tp.set_parameter2D(Te_0 = 10*keV, Te_shape='uniform', ne_shape='Hmode', 
                   dte_te=0.2, dne_ne=0.1, dB_B=0, NR=100, NZ=40, 
                   DownLeft=(-40, 100), UpRight=(40, 300), 
                   timesteps=np.arange(5) )
p2d = tp.create_profile2D(random_fluctuation=True)
p2d.setup_interps()

omega = 8e11
k = omega/c
# single frequency detector
detector1 = GaussianAntenna(omega_list=[1.1*omega], k_list=[1.1*k], 
                            power_list=[1], 
                           waist_x=172, waist_y=2, waist_z=2, w_0y=2, 
                           tilt_v=0, tilt_h=0)
                           
detector2 = GaussianAntenna(omega_list=[omega], k_list=[k], power_list=[1], 
                           waist_x=175, waist_y=2, waist_z=2, w_0y=2, 
                           tilt_v=0, tilt_h=0)
                           
detector3 = GaussianAntenna(omega_list=[omega*0.9], k_list=[0.9*k], 
                            power_list=[1], 
                           waist_x=180, waist_y=2, waist_z=2, w_0y=2, 
                           tilt_v=0, tilt_h=0)

ece = rcp.ECE2D(plasma=p2d, detector=detector1, polarization='X',
                max_harmonic=2, max_power=2, weakly_relativistic=True,
                isotropic=True)
                          
ece_iso = rcp.ECE2D(plasma=p2d,detector=detector1, polarization='X', 
                    max_harmonic=2, max_power=2, weakly_relativistic=False, 
                    isotropic=True)
ece_ani = rcp.ECE2D(plasma=p2d,detector=detector1, polarization='X', 
                    max_harmonic=2, max_power=2, weakly_relativistic=False, 
                    isotropic=False)
X1D = np.linspace(251, 150, 100)
Y1D = np.linspace(-20, 20, 65)
Z1D = np.linspace(-40, 20, 65)

ece_iso.set_coords([Z1D, Y1D, X1D])
ece_ani.set_coords([Z1D, Y1D, X1D])
ece.set_coords([Z1D, Y1D, X1D])

ecei = ECEImagingSystem(p2d, [detector1, detector2, detector3], max_harmonic=2,
                        max_power=2)
                        
ecei.set_coords([Z1D, Y1D, X1D])

#ece.view_point
#Ps = ece.diagnose(debug=True, auto_patch=True)



