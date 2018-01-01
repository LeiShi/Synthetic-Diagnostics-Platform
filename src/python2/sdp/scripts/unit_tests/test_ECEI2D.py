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
import ipyparallel as ipp

from sdp.settings.unitsystem import cgs
import sdp.diagnostic.ecei.ecei2d.ece as rcp
from sdp.diagnostic.ecei.ecei2d.imaging import ECEImagingSystem
from sdp.diagnostic.ecei.ecei2d.detector2d import GaussianAntenna
import sdp.plasma.analytic.testparameter as tp

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
detector1 = GaussianAntenna(omega_list=[omega], k_list=[k],
                            power_list=[1],
                           waist_x=172, waist_y=2, waist_z=2, w_0y=2, w_0z=5,
                           tilt_v=0, tilt_h=np.pi/20)

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
Z1D_fine = np.linspace(-40, 20, 512)

ece_iso.set_coords([Z1D_fine, Y1D, X1D])
ece_ani.set_coords([Z1D_fine, Y1D, X1D])
ece.set_coords([Z1D_fine, Y1D, X1D])


omega_s = np.linspace(0.8, 1.1, 4)*omega
k_s = omega_s/c
detectors = [GaussianAntenna(omega_list=[f], k_list=[k_s[i]],
                             power_list=[1], waist_x=175, waist_y=0,
                             w_0y=2) for i, f in enumerate(omega_s)]


ecei = ECEImagingSystem(p2d, detectors=detectors, max_harmonic=2,
                        max_power=2)

ecei.set_coords([Z1D, Y1D, X1D])

client = ipp.Client()
ecei_para = ECEImagingSystem(p2d, detectors=detectors, max_harmonic=2,
                             max_power=2, parallel=True, client=client)

#ece.view_point
#Ps = ece.diagnose(debug=True, auto_patch=True)



