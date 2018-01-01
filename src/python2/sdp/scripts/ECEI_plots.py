# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:10:43 2016

@author: lei
"""

from __future__ import print_function
import sys

import numpy as np
from scipy.integrate import trapz, cumtrapz
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import rcParams, animation
import ipyparallel as ipp
from IPython.display import HTML

from sdp.settings.unitsystem import cgs
import sdp.plasma.analytic.testparameter as tp
import sdp.diagnostic.ecei.ecei2d.ece as rcp
from sdp.diagnostic.ecei.ecei2d.detector2d import GaussianAntenna
from sdp.diagnostic.ecei.ecei2d.imaging import ECEImagingSystem as ECEI
import sdp.plasma.character as pc
from sdp.visualization.anim2html import display_animation

rcParams['figure.figsize'] = [12, 9]
rcParams['font.size'] = 18
color_array = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

c = cgs['c']
keV = cgs['keV']
e = cgs['e']
me = cgs['m_e']
pi = np.pi
twopi = 2*np.pi

# We will use a uniform Te profile to do the benchmarks
Te0 = 10*keV
ne0 = 6.0e13

# We'll make a high pedestal, so the core region has a flat profile
tp.ShapeTable['Hmode']['PedHightN'] = 0.8

# We need setup fluctuations one by one
freq_fluc = 1e5 # 10 kHz fluctuation
dte_te = tp.Parameter2D['dte_te']
dte_te['type'] = 'siny'
dte_te['params']['k'] = pi/5
dte_te['params']['dx'] = 10
dte_te['params']['level'] = 0.1
dte_te['params']['omega'] = 2*np.pi*freq_fluc
dte_te['params']['phi0'] = 0

dne_ne = tp.Parameter2D['dne_ne']
dne_ne['type'] = 'siny'
dne_ne['params']['k'] = pi/5
dne_ne['params']['dx'] = 10
dne_ne['params']['level'] = 0.03
dne_ne['params']['omega'] = 2*np.pi*freq_fluc
dne_ne['params']['phi0'] = 0

# We set dB to be zero for simplicity
dB_B = tp.Parameter2D['dB_B']
dB_B['params']['level'] = 0

# We devide a whole period of fluctuation into n time steps
ntstep = 20
tp.set_parameter2D(Te_0 = Te0, ne_0=ne0, Te_shape='uniform',
                   ne_shape='Hmode',dte_te=dte_te,
                   dne_ne=dne_ne, dB_B=dB_B,
                   NR=400, NZ=400, timesteps=np.arange(ntstep),
                   dt=1/(ntstep*freq_fluc))
p2d_fluc = tp.create_profile2D(fluctuation=True)
p2d_fluc.setup_interps()

omega = 2*pc.omega_ce(p2d_fluc.get_B0([0, 220]))[0]

x= 220
y_array = np.linspace(-5, 5, 8)
ch_wid = 1

k = omega/c

detector_array = [GaussianAntenna(omega_list=[omega], k_list=[k],
                                  power_list=[1], waist_x=x,
                                  waist_y=y, w_0y=ch_wid, tilt_h=0) \
                  for y in y_array]

x1D = np.linspace(251, 216, 160)
y1D = np.linspace(-30, 30, 65)
z1D = np.linspace(-30, 30, 65)

# channel 3 is the one near the center
try:
    ece_inphase_run
except NameError:
    ece = rcp.ECE2D(plasma=p2d_fluc, detector=detector_array[3], max_harmonic=2,
                    max_power=2)

    ece.set_coords([z1D, y1D, x1D])

    ece.auto_adjust_mesh()

    tsteps = np.arange(len(p2d_fluc.time))
    Te_t = []
    view_spot_t = []
    for t in tsteps:
        print('time {0}.'.format(t))
        ece.diagnose(time=t, mute=True)
        Te_t.append(ece.Te)
        view_spot_t.append(ece.view_spot)
    Te_t = np.array(Te_t)
    ece_inphase_run = True

X = ece.detector.central_beam.waist_loc[2]
Y = ece.detector.central_beam.waist_loc[1]
Te_real = p2d_fluc.get_Te([Y, X], eq_only=False, time=tsteps)
ne = p2d_fluc.get_ne([Y, X], eq_only=False, time=tsteps)

# choose time steps to plot
i = 3
j = 5
k = 10

fig = plt.figure()

ax1 = plt.subplot(211)
ax1_ne = ax1.twinx()
line_tem = ax1.plot(p2d_fluc.time*1e6, Te_t/keV, 'r-', label='Te_measured')
line_ter = ax1.plot(p2d_fluc.time*1e6, Te_real/keV, 'b-', label='Te_real')
line_ne = ax1_ne.plot(p2d_fluc.time*1e6, ne, 'k--', label='ne')
ti = p2d_fluc.time[i]*1e6
tj = p2d_fluc.time[j]*1e6
tk = p2d_fluc.time[k]*1e6
ax1.vlines(x=[ti, tj, tk], ymin=np.min(Te_real/keV), ymax=np.max(Te_real/keV))

lines = line_tem + line_ter + line_ne
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize='small')
ax1.set_xlabel(r'time ($\mu s$)')
ax1.set_ylabel(r'Te (keV)')
ax1_ne.set_ylabel('ne (cm^-3)')
ax1.set_title('(a)')

ax3 = plt.subplot(234)
Te_total = p2d_fluc.Te0 + p2d_fluc.dTe_perp

ax3.contour(ece.X1D, ece.Y1D, view_spot_t[i], [0.368, 0.7, 0.9], colors='k')
ax3.contour(p2d_fluc.grid.R1D, p2d_fluc.grid.Z1D, Te_total[i])
ax3.grid(b=True, which='major')
ax3.set_xlim(210, 230)
ax3.set_ylim(-10, 10)
ax3.set_xlabel('X (cm)')
ax3.set_ylabel('Y (cm)')
ax3.vlines(x=X, ymin=Y-1, ymax=Y+1, linewidths=2)
ax3.hlines(y=Y, xmin=X-1, xmax=X+1, linewidths=2)
ax3.set_title('(b1)')

ax4 = plt.subplot(235)

ax4.contour(ece.X1D, ece.Y1D, view_spot_t[j], [0.368, 0.7, 0.9], colors='k')
ax4.contour(p2d_fluc.grid.R1D, p2d_fluc.grid.Z1D, Te_total[j])
ax4.grid(b=True, which='major')
ax4.set_xlim(210, 230)
ax4.set_ylim(-10, 10)
ax4.set_xlabel('X (cm)')
ax4.vlines(x=X, ymin=Y-1, ymax=Y+1, linewidths=2)
ax4.hlines(y=Y, xmin=X-1, xmax=X+1, linewidths=2)
ax4.set_title('(b2)')

ax5 = plt.subplot(236)

ax5.contour(ece.X1D, ece.Y1D, view_spot_t[k], [0.368, 0.7, 0.9], colors='k')
ax5.contour(p2d_fluc.grid.R1D, p2d_fluc.grid.Z1D, Te_total[k])
ax5.grid(b=True, which='major')
ax5.set_xlim(210, 230)
ax5.set_ylim(-10, 10)
ax5.set_xlabel('X (cm)')
ax5.vlines(x=X, ymin=Y-1, ymax=Y+1, linewidths=2)
ax5.hlines(y=Y, xmin=X-1, xmax=X+1, linewidths=2)
ax5.set_title('(b3)')

plt.tight_layout()
##################################################
# Out of phase run
##################################################

# We need setup fluctuations one by one
"""
freq_fluc = 1e5 # 10 kHz fluctuation
dte_te = tp.Parameter2D['dte_te']
dte_te['type'] = 'siny'
dte_te['params']['k'] = pi/5
dte_te['params']['dx'] = 10
dte_te['params']['level'] = 0.1
dte_te['params']['omega'] = 2*np.pi*freq_fluc
dte_te['params']['phi0'] = 0

dne_ne = tp.Parameter2D['dne_ne']
dne_ne['type'] = 'siny'
dne_ne['params']['k'] = pi/5
dne_ne['params']['dx'] = 10
dne_ne['params']['level'] = 0.03
dne_ne['params']['omega'] = 2*np.pi*freq_fluc
dne_ne['params']['phi0'] = np.pi

# We set dB to be zero for simplicity
dB_B = tp.Parameter2D['dB_B']
dB_B['params']['level'] = 0

# We devide a whole period of fluctuation into n time steps
ntstep = 20
tp.set_parameter2D(Te_0 = Te0, ne_0=ne0, Te_shape='uniform',
                   ne_shape='Hmode',dte_te=dte_te,
                   dne_ne=dne_ne, dB_B=dB_B,
                   NR=200, NZ=200, timesteps=np.arange(ntstep),
                   dt=1/(ntstep*freq_fluc))
p2d_fluc_out = tp.create_profile2D(fluctuation=True)
p2d_fluc_out.setup_interps()

try:
    ece_outphase_run
except NameError:
    ece_out = rcp.ECE2D(plasma=p2d_fluc_out, detector=detector_array[3],
                        max_harmonic=2, max_power=2)

    ece_out.set_coords([z1D, y1D, x1D])

    ece_out.auto_adjust_mesh()

    tsteps = np.arange(len(p2d_fluc_out.time))
    Te_t_out = []
    view_spot_t_out = []
    for t in tsteps:
        print('time {0}.'.format(t))
        ece_out.diagnose(time=t, mute=True)
        Te_t_out.append(ece_out.Te)
        view_spot_t_out.append(ece_out.view_spot)
    Te_t_out = np.array(Te_t_out)
    ece_outphase_run = True

X_out = ece_out.detector.central_beam.waist_loc[2]
Y_out = ece_out.detector.central_beam.waist_loc[1]
Te_real_out = p2d_fluc_out.get_Te([Y_out, X_out], eq_only=False, time=tsteps)
ne_out = p2d_fluc_out.get_ne([Y_out, X_out], eq_only=False, time=tsteps)

# choose time step to plot
i=4

ax2 = axes[0][1]
ax2_ne = ax2.twinx()
line_tem = ax2.plot(p2d_fluc_out.time*1e6, Te_t_out/keV, label='Te_measured')
line_ter = ax2.plot(p2d_fluc_out.time*1e6, Te_real_out/keV, label='Te_real')
line_ne = ax2_ne.plot(p2d_fluc_out.time*1e6, ne_out, 'k--', label='ne')
t = p2d_fluc_out.time[i]*1e6
ax2.vlines(x=t, ymin=np.min(Te_real_out/keV), ymax=np.max(Te_real_out/keV))

lines = line_tem + line_ter + line_ne
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize='x-small')
ax2.set_xlabel(r'time ($\mu s$)')
ax2.set_ylabel(r'Te (keV)')
ax2_ne.set_ylabel('ne (cm^-3)')

ax4 = axes[1][1]
Te_total_out = p2d_fluc_out.Te0 + p2d_fluc_out.dTe_perp
t = p2d_fluc_out.time[i]*1e6
ax4.contour(ece_out.X1D, ece_out.Y1D, view_spot_t_out[i], [0.368, 0.7, 0.9],
            colors='k')
ax4.contour(p2d_fluc_out.grid.R1D, p2d_fluc_out.grid.Z1D, Te_total_out[i])
ax4.grid(b=True, which='major')
ax4.set_xlim(200, 240)
ax4.set_ylim(-20, 20)
ax4.set_xlabel('X (cm)')
ax4.set_ylabel('Y (cm)')
"""


