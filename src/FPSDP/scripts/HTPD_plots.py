
import numpy as np
import matplotlib.pyplot as plt

import FPSDP.Plasma.Analytical_Profiles.TestParameter as tp
import FPSDP.Plasma.PlasmaCharacter as pc
import FPSDP.Diagnostics.ECEI.ECEI2D as ecei2d
from FPSDP.GeneralSettings.UnitSystem import cgs

c = cgs['c']
keV = cgs['keV']


def DIIID_edge_ECE_run():
    tp.set_parameter2D(**tp.Parameter_DIIID)
    tp.Parameter2D['dne_ne']['dx'] = 5
    tp.Parameter2D['dte_te']['dx'] = 5
    tp.Parameter2D['timesteps'] = np.arange(16)
    tp.Parameter2D['dt'] = 2.5e-6/4
    
    p2d_low = tp.create_profile2D(fluctuation=True)
    
    tp.Parameter2D['dne_ne']['level'] = 0.1
    
    p2d_high = tp.create_profile2D(fluctuation=True)
    
    pcpr = pc.PlasmaCharProfile(p2d_low)
    x_array = np.linspace(222, 235, 8)
    omega_array = \
        2*pc.omega_ce(p2d_low.get_B0([[0 for x in x_array],x_array]))    
    k_array = omega_array/c
    
    detector_array = [ecei2d.GaussianAntenna(omega_list=[omega_array[i]], 
                                             k_list=[k_array[i]], 
                                             power_list=[1], waist_x=x, 
                                             waist_y=0, w_0y=2) \
                                             for i, x in enumerate(x_array)]
   
    
    ecei = ecei2d.ECEImagingSystem(p2d_low, detector_array, 
                                   max_harmonic=2, max_power=2 )
    
    
