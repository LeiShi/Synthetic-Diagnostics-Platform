# -*- coding: utf-8 -*-
"""
Contain classes to carry out a full ECE Imaging diagnosis. 

The following classes are included:

    ECEImagingSystem: main class for a complete ECE imaging system
    Configuration: main class reading and writing ECEI configuration file
    

Created on Fri Mar 18 11:07:24 2016

@author: lei
"""
from __future__ import print_function
from os import path
import sys
import json

import numpy as np

from .Reciprocity import ECE2D
from .Detector2D import Detector2D, GaussianAntenna
from ....Plasma.PlasmaProfile import ECEI_Profile


class ECEImagingSystem(object):
    """Main class for a complete ECE imaging system
    
    Initialization
    ***************
        __init__(plasma, detectors, polarization='X', 
                 weakly_relativistic=True, isotropic=True, 
                 max_harmonic=4, max_power=4)
    
    :param plasma: plasma to be diagnosed
    :type plasma: :py:class:`FPSDP.Plasma.PlasmaProfile.ECEIProfile` object
    :param detectors: receiving antennas in whole system 
    :type detectors: 
        list of :py:class`FPSDP.Diagnostics.ECEI.ECEI2D.Detector2D.Detector2D` 
        objects
    :param string polarization: either 'O' or 'X', the chosen polarization.
    :param bool weakly_relativistic: model selection for both dielectric and 
                                     current correlation tensor. If True, 
                                     weakly relativistic formula will be used, 
                                     otherwise non-relativistic formula is 
                                     used. Default is True.
    :param bool isotropic: model selection for current correlation tensor. If 
                           True, isotropic Maxwillian is assumed, and current
                           correlation tensor can be directly obtained from 
                           anti-Hermitian dielectric tensor. Otherwise, 
                           anisotropic formula is needed. Default is True.
                           
    :param bool parallel: parallel run flag. Default is False.
    :param dv: direct view of parallel engines, only needed for parallel runs.
    :type dv: DirectView of Ipython cluster. object created by 
              ipyparallel.Client()[:]
                           
    
    Methods
    *******
    
    set_coords(coordinates): set initial mesh in Z,Y,X 

    auto_adjust_mesh(fine_coeff=1): automatically adjust grid in X    
    
    diagnose(time, detectorID='all'): diagnose plasma of using chosen 
                                      time steps using chosen channels.
    
    direct_map(Te_measured, detectorID='all'): map the measured electron 
                                               temperature onto R-Z plane, 
                                               using real peak emission 
                                               locations.
                                               
    ideal_map(Te_measured, detectorID='all'): map the measured electron 
                                              temperature onto R-Z plane,
                                              using ideal resonance locations.
    """
    
    def __init__(self, plasma, detectors, polarization='X', 
                 weakly_relativistic=True, isotropic=True, 
                 max_harmonic=4, max_power=4, parallel=False, dv=None):
        """Initialize ECEI System
        
        :param plasma: plasma to be diagnosed
        :type plasma: :py:class:`FPSDP.Plasma.PlasmaProfile.ECEIProfile` object
        :param detectors: receiving antennas in whole system 
        :type detectors: list of :py:class`FPSDP.Diagnostics.ECEI.
                         ECEI2D.Detector2D.Detector2D` objects
        :param string polarization: either 'O' or 'X', the chosen polarization.
        :param bool weakly_relativistic: model selection for both dielectric 
                                         and current correlation tensor. If 
                                         True, weakly relativistic formula will
                                         be used, otherwise non-relativistic 
                                         formula is used. Default is True.
        :param bool isotropic: model selection for current correlation tensor. 
                               If True, isotropic Maxwillian is assumed, and 
                               current correlation tensor can be directly 
                               obtained from anti-Hermitian dielectric tensor. 
                               Otherwise, anisotropic formula is needed. 
                               Default is True.
        :param int max_harmonic: maximum harmonic number included in emission 
                                 model.
        :param int max_power: highest order of FLR effect included in weakly 
                              relativistic model. Ignored in non-relativistic
                              model.
                               
        :param bool parallel: parallel run flag. Default is False.
        :param dv: direct view of parallel engines, only needed for parallel 
                   runs.
        :type dv: DirectView of Ipython cluster. object created by 
                  ipyparallel.Client()[:]
        """
        
        self.plasma = plasma
        self.detector_list = detectors
        self._ND = len(detectors)
        self.polarization = polarization
        self.weakly_relativistic = weakly_relativistic
        self.isotropic = isotropic
        if (parallel):
            self._parallel = True
            self._dv = dv
            self._engine_num = len(dv)
        else:
            self._parallel = False
        self.max_harmonic = max_harmonic
        self.max_power = max_power
            
        # Now, create ECE2D object for each channel
        self.channels=[ECE2D(plasma=plasma, detector=detector, 
                             polarization=polarization, 
                             weakly_relativistic=weakly_relativistic, 
                             isotropic=isotropic, max_harmonic=max_harmonic,
                             max_power=max_power) for detector in detectors]
                             
        self._debug_mode = np.zeros((self._ND,), dtype='bool')
                             
        
    def set_coords(self, coordinates, channelID='all'):
        """setup initial calculation mesh in Z,Y,X for chosen channels
        
        :param coordinates: [Z1D, Y1D, X1D] mesh points. Rectanglar mesh 
                            assumed.
        :type coordinates: list of 1D arrays
        :param channelID: Optional, ID for chosen channels. Given as a list of 
                          indices in ``self.channels``. Default is 'all'. 
        :type channelID: list of int.
        """
        if channelID == 'all':
            channelID = np.arange(self._ND)
        for channel_idx in channelID:
            self.channels[channel_idx].set_coords(coordinates)
            
    def auto_adjust_mesh(self, fine_coeff=1, channelID='all'):
        """automatically adjust X grid points based on local emission intensity
        for chosen channels.
        
        :param float fine_coeff: coefficient of fine structure. See 
                                 :py:method:`FPSDP.Diagnostics.ECEI.ECEI2D.
                                 Reciprocity.ECE2D.auto_adjust_mesh` for more 
                                 details.
        :param channelID: Optional, ID for chosen channels. Given as a list of 
                          indices in ``self.channels``. Default is 'all'. 
        :type channelID: list of int.
        """
        if channelID == 'all':
            channelID = np.arange(self._ND)
        for channel_idx in channelID:
            self.channels[channel_idx].auto_adjust_mesh(fine_coeff=fine_coeff)
                                                        
                                                            
    def diagnose(self, time=None, debug=False, auto_patch=False, 
                 channelID='all'):
        """diagnose electron temperature with chosen channels
        
        :param int time: time step in plasma profile chosen for diagnose. if 
                         not given, only equilibrium will be used.
        :param bool debug: debug mode flag. if True, more information will be 
                           kept for examining. Default is False.
        :param bool auto_patch: if True, program will automatically detect the 
                                significant range of x where emission power is
                                originated, and add finer grid patch to that 
                                region. This may cause a decrease of speed, but
                                can improve the accuracy. Default is False, the
                                programmer is responsible to set proper x mesh.
        :param channelID: Optional, ID for chosen channels. Given as a list of 
                          indices in ``self.channels``. Default is 'all'. 
        :type channelID: list of int.
        """
        
        if not self._parallel:
            # single CPU version
            # if no previous Te, initialize with np.nan
            if not hasattr(self, 'Te'):
                self.Te = np.empty((self._ND), dtype='float')
                self.Te[:] = np.nan
            
            if channelID == 'all':
                channelID = np.arange(self._ND)
            
            for channel_idx in channelID:
                self._debug_mode[channel_idx] = debug
                self.Te[channel_idx] = self.channels[channel_idx].\
                                         diagnose(time=time, debug=debug, 
                                                  auto_patch=auto_patch)
                                                  
        else:
            raise NotImplementedError('Parallel diangose is currently not \
implemented.')

    @property
    def view_points(self):
        """ actual viewing location for each channel
        """
        return tuple([c.view_point for c in self.channels])
    
    @property
    def view_spots(self):
        if np.any(~self._debug_mode):
            print('Some of the channels didn\'t run in debug mode. view_spot \
is not available. Call diagnose(debug=True) before accessing this property.', 
                   file=sys.stderr)
        else:
            return tuple([c.view_spot for c in self.channels])
                
    def direct_map(self):
        """create an interpolator of Te based on actual viewing locations
        """
        
        pass
    
        
        
                             
        
    
    




