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
import time

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
    :param client: ipcluster client handler, only needed for parallel 
                   runs.
    :type client: Handler of Ipython cluster. object created by Client()
                           
    
    Methods
    *******
    
    set_coords(coordinates): 
        set initial mesh in Z,Y,X 

    auto_adjust_mesh(fine_coeff=1): 
        automatically adjust grid in X    
    
    diagnose(time, detectorID='all'): 
        diagnose plasma of using chosen 
        time steps using chosen channels.
    
    direct_map(Te_measured, detectorID='all'): 
        map the measured electron temperature onto R-Z plane, using real peak 
        emission locations.
                                               
    ideal_map(Te_measured, detectorID='all'): 
        map the measured electron temperature onto R-Z plane, using ideal 
        resonance locations.
    """
    
    def __init__(self, plasma, detectors, polarization='X', 
                 weakly_relativistic=True, isotropic=True, 
                 max_harmonic=4, max_power=4, parallel=False, client=None):
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
        :param client: ipcluster client handler, only needed for parallel 
                   runs.
        :type client: Handler of Ipython cluster. object created by Client()
        """
        
        self.plasma = plasma
        self.detector_list = detectors
        self._ND = len(detectors)
        self.polarization = polarization
        self.weakly_relativistic = weakly_relativistic
        self.isotropic = isotropic
        self.max_harmonic = max_harmonic
        self.max_power = max_power
        
        if (parallel):
            self._parallel = True
            self._client = client
            self._engine_num = len(client.ids)
            # import and initialize useful modules and variables
            dv = self._client[:]
            dv.execute('\
import FPSDP.Plasma.PlasmaProfile\n\
PlasmaProfile = FPSDP.Plasma.PlasmaProfile\n\
import FPSDP.Diagnostics.ECEI.ECEI2D.Detector2D\n\
Detector2D = FPSDP.Diagnostics.ECEI.ECEI2D.Detector2D\n\
import FPSDP.Diagnostics.ECEI.ECEI2D.Reciprocity\n\
ECE2D = FPSDP.Diagnostics.ECEI.ECEI2D.Reciprocity.ECE2D')
        else:
            self._parallel = False
            
        # Now, create ECE2D object for each channel
        if not self._parallel:
            self._channels=[ECE2D(plasma=plasma, detector=detector, 
                                 polarization=polarization, 
                                 weakly_relativistic=weakly_relativistic, 
                             isotropic=isotropic, max_harmonic=max_harmonic,
                             max_power=max_power) for detector in detectors]
        else:
            # parallel runs needs to create channels on each engine
            status = []
            dv.execute('\
ids = []\n\
detector_parameters = {}\n\
detectors = {}\n\
eces = {}\n')
            dv.push({'p_param':plasma.parameters})
            dv.execute('\
plasma=PlasmaProfile.{0}(**p_param)\n\
plasma.setup_interps()'.format(plasma.class_name), block=True)
            for i,d in enumerate(detectors):
                # distribute channels evenly to each engine
                engine_id = i%self._engine_num
                engine = self._client[engine_id]
                engine.push({'d_param':d.parameters, 'i':i})
                engine.push({'e_param':dict(polarization=polarization,
                                            max_harmonic=max_harmonic,
                                            max_power=max_power,
                                    weakly_relativistic=weakly_relativistic,
                                    isotropic=isotropic)}, block=True)
                sts = engine.execute('\
detector_parameters[i] = d_param\n\
detectors[i] = Detector2D.{0}(**d_param)\n\
eces[i] = ECE2D(plasma=plasma, detector=detectors[i], **e_param)'\
                                     .format(d.class_name))
                status.append(sts)
            # wait until all engines are finished
            wait_time = 0
            for i,d in enumerate(detectors):
                while(not status[i].ready() and wait_time<self._ND):
                    wait_time += 0.01
                    time.sleep(0.01)
            if wait_time >= self._ND:
                raise Exception('Parallel Initialization Time is too long. \
Check if something went wrong! Time elapsed: {0}s'.format(wait_time))
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
        
        if not self._parallel:
            for channel_idx in channelID:
                self._channels[channel_idx].set_coords(coordinates)
        else:
            status=[]
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                engine.push({'coordinates':coordinates})
                sts = engine.execute('\
eces[i].set_coords(coordinates)')
                status.append(sts)
            wait_time = 0
            for i in channelID:
                while(not status[i].ready() and wait_time<len(channelID)):
                    wait_time += 0.01
                    time.sleep(0.01)
            if wait_time >= len(channelID):
                raise Exception('Parallel Set_coords takes too long. Check if \
something went wrong. Time elapsed: {0}s'.format(wait_time))
            
    def auto_adjust_mesh(self, fine_coeff=1, channelID='all', 
                         wait_time_single=120):
        """automatically adjust X grid points based on local emission intensity
        for chosen channels.
        
        :param float fine_coeff: coefficient of fine structure. See 
                                 :py:method:`FPSDP.Diagnostics.ECEI.ECEI2D.
                                 Reciprocity.ECE2D.auto_adjust_mesh` for more 
                                 details.
        :param channelID: Optional, ID for chosen channels. Given as a list of 
                          indices in ``self.channels``. Default is 'all'. 
        :type channelID: list of int.
        :param float wait_time_single: expected execution time for single 
                                       channel, in seconds. Only used in 
                                       parallel mode to avoid infinite waiting.
        """
        if channelID == 'all':
            channelID = np.arange(self._ND)
        if not self._parallel:
            for channel_idx in channelID:
                self.channels[channel_idx].auto_adjust_mesh\
                                           (fine_coeff=fine_coeff)
        else:
            status=[]
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                engine.push({'fine_coeff':fine_coeff})
                sts = engine.execute('\
eces[i].auto_adjust_mesh(fine_coeff=fine_coeff)')
                status.append(sts)
            wait_time = 0
            for i in channelID:
                while(not status[i].ready() and \
                      wait_time < wait_time_single*len(channelID)):
                    wait_time += 0.01
                    time.sleep(0.01)
            if wait_time >= wait_time_single*len(channelID):
                raise Exception('Parallel auto_adjust_mesh takes too long. \
Check if something went wrong. Time elapsed: {0}s'.format(wait_time))
                                                            
    def diagnose(self, time=None, debug=False, auto_patch=False, 
                 oblique_correction=True, channelID='all', 
                 wait_time_single=120):
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
        :param oblique_correction: if True, correction to oblique incident
                                   wave will be added. The decay part will have
                                   :math:`\cos(\theta_h)\cos(\theta_v)` term.
                                   Default is True.
        :type oblique_correction: bool
        :param channelID: Optional, ID for chosen channels. Given as a list of 
                          indices in ``self.channels``. Default is 'all'. 
        :type channelID: list of int.
        :param float wait_time_single: expected execution time for single 
                                       channel, in seconds. Only used in 
                                       parallel mode to avoid infinite waiting.
        """
        if channelID == 'all':
                channelID = np.arange(self._ND)
        if not hasattr(self, 'Te'):
                self.Te = np.empty((self._ND), dtype='float')
                self.Te[:] = np.nan
                
        if not self._parallel:
            # single CPU version
            # if no previous Te, initialize with np.nan
            if not hasattr(self, 'Te'):
                self.Te = np.empty((self._ND), dtype='float')
                self.Te[:] = np.nan
            
            for channel_idx in channelID:
                self._debug_mode[channel_idx] = debug
                self.Te[channel_idx] = self._channels[channel_idx].\
                                         diagnose(time=time, debug=debug, 
                                                  auto_patch=auto_patch,
                                                  oblique_correction=\
                                                  oblique_correction)
                                                  
        else:
            status=[]
            for i in channelID:
                self._debug_mode[i] = debug
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                engine.push(dict(time=time, debug=debug, 
                                  auto_patch=auto_patch,
                                  oblique_correction=oblique_correction))
                sts = engine.execute('\
eces[i].diagnose(time=time, debug=debug, auto_patch=auto_patch, \
oblique_correction=oblique_correction)')
                status.append(sts)
            wait_time = 0
            for i in channelID:
                while(not status[i].ready() and \
                      wait_time < wait_time_single*len(channelID)):
                    wait_time += 0.01
                    self._client.wait(status[i], 0.01)
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                self.Te[i] = engine['eces[{0}].Te'.format(i)]
            if wait_time >= wait_time_single*len(channelID):
                raise Exception('Parallel Set_coords takes too long. Check if \
something went wrong. Time elapsed: {0}s'.format(wait_time))
            
        return self.Te
        
    @property
    def channels(self):
        if not self._parallel:
            return self._channels
        else:
            print('Parallel Channels:')
            channels = []
            for i in xrange(self._ND):
                eid = i%self._engine_num
                engine = self._client[eid]
                channels.append(engine['eces[i].properties'])

    @property
    def view_points(self):
        """ actual viewing location for each channel
        """
        if not self._parallel:
            vp = [c.view_point for c in self.channels]
        else:
            vp = []
            for i in xrange(self._ND):
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                vp.append(engine['eces[{0}].view_point'.format(i)])
        return tuple(vp)
    
    @property
    def view_spots(self):
        if np.any(~self._debug_mode):
            print('Some of the channels didn\'t run in debug mode. view_spot \
is not available. Call diagnose(debug=True) before accessing this property.', 
                   file=sys.stderr)
        else:
            if not self._parallel:
                vs = [c.view_spot for c in self.channels]
            else:
                vs = []
                for i in xrange(self._ND):
                    eid = self._client.ids[i%self._engine_num]
                    engine = self._client[eid]
                    vs.append(engine['eces[{0}].view_spot'.format(i)])
            return tuple(vs)
                
    
    
        
        
                             
        
    
    




