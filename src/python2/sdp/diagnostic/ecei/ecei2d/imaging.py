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
import time as systime

import numpy as np

from .ece import ECE2D
from .detector2d import Detector2D, GaussianAntenna
from ....plasma.profile import ECEI_Profile


class ECEImagingSystem(object):
    """Main class for a complete ECE imaging system
    
    Initialization
    ***************
        __init__(plasma, detectors, polarization='X', 
                 weakly_relativistic=True, isotropic=True, 
                 max_harmonic=4, max_power=4, parallel=False, 
                 client=None)
    
    :param plasma: plasma to be diagnosed
    :type plasma: :py:class:`sdp.plasma.PlasmaProfile.ECEIProfile` object
    :param detectors: receiving antennas in whole system 
    :type detectors: 
        list of :py:class`sdp.diagnostic.ecei.ecei2d.Detector2D.Detector2D`
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
    
    
    """
    
    def __init__(self, plasma, detectors, polarization='X', 
                 weakly_relativistic=True, isotropic=True, 
                 max_harmonic=4, max_power=4, parallel=False, client=None):
        """Initialize ecei System
        
        :param plasma: plasma to be diagnosed
        :type plasma: :py:class:`sdp.plasma.PlasmaProfile.ECEIProfile` object
        :param detectors: receiving antennas in whole system 
        :type detectors: list of :py:class`sdp.diagnostic.ecei.
                         ecei2d.Detector2D.Detector2D` objects
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
import sdp.plasma.PlasmaProfile\n\
PlasmaProfile = sdp.plasma.PlasmaProfile\n\
import sdp.diagnostic.ecei.ecei2d.Detector2D\n\
Detector2D = sdp.diagnostic.ecei.ecei2d.Detector2D\n\
import sdp.diagnostic.ecei.ecei2d.Reciprocity\n\
ECE2D = sdp.diagnostic.ecei.ecei2d.Reciprocity.ECE2D')
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
                    systime.sleep(0.01)
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
        if str(channelID) == 'all':
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
eces[{0}].set_coords(coordinates)'.format(i))
                status.append(sts)
            wait_time = 0
            for i, ci in enumerate(channelID):
                while(not status[i].ready() and wait_time<len(channelID)):
                    wait_time += 0.01
                    systime.sleep(0.01)
            if wait_time >= len(channelID):
                raise Exception('Parallel Set_coords takes too long. Check if \
something went wrong. Time elapsed: {0}s'.format(wait_time))
            return status
            
    def auto_adjust_mesh(self, fine_coeff=1, channelID='all', 
                         wait_time_single=120, mute=False):
        """automatically adjust X grid points based on local emission intensity
        for chosen channels.
        
        :param float fine_coeff: coefficient of fine structure. See 
                                 :py:method:`sdp.diagnostic.ecei.ecei2d.
                                 Reciprocity.ECE2D.auto_adjust_mesh` for more 
                                 details.
        :param channelID: Optional, ID for chosen channels. Given as a list of 
                          indices in ``self.channels``. Default is 'all'. 
        :type channelID: list of int.
        :param float wait_time_single: expected execution time for single 
                                       channel, in seconds. Only used in 
                                       parallel mode to avoid infinite waiting.
        :param bool mute: if True, no output during execution, except warnings.
        """
        tstart = systime.clock()
        
        if str(channelID) == 'all':
            channelID = np.arange(self._ND)
        if not self._parallel:
            if not mute:
                print('Serial run of channel {0} out of total {1} channels.'.\
                       format(channelID, self._ND))
            for channel_idx in channelID:
                if not mute:
                    print('Channel {}:'.format(channel_idx))
                self.channels[channel_idx].auto_adjust_mesh\
                                           (fine_coeff=fine_coeff, mute=mute)
            tend = systime.clock()
            if not mute:
                print('Walltime: {0:.4}s'.format(tend-tstart))
        else:
            if not mute:
                print('Parallel run of channel {0} on {1} engines.'.\
                       format(channelID, self._engine_num) )
            status=[]
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                if not mute:
                    print ('channel #{} on engine #{}.'.format(i, eid))
                engine.push({'fine_coeff':fine_coeff, 'mute':mute})
                sts = engine.execute('\
eces[{0}].auto_adjust_mesh(fine_coeff=fine_coeff, mute=mute)'.format(i))
                status.append(sts)
            wait_time = 0
            for i, ci in enumerate(channelID):
                while(not status[i].ready() and \
                      wait_time < wait_time_single*len(channelID)):
                    wait_time += 0.01
                    systime.sleep(0.01)
            if wait_time >= wait_time_single*len(channelID):
                raise Exception('Parallel auto_adjust_mesh takes too long. \
Check if something went wrong. Time elapsed: {0}s'.format(wait_time))
            tend = systime.clock()
            if not mute:
                print('Walltime: {0:.4}s'.format(tend-tstart))
            return status
                                                            
    def diagnose(self, time=None, debug=False, auto_patch=False, 
                 oblique_correction=True, channelID='all', 
                 wait_time_single=120, mute=False):
        """diagnose electron temperature with chosen channels
        
        :param int time: time steps in plasma profile chosen for diagnose. if 
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
        :param bool mute: if True, no printed output.
        """
        tstart = systime.clock()
        if str(channelID) == 'all':
                channelID = np.arange(self._ND)
        if time is not None:
            time = np.array(time)
            Te_shape = [self._ND]
            Te_shape.extend(time.shape)
            self.Te = np.empty((Te_shape), dtype='float')
            self.Te[...] = np.nan
        else:
            self.Te = np.empty((self._ND,), dtype='float')
                
        if not self._parallel:
            # single CPU version
            # if no previous Te, initialize with np.nan
            if not mute:
                print('Serial run for channel {0} out of total {1} channels.'.\
                       format(channelID, self._ND))
            for channel_idx in channelID:
                if not mute:
                    print('Channel #{}:'.format(channel_idx))
                self._debug_mode[channel_idx] = debug
                self.Te[channel_idx] = np.asarray(self._channels[channel_idx].\
                                         diagnose(time=time, debug=debug, 
                                                  auto_patch=auto_patch,
                                                  oblique_correction=\
                                                  oblique_correction,
                                                  mute=mute))
            tend = systime.clock()
            if not mute:
                print('Walltime: {0:.4}s'.format(tend-tstart))                                                  
        else:
            if not mute:
                print('Parallel run for {0} channels on {1} engines.'.\
                       format(len(channelID), self._engine_num))
            status=[]
            for i in channelID:
                self._debug_mode[i] = debug
                eid = self._client.ids[i%self._engine_num]
                if not mute:
                    print('Channel #{0} on engine #{1}'.format(i, eid))
                engine = self._client[eid]
                engine.push(dict(time=time, debug=debug, 
                                  auto_patch=auto_patch,
                                  oblique_correction=oblique_correction,
                                  mute=mute))
                sts = engine.execute('\
eces[{0}].diagnose(time=time, debug=debug, auto_patch=auto_patch, \
oblique_correction=oblique_correction, mute=mute)'.format(i))
                status.append(sts)
            wait_time = 0
            for i, ci in enumerate(channelID):
                while(not status[i].ready() and \
                      wait_time < wait_time_single*len(channelID)):
                    wait_time += 0.01
                    self._client.wait(status[i], 0.01)
                eid = self._client.ids[ci%self._engine_num]
                engine = self._client[eid]
                self.Te[ci] = np.asarray(engine['eces[{0}].Te'.format(ci)])
            if wait_time >= wait_time_single*len(channelID):
                raise Exception('Parallel diagnose() takes too long. Check if \
something went wrong. Time elapsed: {0}s'.format(wait_time))
            tend = systime.clock()
            if not mute:
                print('Walltime: {0:.4}s'.format(tend-tstart))    
            return status
        
    def get_channels(self, channelID='all'):
        """Detailed information for all/chosen channels"""
        if str(channelID) == 'all':
            channelID = np.arange(self._ND)
        else:
            channelID = np.array(channelID)
        if not self._parallel:
            return [self._channels[i] for i in channelID]
        else:
            print('Parallel Channels:')
            channels = []
            for i in channelID:
                eid = i%self._engine_num
                engine = self._client[eid]
                print('    channel {0} on engine {1}:'.format(i, eid))
                channels.append(engine['eces[i].properties'])
                print('        received.')
            return channels
            
    @property
    def channels(self):
        return self.get_channels()

    def get_view_points(self, channelID='all'):
        """ actual viewing location for each channel
        """
        if str(channelID) == 'all':
            channelID = np.arange(self._ND)
        else:
            channelID = np.array(channelID)
        if not self._parallel:
            vp = [self.channels[i].view_point for i in channelID]
        else:
            vp = []
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                vp.append(engine['eces[{0}].view_point'.format(i)])
        return tuple(vp)
        
    @property
    def view_points(self):
        return self.get_view_points()
    
    def get_view_spots(self, channelID='all'):        
        if str(channelID) == 'all':
            channelID = np.arange(self._ND)
        else:
            channelID = np.array(channelID)
        if not self._parallel:
            vs = [self.channels[i].view_spot for i in channelID]
        else:
            vs = []
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                vs.append(engine['eces[{0}].view_spot'.format(i)])
        return tuple(vs)
        
    @property
    def view_spots(self):
        """view_spot list of all channels"""
        return self.get_view_spots()
        
    def get_diag_x(self, channelID='all'):
        if str(channelID) == 'all':
            channelID = np.arange(self._ND)
        else:
            channelID = np.array(channelID)
        if not self._parallel:
            diag_x = [self.channels[i].diag_x for i in channelID]
        else:
            diag_x = []
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                diag_x.append(engine['eces[{0}].diag_x'.format(i)])
        return tuple(diag_x)
        
    @property
    def diag_xs(self):
        """ diag_x list of all chanels"""
        return self.get_diag_x()
    
    def get_X1Ds(self, channelID='all'):
        
        if str(channelID) == 'all':
            channelID = np.arange(self._ND)
        else:
            channelID = np.array(channelID)
        if not self._parallel:
            x1ds = [self.channels[i].X1D for i in channelID]
        else:
            x1ds = []
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                x1ds.append(engine['eces[{0}].X1D'.format(i)])
        return tuple(x1ds)
    
    @property
    def X1Ds(self):
        """list containing X1D arrays of all channels"""
        return self.get_X1Ds()
    
    def get_Y1Ds(self, channelID='all'):
        
        if str(channelID) == 'all':
            channelID = np.arange(self._ND)
        else:
            channelID = np.array(channelID)
        if not self._parallel:
            y1ds = [self.channels[i].Y1D for i in channelID]
        else:
            y1ds = []
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                y1ds.append(engine['eces[{0}].Y1D'.format(i)])
        return tuple(y1ds)
        
    @property
    def Y1Ds(self):
        """list containing X1D arrays of all channels"""
        return self.get_Y1Ds()
        
    def get_Z1Ds(self, channelID='all'):
        
        if str(channelID) == 'all':
            channelID = np.arange(self._ND)
        else:
            channelID = np.array(channelID)
        if not self._parallel:
            z1ds = [self.channels[i].Z1D for i in channelID]
        else:
            z1ds = []
            for i in channelID:
                eid = self._client.ids[i%self._engine_num]
                engine = self._client[eid]
                z1ds.append(engine['eces[{0}].Z1D'.format(i)])
        return tuple(z1ds)
        
    @property
    def Z1Ds(self):
        """list containing X1D arrays of all channels"""
        return self.get_Z1Ds()
        
    #def save(self, filename='./ecei_save'):
    #    """save all channel information to *filename*.npz
    #    """
        
        
        
    
                
    
    
        
        
                             
        
    
    




