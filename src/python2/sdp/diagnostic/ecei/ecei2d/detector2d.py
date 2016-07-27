# -*- coding: utf-8 -*-
"""
Detector class for ecei2d diagnostic

A detector is specified as a wave complex amplitude on y-z plane at a given x
location. 

The most common approximation is a Gaussian beam focusing at the resonance 
layer. However, for comparison with experiment setups, a Code5 file specifying
the wave amplitude is accepted. 

Created on Wed Mar 09 15:19:07 2016

@author: lei
"""
from __future__ import print_function

import sys
from abc import ABCMeta, abstractproperty, abstractmethod
from collections import OrderedDict

import numpy as np

from ....model.lightbeam import GaussianBeam
from ..detector import Detector


class Detector2D(Detector):
    """abstract base class for Detectors
    """
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def parameters(self):
        """return a dictionary which contains a complete set of initialization
        parameters.
        """
        pass
    
    @abstractproperty
    def class_name(self):
        """return the class name of the detector as a string
        """
        pass
    
    @abstractproperty
    def central_beam(self):
        pass
    
    @abstractproperty
    def beam_list(self):
        pass
    
    @abstractproperty
    def central_E_inc(self):
        pass
    
    @abstractproperty
    def E_inc_list(self):
        pass
    
    @abstractmethod
    def set_inc_coords(self, X_inc, Y1D, Z1D):
        pass
    
    
class GaussianAntenna(Detector2D):
    """Antenna class whose emission wave pattern can be approximated as a 
    Gaussian beam.
    
    Gaussian antennas are most commonly used. They can be specified using a set
    of parameters: (See :py:class:`sdp.math.LightBeam.GaussianBeam` for more
    details about these parameters)
    
    
    :param omega_list: angular frequencies sampled to specify the reception 
                       profile of the antenna.  
    :type omega_list: 1D array of floats
    :param power_list: power reception profile on each frequency in 
                       ``omega_list``. The highest power will be normalized to 
                       1 automatically, and the frequency corresponding to that
                       power will be chosen as the central frequency.
    :type power_list: 1D array of floats
    :param float waist_x: The x' coordinate of waist location in Lab frame
    :param float waist_y: The y' coordinate of waist location in Lab frame 
    :param float waist_z: The z' coordinate of waist location in Lab frame
                          (Optional, Default to be 0)  
    :param float tilt_v: the tilt-angle in vertical plane in radian. 0 is 
                         defined 
                         as along negative x' axis, and tilt_y>0 if the beam
                         is tilted upwards, i.e. towards positive y' direction.
                         (Optional, Default to be 0)
    :param float tilt_h: the tilt-angle in x'-z' plane in radian. 0 is defined 
                         as along negative x' axis, and tilt_z>0 if the beam
                         is tilted upwards, i.e. towards positive z' direction
                         (Optional, Default to be 0)
    :param float rotation: (Optional Default to be 0) 
                           the rotated angle of the elliptic Gaussian. It is 
                           defined as the angle between local y axis and lab y'
                           axis if local z axis has been rotated to align with
                           lab -x' axis. This angle is the tilted angle of the
                           ellipse in the transvers plane. 
    :param float w_0y: waist width in the vertical-like direction (eventually 
                       aligns with y' axis)
    :param float w_0z: waist width in the horizontal-like direction (eventually
                       aligns with z' axis) (Optional, default to be same as 
                       w_0y)  
    
                          
    The antenna then output the electric field amplitude specified on a y-z 
    mesh at a given x location.
    
    Properties:
    ***********
    
    X_inc: x location where incidental wave pattern is specified    
    Y1D: 1D mesh on y direction, E_inc will be given on these mesh points
    Z1D: 1D mesh on z direction, E_inc will be given on these mesh points

    central_omega: central angular frequency of the antenna emission
    omega_list: list of omegas specified
    power_list: list of reception power corresponding to omegas
    central_E_inc: incidental wave's complex electric field amplitude on 
                   Y1D*Z1D mesh at X_inc for central frequency.
    E_inc_list: list of complex electric field amplitudes, 
                corresponding to list of omegas
    central_beam: light beam object containing all beam related information, 
                  for central frequency
    beam_list: list of beams, corresponding to list of omegas
    
    Methods:
    ********
    
    info(self):
        print out information about the current parameters
    __str__(self):
        return information string about the current parameters
    """
    def __init__(self, omega_list, k_list, power_list, waist_x, waist_y, w_0y, 
                 waist_z=0, w_0z=None, tilt_v=0, tilt_h=0, rotation=0):
        omegas = np.array(omega_list)
        ks = np.array(k_list)
        powers = np.array(power_list)             
        sort_arg = np.argsort(omegas)
        self._omega_list = omegas[sort_arg]
        self._k_list = ks[sort_arg]
        self._power_list = powers[sort_arg]
        self._central_index = np.argmax(self._power_list)
        # normalize central power to be 1
        self._power_list /= self._power_list[self._central_index]
        self._beam_parameter = OrderedDict()
        self._beam_parameter['waist_x'] = waist_x
        self._beam_parameter['waist_y'] = waist_y
        self._beam_parameter['waist_z'] = waist_z
        self._beam_parameter['w_0y'] = w_0y
        self._beam_parameter['w_0z'] = w_0z
        self._beam_parameter['tilt_v'] = tilt_v
        self._beam_parameter['tilt_h'] = tilt_h
        self._beam_parameter['rotation'] = rotation
        self._beams = [GaussianBeam(wave_length=2*np.pi/k_list[i],
                                    P_total=self._power_list[i],
                                    omega=omega,
                                    **self._beam_parameter) \
                       for i, omega in enumerate(self._omega_list)]
                           
    @property
    def class_name(self):
        return 'GaussianAntenna'

    @property
    def parameters(self):
        return dict(omega_list=self._omega_list,
                    k_list=self._k_list,
                    power_list = self._power_list,
                    **self._beam_parameter)
                                               
    @property    
    def central_omega(self):
        """central angular frequency of the antenna emission"""
        try:
            return self._omega_list[self._central_index]
        except AttributeError:
            print('Central omega not found!', file=sys.stderr)
            
    @property
    def central_beam(self):
        """light beam object containing all beam related information, for 
        central frequency
        """
        return self._beams[self._central_index]
        
    @property
    def tilt_h(self):
        """horizontal tilted angle
        """
        return self._beam_parameter['tilt_h']
        
    @property
    def tilt_v(self):
        """vertical tilted angle
        """
        return self._beam_parameter['tilt_v']
            
    @property
    def omega_list(self):
        return tuple(self._omega_list)
        
    @property
    def power_list(self):
        return tuple(self._power_list)
        
    @property
    def beam_list(self):
        return tuple(self._beam_list)
    
    @property    
    def central_E_inc(self):
        """central beam's incidental electric field at given X_inc, Y1D and 
        Z1D
        """
        try:
            X = self.X2D
            Y = self.Y2D
            Z = self.Z2D
        except AttributeError:
            print('Incidental coordinates not set! No E_inc output.', 
                  file=sys.stderr)
        return self.central_beam([Z, Y, X])
        
    @property
    def E_inc_list(self):
        """list of E_inc corresponding to list of beams"""
        try:
            X = self.X2D
            Y = self.Y2D
            Z = self.Z2D
        except AttributeError:
            print('Incidental coordinates not set! No E_inc output.', 
                  file=sys.stderr)
        E_list = [beam([Z, Y, X]) for beam in self._beams]
        return tuple(E_list)
        
        
    def set_inc_coords(self, X_inc, Y1D, Z1D):
        """setup coordinates of for E_inc
        
        generate following properties:
            X_inc: float
            
            Y1D: 1D float array
            
            Z1D: 1D float array
            
            X2D: 2D float array, X_inc broadcasted to Z1D*Y1D mesh
            
            Y2D: 2D float array, Y1D broadcasted to Z1D*Y1D mesh
            
            Z2D: 2D float array, Z1D broadcasted to Z1D*Y1D mesh
        """
        self.X_inc = X_inc
        self.Y1D = np.asanyarray(Y1D)
        self.Z1D = np.asanyarray(Z1D)
        self.X2D = np.zeros((len(Z1D), len(Y1D))) + X_inc
        self.Y2D = np.zeros_like(self.X2D) + Y1D[np.newaxis, :]
        self.Z2D = np.zeros_like(self.X2D) + Z1D[:, np.newaxis]
        
    def __str__(self):
        info = 'Gaussian Antenna:\n'
        info += '  central frequency: {0:.4} Hz\n'.format(self.central_omega/\
                                                        (2*np.pi))
        info += '  Gaussian Beam parameters:\n'
        for key,value in self._beam_parameter.items():        
            info += '    {0} : {1}\n'.format(key, value)        
        info += '    divergence : {0:.4}, {1:.4}\n'.\
                format(self.central_beam.divergence[0],
                       self.central_beam.divergence[1])
        info += '    Reighlay range : {0:.4}, {1:.4}'.\
                format(self.central_beam.reighlay_range[0],
                       self.central_beam.reighlay_range[0])
        return info
        
    def info(self):
        print(self)
                                      
        
        
        
    
                         
        
                         
    

        