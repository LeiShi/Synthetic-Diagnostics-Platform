# -*- coding: utf-8 -*-
"""
Base class for ECEI Detectors

Created on Wed Mar 09 17:59:41 2016

@author: lei
"""
from abc import ABCMeta, abstractproperty, abstractmethod

from .ECEI2D.Detector2D import Detector2D, GaussianAntenna
from .ECEI1D.Detector1D import Detector1D

__all__ = [GaussianAntenna]

class Detector(object):
    """abstract base class for Detectors
    """
    __metaclass__ = ABCMeta
    
    @abstractproperty
    def central_omega(self):
        pass
    
    @abstractproperty
    def omega_list(self):
        pass
    
    @abstractproperty
    def power_list(self):
        pass
    
    def info(self):
        print(self.__str__())
        
    @abstractmethod
    def __str__(self):
        pass

