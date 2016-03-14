# -*- coding: utf-8 -*-
"""
Provides various formulations for source current correlation tensor used in 
calculation of electron cyclotron emmision power, using Reciprocity Theorem
in [piliya02]_.

The isotropic Maxwellian plasma's source current correlation tensor can be 
ralated to the anti-Hermitian part of the dielectric tensor by Kirchhoff's Law 
of Radiation [shi16]_. 

Non-isotropic Maxwellian and non-relativistic case is derived in [shi16]_ as 
well.

See the corresponding classes' docstring for detailed formulism.

Reference:
**********

.. [piliya02] On application of the reciprocity theorem to calculation of a 
              microwave radiation signal in inhomogeneous hot magnetized 
              plasmas, A D Piliya and A Yu Popov, Plamsa Phys. Controlled 
              Fusion 44(2002) 467-474
.. [shi16] Development of Fusion Plamsa Synthetic Diagnostics Platform and a 2D
           synthetic electron cyclotron emission imaging module, L. Shi, Ph.D. 
           dissertation (2016), Princeton Plasma Physics Laboratory, Princeton
           University.
           
Created on Mon Mar 07 14:28:10 2016

@author: lei
"""

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from ....Plasma.PlasmaProfile import PlasmaProfile
from ....Plasma.DielectricTensor import HotSusceptibility

class SourceCurrentCorrelationTensor(object):
    r"""Abstract base class for all source current correlation tensors
    
    Derived classes must sustantiate following methods/properties:
    
        __call__(coordinates, omega, k_para, k_perp): 
            :math:`\hat{K}_k` tensor in shape (3,3, Nf, Nk_para, Nk_perp, 
            spatial_shape)
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __call__(self, coordinates, omega, k_para, k_perp):
        r"""return :math:`\hat{K}_k` tensor
        
        :param coordinates: Spatial coordinates in order [Y, X]
        :type coordinates: list of ndarrays, all have same shape 
                           (spatial_shape), depends on plasma's dimension, 
                           length of the list may be 1 or 2. 
        :param omega: wave frequencies, must be positive
        :type omega: 1darray of floats, shape (Nf,)
        :param k_para: parallel wave vector
        :type k_para: 1darray of floats, shape (Nk_para,)
        :param k_perp: perpendicular wave vector
        :type k_perp: 1darray of floats, shape (Nk_perp,)
        
        :return: :math:`\hat{K}_k` tensor
        :rtype: ndarray of shape (3,3, Nf, Nk_para, Nk_perp, spatial_shape)
        """
        pass
    

class IsotropicMaxwellian(SourceCurrentCorrelationTensor):
    r"""Isotropic Maxwellian plasma's source current correlation tensor
    
    In isotropic and locally Maxwellian plasma, emission and absorption are 
    related by Kirchhoff's Law of Radiation. It is shown in [shi16]_ that this
    relation leads to the expression of source current correlation tensor:
    
    .. math::
    
        \hat{K}_k = \frac{4}{\pi} \omega T_e \chi_e^a
        
    where
    
    .. math::
        \chi_e^a \equiv \frac{1}{2i} (\chi_e - \chi_e^\textdagger)
        
    is the anti-Hermitian part of electron's susceptibility tensor 
    :math:`\chi_e`.
    
    This expression is valid for both weakly relativistic and non-relativistic
    electrons.
    
    We will then use the existing susceptibility tensor module to calculate 
    each tensor element.

    Initialization
    ***************
    Initiated with a plasma profile object, and a susceptibility tensor object.
    
    __init__(plasma, suscept_class):
        plasma: :py:class:`FPSDP.Plasma.PlasmaProfile.PlasmaProfile` object
        suscept_class: derived class of 
                       :py:class:`FPSDP.Plasma.DielectricTensor.Susceptibility`
        
    """
    def __init__(self, plasma, suscept_class, max_harmonic=4, max_power=4):
        """Initialization with plasma and susceptibility class
        """    
        assert isinstance(plasma, PlasmaProfile)
        assert issubclass(suscept_class, HotSusceptibility)
        self.plasma = plasma
        self.suscept = suscept_class(plasma, 'e', max_harmonic=max_harmonic,
                                     max_power=max_power)
        
    def __call__(self, coordinates, omega, k_para, k_perp, eq_only=True, 
                 time=0, tol=1e-14):
        r"""Return :math:`\hat{K}_k` tensor
        """
        
        Te = self.plasma.get_Te(coordinates, eq_only=eq_only, 
                                perpendicular=True, time=time)
                                
        chi_e = self.suscept(coordinates, omega, k_para, k_perp, eq_only, time,
                             tol)
        trans_index = np.arange(chi_e.ndim)
        trans_index[0]=1
        trans_index[1]=0
        chi_ea = (chi_e - np.conj(np.transpose(chi_e, axes=trans_index)))/2j
        
        return 4*omega*Te*chi_ea/np.pi 
    