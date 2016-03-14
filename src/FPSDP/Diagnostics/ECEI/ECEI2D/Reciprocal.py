# -*- coding: utf-8 -*-
r"""
main module for ECEI2D package

Received power and the effective temperature will be calculated based on 
the Reciprocity Theorem [piliya02]_ [shi16]_ :

.. math::

    P_e(\omega) = \frac{1}{32\pi} \int \rmd k_z \rmd x \rmd y \; 
                  \vec{E}_0(x, y, k_z,\omega) 
                  \cdot  \hat{K}_k(x, y, k_z; \omega)
                  \cdot \vec{E}^*_0(x, y, k_z,\omega)
                  
and

.. math::

    T_e = 2 \pi P_e(\omega)


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


Created on Tue Mar 08 17:04:59 2016

@author: lei
"""
from __future__ import print_function
import sys

import numpy as np
from scipy.integrate import trapz

from .Detector2D import Detector2D
from ....Models.Waves.Propagator import ParaxialPerpendicularPropagator2D
from ....Plasma.PlasmaProfile import ECEI_Profile
from ....Plasma.DielectricTensor import ConjRelElectronColdIon,\
    ConjHotElectronColdIon,SusceptRelativistic, SusceptNonrelativistic
from .CurrentCorrelationTensor import SourceCurrentCorrelationTensor, \
                                      IsotropicMaxwellian

class ECE2D(object):
    """single channel ECE diagnostic
    
    Using Reciprocity Theorem, the received power and corresponding electron 
    temperature is calculated. 

    Initialize with:
        
    :param plasma: plasma to be diagnosed
    :type plasma: :py:class:`FPSDP.Plasma.PlasmaProfile.ECEIProfile` object
    :param detector: receiving antenna 
    :type detector: 
        :py:class`FPSDP.Diagnostics.ECEI.ECEI2D.Detector2D.Detector2D` object
    :param dielectric_class: chosen formulism for calculating conjugate plasma
                             dielectric tensor
    :type dielectric_class: subclass of :py:class:`FPSDP.Plasma.
                            DielectricTensor.Dielectric`.
    :param source_current_correlation_class:
        chosen formulism for source current correlation tensor calculation
    :type source_current_correlation_class:
        subclass of :py:class:`FPSDP.Diagnostics.ECEI.ECEI2D.
        CurrentCorrelationTensor.SourceCurrentCorrelationTensor`.
    
    
    Method
    *******    
    A full calculation consists of two steps:
    - Step 1: Propagate unit power wave from detector into the conjugate 
              plasma, calculate the wave amplitude at each x,y location for
              each kz component, i.e. :math:`E_0^{+}(\omega, k_z, x, y)`
              
    - Step 2: Calculate the source current correlation tensor 
              :math:`\hat{K}_k(\omega, k_z, x, y)`at each x,y 
              location for each kz component. Then calculate 
              :math:`E_0^{+} \cdot \hat{K}_k \cdot E_0^{+*}`. Finally, 
              integrate over x, y, and kz to obtain the result.
              
    detailed information can be found in [shi16]_
    """
    
    def __init__(self, plasma, detector, polarization='X', 
                 weakly_relativistic=True, isotropic=True, 
                 max_harmonic=4, max_power=4):
        self.plasma = plasma
        self.detector = detector
        self.polarization = polarization
        self.max_harmonic = max_harmonic
        self.max_power = max_power
        self.weakly_relativistic = weakly_relativistic
        self.isotropic = isotropic
        if weakly_relativistic:
            self.dielectric =  ConjRelElectronColdIon
        else:
             self.dielectric = ConjHotElectronColdIon
        if isotropic:
            if weakly_relativistic:
                suscept = SusceptRelativistic
            else:
                suscept = SusceptNonrelativistic
            self.scct = IsotropicMaxwellian(self.plasma, 
                                            suscept,
                                            max_harmonic=max_harmonic,
                                            max_power=max_power)
        else:
            # TODO finish non-isotropic current correlation tensor part
            raise NotImplementedError
            
    def set_coords(self, coords):
        """setup Cartesian coordinates for calculation
        
        :param coords: list of coordinates, [Z1D, Y1D, X1D]. Z1D and Y1D need 
                       to be uniformly spaced and monotonically increasing. 
                       X1D only needs to be monotonic, can be decreasing or 
                       non-uniform, it is assumed that probing wave will be
                       launched from X1D[0] and propagate towards X1D[-1]
        :type coords: list of 1D array of floats
        
        Create Attribute:
        
            X1D, Y1D, Z1D
        """
        self.Z1D = np.asarray(coords[0])
        self.Y1D = np.asarray(coords[1])
        self.X1D = np.asarray(coords[2])
        self.NX = len(self.X1D)
        self.NY = len(self.Y1D)
        self.NZ = len(self.Z1D)
        self.x_start = self.X1D[0]
        self.X2D = np.zeros((self.NY, self.NX)) + self.X1D
        self.Y2D = np.zeros_like(self.X2D) + self.Y1D[:, np.newaxis]
        self.dZ = self.Z1D[1]-self.Z1D[0]
        
    def diagnose(self, time=None, debug=False):
        """launch propagation in conjugate plasma
        
        Calculates E_0, and keep record of kz
        """
        if time is None:
            eq_only = True
        else:
            eq_only = False
            
        self.propagator = \
            ParaxialPerpendicularPropagator2D(self.plasma, 
                                              self.dielectric, 
                                              self.polarization,
                                              direction=-1,
                                ray_y=self.detector.central_beam.waist_loc[1],
                                              max_harmonic=self.max_harmonic,
                                              max_power=self.max_power)
                                              
        try:
            self.detector.set_inc_coords(self.x_start, self.Y1D, self.Z1D)
        except AttributeError:
            print('Calculation mesh not set yet! Call set_coords() to setup\
before running ECE.', file=sys.stderr)
            return None
        
        E_inc_list = self.detector.E_inc_list
        if debug: 
            self.E_inc_list = E_inc_list
            self.E0_list = []
            self.k0_list = []
            self.kz_list = []
            self.K_list = []
            self.eK_ke_list = []
            self.integrand_list = []
        Ps_list = np.empty((len(self.detector.omega_list)), 
                                dtype='complex')
        for i, omega in enumerate(self.detector.omega_list):
            E_inc = E_inc_list[i]
            E0 = self.propagator.propagate(omega, x_start=None, 
                                           x_end=None, nx=None, 
                                           E_start=E_inc, y_E=self.Y1D,
                                           z_E = self.Z1D, 
                                           x_coords=self.X1D, time=time,
                                           keepFFTz=True) * self.dZ
            #E0 = np.fft.fftshift(E0, axes=0)
            kz = self.propagator.kz[:,0,0]
            k0 = self.propagator.k_0[::2]
            K_k = np.empty( (3,3,self.NZ,self.NY,self.NX), dtype='complex')
            for j, x in enumerate(self.X1D):
                X = x + np.zeros_like(self.Y1D)
                K_k[..., j] = self.scct([self.Y1D, X], omega, kz, k0[j], 
                                        eq_only, time)
            if self.polarization == 'X':
                e = np.asarray( [self.propagator.e_x[::2], 
                                 self.propagator.e_y[::2]] )  
                e_conj = np.conj(e)
                # inner tensor product with unit polarization vector and K_k
                eK_ke = 0
                for l in xrange(2):
                    for m in xrange(2):
                        eK_ke += e[l] * K_k[l, m, ...] * e_conj[m]
            elif self.polarization == 'O':
                eK_ke = K_k[2,2]
            integrand = eK_ke * E0 * np.conj(E0)/(32*np.pi)
            # integrate over kz dimension         
            intkz = np.sum(integrand, axis=0)*(kz[1]-kz[0])
            # integrate over y dimension
            inty = trapz(intkz, x=self.Y1D, axis=0)
            # integrate over x dimension
            Ps_list[i] = trapz(inty[::-1], x=self.X1D[::-1], axis=0)
            if debug:
                self.E0_list.append(E0)
                self.k0_list.append(k0)
                self.K_list.append(K_k)
                self.eK_ke_list.append(eK_ke)
                self.kz_list.append(kz)
                self.integrand_list.append(integrand)
        if debug:
            self.Ps_list = Ps_list
        if (len(Ps_list) > 1):        
            # detector has a list of omegas, final result will be integrate 
            # over omega space.
            self.Ps = trapz(Ps_list, x=self.detector.omega_list)
        else:
            # detector has only one omega
            self.Ps = Ps_list[0]
        return np.real(self.Ps)
        
    @property 
    def Te(self):
        try:
            return 2*np.pi*np.real(self.Ps)
        except AttributeError:
            print('Diagnostic has not run! call diagnostic() before retrieving\
 measured temperature.', file=sys.stderr)
                
                
            
            
            
            
    
    

