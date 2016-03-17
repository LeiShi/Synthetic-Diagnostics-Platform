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
from ....Geometry.Grid import Cartesian1D, FinePatch1D
from ....GeneralSettings.UnitSystem import cgs

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
        self._set_propagator()
            
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
        self._set_detector()
        self._auto_coords_adjusted = False
        
    
    def _set_propagator(self):
        """setup propagator for diagnostic
        """
        self.propagator = \
            ParaxialPerpendicularPropagator2D(self.plasma, 
                                              self.dielectric, 
                                              self.polarization,
                                              direction=-1,
                                ray_y=self.detector.central_beam.waist_loc[1],
                                              max_harmonic=self.max_harmonic,
                                              max_power=self.max_power)
        
    def _set_detector(self):
        """setup incidental field mesh for detector 
        """
        try:
            self.detector.set_inc_coords(self.x_start, self.Y1D, self.Z1D)
        except AttributeError:
            print('Calculation mesh not set yet! Call set_coords() to setup\
before running ECE.', file=sys.stderr)
    
    def auto_adjust_coordinates(self, fine_coeff=1):
        
        if self._auto_coords_adjusted:
            if fine_coeff == self._fine_coeff:
                return
            else:
                self._auto_coords_adjusted = False
                self.auto_adjust_coordinates(fine_coeff)
        else:
            # run propagation at cental frequency once to obtain the local 
            # emission pattern
            try:
                x_coord = self.x_coord
            except AttributeError:
                
                omega = self.detector.central_omega
                E_inc = self.detector.central_E_inc
                E0 = self.propagator.propagate(omega,  x_start=None, 
                                               x_end=None, nx=None, 
                                               E_start=E_inc, y_E=self.Y1D,
                                               z_E = self.Z1D, 
                                               x_coords=self.X1D,
                                               keepFFTz=True) * self.dZ
                                               
                kz = self.propagator.kz[:,0,0]
                k0 = self.propagator.k_0[::2]
                K_k = np.empty( (3,3,self.NZ,self.NY,self.NX), dtype='complex')
                for j, x in enumerate(self.X1D):
                    X = x + np.zeros_like(self.Y1D)
                    K_k[..., j] = self.scct([self.Y1D, X], omega, kz, k0[j], 
                                            eq_only=True)
                if self.polarization == 'X':                    
                    e = np.asarray( [self.propagator.e_x[::2], 
                                     self.propagator.e_y[::2]] )  
                    e_conj = np.conj(e)
                    # For X mode, normalization of Poynting vector has an extra
                    # |e_y|^2 term that is not included in detector power 
                    # normalization
                    E0 /= np.sqrt(e[1]*e_conj[1]) 
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
                max_int = np.max(np.abs(inty))
                self._max_idx = np.argmax(np.abs(inty))
                self._x = self.X1D[self._max_idx]
                self._max_idy = np.argmax(np.abs(intkz[:,self._max_idx]))        
                self._y = self.Y1D[self._max_idy]
                self._max_idz =np.argmax(np.abs(np.fft.ifft(\
                                        (eK_ke*E0)[:,self._max_idy,self._max_idx])\
                                       * np.conj(np.fft.ifft(\
                                         E0[:,self._max_idy, self._max_idx]))))
                self._z = self.Z1D[self._max_idz]
                patch_array = np.abs(inty) >= np.exp(-9)*max_int
                #create patched x coordinates
                wave_length = 2*np.pi*cgs['c']/omega
                self.x_coord = FinePatch1D(self.X1D[0], self.X1D[-1], 
                                           ResX=5*wave_length/fine_coeff)
                # search and add patches
                in_patch = False
                for i, patch_flag in enumerate(patch_array):
                    if not in_patch:
                        if not patch_flag:
                            continue
                        else:
                            x_start = self.X1D[i]
                            in_patch = True
                            continue
                    else:
                        if not patch_flag or (i == len(patch_array)):
                            x_end = self.X1D[i]
                            patch = Cartesian1D(x_start, x_end,
                                                ResX=0.5*wave_length/fine_coeff)
                            self.x_coord.add_patch(patch)
                            in_patch = False
                        else:
                            continue
                self._fine_coeff = fine_coeff
                self._auto_coords_adjusted = True
                self.set_coords([self.Z1D, self.Y1D, self.x_coord.X1D])
                print('Automatic coordinates adjustment performed! To reset your \
mesh, call set_coords() again.')
                return
            coeff_ratio = self._fine_coeff/np.float(fine_coeff)
            if not x_coord.reversed:
                Xmin = x_coord.Xmin
                Xmax = x_coord.Xmax
            else:
                Xmin = x_coord.Xmax
                Xmax = x_coord.Xmin
                
            self.x_coord = FinePatch1D(Xmin, Xmax, 
                                       ResX=x_coord.ResX*coeff_ratio)
            
            for p in x_coord.patch_list:
                if not p.reversed:
                    Xmin = p.Xmin
                    Xmax = p.Xmax
                else:
                    Xmin = p.Xmax
                    Xmax = p.Xmin
                self.x_coord.add_patch(Cartesian1D(Xmin, Xmax, 
                                                   ResX=p.ResX*coeff_ratio))
            self.set_coords([self.Z1D, self.Y1D, self.x_coord.X1D])
            print('Automatic coordinates adjustment performed! To reset your \
mesh, call set_coords() again.')
            self._auto_coords_adjusted = True
            
        
    
    def diagnose(self, time=None, debug=False, auto_patch=False):
        r"""Calculates the received power by antenna.
        
        Propagate wave in conjugate plasma, and integrate over the whole space
        to obtain the power using the formula [shi16]_:
        
        .. math::
            P_e(\omega) = \frac{1}{32\pi} \int \,dk_z \,dx \,dy \; 
            \vec{E}_0(x, y, k_z,\omega) 
            \cdot  \hat{K}_k(x, y, k_z; \omega)
            \cdot \vec{E}^*_0(x, y, k_z,\omega)

        where :math:`\hat{K}_k(x, y, k_z; \omega)` is the current correlation 
        tensor calculated in 
        :py:module:`FPSDP.Diagnostics.ECEI.ECEI2D.CurrentCorrelationTensor`.

        :param int time: time step in plasma profile chosen for diagnose. if 
                         not given, only equilibrium will be used.
        :param bool debug: debug mode flag. if True, more information will be 
                           kept for examining.
        :param bool auto_patch: if True, program will automatically detect the 
                                significant range of x where emission power is
                                originated, and add finer grid patch to that 
                                region. This may cause a decrease of speed, but
                                can improve the accuracy. Default is False, the
                                programmer is responsible to set proper x mesh.
        """
        if time is None:
            eq_only = True
        else:
            eq_only = False
        
        try:
            E_inc_list = self.detector.E_inc_list
        except AttributeError:
            print('coordinates need to be setup before diagnose. Call \
set_coords() first.')
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
                                
        if auto_patch:
            if not self._auto_coords_adjusted:
                self.auto_adjust_coordinates()
            else:
                pass
            
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
                # For X mode, normalization of Poynting vector has an extra
                # |e_y|^2 term that is not included in detector power 
                # normalization
                E0 /= np.sqrt(e[1,0]*e_conj[1,0])
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
        return np.real(self.Ps) * 2*np.pi
        
    @property 
    def Te(self):
        """measured electron temperature
        """
        try:
            return 2*np.pi*np.real(self.Ps)
        except AttributeError:
            print('Diagnostic has not run! call diagnose() before retrieving\
 measured temperature.', file=sys.stderr)
 
    @property
    def diag_x(self):
        """list of x coordinates where significant emission came from, as well 
        as width around each points
        """
        try:
            self.x_coord.patch_list
        except AttributeError:
            self.auto_adjust_coordinates()
        x_list=[]
        dx_list=[]
        for patch in self.x_coord.patch_list:
            x_list.append((patch.Xmin + patch.Xmax)/2)
            dx_list.append(np.abs(patch.Xmax - patch.Xmin)/6)
        return (x_list, dx_list)
        
    @property
    def view_point(self):
        try:
            return (self._z, self._y, self._x)
        except AttributeError:
            self.auto_adjust_coordinates()
            return (self._z, self._y, self._x)
            
        
                
                
            
            
            
            
    
    

