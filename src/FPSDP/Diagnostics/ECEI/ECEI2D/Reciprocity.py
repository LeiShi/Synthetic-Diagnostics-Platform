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
import time as systime

import numpy as np
from scipy.integrate import trapz

from .Detector2D import Detector2D
from ....Models.Waves.Propagator import ParaxialPerpendicularPropagator2D
from ....Plasma.PlasmaProfile import ECEI_Profile
from ....Plasma.DielectricTensor import ConjRelElectronColdIon,\
    ConjHotElectronColdIon,SusceptRelativistic, SusceptNonrelativistic
from .CurrentCorrelationTensor import SourceCurrentCorrelationTensor, \
                                      IsotropicMaxwellian, \
                                      AnisotropicNonrelativisticMaxwellian
from ....Geometry.Grid import Cartesian1D, FinePatch1D
from ....GeneralSettings.UnitSystem import cgs

class ECE2D_property(object):
    """Serializable container for main ECE2D properties
    
    This is mainly used for parallel runs when transfering ECE2D objects 
    directly doesn't work.
    
    Initialize with a ECEI2D object    
    
    Attributes:
        
        X1D, Y1D, Z1D (if is_coords_set)
        diag_X (if is_auto_adjusted)
        E0_list, kz_list, integrand_list (if is_debug)
        if is_diagnosed:
            intkz_list, view_point, view_spot 
            propagator:
                E, eps0, deps
            
    Methods:
    
        is_debug : return True if is debug mode.
        is_coords_set: return True if set_coords is called
        is_auto_adjusted: return True if auto_adjust_mesh is called
        is_diagnosed: return True if diagnose is called
    """
    def __init__(self, ece2d):
        assert isinstance(ece2d, ECE2D)
        try:
            self.X1D = ece2d.X1D
            self.Y1D = ece2d.Y1D
            self.Z1D = ece2d.Z1D
            self._auto_coords_adjusted = ece2d._auto_coords_adjusted
            self._coords_set = True
        except AttributeError:
            self._coords_set = False
            self._auto_coords_adjusted = False
            self._debug = False
            self._diagnosed = False
            return
        if self._auto_coords_adjusted:
            self.diag_x = ece2d.diag_x
        try:
            self.E0_list = ece2d.E0_list
            self.kz_list = ece2d.kz_list
            self.integrand_list = ece2d.integand_list
            self._debug = True
        except AttributeError:
            self._debug = False            
        try:
            self.intkz_list = ece2d.intkz_list
            self.view_point = ece2d.view_point
            self.view_spot = ece2d.view_spot
            self.propagator = ece2d.propagator.properties
            self._diagnosed = True
        except AttributeError:
            self._diagnosed = False
        
        
    def is_debug(self):
        return self._debug
    
    def is_diagnosed(self):
        return self._diagnosed
        
    def is_auto_adjusted(self):
        return self._auto_coords_adjusted
        
    def is_coords_set(self):
        return self._coords_set

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
                           
    Methods
    *******
    
    set_coords(coordinates): set initial coordinates for calculation
    auto_adjust_mesh(fine_coeff): 
        automatically adjust coordinates to optimize the calculation
    diagnose(time, debug, auto_patch, oblique_correction):
        run diagnostic. Create received power Ps.
        
    Attributes
    **********
    After Initialization:
        plasma, detector, polarization, max_harmonic, max_power, 
        weakly_relativistic, isotropic, dielectric, scct(Source Current
        Correlation Tensor), propagator
    
    After set_coords call:
        X1D, Y1D, Z1D
        
    After auto_adjust_mesh call:
        view_point
        diag_x
    
    After diagnose call:
        
        non debug:
            Ps, Te, view_spot
            
        debug:
            E0_list, kz_list, integrand_list, Ps_list
    
    
    Algorithm
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
            if weakly_relativistic:
                # anisotropic weakly relativistic current correlation tensor
                # has not been implemented. Theoretical work is needed.
                raise NotImplementedError
            else:
                # anisotropic non-relativistic tensor
                self.scct = AnisotropicNonrelativisticMaxwellian(self.plasma, 
                                                  max_harmonic=max_harmonic)
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
    
    def auto_adjust_mesh(self, fine_coeff=1, mute=False):
        """automatically adjust X mesh to optimize efficiency
        
        :param float fine_coeff: 
            coefficient controling the step sizes Default is 1, corresponds to 
            within emission area, stepsize is 0.5 lambda, outside is 5 lambda. 
            fine_coeff set to 2 will half the step sizes.
        :param bool mute: if True, no standard output. Default is False.
        
        """        
        tstart = systime.clock()        
        
        try:
            # test if set_coords() has been called.
            self._auto_coords_adjusted
        except AttributeError:
            print('Base coordinates not set! Call set_coords() first.', 
                  file=sys.stderr)
            return
        
        if self._auto_coords_adjusted:
            if fine_coeff == self._fine_coeff:
                return
            else:
                self._auto_coords_adjusted = False
                self.auto_adjust_mesh(fine_coeff, True)
        else:
            # run propagation at cental frequency once to obtain the local 
            # emission pattern
            try:
                x_coord = self.x_coord
            except AttributeError:
                
                omega = self.detector.central_omega
                E_inc = self.detector.central_E_inc
                tilt_h = self.detector.tilt_h
                tilt_v = self.detector.tilt_v
                E0 = self.propagator.propagate(omega,  x_start=None, 
                                               x_end=None, nx=None, 
                                               E_start=E_inc, y_E=self.Y1D,
                                               z_E = self.Z1D, 
                                               x_coords=self.X1D,
                                               tilt_h=tilt_h, tilt_v=tilt_v,
                                               keepFFTz=True) * self.dZ
                                               
                kz = self.propagator.masked_kz[:,0,0]
                dkz = self.propagator.kz[1]-self.propagator.kz[0]
                k0 = self.propagator.k_0[::2]
                K_k = np.zeros( (3,3,self.NZ,self.NY,self.NX), dtype='complex')
                
                mask = self.propagator._mask_z
                for j, x in enumerate(self.X1D):
                    X = x + np.zeros_like(self.Y1D)
                    K_k[:,:,mask,:,j] = \
                      np.transpose(self.scct([self.Y1D, X], omega, kz, 
                                             k0[j], eq_only=True), 
                                   axes=(2,0,1,3))
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
                intkz = np.sum(integrand, axis=0)*(dkz)
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
                self.set_coords([self.Z1D, self.Y1D, self.x_coord.X1D])
                print('Automatic coordinates adjustment performed! To reset \
your mesh, call set_coords() with initial mesh again.')
                self._auto_coords_adjusted = True
                if not mute:
                    print('Walltime: {0:.4}s'.format(systime.clock()-tstart))
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
            if not mute:
                print('Automatic coordinates adjustment performed! To reset \
your mesh, call set_coords() with initial mesh again.')
            self._fine_coeff = fine_coeff
            self._auto_coords_adjusted = True
            tend = systime.clock()            
            if not mute:            
                print('Walltime: {0:.4}s'.format(tend-tstart))
    
    def diagnose(self, time=None, debug=False, auto_patch=False, fine_coeff=1,
                 oblique_correction=True, optimize_z=True, mute=False):
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
        :param float fine_coeff: 
            coefficient controling the step sizes in auto_patch. 
            Default is 1, corresponds to within emission area, stepsize is 0.5 
            lambda, outside is 5 lambda. fine_coeff set to 2 will half the step
            sizes.
        :param oblique_correction: if True, correction to oblique incident
                                   wave will be added. The decay part will have
                                   :math:`\cos(\theta_h)\cos(\theta_v)` term.
                                   Default is True.
        :type oblique_correction: bool
        :param bool optimize_z: 
            if True, optimized propagation will be used. 
            See :class:`FPSDP.Model.Waves.Propagator.
                        ParaxialPerpendicularPropagator2D` for more details. 
        :param bool mute: if True, no output. Default is False.
        """
        tstart = systime.clock()        
        
        if not mute:
            print('Diagnose starts.')
        
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
            self.E0_list = []
            self.kz_list = []
            self.integrand_list = []
        self.intkz_list = []
        Ps_list = np.empty((len(self.detector.omega_list)), 
                                dtype='complex')
                                
        if auto_patch:
            try:
                if not self._auto_coords_adjusted:
                    self.auto_adjust_mesh(fine_coeff=fine_coeff, mute=mute)
                else:
                    pass
            except AttributeError:
                print('Base coordinates not set! Call set_coords() first.', 
                      file=sys.stderr)
                return
            
        for i, omega in enumerate(self.detector.omega_list):
            if not mute:
                print('omega = {0:.4}GHz starts.'.format(omega/(2*np.pi*1e9)))
            E_inc = E_inc_list[i]
            tilt_h = self.detector.tilt_h
            tilt_v = self.detector.tilt_v
            E0 = self.propagator.propagate(omega, x_start=None, 
                                           x_end=None, nx=None, 
                                           E_start=E_inc, y_E=self.Y1D,
                                           z_E = self.Z1D, 
                                           x_coords=self.X1D, time=time,
                                           tilt_h=tilt_h, tilt_v=tilt_v,
                                           keepFFTz=True, 
                                           oblique_correction=\
                                           oblique_correction,
                                           optimize_z=optimize_z) * self.dZ
            kz = self.propagator.masked_kz[:,0,0]
            dkz = self.propagator.kz[1]-self.propagator.kz[0]
            k0 = self.propagator.k_0[::2]
            K_k = np.zeros( (3,3,self.NZ,self.NY,self.NX), dtype='complex')
            if optimize_z:
                mask = self.propagator._mask_z
                for j, x in enumerate(self.X1D):
                    X = x + np.zeros_like(self.Y1D)
                    K_k[:,:,mask,:,j] = \
                      np.transpose(self.scct([self.Y1D, X], omega, kz, 
                                             k0[j], eq_only=True), 
                                   axes=(2,0,1,3))
            else:
                for j, x in enumerate(self.X1D):
                    X = x + np.zeros_like(self.Y1D)
                    K_k[...,j] = self.scct([self.Y1D, X], omega, kz, 
                                                  k0[j], eq_only, time)
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
            intkz = np.sum(integrand, axis=0)*(dkz)
            # integrate over y dimension
            inty = trapz(intkz, x=self.Y1D, axis=0)
            # integrate over x dimension
            Ps_list[i] = trapz(inty[::-1], x=self.X1D[::-1], axis=0)
            if debug:
                self.E0_list.append(E0)
                self.kz_list.append(kz)
                self.integrand_list.append(integrand)
            self.intkz_list.append(intkz)
        if debug:
            self.Ps_list = Ps_list
        if (len(Ps_list) > 1):        
            # detector has a list of omegas, final result will be integrate 
            # over omega space.
            self.Ps = trapz(Ps_list, x=self.detector.omega_list)
        else:
            # detector has only one omega
            self.Ps = Ps_list[0]

        tend = systime.clock()            
        if not mute:
            print('Walltime: {0:.4}s'.format(tend-tstart))
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
            a = raw_input('Emission spot hasn\'t been analyzed. Do you want to \
analyze it now, this may take a few minutes? (y/n)')
            if 'y' in a:
                print('That is considered as a YES.')
                self.auto_adjust_mesh()
            else:
                print('That is a NO. Stop diag_x with no return.')
                return None
        x_list=[]
        dx_list=[]
        for patch in self.x_coord.patch_list:
            x_list.append((patch.Xmin + patch.Xmax)/2)
            dx_list.append(np.abs(patch.Xmax - patch.Xmin)/6)
        return (x_list, dx_list)
        
    @property
    def view_point(self):
        """(Z,Y,X) coordinates of the maximum emission intensity"""
        try:
            return (self._z, self._y, self._x)
        except AttributeError:
            self.auto_adjust_mesh()
            return (self._z, self._y, self._x)
            
    @property
    def view_spot(self):
        """observed emission intensity distribution in Y-X plane 
        """
        try:
            integ = self.intkz_list[self.detector._central_index]
        except AttributeError:
            print('view_spot is only available after diagnosing.\
Call diagnose() first.', file=sys.stderr)

        return np.abs(integ)/np.max(np.abs(integ))
        
    @property
    def properties(self):
        """Serializable data for transferring in parallel runs
        """
        return ECE2D_property(self)
        
            
        
                
                
            
            
            
            
    
    

