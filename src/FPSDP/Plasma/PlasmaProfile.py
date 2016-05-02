# -*- coding: utf-8 -*-
"""
This module defines the plasma profile class.

All profile generators -- analytical, experimental, or simulational-- must 
provide output of plasma profile object, which contains all useful information
of certain synthetic diagnostic requires.

Created on Mon Jan 18 13:27:04 2016

@author: lei
"""
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..Geometry.Grid import Grid
from ..GeneralSettings.UnitSystem import UnitSystem, cgs
from ..GeneralSettings.Exceptions import PlasmaWarning

class IonClass(object):
    """General class for a kind of ions
    
    :param string name: name of the ion kind
    :param int mass: mass of the ion, in unit of atomic mass
    :param unit charge: electrical charge of the ion, in unit of elementary 
                        charge
    """
    
    def __init__(self, mass, charge, name='temperary_ion_kind'):
        """
        """        
        
        self._name = name
        self.mass = mass
        self.charge = charge
        
    def __str__(self):
        
        return 'Ion kind: {}\nMass: {} proton mass\nCharge: {} elementary \
charge'.format(self._name, self.mass, self.charge)

    def info(self):
        print str(self)


# some pre-defined ion classes

HYDROGEN = IonClass(1, 1, 'H+')
DEUTERIUM = IonClass(2, 1, 'D+')
TRITIUM = IonClass(3, 1, 'T+')
# more ion species can be added below

class PlasmaProfile(object):
    """Base class for all plasma profiles required for synthetic diagnostics.
        
    In general, a profile can have no plasma quantities, but must have a grid
    layout to contain possible quantities.
    
    :param grid: Grid for the profiles
    :type grid: :py:class:`..Geometry.Grid.Grid` object
    """
    
    def __init__(self, grid, unitsystem):
        assert isinstance(grid, Grid)
        assert isinstance(unitsystem, UnitSystem)
        self.grid = grid
        self.unit_system = unitsystem
        self._name = 'General plasma profile'
        
    @property
    def class_name(self):
        return 'PlasmaProfile'
        
    @property
    def parameters(self):
        return dict(grid=self.grid, unitsystem=self.unit_system)
        
    def physical_quantities(self):
        return 'none'
        
    def __str__(self):
        return '{}:\n\nUnit System:{}\nGrid:{}\nPhysical Quantities:\n{}\n'.\
                format(self._name, str(self.unit_system),str(self.grid), 
                       self.physical_quantities())
        
class ECEI_Profile(PlasmaProfile):
    """Plasma profile for synthetic Electron Cyclotron Emission Imaging.
    
    Initialization
    ---------------

    __init__(self, grid, ne0, Te0, B0, time=None, dne=None, dTe_para=None, 
                 dTe_perp=None, dB=None, unitsystem = cgs)
    
    :var ne0: equilibrium electron density
    :var Te0: equilibrium electron temperature
    :var B0: equilibrium magnetic field
    :var dne: *optional*, electron density perturbation
    :var Te_para:  *optional*, fluctuated electron temperature parallel to B
    :var Te_perp: *optional*, fluctuatied electron temperature perpendicular 
                  to B
    
    These should all be passed in compatible with the ``grid`` specification.
    
    :raises AssertionError: if any of the above quantities are not compatible
    
    Methods
    --------
    
    Following methods are provided:
    
    setup_interps(self, equilibrium_only = False):
        Create interpolators for plasma quantities. This is useful if repeated
        call of "get_*" methods is required.
        
    get_ne0(self, coordinates):
        return ne0 interpolated at *coordinates*
        
    get_Te0(self, coordinates):
        return Te0 interpolated at *coordinates*
        
    get_B0(self, coordinates):
        return B0 interpolated at *coordinates*
        
    get_dne(self, coordinates, time=None):
        return dne interpolated at *coordinates*, for each time step in *time*
        
    get_dB(self, coordinates, time=None):
        return dB interpolated at *coordinates*, for each time step in *time*
        
    get_dTe_para(self, coordinates, time=None):
        return dTe_para interpolated at *coordinates*, for each time step in 
        *time*
        
    get_dTe_perp(self, coordinates, time=None):
        return dTe_perp interpolated at *coordinates*, for each time step in 
        *time*
        
    get_ne(self, coordinates, eq_only=True, time=None):
        wrapper for getting total electron densities        
        
        If eq_only is True, only equilibirum density is returned
        otherwise, the total density is returned.
        
        If time is None, *all* available time steps will be used for fluctuated
        part. This only apply to eq_only=False case.
        
    get_B(self, coordinates, eq_only=True, time=None):
        wrapper for getting total magnetic field
        
    get_Te(self, coordinates, eq_only=True, perpendicular = True, time=None):
        wrapper for getting electron temperature
        
        If eq_only is True, only equilibirum Te is returned
        otherwise, the total Te is returned.
        
        if perpendicular is True, perturbed perpendicular Te is added.
        Otherwise, perturbed parallel Te is added.        
        
        if time is None, all available time steps for perturbations are 
        returned. Otherwise the given time steps are returned.     
    
    physical_quantities(self):
        return info string containing physical quantities included in the
        profile.
    """
    def __init__(self, grid, ne0, Te0, B0, time=None, dne=None, dTe_para=None, 
                 dTe_perp=None, dB=None, unitsystem = cgs):
        assert isinstance(grid, Grid)
        assert isinstance(unitsystem, UnitSystem)
        # test if all equilibrium quantities has same shape as the grid
        assert ne0.shape == grid.shape
        assert Te0.shape == grid.shape
        assert B0.shape == grid.shape
        # test if all perturbed quantities has shape as first dim = len(time), 
        # and the rest == grid.shape
        self.has_dne = False
        if dne is not None:
            assert time is not None
            assert dne.shape[0] == len(time)
            assert dne.shape[1:] == grid.shape[:]
            self.has_dne = True
            self.time = time
            self.dne = dne
            
        self.has_dTe_para = False
        if dTe_para is not None:
            assert time is not None
            assert dTe_para.shape[0] == len(time)
            assert dTe_para.shape[1:] == grid.shape[:]
            self.has_dTe_para = True
            self.time = time
            self.dTe_para = dTe_para
            
        self.has_dTe_perp = False
        if dTe_perp is not None:
            assert time is not None
            assert dTe_perp.shape[0] == len(time)
            assert dTe_perp.shape[1:] == grid.shape[:]
            self.has_dTe_perp = True
            self.time = time
            self.dTe_perp = dTe_perp
            
        self.has_dB = False
        if dB is not None:
            assert time is not None
            assert dB.shape[0] == len(time)
            assert dB.shape[1:] == grid.shape[:]
            self.has_dB = True
            self.time = time
            self.dB = dB
        self._name = 'Electron Cyclotron Emission Imaging Plasma Profile'
        self.unit_system = unitsystem
        self.grid = grid
        self.ne0 = ne0
        self.Te0 = Te0
        self.B0 = B0
        
    @property
    def class_name(self):
        """return the name of the class as a string"""
        return 'ECEI_Profile'
       
    @property   
    def parameters(self):
        """return a whole parameter dictionary that can initialize the profile
        """
        params = dict(grid=self.grid, ne0=self.ne0, Te0=self.Te0, B0=self.B0,
                      unitsystem=self.unit_system)
        if self.has_dB:
            params['time'] = self.time
            params['dB'] = self.dB
        if self.has_dne:
            params['time'] = self.time
            params['dne'] = self.dne
        if self.has_dTe_para:
            params['time'] = self.time
            params['dTe_para'] = self.dTe_para
        if self.has_dTe_perp:
            params['time'] = self.time
            params['dTe_perp'] = self.dTe_perp
        
        return params
        
        
        
    def setup_interps(self, equilibrium_only = False):
        """setup interpolators for frequent evaluation of profile quantities on
        given locations.
        
        """
        mesh = self.grid.get_mesh()
        self.Te0_sp = RegularGridInterpolator(mesh, self.Te0)
        self.ne0_sp = RegularGridInterpolator(mesh, self.ne0)
        self.B0_sp = RegularGridInterpolator(mesh, self.B0)
        if not equilibrium_only:
            if (self.has_dne):
                self.dne_sp = []
                for i in range(len(self.time)):
                    self.dne_sp.append(RegularGridInterpolator(mesh, 
                                                         self.dne[i]))
            if (self.has_dTe_para):
                self.dTe_para_sp = []
                for i in range(len(self.time)):
                    self.dTe_para_sp.append( RegularGridInterpolator(mesh, 
                                                         self.dTe_para[i]))
            if (self.has_dTe_perp):
                self.dTe_perp_sp = []
                for i in range(len(self.time)):
                    self.dTe_perp_sp.append( RegularGridInterpolator(mesh, 
                                                         self.dTe_perp[i]))
                                                         
            if (self.has_dB):
                self.dB_sp = []
                for i in range(len(self.time)):
                    self.dB_sp.append( RegularGridInterpolator(mesh, 
                                                         self.dB[i]))

    def get_ne0(self, coordinates):
        """return ne0 interpolated at *coordinates*
        
        :param coordinates: Coordinates given in (Z,Y,X) *(3D)* or (Z,R) 
                            *(2D)* , or (X,) *(1D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        """
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        transpose_axes = range(1,coordinates.ndim)
        transpose_axes.append(0)
        points = np.transpose(coordinates, transpose_axes)
        try:
            return self.ne0_sp(points)
        except AttributeError:
            print 'ne0_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interps function first.'
            mesh = self.grid.get_mesh()
            ne0_sp = RegularGridInterpolator(mesh, self.ne0)
            return ne0_sp(points)
            

    def get_Te0(self, coordinates):
        """return Te0 interpolated at *coordinates*
        
        :param coordinates: Coordinates given in (Z,Y,X) *(3D)* or (Z,R) 
                            *(2D)* , or (X,) *(1D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        """
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        transpose_axes = range(1,coordinates.ndim)
        transpose_axes.append(0)
        points = np.transpose(coordinates, transpose_axes)
        try:
            return self.Te0_sp(points)
        except AttributeError:
            print 'Te0_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interps function first.'
            mesh = self.grid.get_mesh()
            Te0_sp = RegularGridInterpolator(mesh, self.Te0)
            return Te0_sp(points)


    def get_B0(self, coordinates):
        """return B0 interpolated at *coordinates*
        
        :param coordinates: Coordinates given in (Z,Y,X) *(3D)* or (Z,R) 
                            *(2D)* , or (X,) *(1D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        """
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        transpose_axes = range(1,coordinates.ndim)
        transpose_axes.append(0)
        points = np.transpose(coordinates, transpose_axes)
        try:
            return self.B0_sp(points)
        except AttributeError:
            print 'B0_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interps function first.'
            mesh = self.grid.get_mesh()
            B0_sp = RegularGridInterpolator(mesh, self.B0)
            return B0_sp(points)  
            

    def get_dne(self, coordinates, time=None):
        """return dne interpolated at *coordinates*, for each time step
        
        :param coordinates: Coordinates given in (Z,Y,X) *(3D)* or (Z,R) 
                            *(2D)* , or (X,) *(1D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid* 
        :param time: Optional, the time steps chosen to return. If None, all 
                     available times are returned.
        :type time: array_like or scalar of int
        
        :return: dne time series 
        :rtype: ndarray with shape ``(nt,nc1,nc2,...,ncn)``, where 
                ``nt=len(time)``, ``(nc1,nc2,...,ncn)=coordinates[0].shape``
                if time is scalar, the shape is (nc1, nc2, ..., ncn) only.
        """
        assert self.has_dne
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        
        if time is None:
            time = np.arange(len(self.time))

        time = np.array(time)        
        
        if time.ndim == 0:
            result_shape = coordinates.shape[1:]    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                result = self.dne_sp[time](points)
                
            except AttributeError:
                print 'dne_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interps function first.'
                mesh = self.grid.get_mesh()            
                dne_sp = RegularGridInterpolator(mesh, self.dne[time])
                result = dne_sp(points)
            return result
        elif time.ndim == 1:
            result_shape = [i for i in coordinates.shape]
            nt = len(time)
            result_shape[0] = nt
    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                for i,t in enumerate(time):
                    result[i] = self.dne_sp[t](points)
                
            except AttributeError:
                print 'dne_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interps function first.'
                mesh = self.grid.get_mesh()            
                for i,t in enumerate(time):
                    dne_sp = RegularGridInterpolator(mesh, self.dne[t])
                    result[i] = dne_sp(points)
            return result
        else:
            raise ValueError('time can only be int or 1D array of int.')
            
    def get_dB(self, coordinates, time=None):
        """return dB interpolated at *coordinates*, for each time step
        
        :param coordinates: Coordinates given in (Z,Y,X) *(3D)* or (Z,R) 
                            *(2D)* , or (X,) *(1D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        :param time: Optional, the time steps chosen to return. If None, all 
                     available times are returned.
        :type time: array_like or scalar of int
        
        :return: dne time series 
        :rtype: ndarray with shape ``(nt,nc1,nc2,...,ncn)``, where 
                ``nt=len(time)``, ``(nc1,nc2,...,ncn)=coordinates[0].shape``
                if time is scalar, the shape is (nc1, nc2, ..., ncn) only.
        """
        assert self.has_dB
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        
        if time is None:
            time = np.arange(len(self.time))
        
        time = np.array(time)        
        
        if time.ndim == 0:
            result_shape = coordinates.shape[1:]    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                result = self.dB_sp[time](points)
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interps function first.'
                mesh = self.grid.get_mesh()            
                dB_sp = RegularGridInterpolator(mesh, self.dB[time])
                result = dB_sp(points)
            return result
        
        elif time.ndim == 1:
            # Note that the first dimension in coordinates is the number of 
            # spatial axis. We can simply overwrite it with time steps to get 
            # the desired shape of result. 
            result_shape = [i for i in coordinates.shape]
            nt = len(time)
            result_shape[0] = nt
    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                for i,t in enumerate(time):
                    result[i] = self.dB_sp[t](points)
            except AttributeError:
                print 'dB_sp has not been created. Temperary interpolator \
generated. If this message shows up a lot of times, please consider calling \
setup_interps function first.'
                mesh = self.grid.get_mesh()            
                for i,t in enumerate(time):
                    dB_sp = RegularGridInterpolator(mesh, self.dB[t])
                    result[i] = dB_sp(points)
            return result
        else:
            raise ValueError('time can only be int or 1D array of int.')


    def get_dTe_perp(self, coordinates, time=None):
        """return dTe_perp interpolated at *coordinates*, for each time step
        
        :param coordinates: Coordinates given in (Z,Y,X) *(3D)* or (Z,R) 
                            *(2D)* , or (X,) *(1D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        :param time: Optional, the time steps chosen to return. If None, all 
                     available times are returned.
        :type time: array_like or scalar of int
        
        :return: dTe_perp time series 
        :rtype: ndarray with shape ``(nt,nc1,nc2,...,ncn)``, where 
                ``nt=len(time)``, ``(nc1,nc2,...,ncn)=coordinates[0].shape``
                if time is scalar, the shape is (nc1, nc2, ..., ncn) only.
        """
        assert self.has_dTe_perp
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        
        if time is None:
            time = np.arange(len(self.time)) 
        time = np.array(time)
        if time.ndim == 0:
            result_shape = coordinates.shape[1:]    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                result = self.dTe_perp_sp[time](points)
            except AttributeError:
                print 'dTe_perp_sp has not been created. Temperary \
interpolator generated. If this message shows up a lot of times, please \
consider calling setup_interps function first.'
                mesh = self.grid.get_mesh()            
                dte_sp = RegularGridInterpolator(mesh, self.dTe_perp[time])
                result = dte_sp(points)
            return result
                
        elif time.ndim == 1:
            result_shape = [i for i in coordinates.shape]
            nt = len(time)
            result_shape[0] = nt
    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                for i,t in enumerate(time):
                    result[i] = self.dTe_perp_sp[t](points)
            except AttributeError:
                print 'dTe_perp_sp has not been created. Temperary \
interpolator generated. If this message shows up a lot of times, please \
consider calling setup_interps function first.'
                mesh = self.grid.get_mesh()            
                for i,t in enumerate(time):
                    dte_sp = RegularGridInterpolator(mesh, self.dTe_perp[t])
                    result[i] = dte_sp(points)
            return result
        else:
            raise ValueError('time can only be int or 1D array of int.')

            
    def get_dTe_para(self, coordinates, time=None):
        """return dTe_para interpolated at *coordinates*, for each time step
        
        :param coordinates: Coordinates given in (Z,Y,X) *(3D)* or (Z,R) 
                            *(2D)* , or (X,) *(1D)* order.
        :type coordinates: *dim* ndarrays, *dim* is the dimensionality of 
                           *self.grid*  
        :param time: Optional, the time steps chosen to return. If None, all 
                     available times are returned.
        :type time: array_like or scalar of int
        
        :return: dTe_para time series 
        :rtype: ndarray with shape ``(nt,nc1,nc2,...,ncn)``, where 
                ``nt=len(time)``, ``(nc1,nc2,...,ncn)=coordinates[0].shape``
                if time is scalar, the shape is (nc1, nc2, ..., ncn) only.
        """
        assert self.has_dTe_para
        coordinates = np.array(coordinates)
        assert self.grid.dimension == coordinates.shape[0]
        
        if time is None:
            time = np.arange(len(self.time))  

        time = np.array(time)        
        
        if time.ndim == 0:
            result_shape = coordinates.shape[1:]    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                result = self.dTe_para_sp[time](points)
            except AttributeError:
                print 'dTe_para_sp has not been created. Temperary \
interpolator generated. If this message shows up a lot of times, please \
consider calling setup_interps function first.'
                mesh = self.grid.get_mesh()            
                dte_sp = RegularGridInterpolator(mesh, self.dTe_para[time])
                result = dte_sp(points)
            return result        
        elif time.ndim == 1:
            result_shape = [i for i in coordinates.shape]
            nt = len(time)
            result_shape[0] = nt
    
            result = np.empty(result_shape)        
            
            transpose_axes = range(1,coordinates.ndim)
            transpose_axes.append(0)
            points = np.transpose(coordinates, transpose_axes)
            try:
                for i,t in enumerate(time):
                    result[i] = self.dTe_para_sp[t](points)
            except AttributeError:
                print 'dTe_para_sp has not been created. Temperary \
interpolator generated. If this message shows up a lot of times, please \
consider calling setup_interps function first.'
                mesh = self.grid.get_mesh()            
                for i,t in enumerate(time):
                    dte_sp = RegularGridInterpolator(mesh, self.dTe_para[t])
                    result[i] = dte_sp(points)
            return result
        else:
            raise ValueError('time can only be int or 1D array of int.')
            
    def get_ne(self, coordinates, eq_only=True, time=None):
        """wrapper for getting electron densities
        
        If eq_only is True, only equilibirum density is returned
        otherwise, the total density is returned.
        """
        if eq_only:
            return self.get_ne0(coordinates)
        else:
            if self.has_dne:
                return self.get_ne0(coordinates) + self.get_dne(coordinates, 
                                                                time)
            else:
                warnings.warn('get_ne is called with eq_only=False, but no \
electron density perturbation data available. Equilibrium data is returned.',
                              PlasmaWarning)
                return self.get_ne0(coordinates)
                
    def get_B(self, coordinates, eq_only=True, time=None):
        """wrapper for getting magnetic field strength
        
        If eq_only is True, only equilibirum B is returned
        otherwise, the total B is returned.
        
        if time is None, all available time steps for perturbations are 
        returned. Otherwise the given time steps are returned.
        """
        if eq_only:
            return self.get_B0(coordinates)
        else:
            if self.has_dB:
                return self.get_B0(coordinates) + self.get_dB(coordinates, 
                                                              time)
            else:
                warnings.warn('get_B is called with eq_only=False, but no \
magnetic field perturbation data available. Equilibrium value is returned.',
                               PlasmaWarning)
                return self.get_B0(coordinates)

    def get_Te(self, coordinates, eq_only=True, perpendicular = True, 
               time=None):
        """wrapper for getting electron temperature
        
        If eq_only is True, only equilibirum Te is returned
        otherwise, the total Te is returned.
        
        if perpendicular is True, perturbed perpendicular Te is added.
        Otherwise, perturbed parallel Te is added.        
        
        if time is None, all available time steps for perturbations are 
        returned. Otherwise the given time steps are returned.
        
        
        """
        if eq_only:
            return self.get_Te0(coordinates)
        elif perpendicular:
            if self.has_dTe_perp:
                return self.get_Te0(coordinates) + \
                       self.get_dTe_perp(coordinates, time)
            else:
                warnings.warn('get_Te is called with eq_only=False, \
perpendicular=True but no electron perpendicular temperature perturbation data\
 available. Equilibrium data is returned.', PlasmaWarning)
                return self.get_Te0(coordinates) 
        else:
            if self.has_dTe_para:
                return self.get_Te0(coordinates) + \
                       self.get_dTe_para(coordinates, time)
            else:
                warnings.warn('get_Te is called with eq_only=False, \
perpendicular=False but no electron parallel temperature perturbation data\
 available. Equilibrium data is returned.', PlasmaWarning)
                return self.get_Te0(coordinates) 


    
    def physical_quantities(self):
        """return info string containing physical quantities included in the
        profile.
        """
        keV = cgs['keV']
        
        result = 'Equilibrium:\n\
    Electron density: ne0 (max: {0:.3}, min:{1:.3} cm^-3)\n\
    Electron temperature: Te0 (max:{2:.3}, min:{3:.3} keV)\n\
    Magnetic field: B0 (max:{4:.3}, min:{5:.3} Gauss)\n\
Fluctuation:\n'.format(np.max(self.ne0), np.min(self.ne0), 
                       np.max(self.Te0)/keV, np.min(self.Te0)/keV, 
                       np.max(self.B0), np.min(self.B0))
        if self.has_dne:
            result += '    Electron density: dne (max:{0:.3}, min:{1:.3} \
cm^-3)\n'.format(np.max(self.dne), np.min(self.dne))
        if self.has_dTe_para:
            result += '    Electron parallel temperature: dTe_para \
(max:{0:.3}, min:{1:.3} keV)\n'.format(np.max(self.dTe_para)/keV, 
                                       np.min(self.dTe_para)/keV)
        if self.has_dTe_perp:
            result += '    Electron perpendicular temperature: dTe_perp\
(max:{0:.3}, min:{1:.3} keV)\n'.format(np.max(self.dTe_perp)/keV, 
                                       np.min(self.dTe_perp)/keV)
        if self.has_dB:
            result += '    Magnetic field magnitude: dB (max:{0:.3}, \
min:{1:.3} Gauss)\n'.format(np.max(self.dB), np.min(self.dB))
        return result
