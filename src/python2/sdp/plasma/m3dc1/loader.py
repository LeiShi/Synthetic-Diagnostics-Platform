# -*- coding: utf-8 -*-
"""
Laoding module for M3D-C1 simulation

M3D-C1 output data is assumed to be preprocessed by Gerrit Kramer's loader for
FWR2D code.

Created on Fri May 13 22:26:04 2016

@author: lei
"""
from os import path

import numpy as np
from scipy.io.netcdf import netcdf_file as ncfile

from ...geometry.grid import Cartesian2D
from ...settings.unitsystem import cgs
from ...diagnostic.availdiags import Available_Diagnostics
from ..profile import ECEI_Profile

class M3DC1_Loader(object):
    """Loader class for M3D-C1 code
    
    Provides 2D profile and fluctuation loading form fwr2d data file formats
    rearrange of grid and interface to other diangostic profiles
    
    Initializaiton
    ===============
    
    __init__(self, m3dpath, tor_slice=0)
    """
    
    def __init__(self, m3dpath='./', tor_slice=1):
        """
        :param str path: loading directory of M3DC1 files, default to be current
        :param int tor_slice: toroidal slice number to be loaded, used for 
                              fluctuations
        """
        
        self._path = m3dpath
        self._tor_slice = np.array(tor_slice)
        
        self._load_equilibrium()
        self._load_fluctuation()
        
        
    def _load_equilibrium(self, tol=1e-14):

        eq_file = path.join(self._path, 'C1.h5_equ_ufile.cdf')
        eqf = ncfile(eq_file, 'r')
        
        R = np.copy(eqf.variables['rr'].data) * 100 # convert to cm
        Z = np.copy(eqf.variables['zz'].data) * 100 
        #TODO make sure the R, Z mesh is uniform or not
        # create interpolators and interpolate them on uniform mesh
        
        Rmin = np.min(R)
        Rmax = np.max(R)
        NR = eqf.dimensions['r']
        Zmin = np.min(Z)
        Zmax = np.max(Z)
        NZ = eqf.dimensions['z']
        assert (NR == len(R))
        assert (NZ == len(Z))
        
        self.grid = Cartesian2D(DownLeft=(Zmin, Rmin), UpRight=(Zmax, Rmax), 
                                NR=NR, NZ=NZ)
        # convert from m^-3 to cm^-3
        self.ne0 = np.copy(eqf.variables['ne'].data) * 1e-6
        # convert from keV to erg
        self.Te0 = np.copy(eqf.variables['te'].data) * cgs['keV']
        # set all negative Te (Why do we have that?!) to zero        
        self.Te0[self.Te0 < tol*np.max(self.Te0) ] = tol*np.max(self.Te0)
        # convert from Tesla to Gauss
        self.B0 = np.copy(eqf.variables['bb'].data) * 1e4
        
        eqf.close()
        
    def _load_fluctuation(self, tol=1e-14):
        NR = self.grid.NR
        NZ = self.grid.NZ
        if self._tor_slice.ndim == 0:
            fluc_file = path.join(self._path, 
                             'C1.h5_{0:0>4}_ufile.cdf'.format(self._tor_slice))
            flucf = ncfile(fluc_file, 'r')
            
            self.time = np.array([0])
            self.dne = np.empty((1, NZ, NR))
            self.dTe = np.empty_like(self.dne)
            self.dB = np.empty_like(self.dne)
            
            Tef = np.copy(flucf.variables['te'].data)
            # Similar to equilibrium, any negative total Te should also be set 
            # to marginal positive value
            Te_margin = tol*np.max(Tef)
            Tef[Tef < Te_margin] = Te_margin        
            
            self.dne[0] = flucf.variables['ne'].data*1e-6 - self.ne0
            self.dTe[0] = Tef*cgs['keV'] - self.Te0
            self.dB[0] = flucf.variables['bb'].data*1e4 - self.B0
            
        elif self._tor_slice.ndim == 1:
            nt = len(self._tor_slice)
            self.time = np.arange(nt)
            self.dne = np.empty((nt, NZ, NR))
            self.dTe = np.empty_like(self.dne)
            self.dB = np.empty_like(self.dne)
            
            for i, tor in enumerate(self._tor_slice):
                fluc_file = path.join(self._path, 
                             'C1.h5_{0:0>4}_ufile.cdf'.format(tor))
                flucf = ncfile(fluc_file, 'r')
                Tef = np.copy(flucf.variables['te'].data)
                # Similar to equilibrium, any negative total Te should also be
                # set to marginal positive value
                Te_margin = tol*np.max(Tef)
                Tef[Tef < Te_margin] = Te_margin        
                
                self.dne[i] = flucf.variables['ne'].data*1e-6 - self.ne0
                self.dTe[i] = Tef*cgs['keV'] - self.Te0
                self.dB[i] = flucf.variables['bb'].data*1e4 - self.B0
        else:
            raise ValueError('invalid tor_slice dimension. Only 1D or scalar\
are allowed. tor_slice shape: {}'.format(self._tor_slice.shape))
        
    def create_profile(self, diagnostic):
        """Create required profile object for specific diagnostics
        
        :param diagnostic: Specify the synthetic diagnostic that uses the
                           profile. If not given, a list of available 
                           diagnostics will be printed.
        :type diagnostic: string
        :param grid: The grid on which all required profiles will be given. If 
                     not specified, ``self.grid`` will be used.
        :type grid: :py:class:`<...geometry.Grid.Grid>` derived class
        """
        if (diagnostic is None) or (diagnostic not in Available_Diagnostics):
            raise ValueError('Diagnostic {} not found! Currently available \
            diagnostics are:\n{}'.format(diagnostic, Available_Diagnostics))
            
        if diagnostic in ['ecei1d', 'ecei2d']:
            
            return ECEI_Profile(self.grid, self.ne0, self.Te0, self.B0, 
                                self.time, self.dne, self.dTe, self.dTe, 
                                self.dB)
            
        else:
            raise NotImplementedError('Sorry, M3DC1 interface for {} \
diagnostic is not available yet. Please report you need to the developer of \
the diagnostic module.'.format(diagnostic))
        
        
                                
        

