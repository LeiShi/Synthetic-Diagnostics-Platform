# -*- coding: utf-8 -*-
"""
Interface for creating FWR2D input files

Functions
**********

generate_cdf(plasma, coordinates, filename='./plasma.cdf'): 
    generate netcdf files for FWR2D containing plasma quantities. 

Created on Wed Aug 17 10:50:45 2016

@author: lei
"""

import numpy as np
from scipy.io.netcdf import netcdf_file

from sdp.settings.unitsystem import cgs

def generate_cdf(plasma, coordinates=None, eq_only=True, time=None, 
                 filename='./plasma.cdf'):
    r"""generate netcdf files for FWR2D containing plasma quantities.
    
    :param plasma: plasma profile
    :type plasma: :py:class:`sdp.plasma.profile.PlasmaProfile`
    :param coordinates: coordinates specifying the R-Z mesh, if not given, the
                        mesh in plasma will be used.
    :type coordinates: list of 1d arrays, [Z1D, R1D]
    :param string filename: file name for the output netcdf file
    
    CDF file format
    ***************
    
    Dimensions:
        r_dim: int, number of grid points in R direction.
        z_dim: int, number of grid points in Z direction
    
    Variables:
        rr: 1D array, coordinates in R direction, in Meter
        zz: 1D array, coordinates in Z direction, in Meter
        bb: 2D array, total magnetic field on grids, in Tesla, shape in 
            (z_dim,r_dim)
        ne: 2D array, total electron density on grids, in m^-3
        ti: 2D array, total ion temperature, in keV
        te: 2D array, total electron temperature, in keV
    """
    # check plasma dimension
    assert plasma.grid.dimension == 2
    # check coordinates
    if coordinates is None:
        coordinates = [plasma.grid.Z1D, plasma.grid.R1D]
    assert len(coordinates) == 2
    
    # get the geometry parameter and mesh grids
    Z1D = np.array(coordinates[0])
    R1D = np.array(coordinates[1])
    assert R1D.ndim == 1
    assert Z1D.ndim == 1
    NR = len(R1D)
    NZ = len(Z1D)
    
    # generate 2D mesh to obtain the 2D plasma quantities
    Z2D, R2D = np.meshgrid(Z1D, R1D, indexing='ij')
    
    # obtain the plasma quantities
    ne_prof = plasma.get_ne([Z2D, R2D], eq_only=eq_only, time=time)
    Te_prof = plasma.get_Te([Z2D, R2D], eq_only=eq_only, time=time)/cgs['keV']
    B_prof = plasma.get_B([Z2D, R2D], eq_only=eq_only, time=time)
    
    try:
        Ti_prof = plasma.get_Ti([Z2D, R2D], eq_only=eq_only, time=time)\
                  /cgs['keV']
    except AttributeError as ae:
        if 'get_Ti' in str(ae):
            Ti_prof = np.zeros_like(Z2D)
        else:
            raise ae
    
    # create the cdf file
    f = netcdf_file(filename, 'w')
    f.createDimension('z_dim', NZ)
    f.createDimension('r_dim', NR)
    
    rr = f.createVariable('rr','d',('r_dim',))
    rr[:] = R1D[:]
    zz = f.createVariable('zz','d',('z_dim',))
    zz[:] = Z1D[:]
    rr.units = zz.units = 'Meter'

    bb = f.createVariable('bb','d',('z_dim','r_dim'))
    bb[:,:] = B_prof[:,:]
    bb.units = 'Tesla'
    
    ne = f.createVariable('ne','d',('z_dim','r_dim'))
    ne[:,:] = ne_prof[:,:]
    ne.units = 'per cubic meter'

    te = f.createVariable('te','d',('z_dim','r_dim'))
    te[:,:] = Te_prof[:,:]
    te.units = 'keV'
    
    ti = f.createVariable('ti','d',('z_dim','r_dim'))
    ti[:,:] = Ti_prof[:,:]
    ti.units = 'keV'

    f.close()
    
    

