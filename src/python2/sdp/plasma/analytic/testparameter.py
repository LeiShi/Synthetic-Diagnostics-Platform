
""" create analytical plasma profiles for test use


"""
from collections import OrderedDict

import numpy as np
from numpy.random import random

from ...settings.unitsystem import cgs
from ...settings.exception import FPSDPError
from ...geometry import grid
from ..profile import ECEI_Profile

# Parameter is a keyword dictionary contains all the pre-defined profile
# parameters.
# can be modified via method set_parameter(kwargs).
# Note that all the coordinates are in (Z,R) order.
Parameter2D = OrderedDict()
Parameter2D['R_0']=200
Parameter2D['a']=50
Parameter2D['DownLeft']=(-30,150)
Parameter2D['UpRight']=(30,300)
Parameter2D['NR']=601
Parameter2D['NZ']=720
Parameter2D['ne_0']=2e13
Parameter2D['Te_0']=1*cgs['keV']
Parameter2D['B_0']=20000
Parameter2D['ne_shape']='Hmode'
Parameter2D['Te_shape']='Hmode'
Parameter2D['ShapeTable']={'exp': {'NeDecayScale': 3, 'TeDecayScale':5} ,
              'Hmode':{'PedWidthT': 0.1, 'PedWidthN': 0.1, 'PedHightT': 0.33,
                       'PedHightN': 0.95, 'ne_out': 1e-10, 'Te_out': 1e-10},
              'uniform':{},
              'linear':{'ne_out': 1e-10, 'Te_out': 1e-10}}

# 2D fluctuations can be one of three types: 'sinx', 'siny', or 'random'
# 'sinx' 'params': level, k, omega, y0, dy, x0, phi0
# 'siny' : level, k, omega, x0, dx, y0, phi0
# 'random': level
Parameter2D['dne_ne']={'type':'siny',
                       'params':{'level':0.01, 'k': 2*np.pi/5, 'omega':6.28e5,
                                 'x0': 220, 'dx':5, 'y0':0, 'phi0':0}}
Parameter2D['dte_te']={'type':'siny',
                       'params':{'level':0.01, 'k': 2*np.pi/5, 'omega':6.28e5,
                                 'x0': 220, 'dx':5, 'y0':0, 'phi0':0}}
Parameter2D['dB_B']={'type':'siny',
                     'params':{'level':0.01, 'k': 2*np.pi/5, 'omega':6.28e5,
                               'x0': 220, 'dx':5, 'y0':0, 'phi0':0}}
Parameter2D['timesteps']=[0, 1, 2, 3]
Parameter2D['dt']=2.5e-6

Parameter1D = OrderedDict()
Parameter1D['R_0']=200
Parameter1D['a']=50
Parameter1D['Xmin']=150
Parameter1D['Xmax']=300
Parameter1D['NX']=601
Parameter1D['ne_0']=2e13
Parameter1D['Te_0']=1*cgs['keV']
Parameter1D['B_0']=20000
Parameter1D['ne_shape']='Hmode'
Parameter1D['Te_shape']='Hmode'
Parameter1D['ShapeTable']={'exp': {'NeDecayScale': 3, 'TeDecayScale':5} ,
              'Hmode':{'PedWidthT': 0.1, 'PedWidthN': 0.1, 'PedHightT': 0.33,
                       'PedHightN': 0.95, 'ne_out': 1e-10, 'Te_out': 1e-10},
              'uniform':{},
              'linear':{'ne_out': 1e-10, 'Te_out': 1e-10}}

# fluctuations are specified individually, with 'type' being either 'sin' or
# 'random'
Parameter1D['dne_ne']={'type':'sin', 'params':{'level':0.01, 'k':6.28,
                                               'omega':6.28e5, 'x0':220,
                                               'phi0':0}}
Parameter1D['dte_te']={'type':'sin', 'params':{'level':0.01, 'k':6.28,
                                               'omega':6.28e5, 'x0':220,
                                               'phi0':0}}
Parameter1D['dB_B']={'type':'sin', 'params':{'level':0.01, 'k':6.28,
                                               'omega':6.28e5, 'x0':220,
                                               'phi0':0}}
# timesteps determines the chosen time steps to be created
Parameter1D['timesteps']=[0, 1, 2, 3]
Parameter1D['dt']=2.5e-6

Parameter_DIIID = OrderedDict()
Parameter_DIIID['R_0']=177
Parameter_DIIID['a']=50
# parameter for 2D plasma
Parameter_DIIID['DownLeft']=(-30,127)
Parameter_DIIID['UpRight']=(30,250)
Parameter_DIIID['NR']=493
Parameter_DIIID['NZ']=241
# parameter for 1D plasma
Parameter_DIIID['Xmin']=127
Parameter_DIIID['Xmax']=250
Parameter_DIIID['NX']=493
Parameter_DIIID['ne_0']=3e13
Parameter_DIIID['Te_0']=3*cgs['keV']
Parameter_DIIID['B_0']=20000
Parameter_DIIID['ne_shape']='Hmode'
Parameter_DIIID['Te_shape']='Hmode'
Parameter_DIIID['ShapeTable']={'exp': {'NeDecayScale': 3, 'TeDecayScale':5} ,
              'Hmode':{'PedWidthT': 0.05,'PedWidthN': 0.05 ,'PedHightT': 0.4,
                       'PedHightN': 0.4, 'ne_out': 1e-10, 'Te_out': 1e-10},
              'uniform':{},
              'linear':{'ne_out': 1e-10, 'Te_out': 1e-10}}

Parameter_DIIID['dne_ne']={'type':'siny',
                       'params':{'level':0.01, 'k': 2*np.pi/50, 'omega':6.28e5,
                                 'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_DIIID['dte_te']={'type':'siny',
                       'params':{'level':0.01, 'k': 2*np.pi/50, 'omega':6.28e5,
                                 'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_DIIID['dB_B']={'type':'siny',
                     'params':{'level':0.00, 'k': 2*np.pi/50, 'omega':6.28e5,
                               'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_DIIID['timesteps']=[0, 1, 2, 3]
Parameter_DIIID['dt']=2.5e-6


# JET-like parameters
Parameter_JET = OrderedDict()
Parameter_JET['R_0']=300
Parameter_JET['a']=100
Parameter_JET['DownLeft']=(-30,200)
Parameter_JET['UpRight']=(30,410)
Parameter_JET['NR']=200*4
Parameter_JET['NZ']=60*4
Parameter_JET['Xmin']=200
Parameter_JET['Xmax']=410
Parameter_JET['NX']=800
Parameter_JET['ne_0']=2e13 # 2e13 for typical JET plasma
Parameter_JET['Te_0']=15*cgs['keV'] # 15keV for typical JET plasma
Parameter_JET['B_0']=3e4 # 3 Tesla on axis
Parameter_JET['ne_shape']='Hmode'
Parameter_JET['Te_shape']='Hmode'
Parameter_JET['ShapeTable']={'exp': {'NeDecayScale': 3, 'TeDecayScale':5} ,
              'Hmode':{'PedWidthT': 0.1, 'PedWidthN': 0.1, 'PedHightT': 0.33,
                       'PedHightN': 0.95, 'ne_out': 1e-10, 'Te_out': 1e-10},
              'uniform':{},
              'linear':{'ne_out': 1e-10, 'Te_out': 1e-10}}

# Fluctuations are now set arbitrarily. Based on what kind of phenomena is being studied.
Parameter_JET['dne_ne']={'type':'siny',
                       'params':{'level':0.01, 'k': 2*np.pi/50, 'omega':6.28e5,
                                 'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_JET['dte_te']={'type':'siny',
                       'params':{'level':0.01, 'k': 2*np.pi/50, 'omega':6.28e5,
                                 'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_JET['dB_B']={'type':'siny',
                     'params':{'level':0.00, 'k': 2*np.pi/50, 'omega':6.28e5,
                               'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_JET['timesteps']=[0, 1, 2, 3]
Parameter_JET['dt']=2.5e-6


# ITER-like parameters
Parameter_ITER = OrderedDict()
Parameter_ITER['R_0']=650
Parameter_ITER['a']=170
Parameter_ITER['DownLeft']=(-30,500)
Parameter_ITER['UpRight']=(30,830)
Parameter_ITER['NR']=330*4
Parameter_ITER['NZ']=60*4
Parameter_ITER['Xmin']=500
Parameter_ITER['Xmax']=830
Parameter_ITER['NX']=330*4
Parameter_ITER['ne_0']=6e13 # 6~10 for typical ITER plasma
Parameter_ITER['Te_0']=27*cgs['keV'] # 20~30 for typical ITER plasma
Parameter_ITER['B_0']=5.0e4 # 5 Tesla on axis
Parameter_ITER['ne_shape']='Hmode'
Parameter_ITER['Te_shape']='Hmode'
Parameter_ITER['ShapeTable']={'exp': {'NeDecayScale': 3, 'TeDecayScale':5} ,
              'Hmode':{'PedWidthT': 0.1, 'PedWidthN': 0.1, 'PedHightT': 0.33,
                       'PedHightN': 0.95, 'ne_out': 1e-10, 'Te_out': 1e-10},
              'uniform':{},
              'linear':{'ne_out': 1e-10, 'Te_out': 1e-10}}
# Fluctuations are now set arbitrarily. Based on what kind of phenomena is being studied.
Parameter_ITER['dne_ne']={'type':'siny',
                       'params':{'level':0.01, 'k': 2*np.pi/50, 'omega':6.28e5,
                                 'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_ITER['dte_te']={'type':'siny',
                       'params':{'level':0.01, 'k': 2*np.pi/50, 'omega':6.28e5,
                                 'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_ITER['dB_B']={'type':'siny',
                     'params':{'level':0.00, 'k': 2*np.pi/50, 'omega':6.28e5,
                               'x0': 222, 'dx':2, 'y0':0, 'phi0':0}}
Parameter_ITER['timesteps']=[0, 1, 2, 3]
Parameter_ITER['dt']=2.5e-6


# Loading parameters for XGC_loader on an NSTX case
xgc_test2D ={'DownLeft':(-0.5,0.9),'UpRight':(0.5,1.6),'NR':101,'NZ':101}
xgc_test3D = {'Xmin':0.9,'Xmax':1.6,'Ymin':-0.5, 'Ymax':0.5, 'Zmin':-0.1,
              'Zmax':0.1, 'NX':32,'NY':32,'NZ':16}


# TODO finish 3D parameter dictionary.
# Parameter3D = {'Xmin':1.85,'Xmax':2.0,'Ymin':-0.15,'Ymax':0.15}

# shape table is a dictionary contains the shape parameters
# Do not suggest to change it by outside programs
# DecayScale means within a minor radius, it will decay to exponential of which
# power.

# FlucType provides a list of supported fluctuation types
FlucType2D = ('random', 'siny', 'sinx')
FlucType1D = ('random', 'sinx')

def show_parameter2D():
    """Print out the parameters at the moment
    """
    for key,value in Parameter2D.items():
        print '{0} : {1}'.format(key, value)
        if 'shape' in key:
            print '    {0} Params: {1}'.format(value,
                                              Parameter2D['ShapeTable'][value])

def show_parameter1D():
    """Print out the parameters at the moment
    """
    for key,value in Parameter1D.items():
        print '{} : {}'.format(key, value)
        if 'shape' in key:
            print '    {0} Params: {1}'.format(value,
                                              Parameter1D['ShapeTable'][value])

def set_parameter2D(DownLeft=Parameter2D['DownLeft'],
                    UpRight=Parameter2D['UpRight'],
                    NR=Parameter2D['NR'],
                    NZ=Parameter2D['NZ'],
                    ne_0=Parameter2D['ne_0'],
                    Te_0=Parameter2D['Te_0'],
                    B_0=Parameter2D['B_0'],
                    ne_shape=Parameter2D['ne_shape'],
                    Te_shape=Parameter2D['Te_shape'],
                    a=Parameter2D['a'],
                    R_0=Parameter2D['R_0'],
                    dne_ne=Parameter2D['dne_ne'],
                    dte_te=Parameter2D['dte_te'],
                    dB_B=Parameter2D['dB_B'],
                    timesteps=Parameter2D['timesteps'],
                    dt=Parameter2D['dt'],
                    ShapeTable=Parameter2D['ShapeTable'],
                    **Params):
    """A explicit method to change the parameters.

    Although it's possible to directly assign new values to parameter keywords,
    it may be more comfortable for C programmers to do it this way.
    """
    Parameter2D['ne_0'] = ne_0
    Parameter2D['Te_0'] = Te_0
    Parameter2D['B_0'] = B_0
    Parameter2D['a'] = a
    Parameter2D['R_0'] = R_0
    Parameter2D['ne_shape'] = ne_shape
    Parameter2D['Te_shape'] = Te_shape
    Parameter2D['DownLeft'] = DownLeft
    Parameter2D['UpRight'] = UpRight
    Parameter2D['NR'] = NR
    Parameter2D['NZ'] = NZ
    Parameter2D['dne_ne'] = dne_ne
    Parameter2D['dte_te'] = dte_te
    Parameter2D['dB_B'] = dB_B
    Parameter2D['timesteps'] = timesteps
    Parameter2D['dt'] = dt
    Parameter2D['ShapeTable'] = ShapeTable
    print 'Not used params: {0}'.format(Params)

def set_shape2D(shape, param, value):
    """quick change of one parameter in current shape table.
    :param string shape: shape code, 'Hmode', 'exp', 'uniform', or 'linear'.
    :param string param: parameter name, depends on which shape code, different
                         parameter names are available.
    :param value: value to be set to parameter. normally a float.
    """
    if shape not in Parameter2D['ShapeTable'].keys():
        print '{0} is not a valid shape. Shapes are {1}.'.format(shape,
                                              Parameter2D['ShapeTable'].keys())
        return False
    elif param not in Parameter2D['ShapeTable'][shape].keys():
        print '{0} is not a valid parameter in {1}. Parameters are {2}.'.\
              format(param, shape, Parameter2D['ShapeTable'][shape].keys())
        return False
    else:
        Parameter2D['ShapeTable'][shape][param] = value
        return True

def set_parameter1D(Xmin=Parameter1D['Xmin'],
                    Xmax=Parameter1D['Xmax'],
                    NX=Parameter1D['NX'],
                    ne_0=Parameter1D['ne_0'],
                    Te_0=Parameter1D['Te_0'],
                    B_0=Parameter1D['B_0'],
                    ne_shape=Parameter1D['ne_shape'],
                    Te_shape=Parameter1D['Te_shape'],
                    a=Parameter1D['a'],
                    R_0=Parameter1D['R_0'],
                    dne_ne=Parameter1D['dne_ne'],
                    dte_te=Parameter1D['dte_te'],
                    dB_B=Parameter1D['dB_B'],
                    timesteps=Parameter1D['timesteps'],
                    dt=Parameter1D['dt'],
                    ShapeTable=Parameter1D['ShapeTable'],
                    **Params):
    """A explicit method to change the parameters.

    Although it's possible to directly assign new values to parameter keywords,
    it may be more comfortable for C programmers to do it this way.
    """
    Parameter1D['ne_0'] = ne_0
    Parameter1D['Te_0'] = Te_0
    Parameter1D['B_0'] = B_0
    Parameter1D['a'] = a
    Parameter1D['R_0'] = R_0
    Parameter1D['ne_shape'] = ne_shape
    Parameter1D['Te_shape'] = Te_shape
    Parameter1D['Xmin'] = Xmin
    Parameter1D['Xmax'] = Xmax
    Parameter1D['NX'] = NX
    Parameter1D['dne_ne'] = dne_ne
    Parameter1D['dte_te'] = dte_te
    Parameter1D['dB_B'] = dB_B
    Parameter1D['timesteps'] = timesteps
    Parameter1D['dt']=dt
    Parameter1D['ShapeTable']=ShapeTable
    print 'Not used params: {0}'.format(Params)

def create_profile1D(fluctuation=False):
    """Create the profiles and return it in a dictionary structure

    ne, Te, B values on RZ mesh are returned
    """

    # the return value will be generated by this dictionary
    profile = {}

    # extract the plasma shape information
    a= Parameter1D['a']
    R_0= Parameter1D['R_0']
    XGrid = grid.Cartesian1D(Parameter1D['Xmin'], Parameter1D['Xmax'],
                             Parameter1D['NX'])

    profile['Grid'] = XGrid

    # extract the density information
    nshp = Parameter1D['ne_shape']
    ne_0 = Parameter1D['ne_0']


    # evaluate the density values on each grid point for the given shape
    if (nshp == 'exp') :
        # exponential decay shape
        DecScale= Parameter1D['ShapeTable']['exp']['NeDecayScale']
        DecLength= a/DecScale
        ne_array= ne_0 * np.exp(-np.abs(XGrid.X1D-R_0)/DecLength )
    elif (nshp == 'Hmode') :
        # linear H mode profile
        nped = ne_0  *Parameter1D['ShapeTable']['Hmode']['PedHightN']
        nout = ne_0 * Parameter1D['ShapeTable']['Hmode']['ne_out']
        a_core = a * (1-Parameter1D['ShapeTable']['Hmode']['PedWidthN'])
        a_array = np.abs(XGrid.X1D - R_0)
        # linear function connecting axis, top of pedestal and the vacuum
        ne_array = np.select([a_array <= a_core, a_array >= a,
                              a_array > a_core ],
                             [a_array* (nped - ne_0)/a_core + ne_0 , nout,
                              (a_array - a) * (nped-nout)/(a_core - a)+ nout])
    elif (nshp == 'uniform'):
        # uniform density profile
        ne_array = np.ones_like(XGrid.X1D) * ne_0
    elif (nshp == 'linear'):
        # linear decrease with minor radius
        nout = ne_0 * Parameter1D['ShapeTable']['linear']['ne_out']
        a_array = np.abs(XGrid.X1D - R_0)
        # linear function connecting axis, vacuum
        ne_array = np.select([a_array <= a, a_array > a],
                             [a_array*(nout-ne_0)/a + ne_0, nout])
    else:
        raise KeyError('"{}" is not a valid shape code. Available shapes \
are {}'.format(nshp, Parameter1D['ShapeTable'].keys()))
    profile['ne']= ne_array

    # Te profile
    tshp = Parameter1D['Te_shape']
    Te_0 = Parameter1D['Te_0']

    if ( tshp == 'exp'):
        DecLength= a/Parameter1D['ShapeTable']['exp']['TeDecayScale']
        Te_array= Te_0 * np.exp(-np.abs(XGrid.X1D-R_0)/DecLength )
    elif (tshp == 'Hmode') :
        # linear H mode profile
        tped = Te_0 * Parameter1D['ShapeTable']['Hmode']['PedHightT']
        tout = Te_0 * Parameter1D['ShapeTable']['Hmode']['Te_out']
        a_core = a * (1-Parameter1D['ShapeTable']['Hmode']['PedWidthT'])
        a_array = np.abs(XGrid.X1D - R_0)
        # linear function connecting axis, top of pedestal and the vacuum
        Te_array = np.select([a_array<=a_core, a_array>=a, a_array>a_core],
                             [a_array*(tped-Te_0)/a_core + Te_0, tout,
                              (a_array-a)*(tped-tout)/(a_core-a)+ tout])
    elif (tshp == 'uniform'):
        # uniform temperature profile
        Te_array = np.ones_like(XGrid.X1D) * Te_0
    elif (tshp == 'linear'):
        # linear decrease with minor radius
        tout = Te_0 * Parameter1D['ShapeTable']['linear']['Te_out']
        a_array = np.abs(XGrid.X1D - R_0)
        # linear function connecting axis, vacuum
        Te_array = np.select([a_array <= a, a_array > a],
                             [a_array*(tout-Te_0)/a + Te_0, tout])
    else:
        raise KeyError('"{}" is not a valid shape code. Available shapes \
are {}'.format(tshp, Parameter1D['ShapeTable'].keys()))
    profile['Te']= Te_array

    # B profile
    B_0 = Parameter1D['B_0']
    B_array= B_0 * R_0/XGrid.X1D
    profile['B']= B_array


    if fluctuation:
        dne_type = Parameter1D['dne_ne']['type']
        dte_type = Parameter1D['dte_te']['type']
        dB_type = Parameter1D['dB_B']['type']
        timesteps = np.asarray(Parameter1D['timesteps'])
        time = timesteps*Parameter1D['dt']
        d_shape = [len(time), XGrid.shape[0]]
        if dne_type=='random':
            dne = 2*Parameter1D['dne_ne']['params']['level']*profile['ne']*\
                  (random(d_shape)-0.5)
        elif dne_type=='sin':
            nfluc_level = Parameter1D['dne_ne']['params']['level']
            x0 = Parameter1D['dne_ne']['params']['x0']
            k = Parameter1D['dne_ne']['params']['k']
            omega = Parameter1D['dne_ne']['params']['omega']
            phi0 = Parameter1D['dne_ne']['params']['phi0']
            fluc_pattern = np.sin(omega*time[:, np.newaxis] - k*(XGrid.X1D-x0)\
                                  + phi0)
            dne = nfluc_level*profile['ne']*fluc_pattern
        else:
            raise NotImplementedError('Invalid fluctuation type:  {0}. \
Supported types are {1}'.format(fluctuation, FlucType1D))

        if dte_type=='random':
            dTe_para = 2*Parameter1D['dte_te']['params']['level']*profile['Te']*\
                           (random(d_shape)-0.5)
            dTe_perp = 2*Parameter1D['dte_te']['params']['level']*profile['Te']*\
                           (random(d_shape)-0.5)
        elif dte_type=='sin':
            tfluc_level = Parameter1D['dte_te']['params']['level']
            x0 = Parameter1D['dte_te']['params']['x0']
            k = Parameter1D['dte_te']['params']['k']
            omega = Parameter1D['dte_te']['params']['omega']
            phi0 = Parameter1D['dte_te']['params']['phi0']
            fluc_pattern = np.sin(omega*time[:, np.newaxis] - k*(XGrid.X1D-x0)\
                                  + phi0)
            dTe_para = tfluc_level*profile['Te']*fluc_pattern
            dTe_perp = dTe_para
        else:
            raise NotImplementedError('Invalid fluctuation type:  {0}. \
Supported types are {1}'.format(fluctuation, FlucType1D))

        if dB_type=='random':
            dB = 2*Parameter1D['dB_B']['params']['level']*profile['B']*\
                 (random(d_shape)-0.5)
        elif dB_type=='sin':
            Bfluc_level = Parameter1D['dB_B']['params']['level']
            x0 = Parameter1D['dB_B']['params']['x0']
            k = Parameter1D['dB_B']['params']['k']
            omega = Parameter1D['dB_B']['params']['omega']
            phi0 = Parameter1D['dB_B']['params']['phi0']
            fluc_pattern = np.sin(omega*time[:, np.newaxis] - k*(XGrid.X1D-x0)\
                                  + phi0)
            dB = Bfluc_level*profile['B']*fluc_pattern
        else:
            raise NotImplementedError('Invalid fluctuation type:  {0}. \
Supported types are {1}'.format(fluctuation, FlucType1D))

        return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                            profile['B'], time, dne, dTe_para, dTe_perp, dB)
    else:
        return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                        profile['B'])

def create_profile2D(fluctuation=None, fluc_level=None):
    """Create the profiles and return it in a dictionary structure

    ne, Te, B values on RZ mesh are returned
    """

    # the return value will be generated by this dictionary
    profile = {}

    # extract the plasma shape information
    a= Parameter2D['a']
    R_0= Parameter2D['R_0']
    RZGrid = grid.Cartesian2D(**Parameter2D)

    profile['Grid'] = RZGrid

    # extract the density information
    nshp = Parameter2D['ne_shape']
    ne_0 = Parameter2D['ne_0']


    # evaluate the density values on each grid point for the given shape
    if (nshp == 'exp') :
        # exponential decay shape
        DecScale= Parameter2D['ShapeTable']['exp']['NeDecayScale']
        DecLength= a/DecScale
        ne_array= ne_0 * np.exp(-np.sqrt(((RZGrid.R2D-R_0)**2 +RZGrid.Z2D**2))\
                  /DecLength )
    elif (nshp == 'Hmode') :
        # linear H mode profile
        nped = ne_0 * Parameter2D['ShapeTable']['Hmode']['PedHightN']
        nout = ne_0 * Parameter2D['ShapeTable']['Hmode']['ne_out']
        a_core = a * (1-Parameter2D['ShapeTable']['Hmode']['PedWidthN'])
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, top of pedestal and the vacuum
        ne_array = np.select([a_array <= a_core, a_array >= a,
                              a_array > a_core ],
                             [a_array* (nped - ne_0)/a_core + ne_0 , nout,
                              (a_array - a) * (nped-nout)/(a_core - a)+ nout])
    elif (nshp == 'uniform'):
        ne_array = np.zeros_like(RZGrid.R2D) + ne_0
    elif (nshp == 'linear'):
        # linear decrease with minor radius
        nout = ne_0 * Parameter2D['ShapeTable']['linear']['ne_out']
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, vacuum
        ne_array = np.select([a_array <= a, a_array > a],
                             [a_array*(nout-ne_0)/a + ne_0, nout])
    else:
        raise KeyError('"{}" is not a valid shape code. Available shapes \
are {}'.format(nshp, Parameter2D['ShapeTable'].keys()))
    profile['ne']= ne_array

    # Te profile
    tshp = Parameter2D['Te_shape']
    Te_0 = Parameter2D['Te_0']

    if ( tshp == 'exp'):
        DecLength= a/Parameter2D['ShapeTable']['exp']['TeDecayScale']
        Te_array= Te_0 * np.exp(-np.sqrt(((RZGrid.R2D-R_0)**2 +RZGrid.Z2D**2))\
                  /DecLength )
    elif (tshp == 'Hmode') :
        # linear H mode profile
        tped = Te_0 * Parameter2D['ShapeTable']['Hmode']['PedHightT']
        tout = Te_0 * Parameter2D['ShapeTable']['Hmode']['Te_out']
        a_core = a * (1-Parameter2D['ShapeTable']['Hmode']['PedWidthT'])
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, top of pedestal and the vacuum
        Te_array = np.select([a_array<=a_core, a_array>=a, a_array>a_core],
                             [a_array*(tped-Te_0)/a_core + Te_0, tout,
                              (a_array-a)*(tped-tout)/(a_core-a)+ tout])
    elif (tshp == 'uniform'):
        Te_array = np.zeros_like(RZGrid.R2D) + Te_0
    elif (tshp == 'linear'):
        # linear decrease with minor radius
        tout = Te_0 * Parameter2D['ShapeTable']['linear']['Te_out']
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, vacuum
        Te_array = np.select([a_array <= a, a_array > a],
                             [a_array*(tout-Te_0)/a + Te_0, tout])
    else:
        raise KeyError('"{}" is not a valid shape code. Available shapes \
are {}'.format(tshp, Parameter2D['ShapeTable'].keys()))
    profile['Te']= Te_array

    # B profile
    B_0 = Parameter2D['B_0']
    B_array= B_0 * R_0/RZGrid.R2D
    profile['B']= B_array


    if fluctuation:
        dne_type = Parameter2D['dne_ne']['type']
        dte_type = Parameter2D['dte_te']['type']
        dB_type = Parameter2D['dB_B']['type']

        timesteps = np.asarray(Parameter2D['timesteps'])
        time = timesteps*Parameter2D['dt']
        d_shape = [len(time)]
        d_shape.extend([i for i in RZGrid.R2D.shape])

        if dne_type == 'random':
            dne = 2*Parameter2D['dne_ne']['params']['level']*profile['ne']*\
                  (random(d_shape)-0.5)
        elif dne_type == 'sinx':
            level = Parameter2D['dne_ne']['params']['level']
            x0 = Parameter2D['dne_ne']['params']['x0']
            k = Parameter2D['dne_ne']['params']['k']
            y0 = Parameter2D['dne_ne']['params']['y0']
            dy = Parameter2D['dne_ne']['params']['dy']
            phi0 = Parameter2D['dne_ne']['params']['phi0']
            omega = Parameter2D['dne_ne']['params']['omega']

            fluc_pattern = np.sin(omega*time[:, np.newaxis, np.newaxis] - \
                                  k*(RZGrid.R2D-x0) + phi0) * \
                                  np.exp(-(RZGrid.Z2D-y0)**2/dy**2)
            dne = level*profile['ne']*fluc_pattern
        elif dne_type == 'siny':
            level = Parameter2D['dne_ne']['params']['level']
            y0 = Parameter2D['dne_ne']['params']['y0']
            k = Parameter2D['dne_ne']['params']['k']
            x0 = Parameter2D['dne_ne']['params']['x0']
            dx = Parameter2D['dne_ne']['params']['dx']
            phi0 = Parameter2D['dne_ne']['params']['phi0']
            omega = Parameter2D['dne_ne']['params']['omega']

            fluc_pattern = np.sin(omega*time[:, np.newaxis, np.newaxis] - \
                                  k*(RZGrid.Z2D-y0) + phi0) * \
                                  np.exp(-(RZGrid.R2D-x0)**2/dx**2)
            dne = level*profile['ne']*fluc_pattern
        else:
            raise NotImplementedError('Invalid fluctuation type:  {0}. \
Supported types are {1}'.format(dne_type, FlucType2D))

        if dte_type == 'random':
            dTe_para = 2*Parameter2D['dte_te']*profile['Te']*\
                           (random(d_shape)-0.5)
            dTe_perp = dTe_para

        elif dte_type == 'sinx':
            level = Parameter2D['dte_te']['params']['level']
            x0 = Parameter2D['dte_te']['params']['x0']
            k = Parameter2D['dte_te']['params']['k']
            y0 = Parameter2D['dte_te']['params']['y0']
            dy = Parameter2D['dte_te']['params']['dy']
            phi0 = Parameter2D['dte_te']['params']['phi0']
            omega = Parameter2D['dte_te']['params']['omega']

            fluc_pattern = np.sin(omega*time[:, np.newaxis, np.newaxis] - \
                                  k*(RZGrid.R2D-x0) + phi0) * \
                                  np.exp(-(RZGrid.Z2D-y0)**2/dy**2)
            dTe_para = level*profile['Te']*fluc_pattern
            dTe_perp = dTe_para
        elif dte_type == 'siny':
            level = Parameter2D['dte_te']['params']['level']
            y0 = Parameter2D['dte_te']['params']['y0']
            k = Parameter2D['dte_te']['params']['k']
            x0 = Parameter2D['dte_te']['params']['x0']
            dx = Parameter2D['dte_te']['params']['dx']
            phi0 = Parameter2D['dte_te']['params']['phi0']
            omega = Parameter2D['dte_te']['params']['omega']

            fluc_pattern = np.sin(omega*time[:, np.newaxis, np.newaxis] - \
                                  k*(RZGrid.Z2D-y0) + phi0) * \
                                  np.exp(-(RZGrid.R2D-x0)**2/dx**2)
            dTe_para = level*profile['Te']*fluc_pattern
            dTe_perp = dTe_para
        else:
            raise NotImplementedError('Invalid fluctuation type:  {0}. \
Supported types are {1}'.format(dte_type, FlucType2D))

        if dB_type == 'random':
            dB = 2*Parameter2D['dB_B']*profile['B']*(random(d_shape)-0.5)
        elif dB_type == 'sinx':
            level = Parameter2D['dB_B']['params']['level']
            x0 = Parameter2D['dB_B']['params']['x0']
            k = Parameter2D['dB_B']['params']['k']
            y0 = Parameter2D['dB_B']['params']['y0']
            dy = Parameter2D['dB_B']['params']['dy']
            phi0 = Parameter2D['dB_B']['params']['phi0']
            omega = Parameter2D['dB_B']['params']['omega']

            fluc_pattern = np.sin(omega*time[:, np.newaxis, np.newaxis] - \
                                  k*(RZGrid.R2D-x0) + phi0) * \
                                  np.exp(-(RZGrid.Z2D-y0)**2/dy**2)
            dB = level*profile['B']*fluc_pattern
        elif dB_type == 'siny':
            level = Parameter2D['dB_B']['params']['level']
            y0 = Parameter2D['dB_B']['params']['y0']
            k = Parameter2D['dB_B']['params']['k']
            x0 = Parameter2D['dB_B']['params']['x0']
            dx = Parameter2D['dB_B']['params']['dx']
            phi0 = Parameter2D['dB_B']['params']['phi0']
            omega = Parameter2D['dB_B']['params']['omega']

            fluc_pattern = np.sin(omega*time[:, np.newaxis, np.newaxis] - \
                                  k*(RZGrid.Z2D-y0) + phi0) * \
                                  np.exp(-(RZGrid.R2D-x0)**2/dx**2)
            dB = level*profile['B']*fluc_pattern
        else:
            raise NotImplementedError('Invalid fluctuation type:  {0}. \
Supported types are {1}'.format(dB_type, FlucType2D))


        return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                            profile['B'], time, dne, dTe_para, dTe_perp, dB)

    else:
        return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                        profile['B'])


def simulate_1D(p1d, grid2D):
    """Generate a 2D plasma profile on given grid. The plasma is based on given
    1D profile, it is simply uniform along Y direction.
    """
    X_new = grid2D.R1D
    ne01d = p1d.get_ne0([X_new])
    B01d = p1d.get_B0([X_new])
    Te01d = p1d.get_Te0([X_new])

    ne02d = np.zeros_like(grid2D.Z2D) + ne01d
    B02d = np.zeros_like(grid2D.Z2D) + B01d
    Te02d = np.zeros_like(grid2D.Z2D) + Te01d

    if (p1d.has_dne):
        dne1d = p1d.get_dne([X_new])
        dB1d = p1d.get_dB([X_new])
        dTe_para1d = p1d.get_dTe_para([X_new])
        dTe_perp1d = p1d.get_dTe_perp([X_new])

        dne_shape = (dne1d.shape[0], len(grid2D.Z1D), len(grid2D.R1D))
        dne2d = np.zeros(dne_shape) + dne1d[:, np.newaxis, :]
        dB2d = np.zeros(dne_shape) + dB1d[:, np.newaxis, :]
        dTe_para2d = np.zeros(dne_shape) + dTe_para1d[:, np.newaxis, :]
        dTe_perp2d = np.zeros(dne_shape) + dTe_perp1d[:, np.newaxis, :]
        return ECEI_Profile(grid2D, ne02d, Te02d, B02d, p1d.time, dne2d,
                            dTe_para2d, dTe_perp2d, dB2d)

    else:
        return ECEI_Profile(grid2D, ne02d, Te02d, B02d)


#TODO Build the new test plamsa model center.
# The rest is in development.
class PlasmaModelError(FPSDPError):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Model:
    """base class for all plasma models

    Attributes:
        _name:The name of the model
        _type:The type of the model, 'geometry','Equilibrium', or 'Fluctuation'
        _description: A short description explains the basic idea of the model

    Method:
        info(): prints info about the model

    """
    def __init__(self):

        #Attributes:
        self._name = 'Model_Base'
        self._type = 'Base'
        self._description = 'This is a base model object.'

    def info(self):
        print('name:{0}\ntype:{1}\n{2}').format(self._name,self._type,
                                                self._description)







class TestPlasmaCreator:
    """ A Class that contains all the plasma models, and creates plasma profile
    objects.

    Existing Models
    ===============

    Shape of Magnetic geometry:
        Concentric : all flux surfaces are concentric circles
        Shafranov_Shifted: flux surfaces are circles that have been shifted
                           outwards according to Shafranov shift
    Equilibrium:
        Exponential_Decay: profiles are exponentially decaying radially
        Hmode_linear: profiles are slowly decreasing in the core region, and
                      rapidly decreasing in the pedestal region. Both
                      are linear in radial flux coordinate.
        Hmode_tanh: profiles are in a shape of tanh function, where the half
                    height location corresponds the separatrix.

    Fluctuations:
        Single_Mode: add a single mode to a certain region. specify the mode
                     frequency, amplitude, and mode numbers
        Turbulence: add turbulent structure to a certain region. specify
                    auto-correlation lengths and time, and fluctuation level.

    """






