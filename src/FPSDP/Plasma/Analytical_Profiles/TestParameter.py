
""" create analytical plasma profiles for test use


"""
from collections import OrderedDict

import numpy as np
from numpy.random import random

from ...GeneralSettings.UnitSystem import cgs
from ...Geometry import Grid 
from ..PlasmaProfile import ECEI_Profile

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
Parameter2D['dne_ne']=0.01
Parameter2D['dte_te']=0.01
Parameter2D['dB_B']=0
Parameter2D['siny']={'k': 21, 'omega':6.28e5, 'x0': 220, 'dx':5, 'y0':0}
Parameter2D['sinx']={'k':6.28, 'omega':6.28e5, 'y0': 0, 'dy':20, 'x0':220}
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
Parameter1D['dne_ne']=0.01
Parameter1D['dte_te']=0.01
Parameter1D['dB_B']=0
Parameter1D['sinx']={'k':6.28, 'omega':6.28e5, 'x0':220}
Parameter1D['timesteps']=[0, 1, 2, 3]
Parameter1D['dt']=2.5e-6

Parameter_DIIID = OrderedDict()
Parameter_DIIID['R_0']=177
Parameter_DIIID['a']=50
Parameter_DIIID['DownLeft']=(-30,127)
Parameter_DIIID['UpRight']=(30,250)
Parameter_DIIID['NR']=493
Parameter_DIIID['NZ']=241
Parameter_DIIID['ne_0']=3e13
Parameter_DIIID['Te_0']=3*cgs['keV']
Parameter_DIIID['B_0']=20000
Parameter_DIIID['ne_shape']='Hmode'
Parameter_DIIID['Te_shape']='Hmode'
Parameter_DIIID['dne_ne']=0.01
Parameter_DIIID['dte_te']=0.01
Parameter_DIIID['dB_B']=0
Parameter_DIIID['siny']={'k': 6.28/50, 'omega':6.28e5, 'x0': 222, 'dx':2, 'y0':0}
Parameter_DIIID['sinx']={'k':6.28, 'omega':6.28e5, 'y0': 0, 'dy':20, 'x0':220}
Parameter_DIIID['timesteps']=[0, 1, 2, 3]
Parameter_DIIID['dt']=2.5e-6
               


xgc_test2D ={'DownLeft':(-0.5,0.9),'UpRight':(0.5,1.6),'NR':101,'NZ':101}
xgc_test3D = {'Xmin':0.9,'Xmax':1.6,'Ymin':-0.5, 'Ymax':0.5, 'Zmin':-0.1, 
              'Zmax':0.1, 'NX':32,'NY':32,'NZ':16}


# TODO finish 3D parameter dictionary.
# Parameter3D = {'Xmin':1.85,'Xmax':2.0,'Ymin':-0.15,'Ymax':0.15}

# shape table is a dictionary contains the shape parameters
# Do not suggest to change it by outside programs
# DecayScale means within a minor radius, it will decay to exponential of which
# power.
ShapeTable = {'exp': {'NeDecayScale': 3, 'TeDecayScale':5} , 
              'Hmode':{'PedWidthT': 0.05,'PedWidthN': 0.05 ,'PedHightT': 0.4, 
                       'PedHightN': 0.4, 'ne_out': 1e-10, 'Te_out': 1e-10}, 
              'uniform':None,
              'linear':{'ne_out': 1e-10, 'Te_out': 1e-10}}

# FlucType provides a list of supported fluctuation types
FlucType2D = ('random', 'siny', 'sinx')
FlucType1D = ('random', 'sinx')

def show_parameter2D():
    """Print out the parameters at the moment
    """
    for key,value in Parameter2D.items():
        print '{0} : {1}'.format(key, value)
        if 'shape' in key:
            print '    {0} Params: {1}'.format(value, ShapeTable[value])
    
def show_parameter1D():
    """Print out the parameters at the moment
    """
    for key,value in Parameter1D.items():
        print '{} : {}'.format(key, value) 

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
                    sinx=Parameter2D['sinx'],
                    siny=Parameter2D['siny'],
                    timesteps=Parameter2D['timesteps'],
                    dt=Parameter2D['dt']):
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
    Parameter2D['sinx'] = sinx
    Parameter2D['siny'] = siny
    Parameter2D['timesteps'] = timesteps
    Parameter2D['dt'] = dt
    
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
                    sinx=Parameter1D['sinx'],
                    timesteps=Parameter1D['timesteps'],
                    dt=Parameter1D['dt']):
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
    Parameter1D['sinx'] = sinx
    Parameter1D['timesteps'] = timesteps
    Parameter1D['dt']=dt

def create_profile1D(fluctuation=None, fluc_level=None):
    """Create the profiles and return it in a dictionary structure

    ne, Te, B values on RZ mesh are returned
    """

    # the return value will be generated by this dictionary
    profile = {}

    # extract the plasma shape information
    a= Parameter1D['a']
    R_0= Parameter1D['R_0']
    XGrid = Grid.Cartesian1D(Parameter1D['Xmin'], Parameter1D['Xmax'], 
                             Parameter1D['NX'])

    profile['Grid'] = XGrid

    # extract the density information
    nshp = Parameter1D['ne_shape']
    ne_0 = Parameter1D['ne_0']


    # evaluate the density values on each grid point for the given shape
    if (nshp == 'exp') :
        # exponential decay shape
        DecScale= ShapeTable['exp']['NeDecayScale']
        DecLength= a/DecScale
        ne_array= ne_0 * np.exp(-np.abs(XGrid.X1D-R_0)/DecLength )
    elif (nshp == 'Hmode') :
        # linear H mode profile
        nped = ne_0 * ShapeTable['Hmode']['PedHightN']
        nout = ne_0 * ShapeTable['Hmode']['ne_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthN'])
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
        nout = ne_0 * ShapeTable['linear']['ne_out']
        a_array = np.abs(XGrid.X1D - R_0)
        # linear function connecting axis, vacuum
        ne_array = np.select([a_array <= a, a_array > a], 
                             [a_array*(nout-ne_0)/a + ne_0, nout])
    else:
        raise KeyError('"{}" is not a valid shape code. Available shapes \
are {}'.format(nshp, ShapeTable.keys()))
    profile['ne']= ne_array

    # Te profile
    tshp = Parameter1D['Te_shape']
    Te_0 = Parameter1D['Te_0']

    if ( tshp == 'exp'):
        DecLength= a/ShapeTable['exp']['TeDecayScale']
        Te_array= Te_0 * np.exp(-np.abs(XGrid.X1D-R_0)/DecLength )
    elif (tshp == 'Hmode') :
        # linear H mode profile
        tped = Te_0 * ShapeTable['Hmode']['PedHightT']
        tout = Te_0 * ShapeTable['Hmode']['Te_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthT'])
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
        tout = Te_0 * ShapeTable['linear']['Te_out']
        a_array = np.abs(XGrid.X1D - R_0)
        # linear function connecting axis, vacuum
        Te_array = np.select([a_array <= a, a_array > a], 
                             [a_array*(tout-Te_0)/a + Te_0, tout])
    else:
        raise KeyError('"{}" is not a valid shape code. Available shapes \
are {}'.format(tshp, ShapeTable.keys()))
    profile['Te']= Te_array

    # B profile
    B_0 = Parameter1D['B_0']
    B_array= B_0 * R_0/XGrid.X1D
    profile['B']= B_array
    
    
    if fluctuation is not None:
        if fluctuation=='random':
            time = Parameter1D['timesteps']
            d_shape = [len(time), XGrid.shape[0]]
            if fluc_level is None:
                dne = 2*Parameter1D['dne_ne']*profile['ne']*\
                      (random(d_shape)-0.5)
                dTe_para = 2*Parameter1D['dte_te']*profile['Te']*\
                           (random(d_shape)-0.5)
                dTe_perp = 2*Parameter1D['dte_te']*profile['Te']*\
                           (random(d_shape)-0.5)
                dB = 2*Parameter1D['dB_B']*profile['B']*(random(d_shape)-0.5)
            else:
                dne = 2*fluc_level*profile['ne']*(random(d_shape)-0.5)
                dTe_para = 2*fluc_level*profile['Te']*(random(d_shape)-0.5)
                dTe_perp = 2*fluc_level*profile['Te']*(random(d_shape)-0.5)
                dB = 2*fluc_level*profile['B']*(random(d_shape)-0.5)
            return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                            profile['B'], time, dne, dTe_para, dTe_perp, dB)
        elif fluctuation=='sinx':
            timesteps = np.asarray(Parameter1D['timesteps'])
            time = timesteps*Parameter1D['dt']
            d_shape = [len(time), XGrid.shape[0]]
            if fluc_level is None:
                nfluc_level = Parameter1D['dne_ne']
                tfluc_level = Parameter1D['dte_te']
                Bfluc_level = Parameter1D['dB_B']
            else:
                nfluc_level = fluc_level
                tfluc_level = fluc_level
                Bfluc_level = fluc_level
            x0 = Parameter1D['sinx']['x0']
            k = Parameter1D['sinx']['k']
            omega = Parameter1D['sinx']['omega']
            fluc_pattern = np.sin(omega*time[:, np.newaxis] - k*(XGrid.X1D-x0))
            dne = nfluc_level*profile['ne']*fluc_pattern
            dTe = tfluc_level*profile['Te']*fluc_pattern
            dB = Bfluc_level*profile['B']*fluc_pattern
            return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                            profile['B'], time, dne, dTe, dTe, dB)
        else:
            raise NotImplementedError('Invalid fluctuation type:  {0}. \
Supported types are {1}'.format(fluctuation, FlucType1D))
            
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
    RZGrid = Grid.Cartesian2D(**Parameter2D)

    profile['Grid'] = RZGrid

    # extract the density information
    nshp = Parameter2D['ne_shape']
    ne_0 = Parameter2D['ne_0']


    # evaluate the density values on each grid point for the given shape
    if (nshp == 'exp') :
        # exponential decay shape
        DecScale= ShapeTable['exp']['NeDecayScale']
        DecLength= a/DecScale
        ne_array= ne_0 * np.exp(-np.sqrt(((RZGrid.R2D-R_0)**2 +RZGrid.Z2D**2))\
                  /DecLength )
    elif (nshp == 'Hmode') :
        # linear H mode profile
        nped = ne_0 * ShapeTable['Hmode']['PedHightN']
        nout = ne_0 * ShapeTable['Hmode']['ne_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthN'])
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
        nout = ne_0 * ShapeTable['linear']['ne_out']
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, vacuum
        ne_array = np.select([a_array <= a, a_array > a], 
                             [a_array*(nout-ne_0)/a + ne_0, nout])                              
    else:
        raise KeyError('"{}" is not a valid shape code. Available shapes \
are {}'.format(nshp, ShapeTable.keys()))
    profile['ne']= ne_array

    # Te profile
    tshp = Parameter2D['Te_shape']
    Te_0 = Parameter2D['Te_0']

    if ( tshp == 'exp'):
        DecLength= a/ShapeTable['exp']['TeDecayScale']
        Te_array= Te_0 * np.exp(-np.sqrt(((RZGrid.R2D-R_0)**2 +RZGrid.Z2D**2))\
                  /DecLength )
    elif (tshp == 'Hmode') :
        # linear H mode profile
        tped = Te_0 * ShapeTable['Hmode']['PedHightT']
        tout = Te_0 * ShapeTable['Hmode']['Te_out']
        a_core = a * (1-ShapeTable['Hmode']['PedWidthT'])
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, top of pedestal and the vacuum
        Te_array = np.select([a_array<=a_core, a_array>=a, a_array>a_core],
                             [a_array*(tped-Te_0)/a_core + Te_0, tout,
                              (a_array-a)*(tped-tout)/(a_core-a)+ tout])
    elif (tshp == 'uniform'):
        Te_array = np.zeros_like(RZGrid.R2D) + Te_0   
    elif (tshp == 'linear'):
        # linear decrease with minor radius
        tout = Te_0 * ShapeTable['linear']['Te_out']
        a_array = np.sqrt(((RZGrid.R2D-R_0)**2+RZGrid.Z2D**2))
        # linear function connecting axis, vacuum
        Te_array = np.select([a_array <= a, a_array > a], 
                             [a_array*(tout-Te_0)/a + Te_0, tout])                           
    else:
        raise KeyError('"{}" is not a valid shape code. Available shapes \
are {}'.format(tshp, ShapeTable.keys()))
    profile['Te']= Te_array

    # B profile
    B_0 = Parameter2D['B_0']
    B_array= B_0 * R_0/RZGrid.R2D
    profile['B']= B_array
    
    
    if fluctuation is not None:
        if fluctuation == 'random':
            time = Parameter2D['timesteps']
            d_shape = [len(time)]
            d_shape.extend([i for i in RZGrid.R2D.shape])
            if fluc_level is None:
                dne = 2*Parameter2D['dne_ne']*profile['ne']*\
                      (random(d_shape)-0.5)
                dTe_para = 2*Parameter1D['dte_te']*profile['Te']*\
                           (random(d_shape)-0.5)
                dTe_perp = 2*Parameter1D['dte_te']*profile['Te']*\
                           (random(d_shape)-0.5)
                dB = 2*Parameter2D['dB_B']*profile['B']*(random(d_shape)-0.5)
            else:
                dne = 2*fluc_level*profile['ne']*(random(d_shape)-0.5)
                dTe_para = 2*fluc_level*profile['Te']*(random(d_shape)-0.5)
                dTe_perp = 2*fluc_level*profile['Te']*(random(d_shape)-0.5)
                dB = 2*fluc_level*profile['B']*(random(d_shape)-0.5)
            return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                            profile['B'], time, dne, dTe_para, dTe_perp, dB)
        elif fluctuation=='sinx':
            timesteps = np.asarray(Parameter2D['timesteps'])
            time = timesteps*Parameter2D['dt']
            if fluc_level is None:
                nfluc_level = Parameter2D['dne_ne']
                tfluc_level = Parameter2D['dte_te']
                Bfluc_level = Parameter2D['dB_B']
            else:
                nfluc_level = fluc_level
                tfluc_level = fluc_level
                Bfluc_level = fluc_level
            x0 = Parameter2D['sinx']['x0']
            k = Parameter2D['sinx']['k']
            y0 = Parameter2D['sinx']['y0']
            dy = Parameter2D['sinx']['dy']
            omega = Parameter2D['sinx']['omega']
            
            fluc_pattern = np.sin(omega*time[:, np.newaxis, np.newaxis] - \
                                  k*(RZGrid.R2D-x0)) * \
                                  np.exp(-(RZGrid.Z2D-y0)**2/dy**2)
            dne = nfluc_level*profile['ne']*fluc_pattern
            dTe = tfluc_level*profile['Te']*fluc_pattern
            dB = Bfluc_level*profile['B']*fluc_pattern
            return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                            profile['B'], time, dne, dTe, dTe, dB)
        elif fluctuation=='siny':
            timesteps = np.asarray(Parameter2D['timesteps'])
            time = timesteps*Parameter2D['dt']
            if fluc_level is None:
                nfluc_level = Parameter2D['dne_ne']
                tfluc_level = Parameter2D['dte_te']
                Bfluc_level = Parameter2D['dB_B']
            else:
                nfluc_level = fluc_level
                tfluc_level = fluc_level
                Bfluc_level = fluc_level
            y0 = Parameter2D['siny']['y0']
            k = Parameter2D['siny']['k']
            x0 = Parameter2D['siny']['x0']
            dx = Parameter2D['siny']['dx']
            omega = Parameter2D['siny']['omega']
            
            fluc_pattern = np.sin(omega*time[:, np.newaxis, np.newaxis] - \
                                  k*(RZGrid.Z2D-y0)) * \
                                  np.exp(-(RZGrid.R2D-x0)**2/dx**2)
            dne = nfluc_level*profile['ne']*fluc_pattern
            dTe = tfluc_level*profile['Te']*fluc_pattern
            dB = Bfluc_level*profile['B']*fluc_pattern
            return ECEI_Profile(profile['Grid'],profile['ne'],profile['Te'],
                            profile['B'], time, dne, dTe, dTe, dB)
        else:
            raise NotImplementedError('Invalid fluctuation type:  {0}. \
Supported types are {1}'.format(fluctuation, FlucType2D))
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

class PlasmaModelError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Model:
    """base class for all plasma models

    Attributes:
        _name:The name of the model
        _type:The type of the model, 'Geometry','Equilibrium', or 'Fluctuation'
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
    
    Shape of Magnetic Geometry:
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

    

    
    
    
