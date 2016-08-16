"""Create virtual detectors 
"""
import warnings
from math import sin, cos

import numpy as np

from ....settings.exception import PlasmaWarning
from ....settings.unitsystem import cgs
from ....geometry.grid import path, Path1D, Cartesian2D
from ....model.wave.ray import RayTracer
from ..detector import Detector

class Detector1D(Detector):
    """class of the detector which contains the frequency and path information

    Initialization
    ***************
    
    __init__(self, f_flt, p_flt, location, tilt, polarization='X', pth=None):
    :param float f_ctr: the center frequency ,in Hz
    :param f_flt: the frequencies sampling the pass band
    :type f_flt: array_like of floats
    :param p_flt: the pass coefficient for corresponding frequencies, 
                  may not be normalized
    :type p_flt: array_like of floats
    :param location: [y, x] coordinates of the receiver, starting point of the 
                     light path.
    :type location: array of floats, shape (2,)
    :param float tilt: vertical tilt angle in radian. positive means tilt 
                       upwards.
    :param string polarization: polarization of the antenna, either 'X' or 'O'
    :param path: optional, the ligth path. If given, path will be initially 
                 locked. If not given, paht is unlocked, and must be calculated 
                using ray tracing in cold plasma. See 
                :py:class:`RayTracer<sdp.model.wave.ray.RayTracer>` for details
                about the ray tracing method.
    :type path: :py:class:`path<sdp.geometry.grid.path>`
    
    Attributes
    ***********
    
    central_omega: the central circular frequency of the passing band, defined 
                   as frequency given by f_flt with the largest passing power
    omega_list: list of the circular frequencies in passing band, converted 
                from f_flt. Just :math:`2\pi` times f_flt
    power_list: list of passing power. normalized so the total passing power 
                is 1. Just a normalized version of p_flt
    location: (Z, R) coordinates of the detector, will be the starting point 
              for light path calculation
    tilt: tilt angle of the detector, used for calculating of the light path
    path: light path of the received emission, starting from inside the plasma,
          end at the detector location. If not fixed, will be calculated using
          :py:class:`RayTracer<sdp.model.wave.ray.RayTracer>` in a given 
          plasma.
    
    Methods
    ********
    
    generate_path(self, length, plasma, nodes=51, eq_only=True, time=None, 
                  mute=True): 
        If path is not locked, calculate the light path using ray tracing method
        in the given plasma. If path is locked, no effect.
    lock_path(self):
        lock the light path so the same path can be used in different plasma 
        times.
    unlock_path(self):
        unlock the light path so it can be calculated dynamically in defferent
        plasma times.
    """
    def __init__(self, f_flt, p_flt, location, tilt, polarization='X',
                 path=None):
        self._f_flt = np.copy(f_flt)
        self._p_flt = np.copy(p_flt)
        assert self._f_flt.ndim == 1, 'frequency filter must be specified as a \
1D array of frequencies'
        assert self._f_flt.shape == self._p_flt.shape
        assert np.all(self._f_flt > 0)
        assert np.all(self._p_flt > 0)
        
        self.location = np.array(location, dtype=float)
        self.tilt = tilt        
        
        self._f_ctr = self._f_flt[np.argmax(self._p_flt)]
        self._n_flt = len(self._f_flt)        
        
        # normalize the passing power
        self._p_flt /= np.sum(self._p_flt)
        
        self.polarization = polarization
        
        if path is not None:
            self._lock_path = True
            self.path = path
        else:
            self._lock_path = False
            
    @property
    def central_omega(self):
        return self._f_ctr*2*np.pi
        
    @property
    def omega_list(self):
        return self._f_flt*2*np.pi
        
    @property
    def power_list(self):
        return self._p_flt
        
    def __str__(self):
        info = '1D ECEI detector:\n'
        info += 'Central Frequency (GHz): {0}\n'.format(self._f_ctr)
        info += 'Location: (Z: {0}cm, R: {1}cm)\n'.format(self.location[0], 
                                                          self.location[1])
        info += 'Tilt angle (rad): {0}\n'.format(self.tilt)
        info += 'Band (GHz): (max: {0}, min: {1})'.format(np.max(self._f_flt),
                                                          np.min(self._f_flt))
        return info

    def generate_path(self, length, plasma, nodes=51, eq_only=True, time=None, 
                      mute=True):
        r""" Generate light path in given plasma
        
        :param float length: the approx. total length of light path, in cm
        :param plasma: plasma profile
        :type plasma: :py:class:`PlasmaProfile<sdp.plasma.profile.
                      PlasmaProfile>`
        :param int nodes: total number of points along the light path
        :param bool eq_only: flag for using equilibrium plasma only
        :param int time: time index in plasma fluctuations if eq_only is False.
        :param bool mute: if True, no status printing to screen. Default is 
                          True.
        """
        if self._lock_path:
            warnings.warn('path in locked. No new path will be generated.', 
                          PlasmaWarning)
            return self.path
            
        else:
            if not mute:
                print 'Dynamically generating light path. Detector location \
{0} is assumed in vacuum. Tilt angle {1} used.'.format(self.location,self.tilt)
            k = self.central_omega/cgs['c']
            kx = -k*cos(self.tilt)
            ky = k*sin(self.tilt)
            tracer = RayTracer(plasma=plasma, omega=self.central_omega, 
                               polarization=self.polarization,
                               eq_only=eq_only, time=time)
            # calculate the integration time steps, assume group speed of the 
            # wave is roughly speed of the light
            t_arr = np.linspace(0, length/cgs['c'], nodes)
            # obtain the light path from ray tracing                   
            raw_path = tracer.trace(self.location, [ky, kx], t_arr)
            # Need to invert the path to make it start from inside plasma and
            # end at the receiver
            self.path = path(n=nodes, R=raw_path[::-1, 1], Z=raw_path[::-1, 0])
            return self.path
            
    def lock_path(self):
        print "light path locked."
        self._lock_path = True
        
    def unlock_path(self):
        print "light path unlocked."
        self._lock_path = False
            
                                                
                                                

def create_2D_pointlike_detector_array(plasma):
    """create testing 2D detector array which covers all the given plasma 
    region
    
    :param plasma: plasma profile
    :type plasma: :py:class:`PlasmaProfile<sdp.plasma.profile.PlasmaProfile>`
    
    :return: detectors array
    :rtype: list of detectors
    """
    
    assert isinstance(plasma.grid, Cartesian2D)
    #get 1D coordinates
    Z = plasma.grid.Z1D
    R = plasma.grid.R1D
    
    Detectors = []
    for i, Zi in enumerate(Z):
        if (i%10 == 0):
            for j, Rj in enumerate(R):
                if(j%10==0):
                    #start creating detector correspond to this point
                    f_ctr = cgs['e']*plasma.get_B((Zi, Rj)) \
                            /(cgs['m_e']*cgs['c']* np.pi)
                    f_flt = f_ctr
                    p_flt = [1]
                    # light path is assumed horizontal
                    pth = path(2,[R[0],R[-1]],[Zi,Zi]) 
                    Detectors.append(Detector1D(f_flt,p_flt,[Zi,R[-1]],
                                                0, pth))
    return tuple(Detectors)
    
def create_detector_array(location_list, tilt_list, f_flt_list, p_flt_list, 
                          path_list=None):
    r""" Generate a list of Detector1D objects from the given parameter lists
    
    :param location_list: detector locations, each location is specified with 
                          coordinates [Z, R]
    :type location_list: list of pairs of floats
    :param tilt_list: tilt angles in radian
    :type tilt_list: list of floats
    :param f_flt_list: frequency filter arrays, in GHz
    :type f_flt_list: list of lists of floats
    :param p_flt_list: power list corresponding to the frequency filter list
    :type p_flt_list: list of lists of floats
    :param path_list: specified light paths for detectors, optional. If not 
                      given, light path in a given plasma will be calculated 
                      through ray tracing.
    :type path_list: list of :py:class:`path<sdp.geometry.grid.path>` objects                  
    
    :return: detectors
    :rtype: list of :py:class:`Detector1D<sdp.diagnostics.ecei.ecei1d.
            detector1d.Detector1d>` objects
    """
    n = len(location_list)
    assert len(tilt_list)==n
    assert len(f_flt_list)==n
    assert len(p_flt_list)==n
    if path_list is None:
        return [Detector1D(f_flt_list[i], p_flt_list[i], location_list[i], 
                           tilt_list[i]) for i in xrange(n)]
    else:
        assert len(path_list)==n
        return [Detector1D(f_flt_list[i], p_flt_list[i], location_list[i], 
                           tilt_list[i], path_list[i]) for i in xrange(n)]

def create_spatial_frequency_grid(Detectors, Profile, S=100, ResS=0.05, 
                                  eq_only=True, 
                                  time=None):
    """create grids points on spatial and frequency space for alpha calculation
    suitable for given detector parameters

    :param Detectors: ECEI detectors
    :type Detectors: list of :py:class:`Detector1D
                     <sdp.diagnostics.ecei.ecei1d.detector1d.Detector1D>` 
                     objects
    :param Profile: plasma profile
    :type Profile: :py:class:`PlasmaProfile<sdp.plasma.profile.PlasmaProfile>`
    :param float ResS: resolution in light path, default to be 0.05 cm.
    :param bool eq_only: flag for equilibrium only runs.
    :param int time: time step for non-equilibirum runs. No effect if eq_only 
                     is True.
    
    :return: Profs, an array contains all the quantities dictionaries on grids 
             that feed to alpha calculation function, each profile corresponds 
             to one detector.
    """
    Profs = []
    Profile.setup_interps()
    for dtc in Detectors:
        dtc.generate_path(S, Profile)
        path1D = Path1D(dtc.path, ResS)
        ne_path = Profile.get_ne([path1D.Z1D, path1D.R1D], eq_only=eq_only,
                                 time=time)
        Te_path = Profile.get_Te([path1D.Z1D, path1D.R1D], eq_only=eq_only, 
                                 time=time)
        B_path = Profile.get_B([path1D.Z1D, path1D.R1D], eq_only=eq_only, 
                               time=time)
        new_prof = {}
        new_prof['Profile'] = {}
        new_prof['Profile']['ne'] = ne_path
        new_prof['Profile']['Te'] = Te_path
        new_prof['Profile']['B'] = B_path
        new_prof['omega'] = dtc.omega_list 
        Profs.append(new_prof)
    return tuple(Profs)
        
        
            
