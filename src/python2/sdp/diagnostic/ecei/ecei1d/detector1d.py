"""Create virtual detectors 
"""

#module depends on numpy
import numpy as np
import scipy as sp
from scipy.interpolate import RectBivariateSpline

from ....settings.unitsystem import cgs
from ....geometry.grid import path, Path2D, Cartesian2D
from ....plasma.profile import ECEI_Profile
from ..detector import Detector

#TODO derive Detector1D class from Detector class, and obey all the conventions
# there.
class Detector1D(Detector):
    """class of the detector which contains the frequency and path information

    Initialization
    ***************
    
    __init__(self, f_flt, p_flt, location, tilt, pth=None):
    :param float f_ctr: the center frequency ,in GHz
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
    :param path: optional, the ligth path. If not given, it will be calculated 
                using ray tracing in cold plasma. See 
                :py:class:`RayTracer<sdp.model.wave.ray.RayTracer>` for details
                about the ray tracing method.
    :type path: :py:class:`path<sdp.geometry.grid.path>`

    Methods:
    """
    def __init__(self, f_flt, p_flt, location, tilt, path=None):
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
        if path is not None:
            self._fix_path = True
            self.path = path
        else:
            self._fix_path = False
            
    @property
    def central_omega(self):
        return self._f_ctr*2*np.pi*1e9
        
    @property
    def omega_list(self):
        return self._f_flt*2*np.pi*1e9
        
    @property
    def power_list(self):
        return self._p_flt
        
    @property
    def __str__(self):
        info = '1D ECEI detector:\n'
        info += 'Central Frequency (GHz): {0}\n'.format(self._f_ctr)
        info += 'Location: (Z: {0}cm, R: {1}cm)\n'.format(self.location[0], 
                                                          self.location[1])
        info += 'Tilt angle (rad): {0}\n'.format(self.tilt)
        info += 'Band (GHz): (max: {0}, min: {1})'.format(np.max(self._f_flt),
                                                          np.min(self._f_flt))
        return info

    def generate_path(self, plasma):
#TODO finish the GO ray tracing generating path
        pass
                                                
                                                

def create_2D_pointlike_detector_array(plasma):
    """create testing 2D detector array which covers all the given plasma region
    
    plasma: dictionary, contains all the plasma profiles on 2D grids, see package plasma.TestParameters for detailed format
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
                    Detectors.append(Detector1D(f_flt,p_flt,[R[-1], Zi],
                                                0, pth))
    return tuple(Detectors)
            


def create_spatial_frequency_grid(Detectors, Profile, ResS=0.05, eq_only=True, 
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
        if dtc._fix_path:
            path2D = Path2D(dtc.path, ResS)
        else:
            dtc.generate_path(Profile)
            path2D = Path2D(dtc.path, ResS)
        ne_path = Profile.get_ne([path2D.Z2D[0,:],path2D.R2D[0,:]], 
                                 eq_only=eq_only, time=time)
        Te_para_path = Profile.get_Te([path2D.Z2D[0,:],path2D.R2D[0,:]], 
                                      perpendicular=False, eq_only=eq_only,
                                      time=time)
        Te_perp_path = Profile.get_Te([path2D.Z2D[0,:],path2D.R2D[0,:]], 
                                      perpendicular=True, eq_only=eq_only,
                                      time=time)
        B_path = Profile.get_B([path2D.Z2D[0,:],path2D.R2D[0,:]])
        new_prof = {}
        new_prof['Profile'] = ECEI_Profile(path2D, ne_path[np.newaxis,:], 
                                           Te_para_path[np.newaxis,:],
                                           Te_perp_path[np.newaxis,:], 
                                           B_path[np.newaxis,:])
        new_prof['omega'] = dtc.omega_list 
        Profs.append(new_prof)
    return tuple(Profs)
        
        
            
