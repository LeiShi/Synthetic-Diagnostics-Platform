"""Calculate the recieved light intensity by carrying out the integral along the light path

I(s) = integral(s_0,s)[alpha(s')*exp(-(tau - tau'))*omega^2/(8 pi^3 * c^2) * T(s')] ds' --- ref[1] Eq(2.2.13-2.2.15)

[1] 1983 Nucl. Fusion 23 1153
"""
import numpy as np
from scipy.integrate import cumtrapz, trapz

from .detector1d import create_spatial_frequency_grid
from .detector1d import create_2D_pointlike_detector_array
from .alpha1 import get_alpha_table
from ....settings.unitsystem import cgs


class ECEI1D(object):
    r""" Main class for calculation of 1D synthetic ECEI
    
    Initialization
    **************
    
    Attributes
    **********
    
    Methods
    ********
    
    diagnose
    """
    
    def __init__(self, plasma, detectors, harmonic=2, S=100, ResS=0.05, 
                 lock_path=True):
        self._plasma = plasma
        self.detector_array = detectors
        self.harmonic = harmonic
        self._S = S
        self._ResS = ResS
        if lock_path:
            for detector in self.detector_array:
                detector.generate_path(S, plasma)
                detector.lock_path()
        
    def diagnose(self, time=None):
        if time is None:
            specProfile = create_spatial_frequency_grid(self.detector_array,
                                                        self._plasma, self._S,
                                                        self._ResS)
        else:
            assert isinstance(time, int), 'ECEI1D doesn\'t support time series\
 diagnosis. Please manually pass in one time step at one time.'
            specProfile = create_spatial_frequency_grid(self.detector_array,
                                                        self._plasma, self._S,
                                                        self._ResS, 
                                                        eq_only=False, 
                                                        time=time)
        self._R = [sp['Profile']['grid'].R1D for sp in specProfile]
        self._Z = [sp['Profile']['grid'].Z1D for sp in specProfile]
        self._s = [sp['Profile']['grid'].s1D for sp in specProfile]
        self._power = np.zeros_like(self.detector_array, dtype=float)
        self.Te = np.zeros_like(self._power)
        self._emission_patterns = []
        for i, dtc in enumerate(self.detector_array):
            Profile = specProfile[i]['Profile']
            # extract the path length coordinate array, see detailed format 
            # of s1D in 'sdp.geometry.grid', class Path1D
            s_array = Profile['grid'].s1D[:]
            # electron temperature along the path
            Te_array = Profile['Te'][:]  
            # calculate the 2D alpha array with the first dimention 
            # frequency and second path length
            alpha_array = get_alpha_table(specProfile[i],
                                                self.harmonic)
            for j, fj in enumerate(dtc._f_flt):
                # calculate optical thickness along light path
                tau_array = cumtrapz(alpha_array[j,:], x=s_array, 
                                     initial=0)
                # normalization integrand
                normal_array = alpha_array[j,:]\
                               * np.exp(tau_array - tau_array[-1]) /(2*np.pi)
                # evaluate the integrand on each grid point
                integrand_array = normal_array * Te_array 
                # integration over the path, result is for one given 
                # frequency
                power_f = trapz(integrand_array,x=s_array) 
                # multiply with the pattern ratio, and add on to the total 
                # received intensity of channel i
                self._power[i] += power_f * dtc.power_list[j] 
                # record the emission pattern for the central frequency
                
                if (fj - dtc.central_omega/(2*np.pi) <= dtc.central_omega\
                                                        *0.001/(2*np.pi)):
                    emission_pattern = normal_array * 2*np.pi
                    self._emission_patterns.append(emission_pattern)
            self.Te[i] = self._power[i]*(2*np.pi)
                     
    @property
    def view_spots(self):
        """The shape of sources of ECE along each light path
        
        list of 1d arrays of floats. 
        """
        return self._emission_patterns
        
    @property
    def R(self):
        """ R coordinates along light path
        """
        return self._R
        
    @property
    def Z(self):
        """ Z coordinates along light path
        """
        return self._Z
    

def get_intensity(Dtcs,RawProfile,n=2):
    """Calculate the intensity recieved by Detectors given by Dtcs
    
    Dtcs: list, containing all Detector objects
    RawProfile: plasma profile, Plasma Profile given on 2D grids
    n: harmonic number, default to be 2
    """
    specProfile = create_spatial_frequency_grid(Dtcs,RawProfile)
    intensity = np.zeros((len(Dtcs)))
    norms = np.zeros((len(Dtcs)))
    T_measured = np.zeros((len(Dtcs)))
    tau_arrays = []
    normal_arrays = []
    integrand_arrays = []
    s_arrays = []
    for i, dtc in enumerate(Dtcs):
        Profile = specProfile[i]['Profile']
        # extract the path length coordinate array, see detailed format of s2D 
        # in '..geometry.Grid', class Path2D
        s_array = Profile['grid'].s1D[:] 
        Te_array = Profile['Te'][:] # electron temperature along the path 
        alpha_array = get_alpha_table(specProfile[i],n)[:,:] #extract the 2D alpha array with the first dimention frequency and second path length
        for j, fj in enumerate(dtc._f_flt):
            #alpha = InterpolatedUnivariateSpline(s_array,alpha_array[j,:])
            tau_array = cumtrapz(alpha_array[j,:], x=s_array, initial=0)            
            #tau_array = np.zeros_like(s_array)
            #for k in np.arange( len(s_array) -1 ):
            #    tau_array[k+1] = tau_array[k] + (alpha_array[j,k]+alpha_array[j,k+1])*(s_array[k+1]-s_array[k])/2
            normal_array = alpha_array[j,:]*np.exp(tau_array - tau_array[-1])* fj**2 /(2*np.pi*cgs['c']**2) #normalization integrand
            integrand_array = normal_array * Te_array #evaluate the integrand on each grid point

            tau_arrays.append(tau_array)
            normal_arrays.append(normal_array)
            integrand_arrays.append(integrand_array)
            s_arrays.append(s_array)
            # integration over the path, result is for the given frequency
            intensity_f = trapz(integrand_array,x=s_array) 
            intensity[i] += intensity_f * dtc.power_list[j] # multiply with the pattern ratio, and add on to the total received intensity of channel i
            if (fj - dtc.central_omega/(2*np.pi) <= dtc.central_omega*0.01/(2*np.pi)):
                normalization_f = trapz(normal_array,s_array)
                norms[i]=normalization_f
        T_measured[i] = intensity[i]*(2*np.pi)**3*cgs['c']**2/dtc.central_omega**2

    return (tuple(intensity),tuple(T_measured),tuple(norms),tuple(tau_arrays),
            tuple(normal_arrays),tuple(integrand_arrays),tuple(s_arrays))
    
def get_2D_intensity(plasma):
    Dtcs = create_2D_pointlike_detector_array(plasma)
    intensity_tuple = get_intensity(Dtcs,plasma)
    T_measured = np.array(intensity_tuple[1])
    NZ = (plasma.grid.NZ-1)/10 + 1
    NR = (plasma.grid.NR-1)/10 + 1
    T_m_2D = T_measured.reshape((NZ,NR))
    return tuple(T_m_2D)
    