# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 08:39:25 2016

@author: shilei

Loader and checker for M3D-C1 RMP output files
"""

import numpy as np
from scipy.io.netcdf import netcdf_file
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from sdp.math.smooth import smooth

class RmpLoader(object):
    """Loader for M3D-C1 RMP output files
    
    Initialization
    ****************
    
    __init__()
    Initialize with a filename
    
    Attributes
    ***********
    
    Example
    *******
    """
    
    def __init__(self, filename, mode=r'full'):
        """Initialize the loader
        
        filename should contain the full path to the netcdf file generated by
        M3DC1.
        
        :param string filename: full or relative path to the netcdf file 
                                generated by M3DC1
        :param string mode: loading mode. mode='full' will automatically read
                            all the desired data in the file, and is the 
                            default mode. mode='eq_only' reads only the 
                            equilibrium, no RMP variables. mode='least' only 
                            initialize the object, nothing will be read. 
        """  
        self.filename = filename
        if mode == 'least':            
            return
        else:
            self.load_equilibrium(self.filename)
            if mode == 'full':
                self.load_rmp(self.filename)
                self.generate_interpolators()
    
    def load_equilibrium(self, filename):
        """load the equilibrium data from M3DC1 output netcdf file
        
        :param string filename: full or relative path to the netcdf file 
                                generated by M3DC1
        """
        m3d_raw = netcdf_file(filename, 'r')
        
        # npsi is the number of grid points in psi
        # mpol is the number of grid points in theta
        self.npsi = m3d_raw.dimensions['npsi']
        self.mpol = m3d_raw.dimensions['mpol']
        
        # 1D quantities are function of psi only
        # we have
        
        # poloidal flux, axis value zero, always increase, in weber
        self.poloidal_flux = np.copy(m3d_raw.variables['flux_pol'].data)
        # poloidal flux, in weber/radian
        self.psi_p = np.copy(m3d_raw.variables['psi'].data)
        self.psi_p -= self.psi_p[0]
        if (self.psi_p[-1] < 0):
            self.isign = -1
        else:
            self.isign = 1
        # psi_abs is the flipped psi_p so it's increasing. 
        # WARNING: SPECIAL CARE IS NEEDED WHEN CALCULATING MAGNETIC FIELD LINE
        self.psi_abs = self.psi_p*self.isign
        
        # normalized psi, normalized to psi_wall
        self.psi_n = np.copy(m3d_raw.variables['psi_norm'].data)
        # uniform theta is generated, including the end point at theta=2pi
        self.theta = np.linspace(0, 2*np.pi, self.mpol+1)
        
        # toroidal current enclosed in a flux surface
        self.I = np.copy(m3d_raw.variables['current'].data)
        self.I *= 2e-7
        # R B_phi is also a flux function
        self.F = np.copy(m3d_raw.variables['F'].data)
        # equilibrium electron density
        self.ne = np.copy(m3d_raw.variables['ne'].data)
        # safety factor
        self.q = np.copy(m3d_raw.variables['q'].data)
        # total pressure
        self.p = np.copy(m3d_raw.variables['p'].data)
        # electron pressure
        self.pe = np.copy(m3d_raw.variables['pe'].data)
        
        # 2D quantities will depend on theta, we'll add one end point at 
        # theta=2pi, and assign periodic value there
        
        # R in (R, PHI, Z) coordinates
        self.R = np.empty((self.npsi, self.mpol+1))
        self.R[:, :-1] = m3d_raw.variables['rpath'][:,:]
        self.R[:, -1] = self.R[:, 0]
        # Z in (R, PHI, Z)
        self.Z = np.empty((self.npsi, self.mpol+1))
        self.Z[:, :-1] = m3d_raw.variables['zpath'][:,:]
        self.Z[:, -1] = self.Z[:, 0]
        # poloidal magnetic field
        self.B_p = np.empty((self.npsi, self.mpol+1))
        self.B_p[:, :-1] = m3d_raw.variables['Bp'][:,:]
        self.B_p[:, -1] = self.B_p[:, 0]

        # Jacobian 
        self.Jacobian = np.empty((self.npsi, self.mpol+1))
        self.Jacobian[:, :-1] = m3d_raw.variables['jacobian'][:,:]
        self.Jacobian[:, -1] = self.Jacobian[:, 0]
        
        m3d_raw.close()
        
    def load_rmp(self, filename):
        """load the resonant magnetic perturbations
        :param string filename: full or relative path to the netcdf file 
                                generated by M3DC1
        """
        #todo coordinates convention needs to be sorted out
        
        m3d_raw = netcdf_file(filename, 'r')
        
        # mode numbers in m
        self.m = np.copy(m3d_raw.variables['m'].data)
        
        # In our convention, alpha is a complex number, and the resonant form
        # has cos and sin part on real and imaginary parts respectively
        self.alpha_m = np.copy(m3d_raw.variables['alpha_real'].data) + \
                     1j*np.copy(m3d_raw.variables['alpha_imag'].data)
                     
        # dB_m is the perpendicular component of perturbed magnetic field
        # Fourier decomposition in theta
        self.dB_m = np.copy(m3d_raw.variables['bmn_real'].data) + \
                    1j* np.copy(m3d_raw.variables['bmn_imag'].data)
                    
        self.A = np.copy(m3d_raw.variables['area'].data)
                    
        m3d_raw.close()
        # check if the mode number is inversed, if so, change it back to 
        # increasing order
        if self.m[0]>self.m[-1]:
            self.m = np.fft.ifftshift(self.m[::-1])
            self.alpha_m = np.fft.ifftshift(self.alpha_m[:, ::-1], axes=-1)
            self.dB_m = np.fft.ifftshift(self.dB_m[:,::-1], axes=-1)
        else:
            self.m = np.fft.ifftshift(self.m)
            self.alpha_m = np.fft.ifftshift(self.alpha_m[:, :], axes=-1)
            self.dB_m = np.fft.ifftshift(self.dB_m[:,:], axes=-1)
        
        # for alpha and dalpha_dtheta values in real space, we add the 
        # theta=2pi end point, and assign periodic values
        self.alpha = np.empty((self.npsi, self.mpol+1), dtype=np.complex)
        self.dalpha_dtheta = np.empty((self.npsi, self.mpol+1), 
                                      dtype=np.complex)
        self.dB = np.empty((self.npsi, self.mpol+1), dtype=np.complex)
            
        # Then, the real space alpha can be obtained by FFT. Check Nate's note
        # on the normalization convention, as well as scipy's FFT 
        # documentation.
        
        self.alpha[:, :-1] = np.fft.fft(self.alpha_m, axis=-1)
        self.alpha[:, -1] = self.alpha[:, 0]
        
        # The derivatives respect to theta can also be calculated by FFT
        self.dalpha_dtheta[:, :-1] = np.fft.fft(-1j*self.m*self.alpha_m, 
                                                axis=-1) 
        self.dalpha_dtheta[:, -1] = self.dalpha_dtheta[:, 0]
        # Smooth the derivative for 2 passes of 121
        smooth(self.dalpha_dtheta, periodic=1, passes=2)
        
        # delta_B is also calculated by FFT
        self.dB[:, :-1] = np.fft.fft(self.dB_m, axis=-1)
        self.dB[:, -1] = self.dB[:, 0]
        
        # calculate alpha_m from dB_m and get alpha
        self._calc_alpha_m(3)
        self._alpha_c = np.empty((self.npsi, self.mpol+1), dtype=np.complex)
        self._dalpha_dtheta_c = np.empty((self.npsi, self.mpol+1), 
                                      dtype=np.complex)
        self._alpha_c[:,:-1] = np.fft.fft(self._alpha_mc, axis=-1)
        self._alpha_c[:,-1] = self._alpha_c[:,0]
        self._dalpha_dtheta_c[:, :-1] = np.fft.fft(-1j*self.m*self._alpha_mc, 
                                                axis=-1) 
        self._dalpha_dtheta_c[:, -1] = self._dalpha_dtheta_c[:, 0]

    def _calc_alpha_m(self, n):
        """ Calculate alpha_mn based on the load in B_mn, F, I, A, and a given
        n.
        """
        res = self.m*self.F[:, np.newaxis]+n*self.I[:, np.newaxis]
        self._alpha_mc = -1j*self.A[:, np.newaxis]*self.dB_m/ ((2*np.pi)**4*res)
    def generate_interpolators(self):
        """ Create the interpolators for the loaded quantities
        """
        # 1-D quantities are directly interpolated on psi
        self.q_interp = interp1d(self.psi_abs, self.q)
        self.I_interp = interp1d(self.psi_abs, self.I)
        self.F_interp = interp1d(self.psi_abs, self.F)
        
        
        # 2-D quantities are interpolated on psi-theta plane
        # the periodicity along theta in the values is guaranteed
        self.R_interp = RectBivariateSpline(self.psi_abs, self.theta, self.R)
        self.Z_interp = RectBivariateSpline(self.psi_abs, self.theta, self.Z)
        self.Bp_interp = RectBivariateSpline(self.psi_abs,self.theta,self.B_p)
        
        self.alpha_re_interp = RectBivariateSpline(self.psi_abs, self.theta,
                                                np.real(self.alpha))
        self.alpha_im_interp = RectBivariateSpline(self.psi_abs, self.theta,
                                                np.imag(self.alpha))
        self.dadt_re_interp = RectBivariateSpline(self.psi_abs,self.theta,
                                     np.real(self.dalpha_dtheta))
        self.dadt_im_interp = RectBivariateSpline(self.psi_abs,self.theta,
                                     np.imag(self.dalpha_dtheta))
        
        self._alpha_c_re_interp = RectBivariateSpline(self.psi_abs, self.theta,
                                                np.real(self._alpha_c))
        self._alpha_c_im_interp = RectBivariateSpline(self.psi_abs, self.theta,
                                                np.imag(self._alpha_c))
        self._dadt_c_re_interp = RectBivariateSpline(self.psi_abs,self.theta,
                                     np.real(self._dalpha_dtheta_c))
        self._dadt_c_im_interp = RectBivariateSpline(self.psi_abs,self.theta,
                                     np.imag(self._dalpha_dtheta_c))
        
    
        
    def generate_poincare(self, npsi_start=0, npsi_end=1, npsi=20, ntheta=1,
                          npoints=100, nzeta=100,n=1, m=None):
        """calculate the poincare plot array in (psi, theta) plane
        
        Poincare plot is generated by advancing a point in (psi, theta) along
        the magnetic field line, and after going 2pi in zeta, the point comes 
        back to the initial (psi, theta) plane, and gives a new point. 
        
        Going around for a lot of cycles, we obtain a series of points on the 
        (psi, theta) plane, this is a Poincare plot for the single initial 
        point. 
        
        We can launch an array of initial points, and obtain a full Poincare 
        plot for the whole plane.
        
        :param int npsi_start: the index of the starting psi
        :param int npsi_end: the index of the end psi
        :param int npsi : the total number of sample points in psi 
        :param int ntheta : total number of sample points in theta
        :param int npoints: total poincare cycles, equals the number of points
                            plotted for a single starting point
        :param m: chosen theta mode numbers to make the plot
        :type m: array of ints, if not given, all m numbers available will be 
                 used
        """
        
        psi0 = self.psi_p[-1]*npsi_start
        psi1 = self.psi_p[-1]*npsi_end
        psi_plot = np.linspace(psi0, psi1, npsi)
        theta_plot = np.linspace(0, np.pi*2, ntheta)
        zeta = np.arange(npoints)*np.pi*2/nzeta
        psi_result = np.empty((npsi, ntheta, npoints))
        theta_result = np.empty((npsi, ntheta, npoints))
        for i, psi in enumerate(psi_plot):
            for j, theta in enumerate(theta_plot):
                result = odeint(_FL_prime, [psi, theta], zeta, args=(self,n))
                psi_result[i,j] = result[:, 0]
                theta_result[i, j] = result[:, 1]
        psi_result /= self.psi_p[-1]
        theta_result = np.mod(theta_result, 2*np.pi)
        self.poincare = [psi_result, theta_result]
        
    def _generate_poincare_c(self, npsi_start=0, npsi_end=1, npsi=20, ntheta=1,
                             npoints=1000, nzeta=100, n=1, m=None):
        psi0 = self.psi_p[-1]*npsi_start
        psi1 = self.psi_p[-1]*npsi_end
        psi_plot = np.linspace(psi0, psi1, npsi)
        theta_plot = np.linspace(0, np.pi*2, ntheta)
        zeta = np.arange(npoints)*np.pi*2/nzeta
        psi_result = np.empty((npsi, ntheta, npoints))
        theta_result = np.empty((npsi, ntheta, npoints))
        for i, psi in enumerate(psi_plot):
            for j, theta in enumerate(theta_plot):
                result = odeint(_FL_prime_c, [psi, theta], zeta, args=(self,n))
                psi_result[i,j] = result[:, 0]
                theta_result[i, j] = result[:, 1]
        psi_result /= self.psi_p[-1]
        theta_result = np.mod(theta_result, 2*np.pi)
        self._poincare_c = [psi_result, theta_result]
        

def _FL_prime(psi_theta, zeta, rmp_object, n=3):
    """ field line differentiation along zeta
    
    Calculates dpsi/dzeta, dtheta/dzeta along the field line
    """
    psi, theta = psi_theta
    psi = np.abs(psi)
    theta = np.mod(theta, 2*np.pi)
    
    g = rmp_object.F_interp(psi)
    I = rmp_object.I_interp(psi)
    dadt = rmp_object.dadt_re_interp(psi, theta) * np.cos(zeta) \
           - rmp_object.dadt_im_interp(psi, theta) * np.sin(zeta)
    dadz = -n*(rmp_object.alpha_im_interp(psi, theta) * np.cos(zeta) \
           + rmp_object.alpha_re_interp(psi, theta)*np.sin(zeta) )
    q = rmp_object.q_interp(psi)
    
    dpsi_dzeta = (g*dadt - I*dadz)/q
    dtheta_dzeta = 1/q
    y = [dpsi_dzeta, dtheta_dzeta]
    return y
    
def _FL_prime_c(psi_theta, zeta, rmp_object, n=3):
    """ field line differentiation along zeta
    
    Calculates dpsi/dzeta, dtheta/dzeta along the field line using the derived
    alpha
    """
    psi, theta = psi_theta
    psi = np.abs(psi)
    theta = np.mod(theta, 2*np.pi)
    
    g = rmp_object.F_interp(psi)
    I = rmp_object.I_interp(psi)
    dadt = rmp_object._dadt_c_re_interp(psi, theta) * np.cos(zeta) \
           - rmp_object._dadt_c_im_interp(psi, theta) * np.sin(zeta)
    dadz = -n*(rmp_object._alpha_c_im_interp(psi, theta) * np.cos(zeta) \
           + rmp_object._alpha_c_re_interp(psi, theta)*np.sin(zeta) )
    q = rmp_object.q_interp(psi)
    
    dpsi_dzeta = (g*dadt - I*dadz)/q
    dtheta_dzeta = 1/q
    y = [dpsi_dzeta, dtheta_dzeta]
    return y
    
def poincare_plot(poincare):
    color_table = ['r', 'b', 'm', 'g', 'k', 'c']
    psi_pc, theta_pc = poincare
    npsi = psi_pc.shape[0]
    ntheta = psi_pc.shape[1]

    f,ax = plt.subplots(1)
    for i in range(npsi):
        c = color_table[i%len(color_table)]
        for j in range(ntheta):
            ax.scatter(theta_pc[i,j,:], psi_pc[i,j,:], s=5, linewidth=0, c=c )
        
                     
        
        

            
            
        
