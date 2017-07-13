import json
import scipy.io.netcdf as nc
import numpy as np

class Diagnoser(object):

    def __init__(self, filename='diag_sdp.json'):

        with open(filename) as diagfile:
            raw_diag = json.load(diagfile)
        self.npsi = raw_diag['npsi']
        self.ntheta = raw_diag['ntheta']
        self.spdim = raw_diag['spdim']
        self.x = np.array(raw_diag['x']).reshape((self.ntheta, self.npsi,
                                                  self.spdim))
        self.z = np.array(raw_diag['z']).reshape((self.ntheta, self.npsi,
                                                  self.spdim))
        self.b = np.array(raw_diag['b']).reshape((self.ntheta, self.npsi,
                                                  self.spdim))
        self.g = np.array(raw_diag['g']).reshape(( self.npsi, 3) )
        self.I = np.array(raw_diag['I']).reshape(( self.npsi, 3) )
        self.q = np.array(raw_diag['q']).reshape(( self.npsi, 3) )
        self.jacobian_boozer = np.array(raw_diag['jacobian_boozer']).\
                                       reshape(( self.ntheta, self.npsi, 9) )
        self.jacobian_metric = np.array(raw_diag['jacobian_metric']).\
                                       reshape(( self.ntheta, self.npsi, 9) )
        self._dpsi = 1./(self.npsi-1)
        self._dtheta = np.pi*2/(self.ntheta-1)
        self._psi_separatrix = raw_diag['psi_separatrix']

    def _sp1d(self, psi, ysp):
        n = np.floor(psi/self._dpsi).astype(np.int)
        psi_r = psi - n*self._dpsi
        psi_re = psi_r*self._psi_separatrix

        return ysp[n,0] + ysp[n,1]*psi_re + ysp[n, 2]*psi_re*psi_re

    def g_sp(self, psi):
        return self._sp1d(psi, self.g)
    def I_sp(self, psi):
        return self._sp1d(psi, self.I)
    def q_sp(self, psi):
        return self._sp1d(psi, self.q)

    def _sp2d(self, psi, theta, ysp):
        if(np.any(psi>1)):
            raise ValueError('psi value greater than 1, outside of plasma \
boundary. Check psi values, they should be normalized psi_wall.')
        theta = np.remainder(theta, 2*np.pi)

        n = np.floor(psi/self._dpsi).astype(np.int)
        psi_r = psi - n*self._dpsi

        psi_re = psi_r*self._psi_separatrix

        m = np.floor(theta/self._dtheta).astype(np.int)
        theta_re= theta - m*self._dtheta

        psi_re2 = psi_re*psi_re
        ymn = ysp[m,n,:]
        return (ymn[..., 0]+ymn[...,1]*psi_re+ymn[...,2]*psi_re2) + \
               (ymn[...,3]+ymn[...,4]*psi_re+ymn[...,5]*psi_re2)*theta_re +\
               (ymn[...,6]+ymn[...,7]*psi_re+ymn[...,8]*psi_re2)*theta_re*\
               theta_re


    def x_sp(self, psi, theta):
        return self._sp2d(psi, theta, self.x)
    def z_sp(self, psi, theta):
        return self._sp2d(psi, theta, self.z)
    def b_sp(self, psi, theta):
        return self._sp2d(psi, theta, self.b)
    def jm_sp(self, psi, theta):
        return self._sp2d(psi, theta, self.jacobian_metric)
    def jb_sp(self, psi, theta):
        return self._sp2d(psi, theta, self.jacobian_boozer)

class AlphaDiagnoser(object):
    """ Diagnoser for magnetic perturbations in GTC obtained from M3DC1"""

    def __init__(self, filename='alpha_sdp.nc'):
        raw_alpha = nc.netcdf_file(filename, 'r')
        self.npsi = raw_alpha.dimensions['npsi']
        self.ntheta = raw_alpha.dimensions['ntheta']
        self.nzeta = raw_alpha.dimensions['nzeta']
        self._spdim = raw_alpha.dimensions['spdim']
        self._psi_separatrix = raw_alpha.variables['psi_separatrix'].data
        self._raw_alpha = raw_alpha.variables['alpha'].data
        assert self._raw_alpha.shape==(self.nzeta, self.ntheta, self.npsi,
                                         self._spdim)
        assert self._spdim==27, \
            "Quadratic spline is assumed. Other type not implemented"
        # calculate the step sizes in all dimensions
        self._dpsi = 1./(self.npsi-1)
        self._dtheta = np.pi*2/(self.ntheta-1)
        self._dzeta = np.pi*2/(self.nzeta-1)

    def _sp3d(self, psi, theta, zeta, ysp):
        if(np.any(psi>1)):
            raise ValueError('psi value greater than 1, outside of plasma \
boundary. Check psi values, they should be normalized psi_wall.')
        theta = np.remainder(theta, 2*np.pi)
        zeta = np.remainder(zeta, 2*np.pi)

        # obtain index and remainder in psi
        # note that the GTC spline in done on psi in GTC unit, not normalized
        # to psiw
        i = np.floor(psi/self._dpsi).astype(np.int)
        psi_r = psi - i*self._dpsi
        psi_re = psi_r*self._psi_separatrix
        # special care is needed for i==0 grid, the spline is done with respect
        # to sqrt(psi) in this cell
        psi_re = np.where(i==0, np.sqrt(psi_re), psi_re)

        # obtain index and remainder in theta
        j = np.floor(theta/self._dtheta).astype(np.int)
        theta_re= theta - j*self._dtheta

        # obtain index and remainder in theta
        k = np.floor(zeta/self._dzeta).astype(np.int)
        zeta_re= zeta - k*self._dzeta

        # create the 3D spline vector
        dx_vec = np.transpose([psi_re**p*theta_re**n*zeta_re**m for m in range(3)
                           for n in range(3) for p in range(3)], (1,2,3,0))

        return np.sum(ysp[k,j,i,:]*dx_vec, axis=-1)

    def alpha_sp(self, psi, theta, zeta):
        """evaluate alpha using GTC spline coefficients"""
        return self._sp3d(psi, theta, zeta, self._raw_alpha)

    def fourier_analysis(self, psi, ntheta, nzeta):
        """ Evaluate alpha_mn(psi) based on the GTC alpha spline

        It is important to make sure the Fourier transformation convention used
        in GTC and M3DC1 is the same as here. Check the convention in GTC
        documentations.
        """
        psi_1d = np.array(psi)
        assert psi_1d.ndim <= 1, "Only 1D array of psi is allowed."
        assert nzeta%2==1, 'nzeta need to be odd for FFT shifts'
        theta_1d = np.linspace(0, 2*np.pi, ntheta)
        zeta_1d = np.linspace(0, 2*np.pi, nzeta)

        zeta_mesh, theta_mesh, psi_mesh = np.meshgrid(zeta_1d, theta_1d,
                                                      psi_1d, indexing='ij')
        # spline interpolate the alpha values on 3D mesh
        alpha_arr = self._sp3d(psi_mesh, theta_mesh, zeta_mesh, self._raw_alpha)
        # Calculate the Harmonics
        # Note that the convention was alpha = sum alpha_mn*exp(in zeta-im theta)
        # So the inversed relation is alpha_mn = 1/mn sum alpha*exp(-in zeta +
        # im theta)
        alpha_n = np.fft.fft(alpha_arr,axis=0,norm=None)/nzeta
        # the np.fft.fft convention has the opposite sign for zeta, we need to
        # revert the n harmonics, first shift the array to have increasing
        # order
        alpha_n = np.fft.fftshift(alpha_n, axes=0)
        # revert the whole array
        alpha_n = np.flip(alpha_n, axis=0)
        # shift back to FFT order, nzeta is assumed odd
        alpha_n = np.fft.ifftshift(alpha_n, axes=0)
        # normal FFT on theta and return
        return np.fft.fft(alpha_n, axis=1,norm=None)/ntheta
