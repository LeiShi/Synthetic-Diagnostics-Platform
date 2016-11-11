import json
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
        self._psiw = 5.1879e-2
                                       
    def _sp1d(self, psi, ysp):
        n = np.floor(psi/self._dpsi).astype(np.int)
        psi_r = psi - n*self._dpsi
        psi_re = psi_r*self._psiw
        
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
        
        psi_re = psi_r*self._psiw
        
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
    
