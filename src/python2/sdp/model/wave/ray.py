# -*- coding: utf-8 -*-
r"""

Ray Tracing Model
******************

This module contains model for solving simple ray tracing equations:

.. math::
    \frac{dx}{dt} = \frac{\partial \mathcal{D}/\partial k}
                         {\partial \mathcal{D}/\partial \omega}
    
.. math::
    \frac{dk}{dt} = -\frac{\partial \mathcal{D}/\partial x}
                         {\partial \mathcal{D}/\partial \omega}
    
where x, k, t are configuration coordinates, wave vector coordinates, and time,
and :math:`\mathcal{D}\equiv\mathcal{D}(\omega, x, k, t)` is the function in 
dispersion relation :math:`\mathcal{D}=0` .

For a stationary plasma, the time variation of :math:`\mathcal{D}` can be 
neglected, so the wave frequency is constant. We can solve for the trajectory 
in x,k space.

In particular, when propagation is perpendicular to the magnetic field, we have
two independent polarizaitons: X-mode and O-mode. The dispersion relations are:

O-mode:

.. math::
    \left(\frac{ck}{\omega}\right)^2 = 1-\frac{\omega_{pe}^2}{\omega^2}
    
X-mode:

.. math::
    \left(\frac{ck}{\omega}\right)^2 = \frac{RL}{S}
    
where :math:`R\equiv 1- \frac{\omega_{pe}^2}{\omega(\omega+\Omega_{ce})}`,
:math:`L\equiv 1-\frac{\omega_{pe}^2}{\omega(\omega-\Omega_{ce})}`, and 
:math:`S \equiv \frac{1}{2}(L+R)`. 

Note that the spatial dependent is in the plasma frequency 
:math:`\omega^2_{pe}=4\pi e^2 n_e(x)/m_e`. The partial derivatives need to be
calculated analytically, and then evaluated numerically for a given plasma.

Created on Thu Aug 11 11:38:22 2016

@author: lei
"""

import numpy as np
from scipy.integrate import odeint

from ...settings.unitsystem import cgs
from ...settings.exception import ResonanceError

# some useful constants
e = cgs['e']
m_e = cgs['m_e']
c = cgs['c']

class ColdDispersionDerivatives(object):
    """class providing calculators for partial derivatives of cold dispersion 
    relation.
    
    __init__(self, plasma, omega, polarization='O', equilibrium_only=True)
    
    :param plasma: plasma profile object
    :type plasma: :py:class:`PlasmaProfile<sdp.plasma.profile.PlasmaProfile>`
    :param float omega: circular frequency of the wave
    :param string polarization: polarization of the wave, either 'O' or 'X'.
    :param bool equilibrium_only: True if only equilibrium plasma is used.
    """
    
    def __init__(self, plasma, omega, polarization='O', equilibrium_only=True):
        assert polarization in ['O', 'X']
        self._plasma = plasma
        # setup interpolators for later use
        self._plasma.setup_interps(equilibrium_only)
        
        self._omega = omega
        self._polarization = polarization
        
    def __str__(self):
        info = 'Omega : {0}\n'.format(self._omega)
        info += 'Polarization : {0}\n'.format(self._polarization)
        info += str(self._plasma)
        return info
        
    def _dnedx(self, x, dx=0.01):
        r""" Evaluates dne/dx at given x
        
        center derivative is used by default. If the given x is close to the
        boundary of given plasma, one side derivative is calculated.
        
        :param x: coordinate(s) of the evaluation point
        :type x: list of floats, length equals the dimension of plasma
        :param dx: step size to evaluate derivative of x, default to be 0.01cm
        :type dx: float if same for all directions, list of floats for 
                  different step sizes in different directions.
                  
        :return: derivatives respect to x
        :rtype: list of floats
        """        
        assert len(x) == self._plasma.grid.dimension        
        
        x = np.array(x)
        dx = np.array(dx)
        if (dx.ndim == 0):
            assert dx > 0
            dx = np.zeros_like(x) + dx
        else:
            assert dx.ndims == self._plasma.grid.dimension
            assert np.all(dx > 0)
            
        # before calculating derivatives, we need to identify the near boundary
        # points, where center derivative can not be used, one side derivative
        # must be used instead
        dx_plus = np.copy(dx)
        dx_minus = np.copy(dx)
        ne_plus = np.empty_like(x)
        ne_minus = np.empty_like(x)
        for i,d in enumerate(dx):
            try:
                coords = np.copy(x)
                coords[i] += dx[i]
                ne_plus[i] = self._plasma.get_ne(coords)
            except ValueError:
                dx_plus[i] = 0
                ne_plus[i] = self._plasma.get_ne(x)
            try:
                coords = np.copy(x)
                coords[i] -= dx[i]
                ne_minus[i] = self._plasma.get_ne(coords)
            except ValueError:
                dx_minus[i] = 0
                ne_minus[i] = self._plasma.get_ne(x)
        
        # Every direction must have at least one side within plasma region
        assert np.all(dx_plus+dx_minus > 0)
        return (ne_plus - ne_minus)/(dx_plus + dx_minus)
        
    def _dBdx(self, x, dx=0.01):
        r""" Evaluates dB/dx at given x
        
        center derivative is used by default. If the given x is close to the
        boundary of given plasma, one side derivative is calculated.
        
        :param x: coordinate(s) of the evaluation point
        :type x: list of floats, length equals the dimension of plasma
        :param dx: step size to evaluate derivative of x, default to be 
                   0.01cm
        :type dx: float if same for all directions, list of floats for 
                  different step sizes in different directions.
                  
        :return: derivatives respect to x
        :rtype: list of floats
        """        
        assert len(x) == self._plasma.grid.dimension        
        
        x = np.array(x)
        dx = np.array(dx)
        if (dx.ndim == 0):
            assert dx > 0
            dx = np.zeros_like(x) + dx
        else:
            assert dx.ndims == self._plasma.grid.dimension
            assert np.all(dx > 0)
            
        # before calculating derivatives, we need to identify the near boundary
        # points, where center derivative can not be used, one side derivative
        # must be used instead
        dx_plus = np.copy(dx)
        dx_minus = np.copy(dx)
        B_plus = np.empty_like(x)
        B_minus = np.empty_like(x)
        for i,d in enumerate(dx):
            try:
                coords = np.copy(x)
                coords[i] += dx[i]
                B_plus[i] = self._plasma.get_B(coords)
            except ValueError:
                dx_plus[i] = 0
                B_plus[i] = self._plasma.get_B(x)
            try:
                coords = np.copy(x)
                coords[i] -= dx[i]
                B_minus[i] = self._plasma.get_B(coords)
            except ValueError:
                dx_minus[i] = 0
                B_minus[i] = self._plasma.get_B(x)
        
        # Every direction must have at least one side within plasma region
        assert np.all(dx_plus+dx_minus > 0)
        return (B_plus - B_minus)/(dx_plus + dx_minus)
        
        
        
    def _dPdx(self, x, dx=0.01):
        r""" Evaluates dP/dx at given x
        
        .. math::
            P = 1-\frac{\omega^2_{pe}}{\omega^2}
            
        so 
        
        .. math::
            \frac{dP}{dx} = -\frac{4\pi e^2 n_e'}{m_e \omega^2}
            
        :param x: coordinate(s) of the evaluation point
        :type x: list of floats, length equals the dimension of plasma
        :param dx: step size to evaluate derivative of x, default to be 0.01cm
        :type dx: float if same for all directions, list of floats for 
                  different step sizes in different directions.
                  
        :return: derivatives respect to x
        :rtype: list of floats
        
        """                
        dPdx = -4*np.pi*e*e*self._dnedx(x, dx)/(m_e*self._omega*self._omega)        
        return dPdx
        
    def _dSdx(self, x, dx=0.01, tol=1e-14):
        r""" Evaluate dS/dx
        
        .. math::
            S = 1-\frac{\omega_{pe}^2}{\omega^2-\Omega_{ce}^2}
            
        where :math:`\omega_{pe}^2 = 4\pi e^2 n_e(x)/m_e`, and 
        :math:`\Omega_{ce} = \frac{eB(x)}{m_e c}`.
        
        So, 
        
        .. math::
            \frac{dS}{dx} =-\left(\frac{(\omega_{pe}^2)'}
                                       {\omega^2-\Omega_{ce}^2}
                              + \frac{2\omega_{pe}^2 \Omega_{ce} \Omega'_{ce}}
                                     {(\omega^2 - \Omega_{ce}^2)^2}\right)
                                     
        When :math:`\omega^2-\Omega_{ce}^2=0`, cold resonance occurs, cold 
        dispersion relation can not handle, a :py:Exception:`ResonanceError
        <sdp.settings.exception.ResonanceError>` will be raised.
                                     
        :param x: coordinate(s) of the evaluation point
        :type x: list of floats, length equals the dimension of plasma
        :param dx: step size to evaluate derivative of x, default to be 0.01cm
        :type dx: float if same for all directions, list of floats for 
                  different step sizes in different directions.
        :param float tol: tolerance for checking resonance, when 
                          |omega^2-omega_ce^2|<tol, resonance happens. Default
                          is 1e-14.
                              
        :return: derivatives respect to x
        :rtype: list of floats
        :raise: :py:Exception:`ResonanceError
                               <sdp.settings.exception.ResonanceError>`
        """        
        pe_const = 4*np.pi*e*e/m_e
        omega_pe2 = pe_const*self._plasma.get_ne(x)
        omega_pe2_p = pe_const*self._dnedx(x)
        ce_const = e/(m_e*c)
        omega_ce = ce_const*self._plasma.get_B(x)
        omega_ce_p = ce_const*self._dBdx(x)        
        omega2_m_omegace2 = self._omega*self._omega - omega_ce*omega_ce
        if np.abs(omega2_m_omegace2)<tol:
            raise ResonanceError('Cold X resonance happens, S goes to infinity\
 at {0}.'.format(x))
        dSdx = -(omega_pe2_p/omega2_m_omegace2 + \
                 omega_pe2*omega_ce*omega_ce_p/(omega2_m_omegace2**2))
        return dSdx
        
    def _dDdx(self, x, dx=0.01, tol=1e-14):
        r"""Evaluate dD/dx
        
        .. math::
            D = -\frac{\omega_{pe}^2 \Omega_{ce}}
                      {\omega(\omega^2-\Omega_{ce}^2)}
        
        where :math:`\omega_{pe}^2 = 4\pi e^2 n_e(x)/m_e`, and 
        :math:`\Omega_{ce} = \frac{eB(x)}{m_e c}`.
              
        So, 
        
        .. math::
            \frac{dD}{dx} = -\left( \frac{(\omega_{pe}^2)'\Omega_{ce}+
                                          \omega_{pe}^2\Omega_{ce}'} 
                                         {\omega(\omega^2-\Omega_{ce}^2)}+
                            \frac{2\omega_{pe}^2\Omega_{ce}^2\Omega_{ce}'}
                                 {\omega(\omega^2-\Omega_{ce}^2)^2}\right)
                                 
        When :math:`\omega^2-\Omega_{ce}^2=0`, cold resonance occurs, cold 
        dispersion relation can not handle, a :py:Exception:`ResonanceError
        <sdp.settings.exception.ResonanceError>` will be raised.
        
        :param x: coordinate(s) of the evaluation point
        :type x: list of floats, length equals the dimension of plasma
        :param dx: step size to evaluate derivative of x, default to be 0.01cm
        :type dx: float if same for all directions, list of floats for 
                  different step sizes in different directions.
        :param float tol: tolerance for checking resonance, when 
                          |omega^2-omega_ce^2|<tol, resonance happens. Default
                          is 1e-14.
                              
        :return: derivatives respect to x
        :rtype: list of floats
        :raise: :py:Exception:`ResonanceError
                               <sdp.settings.exception.ResonanceError>`
        """
        pe_const = 4*np.pi*e*e/m_e
        omega_pe2 = pe_const*self._plasma.get_ne(x)
        omega_pe2_p = pe_const*self._dnedx(x)
        ce_const = e/(m_e*c)
        omega_ce = ce_const*self._plasma.get_B(x)
        omega_ce_p = ce_const*self._dBdx(x) 
        omega2_m_omegace2 = self._omega*self._omega - omega_ce*omega_ce
        if np.abs(omega2_m_omegace2)<tol:
            raise ResonanceError('Cold X resonance happens, D goes to infinity\
 at {0}.'.format(x))
 
        dDdx = -((omega_pe2_p*omega_ce + omega_pe2*omega_ce_p)/\
                 (self._omega*omega2_m_omegace2) +\
                 2*omega_pe2*omega_ce*omega_ce*omega_ce_p/\
                 (self._omega*omega2_m_omegace2*omega2_m_omegace2))
                  
        return dDdx
                        
    def pDpk(self, x, k):
        r""" Evaluate partial D over partial k at given (x, k) coordinates

        Since cold dielectric tensor doesn't depend on k, the partial 
        derivative respect to k is simply
        
        .. math::
            \frac{\partial \mathcal{D}}{\partial k} = \frac{2c^2 k}{\omega^2}
            
        """
        k = np.array(k)
        return 2*c*c*k/(self._omega*self._omega)
        
    def pDpw(self, x, k, tol=1e-14):
        r""" Evaluate partial D over partial omega at given (x, k) coordinates
        
        for O-mode, it's simple:
        
        .. math::
            \frac{\partial \mathcal{D}}{\partial \omega} = 
            -2\left(\frac{c^2k^2+\omega_{pe}^2}{\omega^3}\right)
            
        for X-mode, after some algebra, we get:
        
        .. math::
            
        """
        k = np.array(k)
        if self._polarization=='O':
            omega_pe2 = 4*np.pi*e*e*self._plasma.get_ne(x)/m_e
            return -2*(c*c*np.sum(k*k)+omega_pe2)/self._omega**3
            
        elif self._polarization=='X':
            #TODO complete X-mode derivative
            pass
        
    def pDpx(self, x, k, tol=1e-14):
        r""" Evaluate partial D over partial x at given (x, k) coordinates
        
        for O-mode, it's simple:
        
        .. math::
            \frac{\partial \mathcal{D}}{\partial x} = 
                     -\frac{\partial P}{\partial x}
            
            
        for X-mode, after some algebra, we get:
        
        .. math::
        """
        if self._polarization=='O':
            return -self._dPdx(x)
        elif self._polarization=='X':
            omega_ce = e*self._plasma.get_B(x)/(m_e*c)
            if np.abs(self._omega - omega_ce) < tol:
                raise ResonanceError('Cold X resonance happens, S goes to \
infinity at {0}.'.format(x))
            omega_pe2 = 4*np.pi*e*e*self._plasma.get_ne(x)/m_e
            omega2_m_omegace2 = self._omega**2-omega_ce**2
            S = 1-omega_pe2/(omega2_m_omegace2)
            D = -omega_pe2*omega_ce/(self._omega*omega2_m_omegace2)
            if np.abs(S)<tol:
                raise ResonanceError('Cold hybrid resonance happens, S goes to\
 0 at {0}.'.format(x))
            return ((S*S-D*D)/(S*S)-2)*self._dSdx(x) + 2*D/S*self._dDdx(x)
            
        
class RayTracer(object):
    r"""class for solver of ray tracing equations
    
    Starting from (x0, k0), the trajectory of the wave in phase space is traced
    via equations:
    
    .. math::
        \frac{dx}{dt} = \frac{\partial \mathcal{D}/\partial k}
                         {\partial \mathcal{D}/\partial \omega}
    
    .. math::
        \frac{dk}{dt} = -\frac{\partial \mathcal{D}/\partial x}
                         {\partial \mathcal{D}/\partial \omega}
                         
    These first order differential equations are integrated via 
    :py:func:`scipy.integrate.odeint<scipy.integrate.odeint>`.
    """
    
    def __init__(self, plasma, omega, polarization='O'):
        self._dispersion_derivative = ColdDispersionDerivatives(plasma, omega, 
                                                                polarization)
        
    #TODO set up ray tracing function
                
            
            


