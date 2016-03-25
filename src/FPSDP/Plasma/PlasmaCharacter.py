# -*- coding: utf-8 -*-
r"""
Contain simple functions evaluating characteristic plasma parameters.
cgs units only.

Class contained:
    
    PlasmaCharPoint:
        Contain evaluators for plasma characteristics at given set of plasma 
        parameter.
    PlasmaCharProfile:
        Contain evaluators for plasma characteristics on given plasma profile
        
Functions contained:
    
    (formula is based on [NRL09]_ Chap. Collisions and Transport)

    lambd_ee:
        Coulomb Logarithm of thermal electron-electron collision. 
    nu_ee:
        electron-electron collisional rate. base rate for all types. Denoted as
        nu_0 in [NRL09]_
    nu_ee_s:
        slowing down rate 
    nu_ee_perp:
        transverse diffusion rate
    nu_ee_para:
        parallel diffusion rate
    nu_ee_eps:
        energy loss rate    
    
    (formula based on [Stix92]_)  

    omega_pe:
        electron plasma frequency
    omega_ce:
        electron cyclotron frequency
    omega_R:
        R_wave cutoff frequency
    omega_L:
        L_wave cutoff frequency
    omega_UH:
        Upper hybrid resonance frequency
    *omega_lower(Not Implemented):
        Lower hybrid resonance frequency
        

References:

.. [NRL09]:
    NRL Plasma Formulary, Revised 2009, Naval Research Laboratory.

.. [Stix92] "Waves in Plasmas", Chapter 1-3, T.H.Stix, 1992, American Inst. of 
            Physics
       
Created on Wed Mar 02 15:23:04 2016

@author: Lei Shi
"""

import numpy as np
from scipy.special import gammainc

from FPSDP.GeneralSettings.UnitSystem import cgs


# Some common constants all functions will use
# electron rest mass
m = cgs['m_e']
# elementary charge
e = cgs['e']
# vacuum speed of light
c = cgs['c']

def lambd_ee(ne, Te):
    r"""Coulomb Logarithm for thermal electron-electron collision

    .. math::
        
        \lambda_{ee} = 23.5 - {\rm ln}(n_e^{1/2} T_e^{-4/5} - 
            [10^-5 + ({\rm ln}T_e - 2)^2/16]^{1/2} 
    
    This formula is using cgs units, while Te is in eV. However, this function
    receives Te in standard cgs energy unit, i.e. erg.
    
    :param ne: electron density in cm^-3
    :type ne: ndarray of floats
    :param Te: electron temperature in erg
    :type Te: ndarray of floats, must have same shape as ``ne``
    
    :return: :math:`\lambda_ee`
    :rtype: ndarray of floats, same shape as ``ne`` and ``Te``    
    """
    ne = np.array(ne)
    Te = np.array(Te)
    assert ne.shape == Te.shape
    # convert Te into eV
    Te = Te / cgs['keV'] *1e3
    
    return 23.5 - np.log(np.sqrt(ne) * Te**(-1.25)) - \
           np.sqrt(1e-5 + (np.log(Te)-2)**2 / 16)
           

def nu_ee(ne, Te, ve=None):
    r"""Calculate base electron-electron collision rate
    
    .. math::
        
        \nu_0^{ee} = \frac{4\pi e^4 \lambda_{ee} n_e}{m_e^2 v_e^3}
        
    where :math:`\lambda_{ee}` is the Coulomb Logarithm between electrons, 
    :math:`v_e` is the test electron speed. Other notations have normal 
    meaning.
    
    :param ne: electron density in cm^-3
    :type ne: ndarray of floats
    :param Te: electron temperature in erg
    :type Te: ndarray of floats, must have same shape as ``ne``
    :param ve: test electron speed in cm/s. Optional, if not given, thermal 
               electron speed corresponding to ``Te`` is used.
    :type ve: ndarray of floats, same shape as ``ne`` or None.
    
    :return: base electron electron collision rate
    :rtype: ndarray of floats, same shape as ``ne`` and ``Te``.
    """   
    ne = np.array(ne)
    Te = np.array(Te)    
    if ve is None:
        ve = np.sqrt(Te*2/m)
    else:
        ve = np.array(ve)
        
    return 4*np.pi*e**4 * lambd_ee(ne, Te) * ne / (m*m*ve**3)
    

def nu_ee_s(ne, Te, ve=None):
    r"""Calculate electron-electron collision slowing down rate
    
    .. math::
    
        \nu_s^{ee} = 2\psi(x^{ee}) \nu_0^{ee}
        
    where 
    
    .. math::
    
        \psi(x) \equiv \frac{2}{\sqrt{\pi}} \int_0^x dt \, t^{1/2} e^{-t}
        
    and :math:`x^{ee} \equiv m_e v_e^2 / 2T_e`. 
    
    :math:`\psi(x)` is also known as the incomplete gamma function 
    :math:`\gamma(s, x)` with parameter s=3/2.
    
    Here all quantities are in cgs unit, including :math:`T_e`, which is in 
    erg.
    
    :param ne: electron density in cm^-3
    :type ne: ndarray of floats
    :param Te: electron temperature in erg
    :type Te: ndarray of floats, must have same shape as ``ne``
    :param ve: test electron speed in cm/s. Optional, if not given, thermal 
               electron speed corresponding to ``Te`` is used.
    :type ve: ndarray of floats, same shape as ``ne`` or None.
    
    :return: electron electron collision slowing down rate
    :rtype: ndarray of floats, same shape as ``ne`` and ``Te``.
    """
    ne = np.array(ne)
    Te = np.array(Te)    
    if ve is None:
        ve = np.sqrt(Te*2/m)
    else:
        ve = np.array(ve)
    
    x = m*ve*ve/(2*Te)
    psi = gammainc(1.5, x)
    
    return 2*psi*nu_ee(ne, Te, ve)
    
def nu_ee_perp(ne, Te, ve=None):
    r"""Calculate electron-electron collision transverse diffusion rate
    
    .. math::
    
        \nu_\perp^{ee} = 2[(1-\frac{1}{2x^{ee}})\psi(x^{ee}) + \psi'(x^{ee})] 
                     \nu_0^{ee}
        
    where 
    
    .. math::
    
        \psi(x) \equiv \frac{2}{\sqrt{\pi}} \int_0^x dt \, t^{1/2} e^{-t}
        
    and :math:`x^{ee} \equiv m_e v_e^2 / 2T_e`. 
    
    :math:`\psi(x)` is also known as the incomplete gamma function 
    :math:`\gamma(s, x)` with parameter s=3/2.
    
    Here all quantities are in cgs unit, including :math:`T_e`, which is in 
    erg.
    
    :param ne: electron density in cm^-3
    :type ne: ndarray of floats
    :param Te: electron temperature in erg
    :type Te: ndarray of floats, must have same shape as ``ne``
    :param ve: test electron speed in cm/s. Optional, if not given, thermal 
               electron speed corresponding to ``Te`` is used.
    :type ve: ndarray of floats, same shape as ``ne`` or None.
    
    :return: electron electron collision slowing down rate
    :rtype: ndarray of floats, same shape as ``ne`` and ``Te``.
    """
    # square root of pi
    sqrt_pi = 1.7724538509055159
    
    ne = np.array(ne)
    Te = np.array(Te)    
    if ve is None:
        ve = np.sqrt(Te*2/m)
    else:
        ve = np.array(ve)
    
    x = m*ve*ve/(2*Te)
    psi = gammainc(1.5, x)
    psip = 2/sqrt_pi * np.sqrt(x) * np.exp(-x)
    
    return 2*((1-0.5/x)*psi + psip)*nu_ee(ne, Te, ve)


def nu_ee_para(ne, Te, ve=None):
    r"""Calculate electron-electron collision parallel diffusion rate
    
    .. math::
    
        \nu_\parallel^{ee} = \[psi(x^{ee})/x^{ee}] \nu_0^{ee}
        
    where 
    
    .. math::
    
        \psi(x) \equiv \frac{2}{\sqrt{\pi}} \int_0^x dt \, t^{1/2} e^{-t}
        
    and :math:`x^{ee} \equiv m_e v_e^2 / 2T_e`. 
    
    :math:`\psi(x)` is also known as the incomplete gamma function 
    :math:`\gamma(s, x)` with parameter s=3/2.
    
    Here all quantities are in cgs unit, including :math:`T_e`, which is in 
    erg.
    
    :param ne: electron density in cm^-3
    :type ne: ndarray of floats
    :param Te: electron temperature in erg
    :type Te: ndarray of floats, must have same shape as ``ne``
    :param ve: test electron speed in cm/s. Optional, if not given, thermal 
               electron speed corresponding to ``Te`` is used.
    :type ve: ndarray of floats, same shape as ``ne`` or None.
    
    :return: electron electron collision slowing down rate
    :rtype: ndarray of floats, same shape as ``ne`` and ``Te``.
    """
    ne = np.array(ne)
    Te = np.array(Te)    
    if ve is None:
        ve = np.sqrt(Te*2/m)
    else:
        ve = np.array(ve)
    
    x = m*ve*ve/(2*Te)
    psi = gammainc(1.5, x)
    
    return psi/x * nu_ee(ne, Te, ve)


def nu_ee_eps(ne, Te, ve=None):
    r"""Calculate electron-electron collision energy loss rate
    
    .. math::
    
        \nu_\epsilon^{ee} = 2[\psi(x^{ee}) - \psi'(x^{ee})] \nu_0^{ee}
        
    where 
    
    .. math::
    
        \psi(x) \equiv \frac{2}{\sqrt{\pi}} \int_0^x dt \, t^{1/2} e^{-t}
        
    and :math:`x^{ee} \equiv m_e v_e^2 / 2T_e`. 
    
    :math:`\psi(x)` is also known as the incomplete gamma function 
    :math:`\gamma(s, x)` with parameter s=3/2.
    
    Here all quantities are in cgs unit, including :math:`T_e`, which is in 
    erg.
    
    :param ne: electron density in cm^-3
    :type ne: ndarray of floats
    :param Te: electron temperature in erg
    :type Te: ndarray of floats, must have same shape as ``ne``
    :param ve: test electron speed in cm/s. Optional, if not given, thermal 
               electron speed corresponding to ``Te`` is used.
    :type ve: ndarray of floats, same shape as ``ne`` or None.
    
    :return: electron electron collision slowing down rate
    :rtype: ndarray of floats, same shape as ``ne`` and ``Te``.
    """
    # square root of pi
    sqrt_pi = 1.7724538509055159
    
    ne = np.array(ne)
    Te = np.array(Te)    
    if ve is None:
        ve = np.sqrt(Te*2/m)
    else:
        ve = np.array(ve)
    
    x = m*ve*ve/(2*Te)
    psi = gammainc(1.5, x)
    psip = 2/sqrt_pi * np.sqrt(x) * np.exp(-x)
    
    return 2*(psi - psip)*nu_ee(ne, Te, ve)
           
           
def omega_pe(ne):
    """electron plasma frequency
    
    .. math::
        \omega_{pe}^2 \equiv \frac{4\pi e^2 n_e}{m_e}
        
    Gaussian unit is assumed.
    """
    ne = np.array(ne)
    return np.sqrt(4*np.pi*e*e*ne/m)

def omega_ce(B):
    """non-relativistic electron cyclotron frequency
    
    .. math::
        \omega_{ce} \equiv \frac{eB}{m_e c}
        
    Gaussian unit is assumed. Positive frequency is returned.
    """           
    return e*np.abs(B)/(m*c)
    
def omega_R(ne, B):
    """cold plasma R wave cutoff.
    
    The frequency is defined as the positive root of the R-wave dispersion
    
    .. math::
        R \equiv 1 - \frac{\omega^2_{pe}}{\omega(\omega - |\omega_{ce}|)}
        
    the root is easily found:
    
    .. math::
        \omega_R = \frac{|omega_{ce}|+\sqrt(omega_{ce}^2 + 4\omega_{pe}^2)}{2}
        
    Gaussian unit is assumed. 
    """  
    ne = np.array(ne)
    B = np.array(B)
    omega_pe2 = 4*np.pi*e*e*ne/m
    omega_ce = e*np.abs(B)/(m*c)
    return (omega_ce + np.sqrt(omega_ce*omega_ce + 4*omega_pe2))/2
    
def omega_L(ne, B):
    """cold plasma L wave cutoff.
    
    The frequency is defined as the positive root of the L-wave dispersion
    
    .. math::
        L \equiv 1 - \frac{\omega^2_{pe}}{\omega(\omega + |\omega_{ce}|)}
        
    the root is easily found:
    
    .. math::
        \omega_L = \frac{-|omega_{ce}|+\sqrt(omega_{ce}^2 + 4\omega_{pe}^2)}{2}
        
    Gaussian unit is assumed. 
    """  
    ne = np.array(ne)
    B = np.array(B)
    omega_pe2 = 4*np.pi*e*e*ne/m
    omega_ce = e*np.abs(B)/(m*c)
    return (-omega_ce + np.sqrt(omega_ce*omega_ce + 4*omega_pe2))/2
    
def omega_UH(ne, B):
    """upper hybrid resonance frequency
    
    .. math::
        \omega_{UH} = \sqrt(\omega_{ce}^2 + \omega_{pe}^2)
        
    Gaussian unit is assumed.
    """  
    ne = np.array(ne)
    B = np.array(B)
    omega_pe2 = 4*np.pi*e*e*ne/m
    omega_c = e*np.abs(B)/(m*c)
    return np.sqrt(omega_pe2 + omega_c*omega_c)
    

########################################################
# Classes for diagnosing PlasmaProfile
########################################################

class PlasmaCharPoint(object):
    """Plasma Characteristic evaluators for given set of plasma parameter
    
    Only electrons quantities are included. Classes including ions can be 
    derived from this class.
    
    Initialization
    **************
    
    :param ne: electron density
    :type ne: array_like of floats    
    :param B: magnetic field strength
    :type B: array_like of floats
    :param Te: Optional, electron temperature, only used for collision rate 
               calculation.
    :type Te: array_like of floats
               
    properties
    **********
    
    omega_ce: 
        non-relativistic electron cyclotron frequency, in radian/s
    omega_pe: 
        electron plasma frequency, in radian/s
    omega_R: 
        R wave cutoff frequency, in radian/s
    omega_L: 
        L wave cutoff frequency, in radian/s
    omega_UH: 
        upper hybrid resonance frequency, in radian/s
    
    nu_ee_s: 
        electron-electron collision slowing down rate
    nu_ee_perp: 
        electron-electron collision transverse diffusion rate
    nu_ee_para: 
        electron-electron collision parallel diffusion rate
    nu_ee_eps: 
        electron-electron collision energy loss rate
    
    
    """
    def __init__(self, ne, B, Te=None):
        self.ne = np.array(ne)
        self.B = np.array(B)
        if Te is None:
            pass
        else:
            self.Te = Te
    
    @property    
    def omega_ce(self):
        return omega_ce(self.B)
        
    @property
    def omega_pe(self):
        return omega_pe(self.ne)
        
    @property
    def omega_R(self):
        return omega_R(self.ne, self.B)
        
    @property
    def omega_L(self):
        return omega_L(self.ne, self.B)
        
    @property
    def omega_UH(self):
        return omega_UH(self.ne, self.B)
        
    @property
    def nu_ee_s(self):
        return nu_ee_s(self.ne, self.Te)
        
    @property
    def nu_ee_perp(self):
        return nu_ee_perp(self.ne, self.Te)
        
    @property
    def nu_ee_para(self):
        return nu_ee_para(self.ne, self.Te)
        
    @property
    def nu_ee_eps(self):
        return nu_ee_eps(self.ne, self.Te)
        

class PlasmaCharProfile(object):
    """Plasma Characteristic evaluators for given PlasmaProfile object
    
    Only electrons quantities are included. Classes including ions can be 
    derived from this class.
    
    Initialization
    **************
    
    :param plasma: 
        :class:`FPSDP.Plasma.PlasmaProfile.PlasmaProfile` object containing all
        relevant plasma profiles. Including ne, Te, B, and maybe ni, Ti.
        
    Methods
    *******
    
    set_coords(coordinates):
        set the coordinates of locations where all characteristic properties 
        will be evaluated at. Must be called before retrieving any properties.
        coordinates is in [Y2D, X2D] or [X1D] form for 2D or 1D PlasmaProfile 
        respectively.
               
    properties
    **********
    
    omega_ce: 
        non-relativistic electron cyclotron frequency, in radian/s
    omega_pe: 
        electron plasma frequency, in radian/s
    omega_R: 
        R wave cutoff frequency, in radian/s
    omega_L: 
        L wave cutoff frequency, in radian/s
    omega_UH: 
        upper hybrid resonance frequency, in radian/s
    
    nu_ee_s: 
        electron-electron collision slowing down rate
    nu_ee_perp: 
        electron-electron collision transverse diffusion rate
    nu_ee_para: 
        electron-electron collision parallel diffusion rate
    nu_ee_eps: 
        electron-electron collision energy loss rate
    
    
    """
    def __init__(self, plasma):
        self._plasma = plasma
        self._plasma.setup_interps()
        self.ne0 = plasma.ne0
        self.Te0 = plasma.Te0
        self.B0 = plasma.B0
        
    @property
    def plasma(self):
        return self._plasma
        
    @plasma.setter
    def plasma(self, p):
        self._plasma = p
        self.ne0 = p.ne0
        self.Te0 = p.Te0
        self.B0 = p.B0
        self._plasma.setup_interps()
        
    def set_coords(self, coordinates):
        """set the coordinates of locations where all characteristic properties 
        will be evaluated at. Must be called before retrieving any properties.
        
        :param coordinates: [Y2D, X2D] or [X1D] form for 2D or 1D PlasmaProfile 
                            respectively.
        :type cooridnates: list of arrays
        
        Attributes:
        
            ne0, Te0, B0: plasma parameters at given coordinates
            
        :raise ValueError: if any requested coordinates are outside the 
                           original profile's mesh.
        :raise AssertionError: if coordinates has different dimension as plasma
                               profile.
        """
        self._coords = np.asarray(coordinates)
        assert self._plasma.grid.dimension == self._coords.shape[0]
        
        self.ne0 = self._plasma.get_ne0(coordinates)
        self.Te0 = self._plasma.get_Te0(coordinates)
        self.B0 = self._plasma.get_B0(coordinates)
        
    
    @property    
    def omega_ce(self):
        return omega_ce(self.B0)
        
    @property
    def omega_pe(self):
        return omega_pe(self.ne0)
        
    @property
    def omega_R(self):
        return omega_R(self.ne0, self.B0)
        
    @property
    def omega_L(self):
        return omega_L(self.ne0, self.B0)
        
    @property
    def omega_UH(self):
        return omega_UH(self.ne0, self.B0)
        
    @property
    def nu_ee_s(self):
        return nu_ee_s(self.ne0, self.Te0)
        
    @property
    def nu_ee_perp(self):
        return nu_ee_perp(self.ne0, self.Te0)
        
    @property
    def nu_ee_para(self):
        return nu_ee_para(self.ne0, self.Te0)
        
    @property
    def nu_ee_eps(self):
        return nu_ee_eps(self.ne0, self.Te0)