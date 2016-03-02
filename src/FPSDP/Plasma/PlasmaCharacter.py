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
        

References:

.. [NRL09]:
    NRL Plasma Formulary, Revised 2009, Naval Research Laboratory.

Created on Wed Mar 02 15:23:04 2016

@author: Lei Shi
"""

import numpy as np
from scipy.special import gammainc

from FPSDP.GeneralSettings.UnitSystem import cgs


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
    m = cgs['m_e']
    e = cgs['e']    
    
    if ve is None:
        ve = np.sqrt(Te*2/m)
        
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
    m = cgs['m_e']
    if ve is None:
        ve = np.sqrt(Te*2/m)
    
    x = m*ve*ve/(2*Te)
    psi = gammainc(1.5, x)
    
    return 2*psi*nu_ee(ne, Te, ve)
           
           
    
    

