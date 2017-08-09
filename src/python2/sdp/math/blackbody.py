# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 23:09:23 2016

@author: lei

Black Body radiation formula


"""
import numpy as np

from ..settings.unitsystem import cgs

hbar = cgs['hbar']
c = cgs['c']

def planck_formula(omega, T):
    r""" The black body radiation formula given by Planck
    """
    omega = np.array(omega)
    T = np.array(T)
    assert omega.ndim < 2, 'only 1D array of omega is allowed.'
    assert T.ndim < 2, 'only 1D array of T is allowed.'
    # hbar/(8*pi^3*c^2) is just a constant
    C = hbar/(8*np.pi**3*c**2)
    if (omega.ndim == 1 and T.ndim == 1):
        aomega = omega[:,np.newaxis]
        aT = T[np.newaxis, :]
        # 2D array will be returned so omega and T dimensions are all included.
        return C * aomega**3 / (np.exp(hbar*aomega/aT)-1)
    else:
        # at most one input is an array, natural broadcasting is sufficient
        return C * omega**3 / (np.exp(hbar*omega/T)-1)
