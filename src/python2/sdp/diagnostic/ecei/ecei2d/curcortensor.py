# -*- coding: utf-8 -*-
"""
Provides various formulations for source current correlation tensor used in 
calculation of electron cyclotron emmision power, using Reciprocity Theorem
in [piliya02]_.

The isotropic Maxwellian plasma's source current correlation tensor can be 
ralated to the anti-Hermitian part of the dielectric tensor by Kirchhoff's Law 
of Radiation [shi16]_. 

Non-isotropic Maxwellian and non-relativistic case is derived in [shi16]_ as 
well.

See the corresponding classes' docstring for detailed formulism.

Reference:
**********

.. [piliya02] On application of the reciprocity theorem to calculation of a 
              microwave radiation signal in inhomogeneous hot magnetized 
              plasmas, A D Piliya and A Yu Popov, Plamsa Phys. Controlled 
              Fusion 44(2002) 467-474
.. [shi16] Development of Fusion Plamsa Synthetic diagnostic Platform and a 2D
           synthetic electron cyclotron emission imaging module, L. Shi, Ph.D. 
           dissertation (2016), Princeton Plasma Physics Laboratory, Princeton
           University.
           
Created on Mon Mar 07 14:28:10 2016

@author: lei
"""

from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

import numpy as np
from scipy.special import iv, ivp

from ....plasma.profile import PlasmaProfile, ECEI_Profile
from ....plasma.dielectensor import HotSusceptibility
from ....math.pdf import Z
from ....settings.unitsystem import UnitSystem, cgs
from ....settings.exception import ModelInvalidError


class SourceCurrentCorrelationTensor(object):
    r"""Abstract base class for all source current correlation tensors
    
    Derived classes must sustantiate following methods/properties:
    
        __call__(coordinates, omega, k_para, k_perp): 
            :math:`\hat{K}_k` tensor in shape (3,3, Nf, Nk_para, Nk_perp, 
            spatial_shape)
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __call__(self, coordinates, omega, k_para, k_perp):
        r"""return :math:`\hat{K}_k` tensor
        
        :param coordinates: Spatial coordinates in order [Y, X]
        :type coordinates: list of ndarrays, all have same shape 
                           (spatial_shape), depends on plasma's dimension, 
                           length of the list may be 1 or 2. 
        :param omega: wave frequencies, must be positive
        :type omega: 1darray of floats, shape (Nf,)
        :param k_para: parallel wave vector
        :type k_para: 1darray of floats, shape (Nk_para,)
        :param k_perp: perpendicular wave vector
        :type k_perp: 1darray of floats, shape (Nk_perp,)
        
        :return: :math:`\hat{K}_k` tensor
        :rtype: ndarray of shape (3,3, Nf, Nk_para, Nk_perp, spatial_shape)
        """
        pass
    

class IsotropicMaxwellian(SourceCurrentCorrelationTensor):
    r"""Isotropic Maxwellian plasma's source current correlation tensor
    
    In isotropic and locally Maxwellian plasma, emission and absorption are 
    related by Kirchhoff's Law of Radiation. It is shown in [shi16]_ that this
    relation leads to the expression of source current correlation tensor:
    
    .. math::
    
        \hat{K}_k = \frac{\omega T_e}{\pi^2}  \chi_e^a
        
    where
    
    .. math::
        \chi_e^a \equiv \frac{1}{2i} (\chi_e - \chi_e^\textdagger)
        
    is the anti-Hermitian part of electron's susceptibility tensor 
    :math:`\chi_e`.
    
    This expression is valid for both weakly relativistic and non-relativistic
    electrons.
    
    We will then use the existing susceptibility tensor module to calculate 
    each tensor element.

    Initialization
    ***************
    Initiated with a plasma profile object, and a susceptibility tensor object.
    
    __init__(plasma, suscept_class):
        plasma: :py:class:`sdp.plasma.PlasmaProfile.PlasmaProfile` object
        suscept_class: derived class of 
                       :py:class:`sdp.plasma.DielectricTensor.Susceptibility`
        
    """
    def __init__(self, plasma, suscept_class, max_harmonic=4, max_power=4):
        """Initialization with plasma and susceptibility class
        """    
        assert isinstance(plasma, PlasmaProfile)
        assert issubclass(suscept_class, HotSusceptibility)
        self.plasma = plasma
        self.suscept = suscept_class(plasma, 'e', max_harmonic=max_harmonic,
                                     max_power=max_power)
        
    def __call__(self, coordinates, omega, k_para, k_perp, eq_only=True, 
                 time=0, tol=1e-14):
        r"""Return :math:`\hat{K}_k` tensor
        """
        
        Te = self.plasma.get_Te(coordinates, eq_only=eq_only, 
                                perpendicular=True, time=time)
                                
        chi_e = self.suscept(coordinates, omega, k_para, k_perp, eq_only, time,
                             tol)
        trans_index = np.arange(chi_e.ndim)
        trans_index[0]=1
        trans_index[1]=0
        chi_ea = (chi_e - np.conj(np.transpose(chi_e, axes=trans_index)))/2j
        
        return omega*Te*chi_ea/np.pi**2 
        

class AnisotropicNonrelativisticMaxwellian(SourceCurrentCorrelationTensor):
    r"""Current Correlation Tensor for anisotropic Maxwellian electrons
    
    Initialization
    ***************
    
    __init__(plasma, max_harmonic)
    
    :param plasma: plasma profile, supposed to be anisotropic in electron 
                   velocity distribution
    :type plasma: :py:class:`sdp.plasma.PlasmaProfile.ECEIProfile`
    :param int max_harmonic: maximum n included in the following formula
    
    Formula
    *******
    
    The Current correlation tensor :math:`\hat{K}_k` for Nonrelativistic 
    Maxwellian electrons is given as [shi16]_:
    
    .. math::
        \hat{K}_k = 4\omega_{pe}^2 T_\perp  \sum\limits_{n=-\infty}^{+\infty} 
                    {\rm e}^{-\lambda} a_n \mathbf{Y_n}(\lambda),

    where
    
    .. math::
        \mathbf{Y_n}(\lambda) \equiv 
            \begin{pmatrix}
            \frac{n^2}{\lambda}I_n & -{\rm i} n \left( I_n - I_n' \right) & 
            -n \sqrt{\frac{2}{\lambda}} I_n b_n \\
            {\rm i} n \left( I_n - I_n' \right) & \left( \frac{n^2 I_n(\lambda)}
            {\lambda} + 2\lambda I_n - 2\lambda I_n' \right) & 
            -{\rm i} \sqrt{ 2\lambda}  \left( I_n - I_n' \right) b_n  \\
            -n \sqrt{\frac{2}{\lambda}} I_n b_n  & 
            {\rm i} \sqrt{ 2\lambda}  \left( I_n - I_n' \right) b_n  & 
            2 I_n  b_n^2
            \end{pmatrix},

    and

    .. math::
        \begin{split}
        & a_n \equiv \frac{Im(Z_0(\zeta_n))}{k_\parallel v_{\parallel th}} , \\
        & b_n \equiv \frac{\omega + n|\omega_{ce}|}{k_\parallel v_{\perp th}}.
        \end{split}
        
    The symbols are defined:
    
    :math:`I_n \equiv I_n(\lambda)` is the Modified Bessel's Function of the 
    first kind, with argument :math:`\lambda \equiv k_\perp^2v_{\perp th}^2/2
    \omega_{ce}^2`. :math:`v_{th} \equiv 2T/m` for both parallel and 
    perpendicular temperature. :math:`Z_0(\zeta_n)` is the Plasma Dispersion 
    Function with argument :math:`\zeta_n \equiv(\omega + n|\omega_{ce}| - 
    k_\parallel V)/k_\parallel v_{\parallel th}`, and :math:`V` the parallel 
    mean flow velocity. 
    
    Reference
    *********
    
    .. [shi16] Development of Fusion Plamsa Synthetic diagnostic Platform and
               a 2D synthetic electron cyclotron emission imaging module, 
               L. Shi, Ph.D. dissertation (2016), Princeton Plasma Physics 
               Laboratory, Princeton University.
    """
    
    def __init__(self, plasma, max_harmonic=4):
        """
        :param plasma: plasma profile, supposed to be anisotropic in electron 
                   velocity distribution
        :type plasma: :py:class:`sdp.plasma.PlasmaProfile.ECEIProfile`
        :param int max_harmonic: maximum n included in the following formula
        """
        assert isinstance(plasma, ECEI_Profile)
        
        self.plasma = plasma
        self.max_harmonic = max_harmonic
    
    
    def __call__(self, coordinates, omega, k_para, k_perp, 
                 eq_only=True, time = 0, tol=1e-14):
        """Calculates non-relativistic source current correlation tensor at 
        each coordinate given by coordinates.
        
        :param coordinates: spatial coordinates given in the order of (Z,Y,X) 
                            for 3D or (Z,R) for 2D
        :type coordinates: list of array_like
        :param float gamma: the parameter for equation of state, default to be
                            3, i.e. adiabatic process for 1 degree of freedom
        :param omega: frequencies of the waves under study
        :type omega: array of float with shape (Nf, )
        :param k_para: parallel wave vectors of the waves under study
        :type k_para: array of float with shape (Nk_para, )
        :param k_perp: perpendicular wave vectors. 
        :type k_perp: array of float with shape (Nk_perp, )
        :param eq_only: if True, only equilibrium quantities in ``plasma`` will
                        be used.
        :type eq_only: bool
        :param int time: time step for perturbation loading
        :param float tol: the tolerance for determining a float is zero, used
                          to check if resonance is happening. 
                          Default to be 1e-14
        
        :return: source current correlation tensor at each point
        :rtype: ndarray of complex, shape (3,3, Nf, Nk_para, Nk_perp 
                spatial_shape)
                
        :raise ModelInvalidError: if :math:`k_\parallel v_{th,\parallel}` is 
                                  too small. 
        """        
        assert len(coordinates) == self.plasma.grid.dimension
        
        # prepare the result array with the right shape        
        omega = np.array(omega)
        k_para = np.array(k_para)
        k_perp = np.array(k_perp)
        coordinates = np.array(coordinates)
        result_shape = []
        frequency_shape = list(omega.shape)
        wave_vector_para_shape = list(k_para.shape)
        wave_vector_perp_shape = list(k_perp.shape)
        spatial_shape = list(coordinates[0].shape)
        result_shape.extend([3,3])
        result_shape.extend(frequency_shape)
        result_shape.extend(wave_vector_para_shape)
        result_shape.extend(wave_vector_perp_shape)        
        result_shape.extend(spatial_shape)
        result = np.zeros(result_shape, dtype='complex')
        
        # entend frequency array's dimension containing wave vector and spatial
        # dimensions, so it can be directly broadcasted with plasma profile 
        # quantities.
        
        f_dim = len(frequency_shape)
        sp_dim = len(spatial_shape)
        wv_para_dim = len(wave_vector_para_shape)
        wv_perp_dim = len(wave_vector_perp_shape)
        # final dimension equals frequency + wave_vector + spatial dimension
        full_f_shape = frequency_shape
        full_f_shape.extend([1 for i in range(sp_dim + wv_para_dim + \
                                              wv_perp_dim)]) 
        omega = omega.reshape(full_f_shape)
        
        # same dimension expansion for wave vector array.         
        
        full_k_para_shape = []
        full_k_para_shape.extend([1 for i in range(f_dim)])
        full_k_para_shape.extend(wave_vector_para_shape)
        full_k_para_shape.extend([1 for i in range(wv_perp_dim)])
        full_k_para_shape.extend([1 for i in range(sp_dim)])
        k_para = k_para.reshape(full_k_para_shape)
        
        full_k_perp_shape = []
        full_k_perp_shape.extend([1 for i in range(f_dim)])
        full_k_perp_shape.extend([1 for i in range(wv_para_dim)])
        full_k_perp_shape.extend(wave_vector_perp_shape)
        full_k_perp_shape.extend([1 for i in range(sp_dim)])
        k_perp = k_perp.reshape(full_k_perp_shape)
        
        # now we calculate the tensor
        # first, get all particle quantities
                
        c = self.plasma.unit_system['c']
        q = -self.plasma.unit_system['e']
        m = self.plasma.unit_system['m_e']
        pi = np.pi
        
        # profile quantities
        if(eq_only == False):
            n = self.plasma.get_ne(coordinates, False, time=time)
            B = self.plasma.get_B(coordinates, False, time=time)
            # need to use parallel Te perturbation here
            T_para = self.plasma.get_Te(coordinates, eq_only=False, 
                                        perpendicular=False, time=time)
            T_perp = self.plasma.get_Te(coordinates, eq_only=False,
                                        perpendicular=True, time=time)
            try:
                V = self.plasma.get_Ve(coordinates, eq_only=False,
                                       time=time)
            except AttributeError:
                V = 0
        else:
            n = self.plasma.get_ne(coordinates, True)
            B = self.plasma.get_B(coordinates, True)
            T_para = self.plasma.get_Te0(coordinates)
            T_perp = T_para
            try:
                V = self.plasma.get_Ve(coordinates, eq_only=True)
            except AttributeError:
                V = 0
                
        # Calculate physical quantities used in the formula
        
        w_perp2 = 2*T_perp/m
        w_perp = np.sqrt(w_perp2)
        
        w_para2 = 2*T_para/m
        w_para = np.sqrt(w_para2)       
        
        res_width = k_para*w_para
        # check if k_para*w_para is too small for non-relativistic model to be
        # good, also eliminates the potential zero denominator problems
    
        if np.any(res_width < omega*(T_para+T_perp)/(2*m*c*c)) or \
        np.any(res_width < tol):
            raise  ModelInvalidError('k_para*w_para is too small. \
Non-relativistic model may not be valid. Try relativistic models instead.')    
        
        abs_Omega = np.abs(q*B/(m*c))
        lambd = k_perp*k_perp * w_perp2 /(2*abs_Omega*abs_Omega)
        
        # Now, calculate current correlation tensor elements order by order.  
        # We treat positive and negative n together as the same order.
        
        # We also leave the constant coefficients outside the summation, and 
        # multiply them afterwards
        for i in range(self.max_harmonic+1):
            # note that I_n = I_-n, so no need to calculate -n terms           
            I = iv(i,lambd)
            I_p = ivp(i,lambd, 1) 
            
            # first calculate positive i part
            res = (omega - k_para*V + i*abs_Omega)
                
            zeta = res / res_width            
            Ai =  np.imag(Z(zeta)) / res_width
                       
            Bi = (omega + i*abs_Omega)/(k_para * w_perp)
                
            result[0,0] += i*i*I*Ai
            result[1,1] += ((i*i/lambd + 2*lambd)*I - 2*lambd*I_p)*Ai
            result[2,2] += 2*I*Bi*Bi*Ai
            result[0,1] += -1j*i*(I-I_p)*Ai
            result[0,2] += -i*I*Bi*Ai
            result[1,2] += -1j*(I-I_p)*Bi*Ai
            
            if (i != 0):
                # now, negative i part
                i = -i
            
                res = (omega - k_para*V + i*abs_Omega)
                
                zeta = res / res_width            
                Ai =  np.imag(Z(zeta)) / res_width
                           
                Bi = (omega + i*abs_Omega)/(k_para * w_perp)
                    
                result[0,0] += i*i*I*Ai
                result[1,1] += ((i*i/lambd + 2*lambd)*I - 2*lambd*I_p)*Ai
                result[2,2] += 2*I*Bi*Bi*Ai
                result[0,1] += -1j*i*(I-I_p)*Ai
                result[0,2] += -i*I*Bi*Ai
                result[1,2] += -1j*(I-I_p)*Bi*Ai
        
        # now, multiply with each common factors            
        result[0,0] /= lambd
        result[0,2] *= np.sqrt(2/lambd)
        result[1,2] *= np.sqrt(2*lambd)
        
        # fill in the lower half off-diagonal elements
        result[1,0] = -result[0,1]
        result[2,0] = result[0,2]
        result[2,1] = -result[1,2]
        
        # multiply with the all common factor omega_pe^2/omega * exp(-lambd)
        result *= 4*pi*n*q*q/m * T_perp* np.exp(-lambd) /(np.pi*np.pi)
        
        return result
    