# -*- coding: utf-8 -*-
r"""

Ray Tracing Model
******************

This module contains model for solving simple ray tracing equations:

.. math::
    \frac{dx}{dt} = \frac{\partial \omega}{\partial k}
                  =-\frac{\partial \mathcal{D}/\partial k}
                         {\partial \mathcal{D}/\partial \omega}

.. math::
    \frac{dk}{dt} = -\frac{\partial \omega}{\partial x}
                  = \frac{\partial \mathcal{D}/\partial x}
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
from ...plasma.profile import OutOfPlasmaError

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
    :param int time: time step index for plasma fluctuation quantities.
                     Default is None, equilibrium only.
    """

    def __init__(self, plasma, omega, polarization='O', equilibrium_only=True,
                 time=None):
        assert polarization in ['O', 'X']
        self._plasma = plasma
        # setup interpolators for later use
        self._plasma.setup_interps(equilibrium_only)

        self._omega = omega
        self._polarization = polarization
        self._eq_only = equilibrium_only
        self._time = time
        if not self._eq_only:
            assert self._time is not None, 'Time index is required for \
non-equilibrium plasma'

    def __str__(self):
        info = 'Omega : {0}\n'.format(self._omega)
        info += 'Polarization : {0}\n'.format(self._polarization)
        info += 'Eq_only : {0}\n'.format(self._eq_only)
        info += '(time : {0})\n\n'.format(self._eq_only)
        info += 'Plasma Info: \n{0}'.format(str(self._plasma))
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

        x = np.array(x, dtype=float)
        dx = np.array(dx, dtype=float)
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
                ne_plus[i] = self._plasma.get_ne(coords, eq_only=self._eq_only,
                                                 time=self._time)
            except ValueError:
                dx_plus[i] = 0
                ne_plus[i] = self._plasma.get_ne(x, eq_only=self._eq_only,
                                                 time=self._time)
            try:
                coords = np.copy(x)
                coords[i] -= dx[i]
                ne_minus[i] = self._plasma.get_ne(coords,eq_only=self._eq_only,
                                                  time=self._time)
            except ValueError:
                dx_minus[i] = 0
                ne_minus[i] = self._plasma.get_ne(x,eq_only=self._eq_only,
                                                  time=self._time)

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

        x = np.array(x, dtype='float')
        dx = np.array(dx, dtype='float')
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
                B_plus[i] = self._plasma.get_B(coords, eq_only=self._eq_only,
                                                 time=self._time)
            except ValueError:
                dx_plus[i] = 0
                B_plus[i] = self._plasma.get_B(x, eq_only=self._eq_only,
                                                 time=self._time)
            try:
                coords = np.copy(x)
                coords[i] -= dx[i]
                B_minus[i] = self._plasma.get_B(coords, eq_only=self._eq_only,
                                                 time=self._time)
            except ValueError:
                dx_minus[i] = 0
                B_minus[i] = self._plasma.get_B(x, eq_only=self._eq_only,
                                                 time=self._time)

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
        omega_pe2 = pe_const*self._plasma.get_ne(x, eq_only=self._eq_only,
                                                 time=self._time)
        omega_pe2_p = pe_const*self._dnedx(x)
        ce_const = e/(m_e*c)
        omega_ce = ce_const*self._plasma.get_B(x, eq_only=self._eq_only,
                                                 time=self._time)
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
        omega_pe2 = pe_const*self._plasma.get_ne(x, eq_only=self._eq_only,
                                                 time=self._time)
        omega_pe2_p = pe_const*self._dnedx(x)
        ce_const = e/(m_e*c)
        omega_ce = ce_const*self._plasma.get_B(x, eq_only=self._eq_only,
                                                 time=self._time)
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

        :param x: configuration coordinate(s) of the evaluation point
        :type x: list of floats, length equals the dimension of plasma
        :param k: wave-vector coordinate(s) of the evaluation point
        :type k: list of floats, length equals the dimension of plasma

        :return: pDpk
        :rtype: float

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
            \frac{\partial \mathcal{D}}{\partial \omega} =
            -\frac{2c^2k^2}{\omega^3}-\left(1+\frac{D^2}{S^2}\right)
            \frac{\partial S}{\partial \omega} +
            \frac{2D}{S}\frac{\partial D}{\partial \omega}

        where

        .. math::
            \frac{\partial S}{\partial \omega} =
            \frac{2\omega_{pe}^2\omega}{(\omega^2-\Omega_{ce}^2)^2}

        and

        .. math::
            \frac{\partial D}{\partial \omega}=
            \frac{\omega_{pe}^2\Omega_{ce}(3\omega^2-\Omega_{ce}^2)}
                 {\omega^2(\omega^2-\Omega_{ce}^2)^2}

        :param x: configuration coordinate(s) of the evaluation point
        :type x: list of floats, length equals the dimension of plasma
        :param k: wave-vector coordinate(s) of the evaluation point
        :type k: list of floats, length equals the dimension of plasma

        :return: pDpw
        :rtype: float

        """
        k = np.array(k)
        if self._polarization=='O':
            omega_pe2 =4*np.pi*e*e*self._plasma.get_ne(x,eq_only=self._eq_only,
                                                       time=self._time)/m_e
            return -2*(c*c*np.sum(k*k)+omega_pe2)/self._omega**3

        elif self._polarization=='X':
            omega2 = self._omega**2
            omega_pe2 =4*np.pi*e*e*self._plasma.get_ne(x,eq_only=self._eq_only,
                                                        time=self._time)/m_e
            omega_ce = e*self._plasma.get_B(x, eq_only=self._eq_only,
                                            time=self._time)/(m_e*c)
            omega2_m_omegace2 = omega2-omega_ce**2
            S = 1-omega_pe2/(omega2_m_omegace2)
            D = -omega_pe2*omega_ce/(self._omega*omega2_m_omegace2)
            if np.abs(S)<tol:
                raise ResonanceError('Cold hybrid resonance happens, S goes to\
 0 at {0}.'.format(x))
            pSpw = 2*omega_pe2*self._omega/omega2_m_omegace2**2
            pDpw = omega_pe2*omega_ce*(2*omega2 + omega2_m_omegace2)/\
                   (omega2 * omega2_m_omegace2**2)

            return -2*c**2*np.sum(k*k)/(omega2*self._omega) \
                   -(1+D*D/S*S)*pSpw + 2*D/S * pDpw




    def pDpx(self, x, k, tol=1e-14):
        r""" Evaluate partial D over partial x at given (x, k) coordinates

        for O-mode, it's simple:

        .. math::
            \frac{\partial \mathcal{D}}{\partial x} =
                     -\frac{\partial P}{\partial x}


        for X-mode, after some algebra, we get:

        .. math::
            \frac{\partial \mathcal{D}}{\partial x} =
            -\left(1+\frac{D^2}{S^2}\right)\frac{\partial S}{\partial x} +
            \frac{2D}{S}\frac{\partial D}{\partial x}

        :param x: configuration coordinate(s) of the evaluation point
        :type x: list of floats, length equals the dimension of plasma
        :param k: wave-vector coordinate(s) of the evaluation point
        :type k: list of floats, length equals the dimension of plasma

        :return: pDpx
        :rtype: float
        """
        if self._polarization=='O':
            return -self._dPdx(x)
        elif self._polarization=='X':
            omega_ce = e*self._plasma.get_B(x, eq_only=self._eq_only,
                                            time=self._time)/(m_e*c)
            if np.abs(self._omega - omega_ce) < tol:
                raise ResonanceError('Cold X resonance happens, S goes to \
infinity at {0}.'.format(x))
            omega_pe2 =4*np.pi*e*e*self._plasma.get_ne(x,eq_only=self._eq_only,
                                                 time=self._time)/m_e
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
        \frac{dx}{dt} = -\frac{\partial \mathcal{D}/\partial k}
                         {\partial \mathcal{D}/\partial \omega}

    .. math::
        \frac{dk}{dt} = \frac{\partial \mathcal{D}/\partial x}
                         {\partial \mathcal{D}/\partial \omega}

    These first order differential equations are integrated via
    :py:func:`scipy.integrate.odeint<scipy.integrate.odeint>`.

    Initialization
    ***************
    __init__(self, plasma, omega, polarization='O', eq_only=True,
             time=None)

    :param plasma: plasma quantities
    :type plasma: :py:class:`PlasmaProfile<sdp.plasma.profile.PlasmaProfile>`
    :param float omega: wave's circular frequency
    :param string polarization: either 'O' or 'X'
    :param bool eq_only: default is True, flag for using either only
                         equilibrium or with fluctuations in plasma.
    :param int time: time step index for fluctuations chosen. Required if
                     eq_only is False. Do not have effect if eq_only is True.

    Methods
    *******

    :py:method:`trace(self, x0, k0, t)`:
    Tracing the ray along the trajectory

    :param x0: starting point in configuration space
    :type x0: array-like of floats, 1D case also NEED to be an ARRAY
    :param k0: starting point in wave-vector space
    :type k0: array-like of floats, 1D case also NEED to be an ARRAY
    :param 1darray t: [t0, t1, ..., tn], solution will be given at these
                      time points. The first element should correspond to
                      the initial (x0, k0).

    :return: x(t), k(t) as an array
    :rtype: 2darray, shape (n, 2*dimension), n is the number of time points

    Example
    ********
    First we import the necessary modules
        >>> import sdp.model.wave.ray as ray
        >>> import sdp.plasma.analytic.testparameter as tp
    Create a test 2D plasma
        >>> p2d = tp.create_profile2D(fluctuation=True)
    Now we initialize the RayTracer object with our plasma and wave
    information.
        >>> omega = 4e11
        >>> tracer = ray.RayTracer(plasma=p2d, omega=omega, polarization='O',
                                   eq_only=False, time=0)
    Note that we have enabled the fluctuations at time step 0.
    Then we can run the ray tracing from a given starting point (x, k), note
    that these coordinates are all given in the order (Y, X), vertical
    direction is in front of radial direction.

    For example, we launch a wave from [10, 300], which means vertically 10cm
    above mid-plane, and radially at 350cm from the machine axis. The direction
    is purely radially inward, so k=[0, -k], where the k should be calculated
    from the wave frequency at the starting point. Normally, we choose starting
    point in vacuum, so:
        >>> k = omega/ray.c
    should calculate the wave vector properly.

    Let's try trace the light for roughly 60cm, then the total time should be
    more or less 60/c, and let's use 100 time steps
        >>> times = np.linspace(0, 60/ray.c, 100)
        >>> path = tracer.trace([10, 300], [0, -k], times)

    Now, ``path`` should contain the ray information. ``path[:][0]`` contains the
    vertical coordinates, and ``path[:][1]`` the radial ones, ``path[:][2:]``
    contains wave vector coordinates ky, and kx.
        >>> plt.scatter(path[:][1], path[:][0])
    should show the trajectory of the light as a scatter plot.
    """

    def __init__(self, plasma, omega, polarization='O', eq_only=True,
                 time=None):
        self.dimension = plasma.grid.dimension
        self._dispersion_derivative = ColdDispersionDerivatives(plasma, omega,
                                                                polarization)

    def _velocity(self, x, k):
        r""" Evaluate phase space velocity vector

        .. math::
            \frac{dx}{dt} = -\frac{\partial \mathcal{D}/\partial k}
                             {\partial \mathcal{D}/\partial \omega}

        .. math::
            \frac{dk}{dt} = \frac{\partial \mathcal{D}/\partial x}
                             {\partial \mathcal{D}/\partial \omega}

        The velocity is returned in shape (dx/dt, dk/dt), note that these can
        both be vectors when plasma is given in higher dimension configuration
        space.

        :param x: spatial coordinates of the location to be evaluated
        :type x: array_like of float, even if in 1-D space.
        :param k: wave vector coordinates of the location to be evaluated
        :type k: array_like of float, even if in 1-D space.

        :return: (dx/dt, dk/dt)
        :rtype: tuple of floats, length equals 2 times the dimension of plasma
        """
        pDpw = self._dispersion_derivative.pDpw(x, k)
        dxdt = -self._dispersion_derivative.pDpk(x, k) / pDpw
        dkdt = self._dispersion_derivative.pDpx(x, k) / pDpw
        v = np.array([dxdt, dkdt]).flatten()
        return tuple(v)

    def _func(self, *P):
        r""" integrator function used for
        :py:func:`odeint<scipy.integrate.odeint>`

        This is just a wrapper for
        :py:method:`_velocity<sdp.model.wave.ray.RayTracer._velocity>`. The
        arguments are ungrouped to meet odeint format.
        """
        dim = self.dimension
        assert len(P[0])==2*dim, 'Arguments must be given as x0, x1,\
..., xn, k0, k1, ..., kn, and t. Check the diminsion of the plasma!'

        x = P[0][:dim]
        k = P[0][dim:2*dim]
        return self._velocity(x, k)

    def trace(self, x0, k0, t):
        r""" Tracing the ray along the trajectory

        :param x0: starting point in configuration space
        :type x0: array-like of floats, 1D case also NEED to be an ARRAY
        :param k0: starting point in wave-vector space
        :type k0: array-like of floats, 1D case also NEED to be an ARRAY
        :param 1darray t: [t0, t1, ..., tn], solution will be given at these
                          time points. The first element should correspond to
                          the initial (x0, k0).

        :return: x(t), k(t) as an array
        :rtype: 2darray, shape (n, 2*dimension), n is the number of time points

        Tracing uses :py:func:`odeint<scipy.integrate.odeint>` to solve the
        ODEs.

        .. math::
            \frac{dx}{dt} = \frac{\partial \mathcal{D}/\partial k}
                             {\partial \mathcal{D}/\partial \omega}

        .. math::
            \frac{dk}{dt} = -\frac{\partial \mathcal{D}/\partial x}
                             {\partial \mathcal{D}/\partial \omega}



        """
        #TODO finish the Example in doc-string

        init_vec = np.array([x0, k0]).flatten()
        solved = False
        while (len(t)>0 and solved is False):
            try:
                sol = odeint(self._func, init_vec, t)
                solved = True
            except OutOfPlasmaError:
                print "Ray goes out of plasma, trying half time."
                t = t[:len(t)/2]
        if solved is True:
            return sol
        else:
            print "solution not found, check plasma range and initial \
conditions."
            return [[]]









