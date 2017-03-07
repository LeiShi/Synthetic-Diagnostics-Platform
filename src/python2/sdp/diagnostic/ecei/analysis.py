# -*- coding: utf-8 -*-
"""
Post analysis module for ECEI

Class:
    ECEI_Analyzer(ECEI)

Created on Mon May 02 18:11:57 2016

@author: lei
"""
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.tri import triangulation, LinearTriInterpolator
from matplotlib.tri import CubicTriInterpolator

from ...geometry.support import DelaunayTriFinder

class ECEI_Analyzer(object):
    """Provide post analyze methods for ecei diagnostics

    Initialize:
        __init__(self, ecei, X_aim, Y_aim)

        :param ecei: ECE Imaging object, must have Te attribute
        :param X_aim: X coordinates of aim locations, 1D
        :param Y_aim: Y coordinates of aim locations, 1D


    """

    def __init__(self, Te, X_aims, Y_aims):
        self._X = np.array(X_aims)
        self._Y = np.array(Y_aims)
        assert self._X.ndim == 1
        assert self._Y.ndim == 1
        self._Te = Te
        self._name = "ecei analyzer"
        self._points = np.transpose(np.array([self._Y, self._X]))

    def _setup_Te_interpolator(self, kind='linear'):
        """setup interpolator for measured Te

        :param string kind: default is 'linear', can be 'cubic' or 'linear'
        """
        self._Delaunay = Delaunay(self._points)
        self._triangulation = triangulation.Triangulation(self._Y, self._X,
                                         triangles=self._Delaunay.simplices)
        self._trifinder = DelaunayTriFinder(self._Delaunay,
                                            self._triangulation)
        if kind=='linear':
            self._kind = kind
            self._Te_interpolator = LinearTriInterpolator(self._triangulation,
                                                          self._Te,
                                                    trifinder=self._trifinder)
        elif kind == 'cubic':
            self._kind = kind
            self._Te_interpolator = CubicTriInterpolator(self._triangulation,
                                                         self._Te,
                                                   trifinder=self._trifinder)
        else:
            raise ValueError('Wrong kind of interpolation: {0}. Available \
options are "linear" and "cubic".'.format(kind))

    def Te_2D_interp(self, coordinates, kind='linear'):
        """interpolate 2D Te measurement on coordinates

        :param coordinates: contains [Y1D, X1D] arrays, Te will be interpolated
                            on the 2D rectangular mesh.
        :type coordinates: list of 2 1D arrays, in the order (Y, X)
        :param string kind: optional, choose interpolation kind, can be either
                            'linear' or 'cubic'. Default is 'linear'.

        :return: interpolated Te
        :rtype: 2D array of float, in shape (NY, NX), where NY and NX are the
                lengths of input cooridnates Y1D and X1D respectively.
        """
        Y, X = coordinates
        Y = np.asarray(Y)
        X = np.asarray(X)
        assert Y.shape == X.shape
        try:
            if(self._kind == kind):
                return self._Te_interpolator(Y, X)
            else:
                self._setup_Te_interpolator(kind)
                return self._Te_interpolator(Y, X)
        except AttributeError:
            self._setup_Te_interpolator(kind)
            return self._Te_interpolator(Y, X)

