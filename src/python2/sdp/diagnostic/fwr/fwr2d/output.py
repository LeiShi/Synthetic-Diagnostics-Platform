# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:36:54 2016

@author: lei
"""
import numpy as np
from scipy.io.netcdf import netcdf_file


class FWR2DSolution(object):

    def __init__(self, filename):
        self.filename = filename
        self.load_netcdf(filename)

    def load_netcdf(self, filename):
        r"""load the FWR2D output cdf file

        """
        fwrout = netcdf_file(filename, 'r')

        for key, value in fwrout.variables.iteritems():
            self.__setattr__(key, np.copy(value.data))

