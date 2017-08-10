# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:51:53 2016

@author: shile

basic smoothing techniques
"""
from __future__ import print_function

import numpy as np

def smooth(data, method='121', axis=-1, periodic=0, passes=1):
    r""" Simple averaging smooth over raw data, currently only support
    smoothing along the last dimension
    """
    avail_methods=['121']
    if method not in avail_methods:
        raise NotImplementedError('method {0} is not a valid method name, \
available methods are: {1}.'.format(method, avail_methods))
    if axis != -1:
        raise NotImplementedError('smoothing along axis other than the last \
one is currently not available.')
    result = np.copy(data)
    if method == '121':
        for i in xrange(passes):
            d1 = result[..., 1]
            dn2 = result[..., -2]
            result[..., 1:-1] = 0.25*( result[..., :-2] + 2*result[..., 1:-1]\
                                       + result[..., 2:])
            if periodic == 1:
                result[..., 0] = 0.25*(d1 + dn2 + result[..., 0])
                result[..., -1] = result[..., 0]
        return result
    else:
        print("smoothing is not done, data has not changed.")


