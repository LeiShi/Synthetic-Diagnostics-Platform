# -*- coding: utf-8 -*-
"""
generate plasma.cdf for FWR2D runs

Created on Wed Aug 17 13:36:21 2016

@author: lei
"""
import sdp.diagnostic.fwr.fwr2d.input as fwr_input
import sdp.plasma.analytic.testparameter as tp

tp.set_parameter1D(**tp.Parameter_DIIID)
tp.set_parameter2D(**tp.Parameter_DIIID)

p1d = tp.create_profile1D()
p2d = tp.create_profile2D()
p2d_uniform = tp.simulate_1D(p1d, p2d.grid)

fwr_input.generate_cdf(p2d_uniform)

