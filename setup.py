# -*- coding: utf-8 -*-
"""
setup script for SDP

Created on Thu Jun 16 17:38:55 2016

@author: lei
"""

from setuptools import setup, find_packages

setup(name='sdp',
      version='0.1.0',
      description='Synthetic Diagnostics Platform for Fusion Plasmas',
      author='Lei Shi',
      author_email='FPSDP.main@gmail.com',
      url='https://github.com/LeiShi/Fusion_Plasma_Synthetic_Diagnostics_\
Platform-Public-',
      packages = ['sdp'], #find_packages(),
	  package_dir = {'sdp':'src/sdp'},
      include_package_data = True,
      package_data= {'sdp.math' : ['data/*']},
	  data_files = [('', ['./LICENSE', './AUTHORS', './NOTICE'])],
      license = 'revised-BSD',
      
      # dependencies
      install_requires=['numpy>=1.10.4', 'scipy>=0.17.0']
      
)
