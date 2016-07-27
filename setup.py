# -*- coding: utf-8 -*-
"""
setup script for SDP

Created on Thu Jun 16 17:38:55 2016

@author: lei
"""

from setuptools import setup, find_packages

setup(name='sdp',

	  # use git tag as the version number
      use_scm_version = True,
	  
      description='Synthetic Diagnostics Platform for Fusion Plasmas',
      author='Lei Shi',
      author_email='FPSDP.main@gmail.com',
      url='https://github.com/LeiShi/Fusion_Plasma_Synthetic_Diagnostics_\
Platform-Public-',
      packages = find_packages('./src/python2'),
	  package_dir = {'sdp':'src/python2/sdp'},
      #include_package_data = True,
      package_data= {'sdp.math' : ['data/*'],
					 'sdp' : ['./LICENSE', './AUTHORS', './NOTICE']},
	  #data_files = [('src/python2/sdp', ['./LICENSE', './AUTHORS', './NOTICE'])],
      license = 'revised-BSD',
      
      # dependencies
      install_requires=['numpy>=1.10.4', 'scipy>=0.17.0'],
	  setup_requires = ['setuptools_scm', 'setuptools_scm_git_archive']
      
)
