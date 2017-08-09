# -*- coding: utf-8 -*-
"""
publishable figure configuration file

see `plotsettings<https://pypi.python.org/pypi/plotsettings>` for details

Created on Thu Aug 18 11:20:51 2016

@author: lei
"""

journals = {'PUDissertation':{'rcParams':     {'font.size':  18,
                                               'xtick.color': 'k',
                                               'ytick.color': 'k',
                                               'lines.linewidth': 2},
                              'figsize':      {'column_width': 6,
                                               'gutter_width': 0,
                                               'max_height': 9,
                                               'units': 'inch'},
                              'panel_labels': {'case': 'lower',
                                               'prefix': ')'},
                             },
            'RSI':           {'rcParams':     {'font.size': 14,
                                               'xtick.color':'k',
                                               'ytick.color':'k',
                                               'lines.linewidth':1.5},
                              'figsize':      {'column_width': 8.5,
                                               'gutter_width': 0,
                                               'max_height': 16,
                                               'units': 'cm'},
                              'panel_labels': {'case':  'lower',
                                               'prefix': ')'}
                             },
            'Notebook':      {'rcParams':     {'font.size': 14,
                                               'axes.labelcolor': '0.6',
                                               'text.color': '0.6',
                                               'xtick.color':'0.6',
                                               'ytick.color':'0.6',
                                               'lines.linewidth':1.5},
                              'figsize':      {'column_width': 12,
                                               'gutter_width': 0,
                                               'max_height': 9,
                                               'units': 'inch'},
                              'panel_labels': {'case':  'lower',
                                               'prefix': ')'}
                             }
           }

