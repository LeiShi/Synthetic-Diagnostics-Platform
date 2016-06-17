""" Synthetic Diagnostics Platform (sdp)

Developed and maintained by Lei Shi, Princeton Plasma Physics Lab. 
E-mail: fpsdp.main@gmail.com

Package informations can be found in the following variables:

version : Release version number
license : License content
notice : Notice to the users and developers
authors: Complete list of the authors 

Package description is given in variable `info`.
"""



from __future__ import print_function
import pkg_resources

from .diagnostic import Available_Diagnostics as avail_diag
from .plasma import Available_Simulation_Interfaces as avail_interface


version = pkg_resources.get_distribution('sdp').version

license = pkg_resources.resource_string('sdp', 'LICENSE')

notice = pkg_resources.resource_string('sdp', 'NOTICE')

authors = pkg_resources.resource_string('sdp', 'AUTHORS')

info = 'Synthetic Diagnostics Platform, Version {0}.\n\
    Available diagnostics: {1} \n\
    Available simulation interfaces: {2}.'.format(version, 
                                                  avail_diag,
                                                  avail_interface)
                                            
if __name__ == '__main__':
    print(info)