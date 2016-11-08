import json
import numpy as np

class Diagnoser(object):

    def __init__(self, filename='diag_sdp.json'):
        
        with open(filename) as diagfile:
            raw_diag = json.load(diagfile)
        self.npsi = raw_diag['npsi']
        self.ntheta = raw_diag['ntheta']
        self.x = np.array(raw_diag['x']).reshape((self.ntheta, self.npsi, 9))
        self.z = np.array(raw_diag['z']).reshape((self.ntheta, self.npsi, 9))
        self.b = np.array(raw_diag['b']).reshape((self.ntheta, self.npsi, 9))
        self.g = np.array(raw_diag['g']).reshape(( self.npsi, 3) )
        self.I = np.array(raw_diag['I']).reshape(( self.npsi, 3) )
        self.q = np.array(raw_diag['q']).reshape(( self.npsi, 3) )
        self.jacobian_boozer = np.array(raw_diag['jacobian_boozer']).\
                                       reshape(( self.ntheta, self.npsi) )
        self.jacobian_metric = np.array(raw_diag['jacobian_metric']).\
                                       reshape(( self.ntheta, self.npsi) )
    
