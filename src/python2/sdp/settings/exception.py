# -*- coding: utf-8 -*-
"""
Provide definitions of generally used Exception classes:

FPSDPError: base class for all sdp raised errors. Further specific errors
            should all be derived from this class
            
ModelError: base class for all model related errors.

ModelInvalidError: subclass of ModelError. Raised when specific circumstance 
                   doesn't meet model criteria.
                   
ResonanceError: subclass of ModelInvalidError. Raised when wave resonance 
                happens, and can not be handled.



Created on Fri Mar 18 14:02:19 2016

@author: lei
"""

class FPSDPError(Exception):
    """Base class for all sdp raised Exceptions
    """
    def __init__(self, s):
        self.message = s
        
    def __str__(self):
        return self.message
        

class GeometryError(FPSDPError):
    """Base class for Exceptions raised from geometry package
    """  

class MathsError(FPSDPError):
    """Base class for Exceptions raised from math package
    """
    
class PlasmaError(FPSDPError):
    """Base class for Exceptions raised from geometry package"""
        

class ModelError(FPSDPError):
    """Base class for all Model related errors
    """        


class ModelInvalidError(ModelError):
    """Raised when specific circumstance doesn't meet model criteria.
    """
        
        

        
        
class ResonanceError(ModelInvalidError):
    """Raised when wave resonance happens, and can not be handled.
    """
    def __init__(self, s):
        self.message = s

class FPSDPWarning(Warning):
    """Base class for all sdp raised warnings
    """
    
class ECEIWarning(FPSDPWarning):
    """Base class for warnings raised from ecei package
    """
    
class MathsWarning(FPSDPWarning):
    """Base class for warnings raised from math package
    """
    
class PlasmaWarning(FPSDPWarning):
    """Base class for warnings raised from plasma package
    """

class GeometryWarning(FPSDPWarning):
    """Base class for warnings raised from geometry package
    """