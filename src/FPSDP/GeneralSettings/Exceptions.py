# -*- coding: utf-8 -*-
"""
Provide definitions of generally used Exception classes:

FPSDPError: base class for all FPSDP raised errors. Further specific errors 
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
    """Base class for all FPSDP raised Exceptions
    """
    def __init__(self, s):
        self.message = s
        
    def __str__(self):
        return self.message
        

class GeometryError(FPSDPError):
    """Base class for Exceptions raised from Geometry package
    """  

class MathsError(FPSDPError):
    """Base class for Exceptions raised from Maths package
    """
    
class PlasmaError(FPSDPError):
    """Base class for Exceptions raised from Geometry package"""
        

class ModelError(FPSDPError):
    """Base class for all Model related errors
    """
    def __init__(self, s):
        self.message = s
        

class ModelInvalidError(ModelError):
    """Raised when specific circumstance doesn't meet model criteria.
    """
    def __init__(self, s):
        self.message = s
        
        

        
        
class ResonanceError(ModelInvalidError):
    """Raised when wave resonance happens, and can not be handled.
    """
    def __init__(self, s):
        self.message = s

class FPSDPWarning(Warning):
    """Base class for all FPSDP raised warnings
    """
    
class ECEIWarning(FPSDPWarning):
    """Base class for warnings raised from ECEI package
    """
    
class MathsWarning(FPSDPWarning):
    """Base class for warnings raised from Maths package
    """
    
class PlasmaWarning(FPSDPWarning):
    """Base class for warnings raised from Plasma package
    """

class GeometryWarning(FPSDPWarning):
    """Base class for warnings raised from Geometry package
    """