
# Synthetic Diagnostics Platform

The Synthetic Diagnostics Platform (SDP) is a Python package that provides synthetic diagnostic modules on fusion plasma experiments. 

It has interfaces to many plasma simulation codes, including XGC1, GTC, GTS, and M3D-C1. Simple analytic plasmas can also be used for tests and demonstrations. 

Currently, SDP has synthetic [ECEI](ECEI2D tutorial.ipynb), [reflectometry](FWR2D_tutorial) (PPPL cluster access required), and [BES](BES_tutorial) (XGC1 simulation only).

# Prerequisites

1. **Python**
  
   Before you start using the SDP, it is necessary for you to know the basics of Python Programming Language. The [official tutorial](https://docs.python.org/2.7/tutorial/index.html) should be more than enough. SDP is developed on Python 2.7. The Python 3 compatibility is not tested.
    
2. **Numpy & Scipy**
   
   SDP uses Numpy and Scipy packages extensively. Basic knowledge of Numpy is strongly recommended. The [official tutorial](http://docs.scipy.org/doc/numpy/user/quickstart.html) provides a good starting point.

3. **Jupyter Notebook**

    This documentation and other tutorials are written in Jupyter Notebook format. Refer to the [Jupyter documentation](http://jupyter.readthedocs.io/en/latest/index.html) if you need help on using this document.
    
4. **Git and Git-LFS**

    The most convenient way to obtain SDP is through its GitHub repository. You'll need to be familiar with [Git](https://git-scm.com/). Here is an interactive [tutorial](https://try.github.io/) on Git. Another key tool to successfully clone SDP is the [Git-LFS](https://git-lfs.github.com/). **CLONE WITHOUT Git-LFS WILL RESULT IN MISSING KEY DATA FILES IN THE PACKAGE.**

# Recommended IDE

There are a lot of Python Integrated Development Environments (IDEs) available. If you are an experienced Python programmer, and have already gotten familiar with a specific IDE, then you are free to skip this section, and move directly to [installation of SDP](#4. Installation of SDP). The following recommendations are for users of SDP who are still looking for an IDE to start playing with Python.

## Anaconda

[Anaconda](https://www.continuum.io/why-anaconda) is an open source suite of useful Python environments, including an integrated editor [Spyder](https://pythonhosted.org/spyder/), and environments provided by [Jupyter](http://jupyter.readthedocs.io/en/latest/index.html). 

It is straightforward to [install Anaconda](https://www.continuum.io/downloads). Give it a try! 

# Installation of SDP

1. Obtain SDP source code

    SDP is fully open source now! Feel free to checkout the repo and play with it. If you have any questions or suggestions, or want to contribute to SDP, please contact the author: Lei Shi (FPSDP.main@gmail.com).
    
2. Put SDP into Python path

    In order to import SDP from Python interpreter, you need to put SDP into a directory Python recognizes. The most straightforward way is to add the full path of "src/python2/" directory to the [PYTHONPATH](https://docs.python.org/2/using/cmdline.html#envvar-PYTHONPATH) environment variable.   
    
    For example, in Windows, suppose your SDP source package is in "C:\SDP_source\". Then, you only need to add "C:\SDP_source\src\python2\" to the system's environment variable "PYTHONPATH". If no environment variable named "PYTHONPATH" exists, you should create one.
    
SDP should now be correctly recognized by Python and up to work.

SDP depends on some standard scientific libraries: scipy, numpy, matplotlib. You'll need to install them before using SDP. Guide on installation of these packages can be found in their official websites. If you are using Anaconda, you'll be able to obtain these packages easily through the "conda install" command.

# Importing SDP and start using

After setting up the Python path, you should be able to do the following in the Python interpreter.


```python
import sdp.plasma.analytic.testparameter as tp
```


```python
from sdp.settings.unitsystem import cgs
```


```python
print cgs['c']
```

    29979000000.0
    

If you can run the above lines successfully, SDP should be working fine on your machine. Go on and try [Create a simple model plasma](Create_model_plasma.md).



