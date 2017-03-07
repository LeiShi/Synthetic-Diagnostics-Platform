r"""
ECEI2D
=======

contains 2D version of synthetic Electron Cyclotron
Emission Imaging Diagnostic.

Unit Conventions
-----------------
In ECEI2D, Gaussian unit is used by default. The units for common quantities
are:

length:
    centi-meter
time:
    second
mass:
    gram
magnetic field:
    Gauss
temperature:
    erg (we use energy unit for particle temperature)

Usage
------

Preparation
************

A complete ECEI2D run requires knowledge of the plasma, and the receivers.

The former should be provided as an instance of
:py:class:`ECEI_Profile<FPSDP.Plasma.PlasmaProfile.ECEI_Profile>`, and the
latter a list of
:py:class:`Detector2D<FPSDP.Diagnostics.ECEI.ECEI2D.Detector2D.Detector2D>`.

We will assume these two objects have been created and named `plasma2d` and
`detectors`.

First, we import the ECEImagingSystem class::

    >>> from sdp.diagnostic.ecei.ecei2d import ECEImagingSystem

Then, we initialize the ECEI with plasma2d and detectors::

    >>> ecei = ECEImagingSystem(plasma2d, detectors)

Note that some additional parameters can be provided while initialization,
check the doc-string in :py:class:`ECEImagingSystem
<FPSDP.Diagnostics.ECEI.ECEI2D.Imaging.ECEImagingSystem>` for a detailed list
of these parameters.

The next step is to setup the calculation area. ECEI uses 3D Cartesian
coordinates, and assumes rectangular cells. So, three 1D arrays specifying
grids along Z(local toroidal), Y(vertical), and X(Radial) directions is needed.

The detector is always assumed being on the low-field side, and in vacuum. The
calculation area needs to include part of the vacuum region, and large enough
to include all the resonant region. X1D mesh also determines the calculation
start and end points, so its normally from larger X (vacuum region outside of
plasma) to smaller X (inner plasma).

Let's say we choose a uniform XYZ grid, we can create it using
:py:module:`numpy<numpy>` as::

    >>> X1D = numpy.linspace(251, 216, 160)
    >>> Y1D = numpy.linspace(-30, 30, 65)
    >>> Z1D = numpy.linspace(-30, 30, 65)

and set ECEI calculation area::

    >>> ecei.set_coords([Z1D, Y1D, X1D])

It is possible that different detectors need different initial mesh. This is
particularly important if these channels have very different resonance
locations. In this case, we can specify mesh for chosen channels only.

For example, we can set channel 0, and channel 3 only::

    >>> ecei.set_coords([Z1D, Y1D, X1D], channelID=[0, 3])

Note that channelID is numbered from 0.

It is recommended to run the automatic mesh adjustment before diagnosing::

    >>> ecei.auto_adjust_mesh(fine_coeff = 1)

This function run diagnose on the preset mesh and optimize its X grids by
making mesh fine within resonance region, and coarse elsewhere. The fine_coeff
is a parameter controlling the mesh size. The larger this parameter, the finer
resulted mesh overall.

Diagnose
*********

We can now run ECEI and observe the result::

    >>> ecei.diagnose(time=[0, 1, 2])

Running diagnose() without a `time` argument will diagnose the equilibrium
plasma. And a given `time` list will result in a series of diagnosis on
perturbed plasma at corresponding time snaps.

The measured electron temperature is stored in `Te` attribute.

    >>> ecei.Te
    array([[  1.47142490e-08,   1.46694915e-08,   1.46748651e-08],
           [  1.56084333e-08,   1.51977835e-08,   1.48657565e-08],
           [  1.69261271e-08,   1.65879854e-08,   1.61561885e-08],
           [  1.58508369e-08,   1.63720864e-08,   1.68176195e-08],
           [  1.46057450e-08,   1.47844442e-08,   1.50868828e-08],
           [  1.45398116e-08,   1.45283573e-08,   1.45292955e-08],
           [  1.49914189e-08,   1.48120112e-08,   1.47148505e-08],
           [  1.65238937e-08,   1.60221945e-08,   1.55572079e-08]])
    >>> ecei.Te.shape
    (8L, 3L)

The first dimension of Te corresponds to the detectors, and the second
dimension for time.

It is OK to run diagnose multiple times with different parameters, but the
result will be overwritten.

Post analysis
**************

ECEImagingSystem provides additional information about the diagnosing process.

The most useful one is `view_spots`. This attribute stores a list of detailed
emission spot information for each channel in the most recent time snap.

    >>> vs = ecei.view_spots
    >>> len(vs)
    8
    >>> vs[0].shape
    (65L, 63L)

The shape of each view_spot is [NY, NX], it contains the instrumental function
on the 2D plane, with the largest point normalized to 1. This means the
measured Te is just a weighted average of the Te on the 2D plane under this
weighting function.

More information can be obtained from `channels` attribute, which is literally
the ECE2D objects that carry out the diagnostic.

The propogation and absorption of the probing waves can be found in
`propagator` attribute in each channel.

Modules
--------

CurrentCorrelationTensor:
    Contains classes for calculating current correlation tensor. Mainly
    includes non-relativistic and relativistic versions.
Detector2D:
    Contains Detector class for ECEI2D. Now it has GaussianAntenna type
    detector.
Reciprocity:
    Main module carrying out 2D ECE calculation. ECE2D class is the main
    class.
Imaging:
    Contains multi-channel ECE Imaging class. ECEImagingSystem is the main
    class.


"""

from .imaging import ECEImagingSystem
from .ece import ECE2D
from .detector2d import GaussianAntenna

