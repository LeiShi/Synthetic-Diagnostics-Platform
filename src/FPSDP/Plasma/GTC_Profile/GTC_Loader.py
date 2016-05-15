# -*- coding: utf-8 -*-
"""
GTC loading module

Read in equilibrium and fluctuation data from GTC output files, interpolate and 
extrapolate necessary data onto a Cartesian grid specified by user. 
Data structure is compatible with Diagnostic modules.



Created on Mon Sep 21 14:07:39 2015

@author: lshi
"""
import json
import re
import os
import warnings

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import LinearNDInterpolator, interp1d
from matplotlib.tri import triangulation 
from matplotlib.tri import LinearTriInterpolator as linear_interp

from ...Geometry.Grid import Cartesian2D, Cartesian3D
from ...Geometry.Support import DelaunayTriFinder
from ...IO import f90nml
from ...Maths.Funcs import poly2_curve
from ...Diagnostics.AvailableDiagnostics import Available_Diagnostics
from ..PlasmaProfile import ECEI_Profile, IonClass
from ...GeneralSettings.Exceptions import PlasmaError, PlasmaWarning
from ...GeneralSettings.UnitSystem import cgs


# unit converting dictionary
# The quantities read from GTC output file should be MULTIPLIED by the 
# corresponding entries to get into cgs unit
GTC_to_cgs = {
    'length':1,
    'density':1,
	# equilibrium temperature is in energy unit, originally eV
    'temperature':cgs['keV']/1000,
        # magnetic field is already in Gauss in GTC output
    'magnetic_field':1,
	# potential is using energy unit, which is gotten by multiplying it with 
	# elementary charge
    'electric_potential':cgs['keV']/1000,
	# magnetic vector potential is originally normalized to the same unit as 
	# electric potential.
    'magnetic_potential':cgs['keV']/1000,
	# pressure is special due to the changable normalization in GTC
        # normalization is explicitly done in interpolate_fluc_2D method
    'pressure': 1}

class GTC_Loader_Error(PlasmaError):
    def __init__(self,message):
        self.message = message
    def __str__(self):
        return str(self.message)
        
class GTC_Loader_Warning(PlasmaWarning):
    """class for GTC_Loader induced Warnings
    """

def get_interp_planes(loader):
    """Get the plane numbers used for interpolation for each point 
    """
    dPHI = 2 * np.pi / loader.n_plane
    phi_planes = np.arange(loader.n_plane)*dPHI
    if(loader.CO_DIR):
        nextplane = np.searchsorted(phi_planes,loader.grid.phi3D, side='right')
        prevplane = nextplane - 1
        nextplane[np.nonzero(nextplane == loader.n_plane)] = 0
    else:
        prevplane = np.searchsorted(phi_planes,loader.grid.phi3D, side='right')
        nextplane = prevplane - 1
        prevplane[np.nonzero(prevplane == loader.n_plane)] = 0

    return (prevplane,nextplane)

def calculate_needed_planes(my_gtc):
    pass

def check_time_availability(path, required_times, fname_pattern):
    """Check GTC output files for available time steps
    
    We use regular expression to extract the time step information from file 
    names. Note that if non-default file names are used, we need to modify the 
    regular expression pattern.
    
    More information about regular expression module in python, check out the 
    documentation :https://docs.python.org/2/library/re.html 
    
    :param string path: the path within which file names are being checked
    :param required_times: time steps that are expected to be existing
    :type required_times: python or numpy array of int.
    :param fname_pattern: regular expression pattern that fits the filenames, 
                          and extract the time information with the group name 
                          *time*.        
    :type fname_pattern: raw string
    """
    
    fname_re = re.compile(fname_pattern)
    fnames = os.listdir(path)
    time_all = []
    for name in fnames:
        m = fname_re.match(name)
        if m:
            time_all.append(int(m.group('time')))
        
    time_all = np.sort(time_all)
    for t in required_times:
        if t not in time_all:
            raise GTC_Loader_Error(('Time {} not available!'.format(t), 
                                    time_all))
    print 'All time available.'
    return time_all
    

class GTC_Loader:
    """Main class for loading GTC data. 
    
    Instances are initialized with a given "grid" containing user specified 
    mesh, and a path containing all GTC output files. After initialization, GTC
    data on the mesh will be ready to use.
    
    Initialization
    ---------------
    
    __init__(self, gtc_path, grid, tsteps, 
             fname_pattern_2D=r'snap(?P<time>\d+)_fpsdp.json', 
             fname_pattern_3D = r'PHI_(?P<time>\d+)_\d+.ncd', 
             Mode = 'full')
        
    :param string gtc_path: The path where GTC output files are located. 
    :param grid: User defined spatial grid. All GTC data will be 
                 interpolated onto this grid.
    :type grid: :class:`Grid <FPSDP.Geometry.Grid>` object. Only 
                Cartesian2D and Cartesian3D are supported for now.
    :param tsteps: 
        time steps requested. Use GTC simulation step counting. 
        Will be checked at the beginning to make sure all requested time 
        steps have been outputted. 
    :type tsteps: list or 1D numpy array of int
    :param fname_pattern_2D: 
        (Optional) Regular Expression to fit 2D output files. Do not change
        unless you have changed the output file name convention.
    :type fname_pattern_2D: raw string
    :param fname_pattern_3D: 
        (Optional) Regular Expression to fit 3D output files. Do not change
        unless you have changed the output file name convention.
    :type fname_pattern_3D: raw string
    :param string Mode:
        (Optional) choice among 3 possible iniialization modes: 
        
        **full**(Default): 
            load both equilibrium and fluctuations, and interpolate them on
            given *grid*            
        **eq_only**: 
            load only equilibrium, and interpolate it on given *grid*
        **least**: 
            DO NOT load any GTC output files, only initialize the loader 
            with initial parameters. This mode is mainly used for debug.
    
    Calculation of dTe
    -------------------
    
    Since we have 2 degrees of freedom in perpendicular direction, and only
    1 degree of freedom in parallel direction. The perpendicular energy 
    should be 2*(1/2 Te), and parallel energy is 1/2 Te.
    
    .. math::
        T_{e\perp}(x) 
        \equiv \frac{\int (1/2)m_e v_\perp^2 f(x, v)dv}{\int f(x, v) dv}
        = \frac{P_{e\perp}(x)}{n_e(x)}
    
    .. math::
        \frac{1}{2}T_{e\parallel}(x) \equiv 
        \frac{\int (1/2)m_e v_\parallel^2 f(x, v)dv}{\int f(x, v) dv}
        = \frac{P_{e\parallel}(x)}{n_e(x)}
                
    Then :math:`dT_e(x) = T_e(x)-T_{e0}(x)`, where :math:`T_{e0}(x)` is 
    defined by the equilibrium distribution function :math:`f_0(x, v)`.
    
    We write :math:`f(x, v) = f_0(x, v) + df(x, v)`, and define the 
    corresponding :math:`dP_e` and :math:`dn_e` as the integration of 
    :math:`df`. We finally have the expression for :math:`dT_e`:
    
    .. math::
       dT_{e\perp}(x) = 
       \left(\frac{P_{e\perp}}{n_e} -\frac{P_{e0\perp}}{n_{e0}}\right)
       = \frac{dP_{e\perp} - T_{e0}dn_e}{n_{e0}} 
       
    .. math::
       \frac{1}{2}dT_{e\parallel}(x) = 
       \left(\frac{P_{e\parallel}}{n_e}
       -\frac{P_{e0\parallel}}{n_{e0}}\right)
       = \frac{2 \, dP_{e\parallel} - T_{e0}dn_e}{n_{e0}}
                
    Note that we use a isotropic equilibrium temperature, so 
    
    .. math::
        T_{e0\perp} = T_{e0\parallel} = T_{e0}
            
    
    """
    
    def __init__(self, gtc_path, grid, tsteps=None, 
                 fname_pattern_2D=r'snap(?P<time>\d+)_fpsdp.json', 
                 fname_pattern_3D = r'PHI_(?P<time>\d+)_\d+.ncd', 
                 Mode = 'full'):
        """Initialize a GTC loader with the following parameters:
        
        :param string gtc_path: The path where GTC output files are located. 
        :param grid: User defined spatial grid. All GTC data will be 
                     interpolated onto this grid.
        :type grid: :class:`Grid <FPSDP.Geometry.Grid>` object. Only 
                    Cartesian2D and Cartesian3D are supported for now.
        :param tsteps: 
            time steps requested. Use GTC simulation step counting. 
            Will be checked at the beginning to make sure all requested time 
            steps have been outputted. 
        :type tsteps: list or 1D numpy array of int
        :param fname_pattern_2D: 
            (Optional) Regular Expression to fit 2D output files. Do not change
            unless you have changed the output file name convention.
        :type fname_pattern_2D: raw string
        :param fname_pattern_3D: 
            (Optional) Regular Expression to fit 3D output files. Do not change
            unless you have changed the output file name convention.
        :type fname_pattern_3D: raw string
        :param string Mode:
            (Optional) choice among 3 possible iniialization modes: 
            
            **full**(Default): 
                load both equilibrium and fluctuations, and interpolate them on
                given *grid*            
            **eq_only**: 
                load only equilibrium, and interpolate it on given *grid*
            **least**: 
                DO NOT load any GTC output files, only initialize the loader 
                with initial parameters. This mode is mainly used for debug.
        """        
        
        
        self.path = gtc_path
        self.grid = grid
        self.tsteps = tsteps
        
        if isinstance(grid, Cartesian2D):
            print('2D grid detected.')
            self.dimension = 2
            if (Mode == 'full'):
            # First, use :py:func:`<check_time_availability>` to analyze the 
            # existing files and check if all required time steps are 
            # available.
            # If any requested time steps are not there, raise an error and 
            # print out all existing time steps.
                self.Mode = Mode
                try:
                    self._time_all =check_time_availability(self.path, 
                                                            self.tsteps,
                                                            fname_pattern_2D)
                except GTC_Loader_Error as e:
                    print e.message[0]
                    print 'Available time steps are:'
                    print repr(e.message[1])
                    raise 
            elif (Mode == 'eq_only' or Mode == 'least'):
                self.Mode = Mode
            else:
                raise GTC_Loader_Error('Invalid Mode: {0}. \n\
   Available Modes are: "full", "eq_only", "least"'.format(Mode))
        elif isinstance(grid, Cartesian3D):
            print('3D grid detected.')
            self.dimension = 3
            try:
                self._time_all =check_time_availability(os.path.join(self.path,
                                                                    'phi_dir'),
                                                        self.tsteps,
                                                        fname_pattern_3D)
            except GTC_Loader_Error as e:
                print e.message[0]
                print 'Available time steps are:'
                print str(e.message[1])
                raise 
        else:
            raise(GTC_Loader_Error('Invalid grid: Only Cartesian2D or \
Cartesian3D grids are supported.'))
            
        
        if ((Mode == 'full') or (Mode == 'eq_only')):
            # read gtc.in.out and gtc.out, obtain run specifications like: 
            # adiabatic/non-adiabatic electrons, electrostatic/electromagnetic,
            # time step, ion gyro-radius, and snap output frequency.
            self.load_gtc_specifics() 
        
            # read grid_fpsdp.json, create triangulated mesh on GTC grids and 
            # extended grids for equilibrium.
            self.load_grid() 
        
            # read equilibriumB_fpsdp.json and equilibrium1D_fpsdp.json, create
            # interpolators for B_phi,B_R,B_Z on extended grids, and 
            # interpolators for equilibrium density and temperature over 'a'.
            self.load_equilibrium(SOL_width=0.1) 
            
            # check the grid resolution to warn user if grid is not optimal
            self.check_grid_resolution()
            # interpolate equilibrium quantities
            self.interpolate_eq()
            
            if(Mode == 'full'):
                # For fluctuations, 2D and 3D loaders are different
                if(self.dimension == 2):
                    # 2D is simple, read snap{time}_fpsdp.json and interpolate 
                    # the data onto 2D grid
                    self.load_fluctuations_2D()
                    self.interpolate_fluc_2D()
                    
                if(self.dimension == 3):
                    # 3D loader is current not implemented
                    raise NotImplementedError
                    # 3D is much harder to do. First, we calculate the needed 
                    # cross-section numbers
                    # get interpolation plane numbers for each grid point
                    self.prevplane,self.nextplane = get_interp_planes(self)
                    # calculate needed planes
                    planes = calculate_needed_planes(self)
                    # load fluctuations on those cross-sections            
                    self.load_fluctuations_3D(planes)
        
                    # interpolate onto our 3D mesh
                    self.interpolate_fluc_3D()
            
    def load_gtc_specifics(self):
        """ read relevant GTC simulation settings from gtc.in and gtc.out 
        files.
        
        Create Attributes:
            :var bool HaveElectron: 
                if True, GTC is simulating non-adiabatic electrons. 
                Otherwise only adiabatic electrons are used.
            :var bool isEM: 
                if True, GTC is in Electro-magnetic mode. 
                Otherwise in electro-static mode.
            :var double dt: GTC simulation time step length, unit: second
            :var double rho_i: 
                typical ion gyro-radius in GTC simulation, unit:meter
            :var int snapstep: 
                GTC snapshot output time, every *snapstep* timesteps, 
                a *snap###_fpsdp.json* file is written.
        """
        
        GTCin_fname = os.path.join(self.path, 'gtc.in')
        gtcin_nml = f90nml.read(GTCin_fname)        
        GTCout_fname = os.path.join(self.path, 'gtc.out')
        gtcout_nml = f90nml.read(GTCout_fname)
        
        self.HaveElectron = (gtcin_nml['input_parameters']['nhybrid'] > 0)
        self.isEM = (gtcin_nml['input_parameters']['magnetic'] == 1)
        self.snapstep = gtcin_nml['input_parameters']['msnap']
        
        # time step in GTC is normalized to R0/cs, cs = sqrt(Te0/m_i)
        # We will need the equilibrium data to get dt in seconds
        # Now, we store the GTC normalized time step for future use
        self._dt_gtc = gtcin_nml['input_parameters']['tstep']
        
        # Ion information is NOT fully implemented. Further work is needed 
        # before using ion related information.
        ion_mass = gtcin_nml['input_parameters']['aion']
        ion_charge = gtcin_nml['input_parameters']['qion']
        fast_ion_mass = gtcin_nml['input_parameters']['afast']
        fast_ion_charge = gtcin_nml['input_parameters']['qfast']
        self.ions = [IonClass(ion_mass, ion_charge, 'thermal_ion'), 
                     IonClass(fast_ion_mass, fast_ion_charge, 'fast_ion')]

        # iload, eload determines the normalization of perturbed quantities.
        # iload(eload) = 1 normalize all quantities to the on-axis value
        # iload(eload) = 2 has density normalized to local equilibrium density, but 
        # temperature to on-axis temperature
        # iload(eload) = 3 has both density and temperature normalized to local 
        # equilibrium
        self.iload = gtcin_nml['input_parameters']['iload']
        self.eload = gtcin_nml['input_parameters']['eload']

        self.psi1 = gtcout_nml['input_parameters']['psi1']        
        #NEED INFO on gtc.in and gtc.out
        
    def load_grid(self):
        """ Read in Grid information from grid_fpsdp.json file.
        Create Attributes:
            :var R_gtc: R coordinates for GTC simulation grid. unit: meter
            :vartype R_gtc: 1D double array of length N
            :var Z_gtc: R coordinates for GTC simulation grid. unit: meter
            :vartype Z_gtc: 1D double array of length N
            :var points_gtc: (R,Z) pairs for each GTC simulation grid point
            :vartype points_gtc: (N,2) shape double array
            
            :var a_gtc: 
                corresponding normalized flux radial coordinates on GTC grid
            :vartype a_gtc: 1D double array of length N
            :var theta_gtc: corresponding flux poloidal coordinates on GTC grid
            :vartype theta_gtc: 1D double array of length N

            :var R_eq: 
                R cooridnate on larger mesh for interpolating equilibrium
            :vartype R_eq: 1D double array of length N_eq
            :var Z_eq: 
                Z cooridnate on larger mesh for interpolating equilibrium
            :vartype Z_eq: 1D double array of length N_eq
            :var points_eq: (R,Z) pairs for each grid point on equilibrium mesh
            :vartype points_eq: (N_eq,2) shape double array

            :var a_eq: corresponding *a* values 
            :vartype a_eq: 1D double array of length N_eq
            
            :var R_LCFS: 
                R coordinates for sampled last closed flux surface, i.e. a=1
            :vartype R_LCFS: 1D double array of length N_LCFS
            :var Z_LCFS: 
                Z coordinates for sampled last closed flux surface, i.e. a=1
            :vartype Z_LCFS: 1D double array of length N_LCFS
            
            :var double R0: R coordinate of magnetic axis
            :var double Z0: Z coordinate of magnetic axis
            
            :var Delaunay_gtc: 
                trangulation of GTC grid on (R,Z) plane, 
                created by :class:`Delaunay <scipy.spatial.Delaunay>`
            :vartype Delaunay_gtc: scipy.spatial.Delaunay object
            :var Delaunay_eq: 
                trangulation of equilibrium grid on (R,Z) plane, created by 
                :class:`Delaunay <scipy.spatial.Delaunay>`
            :vartype Delaunay_eq: scipy.spatial.Delaunay object
            
            :var a_eq_interp: 
                2D linear interpolator of *a* on (R,Z) plane, using *a_eq* and 
                *Delaunay_eq*. Out_of_bound value is set to be *nan* and will 
                be dealt with later.
            :vartype a_eq_interp: scipy.interpolate.LinearNDInterpolator object
            
            
        """
        l2cgs = GTC_to_cgs['length']
        b2cgs = GTC_to_cgs['magnetic_field']
        grid_fname = os.path.join(self.path, 'grid_fpsdp.json')
        with open(grid_fname) as gridfile:
            raw_grids = json.load(gridfile)
        self.R0= raw_grids['R0'] * l2cgs
        self.Z0 = raw_grids['Z0'] * l2cgs
        self.B0 = raw_grids['B0'] * b2cgs       
        self.R_gtc = np.array(raw_grids['R_gtc'])*self.R0
        self.Z_gtc = np.array(raw_grids['Z_gtc'])*self.R0
        self.points_gtc = np.transpose(np.array([self.Z_gtc,self.R_gtc]))
        self.a_gtc = np.array(raw_grids['a_gtc'])
        self.theta_gtc = np.array(raw_grids['theta_gtc'])
        self.R_eq = np.array(raw_grids['R_eq'])*self.R0
        self.Z_eq = np.array(raw_grids['Z_eq'])*self.R0
        self.points_eq = np.transpose(np.array([self.Z_eq,self.R_eq]))
        self.a_eq = np.array(raw_grids['a_eq'])
      
        # use Delaunay Triangulation package provided by **scipy** to do the 2D
        # triangulation on GTC mesh
        self.Delaunay_gtc = Delaunay(self.points_gtc)
        self.triangulation_gtc = triangulation.Triangulation(self.Z_gtc, 
                                                             self.R_gtc,
                                        triangles=self.Delaunay_gtc.simplices)
        self.trifinder_gtc = DelaunayTriFinder(self.Delaunay_gtc,
                                               self.triangulation_gtc)
        
        # For equilibrium mesh, we use Triangulation package provided by 
        # **matplotlib** to do a cubic interpolation on *a* values and B field.
        # The points outside the convex hull of given set of points will be 
        # treated later.
        self.Delaunay_eq = Delaunay(self.points_eq)
        self.triangulation_eq = triangulation.Triangulation(self.Z_eq,
                                                            self.R_eq, 
                                        triangles = self.Delaunay_eq.simplices)
        self.trifinder_eq = DelaunayTriFinder(self.Delaunay_eq, 
                                              self.triangulation_eq)
        
        # interpolate flux surface coordinate "a" onto grid_eq, save the 
        # interpolater for future use.
        # Default fill value is "nan", points outside eq mesh will be dealt 
        # with care
        self.a_eq_interp = linear_interp(self.triangulation_eq, self.a_eq, 
                                         trifinder = self.trifinder_eq)
        
    def load_equilibrium(self, SOL_width = 0.1, extrapolation_points = 20, 
                         fill_coeff = 1e-5):
        """ read in equilibrium field data from equilibriumB_fpsdp.json and 
        equilibrium1D_fpsdp.json
        
        :param double SOL_width: 
            *(optional)*, width of the scrape-off layer outside the closed flux
            surface region, used to determine the equilibrium decay length. 
            Default to be 0.1. Unit: minor radius *a*=1
        :param int extrapolation_points: 
            *(optional)* number of sample points added within SOL region, used 
            for extrapolation of equilibrium density and temperature. 
            Default to be 20.
        :param float fill_coeff:
            *(optional)*, residule coefficient for ne0 and Te0 values outside 
            extrapolation area. The outside values will be the coefficient 
            times the LCFS value read from GTC. Default is 1e-5. 
        
        Create Attributes:
            :var B_phi: :math:`\\Phi` component of equilibrium magnetic field 
                        on eq grid
            :vartype B_phi: 1D double array of length N_eq            
            :var B_R: *R* component of equilibrium magnetic field on eq grid
            :vartype B_R: 1D double array of length N_eq 
            :var B_Z: *Z* component of equilibrium magnetic field
            :vartype B_Z: 1D double array of length N_eq 
            
            :var B_phi_interp: interpolated B_phi. Out_of_bound value is set to
                               be *nan* and will be dealt with later.
            :vartype B_phi_interp: 
                :class:`<scipy.interpolate.LinearNDInterpolator>` object
            :var B_R_interp: interpolated B_R. Out_of_bound value is set to be 
                             *nan* and will be dealt with later.
            :vartype B_R_interp: 
                :class:`<scipy.interpolate.LinearNDInterpolator>` object
            :var B_Z_interp: interpolated B_Z. Out_of_bound value is set to be
                             *nan* and will be dealt with later.
            :vartype B_Z_interp: 
                :class:`<scipy.interpolate.LinearNDInterpolator>` object
            
            :var a_1D: *a* coordinate for 1D interpolation, includes *a* array 
                       read from *equilibrium1D_fpsdp.json* file, and extension
                       on larger *a* values. 
            :vartype a_1D: 1D double array of length *N_1D*
            :var ne0_1D: equilibrium electron density on *a_1D*, read in from 
                         *equilibrium1D_fpsdp.json* file, and use 3rd order 
                         polynomial to extrapolate outside values. 
                         Unit: :math:`m^{-3}`
            :vartype ne0_1D: 1D double array of length *N_1D*
            :var Te0_1D: equilbirium electron temperature on *a_1D*, read in 
                         from *equilibrium1D_fpsdp.json* file, and use 3rd 
                         order polynomial to extrapolate outside values. 
                         Unit: keV
            :vartype Te0_1D: 1D double array of length *N_1D*
            
            :var ne0_interp: 
                linear interpolator created with *ne0_1D* on *a_1D*. 
                Out_of_bound value set to 0.
            :vartype ne0_interp: :class:`<scipy.interpolate.interp1d>` object
            :var Te0_interp:
                linear interpolator created with *Te0_1D* on *a_1D*. 
                Out_of_bound value set to 0.
            :vartype ne0_interp: :class:`<scipy.interpolate.interp1d>` object
        """
        n2cgs = GTC_to_cgs['density']
        T2cgs = GTC_to_cgs['temperature']
        
        eqB_fname = os.path.join( self.path, 'equilibriumB_fpsdp.json')
        with open(eqB_fname,'r') as eqBfile:
            raw_eqB = json.load(eqBfile)
        
        # data is normalized to B0, so need to multiply with B0 to get actual 
        # value
        self.B_phi = np.array(raw_eqB['B_phi_eq'])*self.B0
        self.B_total = np.array(raw_eqB['B_eq'])*self.B0
        
        # outside points will be masked and dealt with later.
        self.B_phi_interp = linear_interp(self.triangulation_eq, self.B_phi,
                                          trifinder=self.trifinder_eq) 
        self.B_total_interp = linear_interp(self.triangulation_eq, 
                                            self.B_total,
                                            trifinder=self.trifinder_eq)
        # self.B_R_interp = linear_interp(self.triangulation_eq, self.B_R, 
        #                                trifinder=self.trifinder_eq)
        # self.B_Z_interp = linear_interp(self.triangulation_eq, self.B_Z, 
        #                                trifinder=self.trifinder_eq)

        # Now read in 1D equilibrium quantities        
        eq1D_fname = os.path.join(self.path, 'equilibrium1d_fpsdp.json')
        with open(eq1D_fname,'r') as eq1Dfile:
            raw_eq1D = json.load(eq1Dfile)
        
        # a, ne0, Te0 inside LCFS, sorted in *a* increasing order
        raw_a = np.array(raw_eq1D['a'])
        raw_a_max = np.max(raw_a)
        sorting = np.argsort(raw_a)
        sorted_a = raw_a[sorting]/raw_a_max
        raw_ne0 = np.array(raw_eq1D['ne0']) * n2cgs
        sorted_ne0 = raw_ne0[sorting]
        raw_Te0 = np.array(raw_eq1D['Te0']) * T2cgs
        sorted_Te0 = raw_Te0[sorting]
        
        
        # outside needs extrapolation
        a_out = np.linspace(1,1+SOL_width,extrapolation_points)      
        # ne0 and Te0 will be extrapolated using 2nd order polynomial curves 
        # that fits the value at a=1, and decays to 0 and flat at SOL outter 
        # edge.        
        
        #set up the polynomial curve
        ne_fill = fill_coeff * sorted_ne0[-1]
        Te_fill = fill_coeff * sorted_Te0[-1]
        ne_curve = poly2_curve(sorted_a[-1],sorted_ne0[-1],a_out[-1], ne_fill)
        Te_curve = poly2_curve(sorted_a[-1],sorted_Te0[-1],a_out[-1], Te_fill)
        #evaluate extrapolated curve at sample points
        ne0_out = ne_curve(a_out)
        Te0_out = Te_curve(a_out)        
        #append outside values to original sorted array
        self.a_1D = np.append(sorted_a,a_out)*raw_a_max
        self.ne0_1D = np.append(sorted_ne0,ne0_out)
        self.Te0_1D = np.append(sorted_Te0,Te0_out)
        #set up interpolators using extrapolated samples, points outside the 
        # extended *a* range can be safely set to 0.
        self.ne0_interp = interp1d(self.a_1D, self.ne0_1D, bounds_error=False, 
                                   fill_value=ne_fill)
        self.Te0_interp = interp1d(self.a_1D, self.Te0_1D, bounds_error=False, 
                                   fill_value=Te_fill)
                                   
        # Generate equilibrium data on GTC grid, for later use in calculating
        # dTe on this mesh
        self.ne0_gtc = self.ne0_interp(self.a_gtc)
        self.Te0_gtc = self.Te0_interp(self.a_gtc)
                                   
    @property
    def time(self):
        """The time for all time steps in real unit, i.e. seconds
        """
        # GTC time step is normalized to R0/cs, cs=sqrt(Te0/mi)
        m_i = self.ions[0].mass*cgs['m_p']
        cs = np.sqrt(self.Te0_1D[0]/m_i)
        time_unit = self.R0/cs
        return np.array(self.tsteps) * self._dt_gtc * time_unit

    def check_grid_resolution(self):
        """Check grid resolution by comparing it to on-axis rho_s
        rho_s = c_s/omega_ci
        """
        m_i = self.ions[0].mass * cgs['m_p']
        cs = np.sqrt(self.Te0_1D[0]/m_i)
        e = cgs['e']
        c = cgs['c']
        omega_ci = self.B0*e/(m_i*c)
        rho_s = cs/omega_ci
        if(self.grid.ResR > rho_s):
            warnings.warn('Grid may be too coarse in R: ResR = {0:.3}cm, rho_s = {1:.3}cm'.format(self.grid.ResR, rho_s))
        if(self.grid.ResZ > rho_s):
            warnings.warn('Grid may be too coarse in Z: ResZ = {0:.3}cm, rho_s = {1:.3}cm'.format(self.grid.ResZ, rho_s))
        if(self.grid.ResR < rho_s/10):
            warnings.warn('Grid may be too fine in R: ResR = {0:.3}cm, rho_s = {1:.3}cm'.format(self.grid.ResR, rho_s))
        if(self.grid.ResZ < rho_s/10):
            warnings.warn('Grid may be too fine in Z: ResZ = {0:.3}cm, rho_s = {1:.3}cm'.format(self.grid.ResZ, rho_s))
        
    def interpolate_eq(self):
        r"""Interpolate equilibrium quantities on given grid. 
        
        *B_R*, *B_Z*, *B_phi* and *a* are interpolated over (Z_eq,R_eq) mesh, 
        and *ne0*, *Te0* are interpolated on *a* space.
        
        For interpolation over (Z_eq,R_eq), Grid points outside Equilibrium 
        mesh(i.e. outside LCFS) will be approximated using the following 
        method:
            
        For an outside point :math:`(Z_{out},R_{out})`, we search for the 
        closest vertex on the convex hull of the interpolation set, 
        :math:`(Z_n,R_n)`, and the corresponding :math:`a=a_n`. 
            
        From the cubic interpolation, we can obtain the derivatives of *a* 
        respect to Z and R at :math:`(Z_n,R_n)`, :math:`\partial a/\partial Z` 
        and :math:`\partial a/\partial R`.
            
        Now the *a* value at :math:`(Z_{out}, R_{out})` will be approximated 
        by:
                
        ..math::
        
            a(Z_{out},R_{out}) = a_n + (Z_{out}-Z_n) \cdot \frac{\partial a}
            {\partial Z} + (R_{out}-R_n) \cdot \frac{\partial a}{\partial R}
            
        This first order approximation is good if :math:`(Z_{out},R_{out})` is 
        not far from :math:`(Z_n,R_n)`. In our case, since we are assuming 
        :math:`n_e` and :math:`T_e` are rapidly decaying in *a* outside the 
        LCFS, this approximation is good enough.
        """        
        # outside points are obtained by examining the mask flag from the 
        # returned masked array of "linear_interp"
        Zwant = self.grid.Z2D
        Rwant = self.grid.R2D        
        self.a_on_grid = self.a_eq_interp(Zwant,Rwant)
        out_mask = np.copy(self.a_on_grid.mask)
        
        Zout = Zwant[out_mask]
        Rout = Rwant[out_mask]
        
        # boundary points are obtained by applying ConvexHull on equilibrium 
        # grid points
        hull = ConvexHull(self.points_eq)
        p_boundary = self.points_eq[hull.vertices]
        Z_boundary = p_boundary[:,0]
        R_boundary = p_boundary[:,1]
        
        # Now let's calculate *a* on outside points, first, get the nearest 
        # boundary point for each outside point
        nearest_indices = []
        for i in range(len(Zout)):
            Z = Zout[i]
            R = Rout[i]
            nearest_indices.append(np.argmin((Z-Z_boundary)**2 + \
                                   (R-R_boundary)**2) )
            
        # Then, calculate *a* based on the gradient at these nearest points
        Zn = Z_boundary[nearest_indices]
        Rn = R_boundary[nearest_indices]
        # The value *a* and its gradiant at this nearest point can by easily 
        # obtained            
        an = self.a_eq_interp(Zn,Rn)            
        gradaZ_bdy,gradaR_bdy = self.a_eq_interp.gradient(Zn,Rn)
        
        # Now deal with points that didn't get a finite gradient from the 
        # interpolator
        gradaR_bdy, gradaZ_bdy = self._correct_interpolator_gradient\
                                     (self.a_eq_interp, gradaR_bdy, gradaZ_bdy,
                                      Rn, Zn, an) 
        # Finally, we can evaluate the outside a values by gradient at the 
        # closest boundary point.                
        a_out = an + (Zout-Zn)*gradaZ_bdy + (Rout-Rn)*gradaR_bdy
        
        # Finally, assign these outside values to the original array
        self.a_on_grid[out_mask] = a_out
        
        # Now we are ready to interpolate ne and Te on our grid
        self.ne0_on_grid = self.ne0_interp(self.a_on_grid.data)
        self.Te0_on_grid = self.Te0_interp(self.a_on_grid.data)
        
        # B_total and B_phi can be interpolated exactly like *a*               
        self.Btotal_on_grid = self.B_total_interp(Zwant, Rwant)
        self.Bphi_on_grid = self.B_phi_interp(Zwant,Rwant)
        
        # B_R and B_Z can by obtained from gradaR and gradaZ since *a* is the
        # poloidal flux. 
        R2D = self.grid.R2D
        Z2D = self.grid.Z2D
        gradaZ, gradaR = self.a_eq_interp.gradient(Z2D, R2D)
        self.BR_on_grid = -1/R2D * gradaZ * self.R0 * self.R0 * self.B0
        self.BZ_on_grid = 1/R2D * gradaR * self.R0 * self.R0 * self.B0
        
        # We use the boundary gradient to calculate the outside BR and BZ field        
        BR_out = -gradaZ_bdy/Rout *self.R0*self.R0*self.B0
        self.BR_on_grid[out_mask] = BR_out
        
        BZ_out = gradaR_bdy/Rout*self.R0*self.R0*self.B0 # psi is normalized
        self.BZ_on_grid[out_mask] = BZ_out
        
        # Outside B_phi is taken to be simply decaying as 1/R
        Bphi_out = self.B0*self.R0/Rout        
        self.Bphi_on_grid[out_mask] = Bphi_out
        
        # Outside B_total is now calculated from its three components.
        Btotal_out = np.sqrt(BR_out**2 + BZ_out**2 + Bphi_out**2)
        self.Btotal_on_grid[out_mask] = Btotal_out
        
    def _correct_interpolator_gradient(self, interpolator, gradR, gradZ, Rn, 
                                       Zn,fn, tol=(1e-2,1e1), resR=0.01, 
                                       resZ=0.01):
        """ A private use function that make necessary corrections to the 
        gradients calculated by linear interpolators
        
        :param interpolator: 
            The interpolaor object we are using, normally just 
            *self.a_eq_interp*
        :param gradR: 
            The gradient respect to R array obtained by interpolator, need to 
            be updated
        :param gradZ: 
            The gradient respect to Z array obtained by interpolator, need to 
            be updated
        :param Rn: 
            corresponding R coordinates of the points
        :param Zn: 
            corresponding Z coordinates of the points
        :param fn: 
            the value of the quantity at these points
        :param tol: 
            tolarance for checking zero/infinity gradient points. 
            
            If the calculated gradient at one boundary point is less than 
            tol[0]*averaged_gradient, it's considered 0; 
            
            if it's greater than tol[1]*averaged_gradient, 
            it's considered infinity. 
            All these points will be recalculated by 
            :math:`f'=\frac{f(x+\Delta x)-f(x)}{\Delta x}`, with `\Delta x` 
            taken to be either positive or negative so that `x+\ Delta x` is 
            still inside the convex hull of original data set.        
        
        :type tol: tuple of 2 floats        
        :param float resR: 
            relative resolution in R when evaluating gradient. 
            :math:`\Delta R = resR \cdot \Delta R_{grid}`, 
            where `\Delta R_{grid}` is the resolution of the R grid given by 
            *self.grid.ResR*
        :param float resZ: same meaning as *resR*, but in Z direction. 
        """
        
        gradient= gradZ**2+gradR**2
        average_gradient = np.median(gradient) 
        
        # check if the gradient is too small or too large.
        no_gradient_idx = \
            np.logical_or(gradient < tol[0]*tol[0]*average_gradient, 
                          gradient > tol[1]*tol[1]*average_gradient)
        Z_null = Zn[no_gradient_idx]
        R_null = Rn[no_gradient_idx]
        f_null = fn[no_gradient_idx]
        
        new_gradR = np.zeros_like(gradR[no_gradient_idx])
        new_gradZ = np.zeros_like(gradZ[no_gradient_idx])
        
        # manually calculate the da/dR by setting a small increment in R, 
        # and calculate the first order difference approximation
        dR = self.grid.ResR * resR 
        R_p = R_null + dR
        f_pR = interpolator(Z_null,R_p)
        new_gradR[~f_pR.mask] = (f_pR-f_null)[~f_pR.mask]/dR
        changed_mark_R = ~f_pR.mask # not masked values are changed
        
        # If not all points changed, then positive dR make some points out of 
        # the convex hull, try the negative dR
        if(not changed_mark_R.all()): 
            R_n = R_null -dR
            f_nR = interpolator(Z_null,R_n)
            new_gradR[~f_nR.mask] = (f_null - f_nR)[~f_nR.mask]/dR
            # just in case there still are points left out, keep track of all 
            # the points that have been covered.
            changed_mark_R = np.logical_or(~f_nR.mask,~f_pR.mask)
        gradR[no_gradient_idx] = new_gradR
        
        # manually calculate the da/dZ by setting a small increment in Z, and 
        # calculate the first order difference approximation
        dZ = self.grid.ResZ * resZ 
        
        Z_p = Z_null + dZ
        f_pZ = interpolator(Z_p,R_null)
        new_gradZ[~f_pZ.mask] = (f_pZ-f_null)[~f_pZ.mask]/dZ
        changed_mark_Z = ~f_pZ.mask # not masked values are changed
        if(not changed_mark_Z.all()): 
        # If not all points changed, then positive dZ make some points out of 
        # the convex hull, try the negative dZ
            Z_n = Z_null -dZ
            f_nZ = interpolator(Z_n,R_null)
            new_gradZ[~f_nZ.mask] = (f_null - f_nZ)[~f_nZ.mask]/dZ
            # just in case there still are points left out, keep track of all 
            # the points that have been covered.
            changed_mark_Z = np.logical_or(~f_nZ.mask,~f_pZ.mask)
        all_changed = np.logical_or(changed_mark_R,changed_mark_Z)    
        gradZ[no_gradient_idx] = new_gradZ
        # Now ALL points should be changed either in R or Z gradient, if not, 
        # something really weird must have happened.
        if(not all_changed.all()):
            raise GTC_Loader_Error('Strange thing happened! Some point has \
been left isolated. Check the input R_eq and Z_eq mesh, and see how its convex\
 hull looks like.')
        return (gradR,gradZ)

    
    def load_fluctuations_2D(self, fname_format = 'snap{0:0>7}_fpsdp.json'):
        """ Read fluctuation data from **snap{time}_fpsdp.json** files
        Read data into an array with shape (NT,Ngrid_gtc), NT the number of 
        requested timesteps, corresponds to *self.tstep*, Ngrid_gtc is the GTC 
        grid number on each cross-section.
        Create Attribute:
            :var phi: fluctuating electro-static potential on GTC grid for each 
                      requested time step, unit: V
            :vartype phi: double array with shape (NT,Ngrid_gtc)
            :var Te: electron temperature fluctuation, unit: keV
            :vartype Te: double array with shape (NT,Ngrid_gtc)
            if *HaveElectron*:
            :var nane: non-adiabatic electron density fluctuation, unit: 
                       :math:`m^{-3}`
            :vartype nane: double array with shape (NT,Ngrid_gtc)                
            if *isEM*:
            :var A_par: parallel vector potential fluctuation
            :vartype A_par:double array with shape (NT,Ngrid_gtc)
            
        """
        NT = len(self.tsteps)
        Ngrid_gtc = len(self.R_gtc)
        self.phi = np.empty((NT,Ngrid_gtc))
               
        self.dPe_perp = np.empty_like(self.phi)
        self.dPe_para = np.empty_like(self.phi)
        self.dni = np.empty_like(self.phi)
        self.dne_ad = np.empty_like(self.phi)        
        if self.HaveElectron:
            self.nane = np.empty_like(self.phi)
        if self.isEM:
            self.A_para = np.empty_like(self.phi)
            self.dTe_ad = np.empty_like(self.phi)
            self.dpsi = np.empty_like(self.phi)
        
        
        for i in range(NT):
            snap_fname = os.path.join(self.path,
                                      fname_format.format(self.tsteps[i]))
            with open(snap_fname,'r') as snap_file:
                raw_snap = json.load(snap_file)
            # our phi is in energy unit, which is actually e*phi. It's 
			# originally in eV, we'll cast it into erg.
            self.phi[i] = np.array(raw_snap['phi']) \
				          * GTC_to_cgs['electric_potential']
            
            self.dni[i] = np.array(raw_snap['densityi'])\
								* GTC_to_cgs['density']
            self.dne_ad[i] = np.array(raw_snap['fluidne'])\
								* GTC_to_cgs['density']
            if self.HaveElectron:
                self.nane[i] = np.array(raw_snap['densitye'])\
								* GTC_to_cgs['density']
                # dPe output is normalized based on GTC running parameter 
                # It will be taken care later
                self.dPe_perp[i] = np.array(raw_snap['Pe_perp'])\
								* GTC_to_cgs['pressure']
                self.dPe_para[i] = np.array(raw_snap['Pe_para'])\
								* GTC_to_cgs['pressure']
            if self.isEM:
                warnings.warn('Magnetic perturbations are not properly \
normalized. Quantative calculation using perturbed vector potential requires \
special attention.', GTC_Loader_Warning)
                self.A_para[i] = np.array(raw_snap['apara'])\
						     * GTC_to_cgs['magnetic_potential']
                self.dTe_ad[i] = np.array(raw_snap['Te_adiabatic'])
                self.dpsi[i] = np.array(raw_snap['delta_psi'])
        # Now, we take care of dPe and dne normalization
        ni_norm = self.ne0_gtc / self.ions[0].charge
        ne_norm = self.ne0_gtc
        # Temperature normalization is to on-axis equilibrium Te0
        Te_norm = self.Te0_1D[0]
            
        # Note that dne_ad is always normalized to local equilibrium density
        self.dne_ad *= self.ne0_gtc
        self.dni *= ni_norm
        self.nane *= ne_norm            

        Pe_norm = ne_norm * Te_norm
        self.dPe_para *= Pe_norm
        self.dPe_perp *= Pe_norm
                      
    def interpolate_fluc_2D(self):
        """Interpolate 2D fluctuations onto given Cartesian grid. Grids outside
        the GTC simulation domain will be given 0.
        """
        NT = len(self.tsteps)
        NZ = self.grid.NZ
        NR = self.grid.NR
        
        points_on_grid =np.transpose( np.array([self.grid.Z2D,self.grid.R2D]),
                                     (1,2,0))        
        
        self.phi_on_grid = np.empty((NT,NZ,NR))
        
        self.dPe_perp_on_grid = np.empty_like(self.phi_on_grid)
        self.dPe_para_on_grid = np.empty_like(self.phi_on_grid)
        self.dni_on_grid = np.empty_like(self.phi_on_grid)
        self.dne_ad_on_grid = np.empty_like(self.phi_on_grid)
        if self.HaveElectron:
            self.nane_on_grid = np.empty_like(self.phi_on_grid)
        if self.isEM:
            self.A_para_on_grid = np.empty_like(self.phi_on_grid)
        
        
        for i in range(NT):
            phi_interp = LinearNDInterpolator(self.Delaunay_gtc, self.phi[i],
                                              fill_value = 0)
            self.phi_on_grid[i] = phi_interp(points_on_grid)
            
            dPe_perp_interp = LinearNDInterpolator(self.Delaunay_gtc,
                                                  self.dPe_perp[i], 
                                                  fill_value=0)
            self.dPe_perp_on_grid[i] = dPe_perp_interp(points_on_grid)

            dPe_para_interp = LinearNDInterpolator(self.Delaunay_gtc,
                                                  self.dPe_para[i],
                                                  fill_value=0)
            self.dPe_para_on_grid[i] = dPe_para_interp(points_on_grid)  
            
            dni_interp = LinearNDInterpolator(self.Delaunay_gtc,
                                              self.dni[i], fill_value=0)
            self.dni_on_grid[i] = dni_interp(points_on_grid)
            
            dne_ad_interp = LinearNDInterpolator(self.Delaunay_gtc,
                                                 self.dne_ad[i], fill_value=0)
            self.dne_ad_on_grid[i] = dne_ad_interp(points_on_grid)
            
            if self.HaveElectron:
                nane_interp = LinearNDInterpolator(self.Delaunay_gtc,
                                                   self.nane[i], fill_value=0)
                self.nane_on_grid[i] = nane_interp(points_on_grid)
            if self.isEM:   
                A_para_interp = LinearNDInterpolator(self.Delaunay_gtc,
                                                     self.A_para[i],
                                                     fill_value=0)
                self.A_para_on_grid[i] = A_para_interp(points_on_grid)

    @property
    def dne(self):
        """ total electron density perturbation on GTC mesh """
        try:
            if not self.HaveElectron:
                return self.dne_ad
            else:
                return self.dne_ad + self.nane
        except AttributeError as e:
            print('dne_on_grid is only available when fluctuations are loaded.\
 Use full mode in initialization to enable fluctuation loading.')
            raise e
            
    @property         
    def dne_on_grid(self):
        """ total electron density perturbation on grid
        """
        try:
            if not self.HaveElectron:
                return self.dne_ad_on_grid
            else:
                return self.dne_ad_on_grid + self.nane_on_grid
        except AttributeError as e:
            print('dne_on_grid is only available when fluctuations are loaded.\
 Use full mode in initialization to enable fluctuation loading.')
            raise e
            
    @property
    def ne_on_grid(self):
        """ total electron density on grid"""
        try:
            return self.ne0_on_grid + self.dne_on_grid
        except AttributeError:
            warnings.warn('fluctuation is no loaded, ne0 is returned.', 
                          GTC_Loader_Warning)
            return self.ne0_on_grid
            
    def calculate_fluc_Te(self, component, mesh,tol=1e-14):
        r"""
        Calculate electron temperature fluctuation based on pressure 
        perturbation and density perturbation read from GTC output data.
        
        Since we have 2 degrees of freedom in perpendicular direction, and only
        1 degree of freedom in parallel direction. The perpendicular energy 
        should be 2*(1/2 Te), and parallel energy is 1/2 Te.
        
        .. math::
            T_{e\perp}(x) 
            \equiv \frac{\int (1/2)m_e v_\perp^2 f(x, v)dv}{\int f(x, v) dv}
            = \frac{P_{e\perp}(x)}{n_e(x)}
        
        .. math::
            T_{e\parallel}(x) \equiv 
            \frac{\int m_e v_\parallel^2 f(x, v)dv}{\int f(x, v) dv}
            = \frac{P_{e\parallel}(x)}{n_e(x)}
                    
        Then :math:`dT_e(x) = T_e(x)-T_{e0}(x)`, where :math:`T_{e0}(x)` is 
        defined by the equilibrium distribution function :math:`f_0(x, v)`.
        
        We write :math:`f(x, v) = f_0(x, v) + df(x, v)`, and define the 
        corresponding :math:`dP_e` and :math:`dn_e` as the integration of 
        :math:`df`. We finally have the expression for :math:`dT_e`:
        
        .. math::
           dT_{e\perp}(x) = 
           \left(\frac{P_{e\perp}}{n_e} -\frac{P_{e0\perp}}{n_{e0}}\right)
           = \frac{dP_{e\perp} - T_{e0}dn_e}{n_{e0}} 
           
        .. math::
           \fracdT_{e\parallel}(x) = 
           \left(\frac{P_{e\parallel}}{n_e}
           -\frac{P_{e0\parallel}}{n_{e0}}\right)
           = \frac{dP_{e\parallel} - T_{e0}dn_e}{n_{e0}}
                    
        Note that we use a isotropic equilibrium temperature, so 
        
        .. math::
            T_{e0\perp} = T_{e0\parallel} = T_{e0}
        """
        if mesh == 'GTC':
            dT = np.zeros_like(self.dPe_perp)
            if component == 'perp':
                dPe = self.dPe_perp
            elif component == 'para':
                dPe = self.dPe_para
            else:
                raise ValueError('component {0} not valid. options are "perp" or \
"para"'.format(component))
            for i, t in enumerate(self.tsteps):
                dT[i] = (dPe[i] - self.Te0_gtc*self.nane)[i] / self.ne0_gtc
            return dT
        elif mesh == 'grid':
            in_idx = np.abs(self.ne0_on_grid) > tol
            dT = np.zeros_like(self.dPe_perp_on_grid)
            if component == 'perp':
                dPe = self.dPe_perp_on_grid
            elif component == 'para':
                dPe = self.dPe_para_on_grid
            else:
                raise ValueError('component {0} not valid. options are "perp" or \
"para"'.format(component))
            for i, t in enumerate(self.tsteps):
                dT[i][in_idx] = (dPe[i][in_idx] - (self.Te0_on_grid\
                                 *self.nane_on_grid)[i][in_idx]) \
                                 / self.ne0_on_grid[in_idx]
            return dT
        else:
            raise ValueError('mesh {0} not valid. options are "GTC" or \
"grid"'.format(mesh))
    
    @property
    def dTe_perp(self):
        return self.calculate_fluc_Te('perp', 'GTC')        
        
    @property
    def dTe_para(self):
        return self.calculate_fluc_Te('para', 'GTC')
    
    @property
    def dTe_perp_on_grid(self):
        return self.calculate_fluc_Te('perp', 'grid')
        
    @property
    def dTe_para_on_grid(self):
        return self.calculate_fluc_Te('para', 'grid')
        
    
    def interpolate_on_grid(self, grid=None):
        """Interpolate required quantities on new grid. Useful for loading same
        simulation data for multiple diagnostics which requires different 
        grids.
        """
        # TODO complete the interpolation function
        pass
        
    def create_profile(self, diagnostic=None, grid=None):
        """Create required profile object for specific diagnostic
        
        :param diagnostc: Specify the synthetic diagnostic that uses the 
                           profile. If not given, a list of available 
                           diagnostics will be printed.
        :type diagnostic: string
        :param grid: The grid on which all required profiles will be given. If 
                     not specified, ``self.grid`` will be used.
        :type grid: :py:class:`<...Geometry.Grid.Grid>` derived class
        """
        if (diagnostic is None) or (diagnostic not in Available_Diagnostics):
            raise ValueError('Diagnostic not specified! Currently available \
            diagnostics are:\n{}'.format(Available_Diagnostics))
        
        if self.Mode == 'least':
            warnings.warn('least mode has no data loaded. No plasma profile \
created.')
            return
            
        if (diagnostic in ['ECEI1D', 'ECEI2D']):
            self.interpolate_on_grid(grid)
            if grid is None:
                grid = self.grid
            if self.Mode == 'full': 
                return ECEI_Profile(grid, self.ne0_on_grid, self.Te0_on_grid, 
                                    self.Btotal_on_grid, self.time, 
                                    self.dne_on_grid, self.dTe_para_on_grid, 
                                    self.dTe_perp_on_grid)
            elif self.Mode == 'eq_only':
                return ECEI_Profile(grid, self.ne0_on_grid, self.Te0_on_grid,
                                    self.Btotal_on_grid)
                                
        # TODO finish FWR2D profile class and GTC writing fun
        else:
            raise NotImplemented('GTC profile generator for {} is not \
implemented! Modify FPSDP.Plasma.GTC_Profile.GTC_Loader.create_profile to \
implement it.')
        
        
        
            
        
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        
