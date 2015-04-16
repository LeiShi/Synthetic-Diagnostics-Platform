import math
import json
import numpy as np
import FPSDP.Plasma.Collisions.collisions as col
import FPSDP.Maths.Integration as integ
import ConfigParser as psr
import FPSDP.Maths.Interpolation as Fint
from scipy import interpolate
"""  BE CAREFULL WITH THE LOAD XGC, THE CARTESIAN GRID IS NOT IN
     A CLASSICAL POSITION (Y for the heigth)
     In the code of the BES, the Z axis is for the height (so switch between Y and Z)
     data_on_grid[Z,Y,X]
"""


def to_other_index(pos):
    " Change the order of the index "
    if len(pos.shape) == 2:
        a = np.array([pos[:,1],pos[:,2],pos[:,0]])
    elif len(pos.shape) == 1:
        a = np.array([pos[1],pos[2],pos[0]])
    else:
        raise NameError('Wrong shape of pos')
    return a 

class Beam1D:
    """ Simulate a 1D beam with the help of datas from simulation

        Methods:
        __init__(config_file,timesteps,data)
                            -- config_file is the name of the config file,
                               timesteps is the list of the timesteps wanted
                               and data are the data from XGC1 (for example)
       
        get_width(dist)     -- return the standard deviation at the distance given

        create_mesh()       -- create the points of the mesh

        find_wall(eps=1e-6) -- find the end (for the beam) of the mesh from the data
                               eps is for avoiding the end of the mesh (relative length)

        get_electron_density(pos,t_) 
                            -- function simplificating the acces to the data
                               (avoid the problem of the two representations)
                               pos is a 1D or 2D array (first index is the dimension)
                               the coordinate are the same than in the beam config file

        get_ion_density(pos,t_)
                            -- same as before
                    
        get_electron_temp(pos)
                            -- same as before (the perturbation of the temperature)
                               are not taken in account (small effect)

        get_ion_temp(self,pos)
                            -- same as before

        get_electron_density_fluc(pos,t_)
                            -- same as before


        compute_beam_on_mesh()
                            -- compute the density of the beam on the mesh with a
                               Gauss-Legendre formula (two points)
 
        get_mesh()          -- return the mesh, same coordinate than the beam config file
                               first index is for the dimension
 
        get_origin()        -- return the origin of the beam
        
        get_beam_density(pos,t_)
                            -- use a gaussian profile and an interpolation in order to
                               compute the beam density at the position. t_ is the index
                               for the time


        get_emis(pos,t_)    -- compute the emissivity \epsilon 

        get_emis_fluc(pos,t_)
                            -- compute the fluctuation of the emissivity with only
                               the fluctuation part of the density
        get_emis_ave(pos)   -- compute the fluctuation using the average over time

        get_emis_lifetime(pos,t_)
                            -- same as get_emis but compute the lifetime effect

        Attribut:
    
        cfg_file     --  name of the config file
        data         --  data from the simulation
        timesteps    --  list of the timesteps considered
        adas_atte    --  list of file for the attenuation
        adas_emis    --  list of file for the emission
        collisions   --  class containing all the informations for the collisions
        coll_atte    --  tuple relating the adas file with the beams (choice of collision)
                         for the attenuation
        coll_emis    --  same as before but with the emission
        mass_b       --  mass of the particles in the beam (in amu)
        beam_comp    --  energy of each beam components
        power        --  total power of the beam
        frac         --  fraction of power for each components
        pos          --  origin of the beam
        direc        --  direction of the beam
        beam_width   --  FWHM of the beam at the origin
        Nz           --  number of point on the beam mesh
        Nlt          --  number of point for the lifetime effect
        t_max        --  upper boundary of the integral for the lifetime effect
                         (in unit of the lifetime)

        inters       --  ending point of the mesh
        mesh         --  mesh (first index for the dimension)

        density_beam --  beam density at the grid points
        speed        --  speed of each particles in a beam component
        std_dev      --  standard deviation of the beam at the origin
        std_dev2     --  square of std_dev
        dens         --  electron density at the grid points

    """
    
    def __init__(self,config_file):
        """ Load everything from the config file"""
        self.cfg_file = config_file                                          #!
        print 'Reading config file (beam)'
        config = psr.ConfigParser()
        config.read(self.cfg_file)

        # The example input is well commented
        
        # load data for collisions
        self.adas_atte = json.loads(config.get('Collisions','adas_atte'))    #!
        self.adas_emis = json.loads(config.get('Collisions','adas_emis'))    #!
        n_low = json.loads(config.get('Collisions','n_low'))
        n_high = json.loads(config.get('Collisions','n_high'))
        self.collisions = col.Collisions(self.adas_atte,self.adas_emis,
                                         (n_low,n_high))                     #!
        self.coll_atte = json.loads(config.get('Collisions','coll_atte'))    #!
        self.coll_emis = json.loads(config.get('Collisions','coll_emis'))    #!
        self.Nlt = int(json.loads(config.get('Collisions','Nlt')))           #!
        self.t_max = json.loads(config.get('Collisions','t_max'))            #!
        

        # load data about the beam energy
        self.mass_b = json.loads(config.get('Beam energy','mass_b'))         #!
        self.mass_b = np.array(self.mass_b)
        self.beam_comp = json.loads(config.get('Beam energy','E'))           #!
        self.beam_comp = 1000*np.array(self.beam_comp)
        self.power = float(config.get('Beam energy','power'))                #!
        self.power = np.array(self.power)
        self.frac = json.loads(config.get('Beam energy','f'))                #!
        self.frac = np.array(self.frac)
        if np.sum(self.frac) > 100: # allow the possibility to simulate only
            # a part of the beam
            raise NameError('Sum of f is greater than 100')

        # load data about the geometry of the beam
        self.pos = json.loads(config.get('Beam geometry','position'))        #!
        self.pos = np.array(self.pos)
        self.direc = json.loads(config.get('Beam geometry','direction'))     #!
        self.direc = np.array(self.direc)
        self.direc = self.direc/np.sqrt(np.sum(self.direc**2))
        self.beam_width_h = json.loads(
            config.get('Beam geometry','beam_width_h'))                      #!
        self.beam_width_v = json.loads(
            config.get('Beam geometry','beam_width_v'))                      #!
        self.Nz = int(config.get('Beam geometry','Nz'))                      #!

        # get the standard deviation at the origin
        self.stddev_h = self.beam_width_h/(2.0*np.sqrt(2.0*np.log(2.0)))     #!
        self.stddev_v = self.beam_width_v/(2.0*np.sqrt(2.0*np.log(2.0)))     #!
        self.stddev2_h = self.stddev_h**2                                    #!
        self.stddev2_v = self.stddev_v**2                                    #!

        # speed of each beam
        self.speed = np.zeros(len(self.beam_comp))                           #!
        self.speed = np.sqrt(2*self.beam_comp*9.6485383e7/self.mass_b)



    def set_data(self,data):
        """ Split the initialization in two part due to bes
        """
        self.data = data                                                     #!
        """grid_ = (self.data.grid.Z1D,self.data.grid.Y1D,
                 self.data.grid.X1D)
        self.ne_int = []
        self.ni_int = []
        for i in range(len(data.time_steps)):
            self.ne_int.append(interpolate.RegularGridInterpolator(
                grid_,self.data.ne_on_grid[0,i,:,:,:]))
            
            self.ni_int.append(interpolate.RegularGridInterpolator(
                grid_,self.data.ni_on_grid[0,i,:,:,:]))
            
        self.Ti_int = interpolate.RegularGridInterpolator(
            grid_,self.data.ti_on_grid)
        
        self.Te_int = interpolate.RegularGridInterpolator(
            grid_,self.data.te_on_grid)
        """ 
        print 'Creating mesh'
        self.create_mesh()
        print 'Computing density of the beam'
        self.compute_beam_on_mesh()

    def get_width(self,dist):
        """ Return the width of the beam at the distance "dist" (projected 
            against the direction of the beam)
            Used for simplification in the case that someone want to add the
            beam divergence
        """
        return np.array([self.stddev_h*np.ones(dist.shape),
                         self.stddev_v*np.ones(dist.shape)])

    def create_mesh(self):
        """ create the 1D mesh between the source of the beam and the end 
            of the mesh
        """
        # intersection between end of mesh and beam
        self.inters = self.find_wall()                                       #!
        length = np.sqrt(np.sum((self.pos-self.inters)**2))
        # distance to the origin along the central line
        self.dl = np.linspace(0,length,self.Nz)
        self.mesh = np.zeros((self.Nz,3))                                    #!
        # the second index corresponds to the dimension (X,Y,Z)
        self.mesh = self.dl[:,np.newaxis]*self.direc + self.pos              #!
        
                    
    def find_wall(self, eps=1e-6):
        """ find the wall (of the mesh) that will stop the beam and return
            the coordinate of the intersection with the beam
            eps is used to avoid the end of the mesh
        """
        # X-direction
        if self.direc[0] == 0.0:
            tx1 = -np.inf
            tx2 = -np.inf
        else:
            tx1 = abs((self.data.Xmax-self.pos[0])/self.direc[0])
            tx2 = abs((self.data.Xmin-self.pos[0])/self.direc[0])
        
        # Y-directio
        if self.direc[1] == 0.0:
            ty1 = -np.inf
            ty2 = -np.inf
        else:
            ty1 = abs((self.data.Zmax-self.pos[1])/self.direc[1])
            ty2 = abs((self.data.Zmin-self.pos[1])/self.direc[1])
        
        # Z-direction
        if self.direc[2] == 0.0:
            tz1 = -np.inf
            tz2 = -np.inf
        else:
            tz1 = abs((self.data.Ymax-self.pos[2])/self.direc[2])
            tz2 = abs((self.data.Ymin-self.pos[2])/self.direc[2])

        t_ = [tx1,tx2,ty1,ty2,tz1,tz2]
        t = np.argmax(t_)
        
        return self.pos + self.direc*t_[t]*(1-eps)

    def get_quantities(self,pos,t_,quant,eq=False):
        """ get the wanted quantities inside the simulation data
            at pos and the timestep(s) number t_

            Arguments:
            pos   --  1D array with (X,Y,Z) or 2D array with (X,Y,Z)
                      as the second index
            t_    --  time step
            quant --  list containing the wanted quantities (see data class
                      for more information)
            eq    --  choice between equilibrium or exact quantities
        """
        if isinstance(t_,list):
            raise NameError('Only one time should be given')
        return self.data.interpolate_data(pos,t_,quant,eq)
        
        
    def get_electron_density(self,pos,t_,eq=False):
        """ get the electron density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        raise NameError('Old version, should not be used')
        if eq: # read the data from the equilibrium
            if len(pos.shape) == 1:
                R = np.sqrt(pos[0]**2 + pos[1]**2)
                psi = self.data.psi_interp(pos[2],R)
            else:
                R = np.sqrt(np.sum(pos[:,0:2]**2,axis=1))
                psi = self.data.psi_interp(pos[:,2],R)
            ret = self.data.ne0_sp(psi)
        else:
            a = to_other_index(pos).T
            ret = self.ne_int[t_](a)
        """if len(pos.shape) == 1:
            if ret == 0.0:
                print pos
                raise NameError('Error point outside the mesh')
        else:
            if (ret == 0.0).any():
                print pos
                raise NameError('Error point outside the mesh')"""
        return ret
        

    def get_electron_density_fluc(self,pos,t_):
        """ get the fluctuation in the electron density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        raise NameError('Old version, should not be used')
        a = to_other_index(pos).T
        print 'Warning inefficient, should not be used' # the copy of next line is done at each call
        dne = self.data.ne_on_grid[0,t_,...] - self.data.ne0_on_grid
        # the order of the grid is due to the XGC loading coordinate
        return Fint.trilinear_interp(self.data.grid.Z1D,self.data.grid.Y1D,
                                         self.data.grid.X1D,
                                         dne,a)
    
    def get_ion_density(self,pos,t_, eq=False):
        """ get the ion density (from the data) at pos
           
            Argument:
            pos  --  position (3D array) where to compute the density
        """
        raise NameError('Old version, should not be used')
        # FOR ADDING MUTLIPLE IONS SPECIES CHANGE HERE [add an argument and see
        # where it does not work, and after add the loop over element]
        if eq:
            if len(pos.shape) == 1:
                R = np.sqrt(pos[0]**2 + pos[1]**2)
                psi = self.data.psi_interp(pos[2],R)
            else:
                R = np.sqrt(np.sum(pos[:,0:2]**2,axis=1))
                psi = self.data.psi_interp(pos[:,2],R)
            ret = self.data.ni0_sp(psi)
        # the order of the grid is due to the XGC loading coordinate
        else:
            a = to_other_index(pos).T
            ret = self.ni_int[t_](a)
        """if len(pos.shape) == 1:
            if ret == 0.0:
                print pos
                raise NameError('Error point outside the mesh')
        else:
            if (ret == 0.0).any():
                print pos
                raise NameError('Error point outside the mesh')"""
        return ret
    
    def compute_beam_on_mesh(self,eq=True):
        """ compute the beam intensity at each position of the mesh 
            with the Gauss-Legendre quadrature (2 points)
        """
        
        self.density_beam = np.zeros((len(self.beam_comp),self.Nz))  #!
        print('If keep attenuation with eq, should remove time loop')
        # density of the beam at the origin
        n0 = np.zeros(len(self.beam_comp))
        n0 = np.sqrt(2*self.mass_b*1.660538921e-27)*self.power
        n0 *= self.frac/(self.beam_comp*1.60217733e-19)**(1.5)
        
        # define the quadrature formula for this method
        quad = integ.integration_points(1,'GL3') # Gauss-Legendre order 3
        for j in range(len(self.mesh[:,0])):
            # density over the central line (usefull for some check)
            if j is not 0:
                temp_beam = np.zeros(len(self.beam_comp))
                for k in self.coll_atte:
                    file_nber = k[0]
                    beam_nber = k[1]
                    # limit of the integral
                    a = self.mesh[j-1,:]
                    b = self.mesh[j,:]
                    # average
                    av = (a+b)/2.0
                    # difference
                    diff = (b-a)/2.0
                    # integration point
                    pt = quad.pts[:,np.newaxis]*diff + av
                    # compute all the values needed for the integral
                    ne, T = self.get_quantities(pt,None,['ne','Ti'],eq)
                    
                    # attenuation coefficient from adas
                    S = self.collisions.get_attenutation(
                        self.beam_comp[beam_nber],ne,self.mass_b[beam_nber],
                        T,file_nber)
                    
                    # half distance between a & b
                    norm_ = 0.5*np.sqrt(np.sum((b-a)**2))
                    temp1 = np.sum(ne*S*quad.w)
                    temp1 *= norm_/self.speed[beam_nber]
                    temp_beam[beam_nber] += temp1
                    
                self.density_beam[:,j] = self.density_beam[:,j-1] - \
                                                temp_beam

        self.nb_tck = []
        # interpolant for the beam
        for i in range(len(self.beam_comp)):
            self.density_beam[i,:] = n0[i]*np.exp(self.density_beam[i,:])
            self.nb_tck.append(interpolate.splrep(self.dl,self.density_beam[i,:]))
                
    def get_electron_temp(self,pos, eq=False):
        """ Return the value of the electron temperature from the
            simulation
            eq is used for computing the value of the equilibrium
        """
        raise NameError('Old version, should not be used')
        if eq:
            if len(pos.shape) == 1:
                R = np.sqrt(pos[0]**2 + pos[1]**2)
                psi = self.data.psi_interp(pos[2],R)
            else:
                R = np.sqrt(np.sum(pos[:,0:2]**2,axis=1))
                psi = self.data.psi_interp(pos[:,2],R)
            ret = self.data.te0_sp(psi)
        # the order of the grid is due to the XGC loading coordinate
        else:
            a = to_other_index(pos).T
            ret = self.Ti_int(a)

        """if len(pos.shape) == 1:
            if ret == 0.0:
                print pos
                raise NameError('Error point outside the mesh')
        else:
            if (ret == 0.0).any():
                print pos
                raise NameError('Error point outside the mesh')"""
        return ret


    def get_ion_temp(self,pos, eq=False):
        """ Return the value of the ion temperature from the
            simulation 
        """
        raise NameError('Old version, should not be used')
        if eq:
            if len(pos.shape) == 1:
                R = np.sqrt(pos[0]**2 + pos[1]**2)
                psi = self.data.psi_interp(pos[2],R)
            else:
                R = np.sqrt(np.sum(pos[:,0:2]**2,axis=1))
                psi = self.data.psi_interp(pos[:,2],R)
            ret = self.data.ti0_sp(psi)
        # the order of the grid is due to the XGC loading coordinate
        else:
            a = to_other_index(pos).T
            ret = self.Ti_int(a)

        """len(pos.shape) == 1:
            if ret == 0.0:
                print pos
                raise NameError('Error point outside the mesh')
        else:
            if (ret == 0.0).any():
                print pos
                raise NameError('Error point outside the mesh')"""
        return ret

            
    def get_mesh(self):
        """ Return the mesh (3D array)"""
        return self.mesh

    def get_origin(self):
        """ Return the origin of the beam"""
        return self.pos

    def get_beam_density(self,pos):
        """ Return the beam density at the position and time step wanted
            assuming a gaussian profile
        """
        if len(pos.shape) == 1:
            # array to return
            nb = np.zeros(len(self.beam_comp))
            # vector from beam origin to the wanted position
            dist = pos - self.get_origin()
            # projection along the axis
            proj = np.dot(dist,self.direc)
            # width of the beam
            stddev = self.get_width(proj)
            stddev2 = stddev**2
            if proj < 0:
                raise NameError('Point before the origin of the beam')
            # cubic spline for finding the value along the axis
            for i in range(len(self.beam_comp)):
                nb[i] = interpolate.splev(proj,self.nb_tck[i])
            # radius^2 on the plane perpendicular to the beam
            R2 = dist - proj*self.direc
            # radius in the horizontal plane
            xy = np.sum(R2[0:2]**2)
            # radius in the vertical plane
            z = R2[2]**2
            print 'need to check'
            # gaussian beam
            nb = nb*np.exp(-xy/(2*stddev2[0]) - z/(2*stddev2[1]))/(
                2*np.pi*stddev[0]*stddev[1])
            return nb
        
        else:
            pos = np.reshape(pos,(-1,3))
            nb = np.zeros((self.beam_comp.shape[0],pos.shape[0]))
            # vector from beam origin to the wanted position
            dist = pos - self.get_origin()
            # result is a 1D array containing the projection of the distance
            # from the origin over the direction of the beam
            proj = np.einsum('ij,j->i',dist,self.direc)
            stddev = self.get_width(proj)
            stddev2 = stddev**2

            # cubic spline for finding the value along the axis
            for i in range(len(self.beam_comp)):
                for j in range(pos.shape[0]):
                    nb[i,j] = interpolate.splev(proj[j],self.nb_tck[i], ext=1)
            # radius^2 on the plane perpendicular to the beam
            R2 = dist - proj[...,np.newaxis]*self.direc
            # compute the norm of each position
            xy = np.sum(R2[...,0:2]**2, axis=-1)
            z = R2[...,2]**2
            nb = nb*np.exp(-xy/(2*stddev2[0]) - z/(2*stddev2[1]))/(
                2*np.pi*stddev[0]*stddev[1])
            return nb

                    
        
    def get_emis(self,pos,t_):
        """ Return the emissivity at pos and time t_ 
            epsilon = <sigma*v> n_b n_e
            Argument:
            pos   -- 2D array, first index is for X,Y,Z
        """
        # first take all the value needed for the computation
        if len(pos.shape) == 1:
            pos = pos[np.newaxis,:]
        emis = np.zeros((len(self.beam_comp),pos.shape[0]))
        n_e,Ti = self.get_quantities(pos,t_,['ne','Ti'])
        # loop over all the type of collisions
        n_b = self.get_beam_density(pos)
        for k in self.coll_emis:
            file_nber = k[0]
            beam_nber = k[1]
            # compute the emission coefficient
            emis[beam_nber,:] += self.collisions.get_emission(
                self.beam_comp[beam_nber],n_e,self.mass_b[beam_nber],Ti,file_nber)
        # compute the emissivity
        emis = emis*n_e*n_b
        return emis


    def get_emis_lifetime(self,pos,t_):
        """ Return the emissivity at pos and time t_ 
            epsilon = <sigma*v> n_b n_e with the effect
            of the lifetime (depends on the position))
            Argument:
            pos   -- 2D array, first index is for X,Y,Z
        """
        print 'wavelength!!!'

        quad = integ.integration_points(1,'GL19') # Gauss-Legendre order 3
        emis = np.zeros((len(self.beam_comp),pos.shape[0]))
        # avoid the computation at each time
        ne_in, Ti_in,Te_in = self.get_quantities(pos,t_,['ne','Ti','Te'])
        for k in self.coll_emis:
            file_nber = k[0]
            beam_nber = k[1]
            # loop over all the position
            l = self.collisions.get_lifetime(ne_in,Te_in,Ti_in,
                                             self.beam_comp[beam_nber],
                                             self.mass_b[beam_nber],file_nber)
            dist = np.sqrt(np.sum((pos-self.pos)**2,axis=-1))
            # used for avoiding the discontinuity before the origin
            # of the beam
            up_lim = np.minimum(l*self.t_max*self.speed[beam_nber],dist)

            # split the distance in interval
            # copy and modify the source code of linspace
            step = up_lim/float(self.Nlt)
            delta = step[...,np.newaxis]*np.arange(0, self.Nlt)\
                    *np.ones(l.shape)[...,np.newaxis]
            
            # average position (a+b)/2
            av = 0.5*(delta[...,:-1] + delta[...,1:])
            # half distance (b-a)/2
            diff = 0.5*(-delta[...,:-1] + delta[...,1:])
            # integration points at each interval
            pt = av[...,np.newaxis] + diff[...,np.newaxis]*quad.pts

            # points in 3D space
            x = pos[np.newaxis,:,np.newaxis,np.newaxis,:] \
                - pt[...,np.newaxis]*self.direc

            n_e, Ti = self.get_quantities(x,t_,['ne','Ti'])

            n_e = np.reshape(n_e,pt.shape)
            Ti = np.reshape(Ti,pt.shape)

            n_b = self.get_beam_density(x)[beam_nber,...]
            n_b = np.reshape(n_b,pt.shape)
        
            
            f = self.collisions.get_emission(self.beam_comp[beam_nber],n_e.flatten()
                                             ,self.mass_b[beam_nber],Ti.flatten(),file_nber)
            f = np.reshape(f,pt.shape)
            
            f = n_b*f*n_e*np.exp(-pt/(l[...,np.newaxis,np.newaxis]*
                                      self.speed[beam_nber]))/self.speed[beam_nber]

            if np.isnan(f).any():
                print np.isnan(f)
                raise NameError('Mesh not well computed')
            
            f = np.einsum('kmn,n->km',f,quad.w)

            emis[beam_nber,:] = np.sum(diff*f,axis=-1)/l
        return emis