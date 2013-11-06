"""Class definitions of grids 
"""
#moudle depends on numpy package
import numpy as np

class GridError(Exception):
    def __init__(this,*p):
        this.args = p

class Grid:
    """Base class for all kinds of grids

    contains name of the grid
    """
    def __init__(this):
        this._name = 'General Grids'

    def tell(this):
        return this._name

class Cartesian2D(Grid):
    """Cartesian grids in 2D space. Generally corresponds a toroidal slice, i.e. R-Z plane. Rectangular shape assumed.

    Attributes:
        float ResR: the resolution in R direction. 
        float ResZ: the resolution in Z direction.
        point DownLeft: the coordinates of the down left cornor point, given in (Zmin,Rmin) value pair form.
        point UpRight: the coordinates of the up right cornor point, given in (Zmax,Rmax) form.
        int NR,NZ: The gird number in R,Z directions. Can be specified initially or derived from other parameters.
        2darray R2D: R values on 2D grids. R(0,:) gives the 1D R values.
        2darray Z2D: Z values on 2D grids. R is the fast changing variable so Z2D(:,0) gives the 1D Z values.

        
    """
    def __init__(this, **P):
        """initialize the cartesian grid object.

        If either DownLeft or UpRight is not specified, a GridError exception will be raised.
        Either NR or ResN can be specified. If none or both, a GridError exception will be raised. Same as NZ and ResZ
        """
        this._name = '2D Cartesian Grids'
        try:
            if ( 'DownLeft' in P.keys() and 'UpRight' in P.keys() ):
                this.DownLeft ,this.UpRight = P['DownLeft'], P['UpRight']
                Zmax,Rmax = this.UpRight
                Zmin,Rmin = this.DownLeft
                rangeR = float(Rmax - Rmin)
                rangeZ = float(Zmax - Zmin)
                if ( 'NR' in P.keys() and not 'ResR' in P.keys() ):
                    this.NR = P['NR']
                    this.ResR = rangeR / this.NR                
                elif ('ResR' in P.keys() and not 'NR' in P.keys() ):
                    this.NR = int ( rangeR/P['ResR'] + 2 ) # make sure the actual resolution is finer than the required one
                    this.ResR = rangeR / this.NR
                else:
                    raise GridError('NR and ResR missing or conflicting, make sure you specify exactly one of them.')
                if ( 'NZ' in P.keys() and not 'ResZ' in P.keys() ):
                    this.NZ = P['NZ']
                    this.ResZ = rangeZ / this.NZ                
                elif ('ResZ' in P.keys() and not 'NZ' in P.keys() ):
                    this.NZ = int ( rangeZ/P['ResZ'] + 2 ) # make sure the actual resolution is finer than the required one
                    this.ResZ = rangeZ / this.NZ
                else:
                    raise GridError('NZ and ResZ missing or conflicting, make sure you specify exactly one of them.')
            else:
                raise GridError("Initializing Grid fails: UpLeft or DownRight not set.")
        except GridError:
            #save for further upgrades, may handle GridError here
            raise
        except:
            print 'Unexpected error in grid initialization! During reading and comprehensing the arguments.'
            raise

        #now create the 2darrays for R and Z
        this.R2D = np.zeros( (this.NZ,this.NR) ) + np.linspace(Rmin,Rmax,this.NR)[ np.newaxis , : ]
        this.Z2D = np.zeros( (this.NZ,this.NR) ) + np.linspace(Zmin,Zmax,this.NZ)[ : , np.newaxis ]    

    def tell(this):
        """returns the key informations of the grids
        """
        info = this._name + '\n'
        info += 'DownLeft :' + str(this.DownLeft) +'\n'
        info += 'UpRight :' + str(this.UpRight) +'\n'
        info += 'NR,ResR :' + str( (this.NR,this.ResR) ) +'\n'
        info += 'NZ,ResZ :' + str( (this.NZ,this.ResZ) ) +'\n'
        return info


class path:
    """class of the light path, basically just a series of points

    Attributes:
    n: int, number of points on the path
    R: double[n], R coordinates of the points
    Z: double[n], Z coordinates of the points 
    """    
    def __init__(this, n=0, R=np.zeros(1), Z=np.zeros(1)):
        this.n = n
        this.R = R
        this.Z = Z
    def __setitem__(this,p2):
        this.n = p2.n
        this.R = np.copy(p2.R)
        this.Z = np.copy(p2.Z)


class Path2D(Grid):
    """ R-Z Grid created based on an light path

        Attributes:
        ResS : double, resolution in light path length variable s
        R2D : double[], R coordinates still stored in 2D array, but one dimension is actually not used
        Z2D : double[], Corresponding Z coordinates,
        s : double[], path length coordinates, start with s=0
        N : int[], number of grid points, accumulated in each section
                
    """
    def __init__(this, pth, ResS):
        """initialize with a path object pth, and a given resolution ResS
        """
        this._name = "2D Light Path Grid"
        this.pth = pth
        n = pth.n
        this.ResS = ResS
        this.s = np.empty((n)) #s is the array stores the length of path variable
        this.s[0]=0 # start with s=0
        this.N = np.empty((n)) #N is the array stores the number of grid points 
        this.N[0]=1 # The starting point is considered as 1 grid
        for i in range(1,n):
            this.s[i]=( np.sqrt((pth.R[i]-pth.R[i-1])**2 + (pth.Z[i]-pth.Z[i-1])**2) + this.s[i-1] ) # increase with the length of each section
            this.N[i]=( np.ceil((this.s[i]-this.s[i-1])/ResS)+ this.N[i-1] ) # increase with the number that meet the resolution requirement
        this.R2D = np.empty((1,this.N[n-1]))
        this.Z2D = np.empty((1,this.N[n-1]))
        this.s2D = np.empty((1,this.N[n-1]))
        for i in range(1,n):
            this.R2D[ 0, (this.N[i-1]-1): this.N[i]] = np.linspace(pth.R[i-1],pth.R[i],this.N[i]-this.N[i-1]+1) #fill in the middle points with equal space
            this.Z2D[ 0, (this.N[i-1]-1): this.N[i]] = np.linspace(pth.Z[i-1],pth.Z[i],this.N[i]-this.N[i-1]+1)
            this.s2D[ 0, (this.N[i-1]-1): this.N[i]] = this.s[i-1]+ np.sqrt( (this.R2D[0,(this.N[i-1]-1): this.N[i]] - this.pth.R[i-1])**2 + (this.Z2D[0,(this.N[i-1]-1): this.N[i]] - this.pth.Z[i-1])**2 )

    def tell(this):
        """display information
        """
        info = this._name + "\n"
        info += "created by path:\n"
        info += "\tnumber of points: "+ str(this.pth.n)+"\n"
        info += "\tR coordinates:\n\t"+ str(this.pth.R)+"\n"
        info += "\tZ coordinates:\n\t"+ str(this.pth.Z)+"\n"
        info += "with resolution in S: "+str(this.ResS)+"\n"
        info += "total length of path: "+str(this.s[this.pth.n-1])+"\n"
        return info
