#!/usr/bin/env python

import sdp.plasma.xgc.loader as xgc
import sdp.geometry.grid as Grid
import sdp.plasma.TestParameter as tp

reload(xgc)
reload(tp)

xgc_path ='/p/gkp/lshi/XGC1_NSTX_Case/XGC_output/'
my_grid3D = Grid.Cartesian3D(**tp.xgc_test3D)
my_grid2D = Grid.Cartesian2D(**tp.xgc_test2D)
xgc1 = xgc.XGC_loader(xgc_path,my_grid3D,1000,1001,1)

#print xgc1.grid.tell()

#xgc1.change_grid(my_grid2D)

#print xgc1.grid.tell()

R = xgc1.mesh['R']
Z = xgc1.mesh['Z']




