import numpy as np 
import tools 
from mayavi import mlab 
import vis 


removed_points,points = tools.remove_outside_point()
fig = vis.visualize_pts(points)
vis.draw_sphere_pts(removed_points,fig= fig)
mlab.show()