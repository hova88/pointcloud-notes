
import numpy as np 
import mayavi.mlab as mlab 
from script import create_anchors_3d_stride
import vis 

gt_boxes = create_anchors_3d_stride([1,200,176]).reshape(-1,7)[:4]
corners3d = vis.boxes_to_corners_3d(gt_boxes)
fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(600, 600))
fig = vis.draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

mlab.show()