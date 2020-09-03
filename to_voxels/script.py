import numpy as np 
from ops import points_to_voxel
from tools import load 
import time 


def main():

    points,p2,rect,Trv2c,image_shape,annos = load()
    #params
    point_cloud_range = [0, -40, -3, 70.4, 40, 3]
    voxel_size = [0.16, 0.16, 2]
    max_num_points = 35
    max_voxels = 12000

    #process
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    # print(grid_size)
    #compute
    for _ in range(10):
        a = time.time()
        voxels,_,_ = points_to_voxel(points,
                                    voxel_size,
                                    point_cloud_range,
                                    max_num_points,
                                    False,
                                    max_voxels)
        b = time.time() - a 
        print('frame speed: {} ms/id'.format(b * 1000))

    return voxels.reshape(-1,4),points


if __name__ == '__main__':
    import mayavi.mlab as mlab 
    import vis 
    voxels,points = main()
    fig = vis.visualize_pts(points)
    vis.draw_cube_pts(voxels,fig= fig)
    mlab.show()
# frame speed: 2.462148666381836 ms/id
# frame speed: 4.048585891723633 ms/id
# frame speed: 3.7550926208496094 ms/id
# frame speed: 2.560138702392578 ms/id
# frame speed: 1.9612312316894531 ms/id
# frame speed: 1.8930435180664062 ms/id
# frame speed: 1.9588470458984375 ms/id
# frame speed: 1.8994808197021484 ms/id
# frame speed: 2.1104812622070312 ms/id

