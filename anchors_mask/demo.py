import numpy as np 
from ops import points_to_voxel
from tools import load 
import anchors_mask
import anchors
import time 


def main(coords, anchors_bv, voxel_size, point_cloud_range, grid_size, threshold = 10):
    

    shape = tuple(grid_size[::-1][1:])
    ret = np.zeros(shape,dtype=np.float32)

    dense_voxel_map = anchors_mask.sparse_sum_for_anchors_mask(coords, ret)
    dense_voxel_map = dense_voxel_map.cumsum(0)
    dense_voxel_map = dense_voxel_map.cumsum(1)
    anchors_area = anchors_mask.fused_get_anchors_area(
        dense_voxel_map, anchors_bv, voxel_size, point_cloud_range, grid_size)
    return anchors_area >= threshold


def run():
    points,p2,rect,Trv2c,image_shape,annos = load()
    #params
    point_cloud_range = [0, -40, -3, 70.4, 40, 3]
    voxel_size = [0.16, 0.16, 2]
    max_num_points = 35
    max_voxels = 12000
    threshold = 10

    #process
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    #anchors
    Anchors = anchors.create_anchors_3d_stride()
    anchors_bv = anchors.rbbox2d_to_near_bbox(Anchors[:, [0, 1, 3, 4, 6]]).astype(np.float32)

    voxels,coords,num_points_pre_voxels = points_to_voxel(points,voxel_size,
                                            point_cloud_range,
                                            max_num_points,
                                            False,
                                            max_voxels)


    shape = tuple(grid_size[::-1][1:])
    ret = np.zeros(shape,dtype=np.float32)

    dense_voxel_map = anchors_mask.sparse_sum_for_anchors_mask(coords, ret)
    dense_voxel_map = dense_voxel_map.cumsum(0)
    dense_voxel_map = dense_voxel_map.cumsum(1)
    anchors_area = anchors_mask.fused_get_anchors_area(
        dense_voxel_map, anchors_bv, voxel_size, point_cloud_range, grid_size)
    return anchors_area >= threshold





















if __name__=='__main__':

    points,p2,rect,Trv2c,image_shape,annos = load()
    #params
    point_cloud_range = [0, -40, -3, 70.4, 40, 3]
    voxel_size = [0.16, 0.16, 2]
    max_num_points = 35
    max_voxels = 12000
    threshold = 10

    #process
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    #anchors
    Anchors = anchors.create_anchors_3d_stride()
    anchors_bv = anchors.rbbox2d_to_near_bbox(Anchors[:, [0, 1, 3, 4, 6]]).astype(np.float32)

    voxels,coords,num_points_pre_voxels = points_to_voxel(points,voxel_size,
                                            point_cloud_range,
                                            max_num_points,
                                            False,
                                            max_voxels)


    for i in range(10):
        a = time.time()
        main(coords, anchors_bv, voxel_size, point_cloud_range, grid_size)
        b = time.time()-a

        print("anchor_mask frame take: {} ms ".format(b*1000))

    # anchor_mask took 273.84138107299805 ms
    # anchor_mask took 1.6858577728271484 ms
    # anchor_mask took 1.6162395477294922 ms
    # anchor_mask took 1.8193721771240234 ms
    # anchor_mask took 1.4204978942871094 ms
    # anchor_mask took 1.417398452758789 ms
    # anchor_mask took 1.417398452758789 ms
    # anchor_mask took 1.4157295227050781 ms
    # anchor_mask took 1.5590190887451172 ms
    # anchor_mask took 1.4972686767578125 ms
