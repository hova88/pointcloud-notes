import numpy as np


def create_anchors_3d_stride(feature_size=  [1,200,176],
                             sizes=[1.6, 3.9, 1.56],
                             anchor_strides=[0.4, 0.4, 0.0],
                             anchor_offsets=[0.2, -39.8, -1.78],
                             rotations=[0, np.pi / 2],
                             dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    # almost 2x faster than v1
    x_stride, y_stride, z_stride = anchor_strides
    x_offset, y_offset, z_offset = anchor_offsets
    z_centers = np.arange(feature_size[0], dtype=dtype)
    y_centers = np.arange(feature_size[1], dtype=dtype)
    x_centers = np.arange(feature_size[2], dtype=dtype)
    z_centers = z_centers * z_stride + z_offset
    y_centers = y_centers * y_stride + y_offset
    x_centers = x_centers * x_stride + x_offset
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5]).reshape(-1,7)


def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)
    
    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
    
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners

######################################
############ anchors_bs ##############
######################################



def rbbox2d_to_near_bbox(rbboxes):
    """convert rotated bbox to nearest 'standing' or 'lying' bbox.
    Args:
        rbboxes: [N, 5(x, y, xdim, ydim, rad)] rotated bboxes
    Returns:
        bboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    rots = rbboxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def center_to_minmax_2d(centers, dims, origin=0.5):
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def center_to_minmax_2d_0_5(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)
    
    Args:
        centers (float array, shape=[N, 2]): locations in kitti label file.
        dims (float array, shape=[N, 2]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
    
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners



    
