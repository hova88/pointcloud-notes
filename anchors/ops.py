import numpy as np

def create_anchors_3d_stride(feature_size,
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
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


if __name__ == "__main__":
    import time 
    feature_map_size = [1,200,176]
    for _ in range(10):
        s = time.time()
        anchors = create_anchors_3d_stride(feature_map_size)
        e = time.time() - s
        print(e*1000,'ms')
    # print(anchors.shape) #(1, 200, 176, 1, 2, 7)
    # print(anchors[0])