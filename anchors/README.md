# Anchor / RPN 相关记录

## parameters
```python
    feature_map_size =  grid_size[:2] // out_size_factor   #grid_size  =   []   out_size_factor = int
    sizes=[1.6, 3.9, 1.56]
    anchor_strides=[0.4, 0.4, 1.0]
    anchor_offsets=[0.2, -39.8, -1.78]
    rotations=[0, np.pi / 2]
    class_id=None
    match_threshold=-1
    unmatch_threshold=-1
```


## algorithm
```python
    # 1.stride , offset
    x_stride, y_stride, z_stride = anchor_strides
    x_offset, y_offset, z_offset = anchor_offsets

    # 2.use feature_size to find anchor's centers point 锚点 
    z_centers = np.arange(feature_size[0], dtype=dtype)
    y_centers = np.arange(feature_size[1], dtype=dtype)
    x_centers = np.arange(feature_size[2], dtype=dtype)
    z_centers = z_centers * z_stride + z_offset
    y_centers = y_centers * y_stride + y_offset
    x_centers = x_centers * x_stride + x_offset
    # 3.anchor's size and rotations   
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)

    # 4.generate rectangle
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])   #====>np.tile 平铺过去，复制
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)   
    anchors = np.transpose(ret, [2, 1, 0, 3, 4, 5])   #(1, 200, 176, 1, 2, 7)
    anchors = anchors.reshape([-1, 7])    #(num_anchors,7)    ===>[x,y,z,w,l,h,yaw]                                       
```