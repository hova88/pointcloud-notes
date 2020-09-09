
# Anchors mask

生成 `anchors mask` (bool矩阵，[440,500] == grid size)：计算anchors具有的点云个数，并判断是否大于阈值。将大于阈值的选为可用anchors。


# Inputs
```
arg:
    --coords  ---->(5785, 3), [voxels的个数,以及每个voxels对应的坐标]
    --grid_size ---->array([440, 500,   3], dtype=int32)
    --anchors_bv ---->(70400, 4), [anchors的个数，以及每个voxels对应的xmin, ymin, xmax, ymax
    --pc_range ---->array([  0. , -40. ,  -3. ,  70.4,  40. ,   3. ], dtype=float32)
    --voxel_size --->array([0.16, 0.16, 2.  ], dtype=float32)
param:
    --stride: voxel_size
    --offset: pc_rang
```


# Algorithm

```python

# 1. 生成voxel map，将voxel（5000+）按照grid_size(440,500)展开，Voxels存在的[x,y]坐标上进行累加，不存在的地方为0
def sparse_sum_for_anchors_mask(coors, shape):
    ret = np.zeros(shape, dtype=np.float32)
    for i in range(coors.shape[0]):
        ret[coors[i, 1], coors[i, 2]] += 1
    return ret
dense_voxel_map = sparse_sum_for_anchors_mask(coords, tuple(self.grid_size[::-1][1:]))


# 2. 生成anchors_bv
def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

def center_to_minmax_2d(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)

rbboxes = anchors[:, [0, 1, 3, 4, 6]]
rots = rbboxes[..., -1]
rots_0_pi_div_2 = np.abs(limit_period(rots,0.5,np.pi))
cond = (rots_0_pi_div_2 > np.pi /4)[...,np.newaxis]
bboxes_center = np.where(cond,rbboxes[:,[0,1,3,2]],rbboxes[:,:4])
anchor_bv = np.concatenate([bboxes_center[:,:2] - [bboxes_center[:,2:] / 2, [bboxes_center[:,:2] + [bboxes_center[:,2:] / 2], axis=-1)


# 3. 列向累加、行向累加，summed-area tabel 
dense_voxel_map = dense_voxel_map.cumsum(0)
dense_voxel_map = dense_voxel_map.cumsum(1)

@numba.jit(nopython=True)
def fused_get_anchors_area(dense_map, anchors_bv, stride, offset,
                           grid_size):
    anchor_coor = np.zeros(anchors_bv.shape[1:], dtype=np.int32)   #---->(4,)
    grid_size_x = grid_size[0] - 1     #--->440-1=439
    grid_size_y = grid_size[1] - 1     #--->500-1=499
    N = anchors_bv.shape[0]            #---->N=70400
    ret = np.zeros((N), dtype=dense_map.dtype)
    for i in range(N):
        anchor_coor[0] = np.floor(
            (anchors_bv[i, 0] - offset[0]) / stride[0])    #------>将anchors平移到原始点云上
        anchor_coor[1] = np.floor(
            (anchors_bv[i, 1] - offset[1]) / stride[1])
        anchor_coor[2] = np.floor(
            (anchors_bv[i, 2] - offset[0]) / stride[0])
        anchor_coor[3] = np.floor(
            (anchors_bv[i, 3] - offset[1]) / stride[1])
        anchor_coor[0] = max(anchor_coor[0], 0)           #----->判断区域大小，裁减 grid_size以外的anchors
        anchor_coor[1] = max(anchor_coor[1], 0)
        anchor_coor[2] = min(anchor_coor[2], grid_size_x)
        anchor_coor[3] = min(anchor_coor[3], grid_size_y)
        ID = dense_map[anchor_coor[3], anchor_coor[2]]    #---->将anchor坐标映射到dense_map，判断anchor内所含voxel个数
        IA = dense_map[anchor_coor[1], anchor_coor[0]]
        IB = dense_map[anchor_coor[3], anchor_coor[0]]
        IC = dense_map[anchor_coor[1], anchor_coor[2]]
        ret[i] = ID - IB - IC + IA                        #----->summed_area tabel算法
    return ret
    
anchors_area = fused_get_anchors_area(
    dense_voxel_map, anchors_bv,voxel_size, pc_range, grid_size)     #---->准确来说还不是area,应该是每个anchor所含voxel个数，再整体成单个voxel的面积（0.16*0.16）就是面积
    
# 4. 通过判断anchors_area是否大于阈值，生成anchors_mask
anchors_mask = anchors_area >= 阈值
example['anchors_mask'] = anchors_mask

# ps. 辅助理解
def image_box_region_area(img_cumsum, bbox):
    """check a 2d voxel is contained by a box. used to filter empty
    anchors.
    Summed-area table algorithm:
    ==> W
    ------------------
    |      |         |
    |------A---------B
    |      |         |
    |      |         |
    |----- C---------D
    Iabcd = ID-IB-IC+IA
    Args:
        img_cumsum: [M, H, W](yx) cumsumed image.
        bbox: [N, 4](xyxy) bounding box, 
    """
    N = bbox.shape[0]
    M = img_cumsum.shape[0]
    ret = np.zeros([N, M], dtype=img_cumsum.dtype)
    ID = img_cumsum[:, bbox[:, 3], bbox[:, 2]]
    IA = img_cumsum[:, bbox[:, 1], bbox[:, 0]]
    IB = img_cumsum[:, bbox[:, 3], bbox[:, 0]]
    IC = img_cumsum[:, bbox[:, 1], bbox[:, 2]]
    ret = ID - IB - IC + IA
    return ret

```
    
# Usage
```python
python demo.py
```
