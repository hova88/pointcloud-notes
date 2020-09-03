import numpy as np 
from skimage import io
import ops 


def load(   v_path = 'dataset/kitti/000050.bin',
            calib_path = 'dataset/kitti/calib_000050.txt',
            label_path = 'dataset/kitti/label_000050.txt',
            img_path = 'dataset/kitti/000050.png'):

    points_v = np.fromfile(v_path,dtype=np.float32,count=-1).reshape([-1,4])
    anno = get_label_anno(label_path)
    with open(calib_path,'r') as f:
        lines = f.readlines()
    p2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3,4])

    p2 = np.concatenate([p2,np.array([[0.,0.,0.,1.]])],axis=0)
    R0_rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
    rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
    rect_4x4[3, 3] = 1.
    rect_4x4[:3, :3] = R0_rect
    image_shape = np.array(io.imread(img_path).shape[:2],dtype=np.int32)
    Tr_velo_to_cam = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
    Trv2c = np.concatenate([Tr_velo_to_cam,np.array([[0.,0.,0.,1.]])],axis=0)
    return points_v,p2,rect_4x4,Trv2c,image_shape,anno  


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name':[],
        'truncated':[],
        'occluded':[],
        'alpha':[],
        'bbox':[],
        'dimensions':[],
        'location':[],
        'rotation_y':[]
    })
    with open(label_path,'r') as f:
        lines = f.readlines()
    
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)
    
    if len(content) != 0 and len(content[0]) == 16: #score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

