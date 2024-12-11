import torch
import numpy as np
import pickle
import torchvision.transforms as transforms
# from scipy.spatial import procrustes

from .dataset import Dataset, FreiHandDataset
from FreiHand.utils.dataset import MultipleDatasets
    

def ho3d_collate_fn(batch):
    # print(batch, '\n--------------------\n')
    # print(len(batch))
    return batch

def h2o_collate_fn(samples):
    output_list = []
    for sample in samples:
        sample_dict = {
            'path': sample[0],
            'inputs': sample[1],
            'keypoints2d': sample[2],
            'keypoints3d': sample[3].unsqueeze(0),
            'mesh2d': sample[4],
            'mesh3d': sample[5].unsqueeze(0),
            'boxes': sample[6],
            'labels': sample[7],
            'keypoints': sample[8]
        }
        output_list.append(sample_dict)
    return output_list

def create_loader(dataset_name, root, split, batch_size, num_kps3d=21, num_verts=778, h2o_info=None):

    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name == 'freihand111':
        dataset = FreiHandDataset(root=root, load_set=split, transform=transform)
        # trainset_loader = MultipleDatasets(dataset, make_same_len=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=ho3d_collate_fn, drop_last=True)

        return loader, dataset
    else :  #ho3d_v2. ho3d_v3, freihand
        dataset = Dataset(root=root, load_set=split, transform=transform, num_kps3d=num_kps3d, num_verts=num_verts)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=ho3d_collate_fn, drop_last=True)
        
        return loader, dataset

def freeze_component(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
def calculate_keypoints(dataset_name, obj):

    if dataset_name == 'ho3d_v2' or dataset_name == 'ho3d_v3' or dataset_name == 'freihand':
        num_verts = 1778 if obj else 778
        num_kps3d = 29 if obj else 21
        num_kps2d = 29 if obj else 21

    else:
        num_verts = 2556 if obj else 1556
        num_kps3d = 50 if obj else 42
        num_kps2d = 50 if obj else 42

    return num_kps2d, num_kps3d, num_verts

def mpjpe(predicted, target):   #non-aligned MPJPE
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def pa_mpjpe(predicted, target):    #aligned MPJPE
    # mtx1, mtx2, disparity, R, s = procrustes(target, predicted)   #对应导入procrustes包
    aligned_predicted = procrustes(target, predicted)[1]
    # aligned_predicted = s * R.dot(predicted.T)+ disparity    #对应导入procrustes包
    # aligned_predicted = s * np.dot(predicted, R)+ (mtx1 - mtx2)   #效果很差    #对应导入procrustes包
    # aligned_predicted = s * R.dot(predicted.T)+ (mtx1.T - mtx2.T)   #对应导入procrustes包
    # return mpjpe(torch.Tensor(aligned_predicted.T), torch.Tensor(target))    #对应导入procrustes包
    return mpjpe(torch.Tensor(aligned_predicted), torch.Tensor(target))

def save_calculate_error(predictions, labels, path, errors, output_dicts, c, xyz_err_procrustes_al, verts_err_procrustes_al, xyz_auc_procrustes_al, verts_auc_procrustes_al, F_score_al, PCK, num_classes=2, dataset_name='h2o3d', obj=True, generate_mesh=False):
    """Stores the results of the model in a dict and calculates error in case of available gt"""

    predicted_labels = list(predictions['labels'])

    rhi, obji = 0, 21
    rhvi, objvi = 0, 778

    if dataset_name == 'h2o3d':
        rhi, obji = 21, 42
        rhvi, objvi = 778, 778*2

    if (num_classes > 2 and set([1, 2, 3]).issubset(predicted_labels)) or (num_classes == 2 and 1 in predicted_labels):

        keypoints = predictions['keypoints3d'][0]
        keypoints_gt = labels['keypoints3d'][0]
        
        if generate_mesh:
            mesh = predictions['mesh3d'][0][:, :3]
            mesh_gt = labels['mesh3d'][0]
        else:
            mesh = np.zeros((2556, 3))
            mesh_gt = np.zeros((2556, 3))

        rh_pose, rh_pose_gt = keypoints[rhi:rhi+21], keypoints_gt[rhi:rhi+21]
        rh_mesh, rh_mesh_gt = mesh[rhvi:rhvi+778], mesh_gt[rhvi:rhvi+778]

        if obj:
            obj_pose, obj_pose_gt = keypoints[obji:], keypoints_gt[obji:]
            obj_mesh, obj_mesh_gt = mesh[objvi:], mesh_gt[objvi:]
        else:
            obj_pose, obj_pose_gt = np.zeros((8, 3)), np.zeros((8, 3))
            obj_mesh, obj_mesh_gt = np.zeros((1000, 3)), np.zeros((1000, 3))
            

        if dataset_name == 'h2o3d':
            lh_pose, lh_pose_gt = keypoints[:21], keypoints_gt[:21]
            lh_mesh, lh_mesh_gt = mesh[:778], mesh_gt[:778]
        else:
            lh_pose, lh_pose_gt = np.zeros((21, 3)), np.zeros((21, 3))
            lh_mesh, lh_mesh_gt = np.zeros((778, 3)), np.zeros((778, 3))

        if dataset_name =='ho3d_v2' or dataset_name == 'ho3d_v3':
            pair_list = [
                (rh_pose, rh_pose_gt),
                (rh_mesh, rh_mesh_gt),
                (obj_pose, obj_pose_gt),
                (obj_mesh, obj_mesh_gt)
            ]

        if dataset_name == 'h2o3d':
            pair_list = [
                (lh_pose, lh_pose_gt),
                (lh_mesh, lh_mesh_gt),
                (rh_pose, rh_pose_gt),
                (rh_mesh, rh_mesh_gt),
                (obj_pose, obj_pose_gt),
                (obj_mesh, obj_mesh_gt)
            ]

        for i in range(len(pair_list)):

            # error = mpjpe(torch.Tensor(pair_list[i][0]), torch.Tensor(pair_list[i][1]))
            error = pa_mpjpe(pair_list[i][0], pair_list[i][1])
            errors[i].append(error)

        # error = mpjpe(torch.Tensor(mesh), torch.Tensor(mesh_gt))
        error = pa_mpjpe(mesh, mesh_gt)
    else:
        c += 1
        error = 1000
        keypoints = np.zeros((50, 3))
        mesh = np.zeros((2556, 3))
        print(c)
      
    output_dicts[0][path] = keypoints
    output_dicts[1][path] = mesh   

    # Object pose
    # output_dicts[1][path] = keypoints_gt[42:]

    ###------------------------------------------------
    #more metric
    from innovation.more_metric import EvalUtil, align_w_scale, calculate_fscore
    eval_xyz, eval_xyz_procrustes_aligned = EvalUtil(), EvalUtil()
    eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
    f_score_aligned = []
    f_threshs = [5, 15]

    xyz, verts = rh_pose_gt, rh_mesh_gt
    xyz, verts = [np.array(x) for x in [xyz, verts]]

    xyz_pred, verts_pred = rh_pose, rh_mesh
    xyz_pred, verts_pred = [np.array(x) for x in [xyz_pred, verts_pred]]

    xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
    verts_pred_aligned = align_w_scale(verts, verts_pred)

    #joints_err_aligned_procrustes
    eval_xyz_procrustes_aligned.feed(
        xyz,
        np.ones_like(xyz[:,0]),
        xyz_pred_aligned
    )

    #mesh_err_aligned_procrustes
    eval_mesh_err_aligned.feed(
        verts,
        np.ones_like(verts[:, 0]),
        verts_pred_aligned
    )

    #F-scores_aligned
    la = []
    for t in f_threshs:
        f, _, _ = calculate_fscore(verts, verts_pred_aligned, t)
        la.append(f)
    f_score_aligned.append(la)

    #PCK
    rh_mesh_aligned = align_w_scale(rh_mesh_gt, rh_mesh)
    rh_mesh_err_aligned = np.abs(rh_mesh_aligned - rh_mesh_gt)
    rh_joint_aligned = align_w_scale(rh_pose_gt, rh_pose)
    rh_joint_err_aligned = np.abs(rh_joint_aligned - rh_pose_gt)
    result = []
    pck_threshs = [0, 10, 20, 30, 40, 50]
    for t in pck_threshs:
        pck = np.mean((rh_mesh_err_aligned <= t).astype('float'))
        result.append(pck)
    PCK[0].append(result)
    result = []
    for t in pck_threshs:
        pck = np.mean((rh_joint_err_aligned <= t).astype('float'))
        result.append(pck)
    PCK[1].append(result)

    #Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 50.0, 100)

    xyz_procrustes_al_mean3d, _, xyz_procrustes_al_auc3d, pck_xyz_procrustes_al, thresh_xyz_procrustes_al = eval_xyz_procrustes_aligned.get_measures(0.0, 50.0, 100)

    mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 50.0, 100)

    mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = eval_mesh_err_aligned.get_measures(0.0, 50.0, 100)

    xyz_err_procrustes_al.append(xyz_procrustes_al_mean3d)
    verts_err_procrustes_al.append(mesh_al_mean3d)
    xyz_auc_procrustes_al.append(xyz_procrustes_al_auc3d)
    verts_auc_procrustes_al.append(mesh_al_auc3d)
    F_score_al.append(f_score_aligned)

    ###------------------------------------------------

    return c

def save_dicts(output_dicts, split):
    
    output_dict = dict(sorted(output_dicts[0].items()))
    output_dict_mesh = dict(sorted(output_dicts[1].items()))
    print('Total number of predictions:', len(output_dict.keys()))

    with open(f'./outputs/rcnn_outputs/rcnn_outputs_29_{split}_3d_v3.pkl', 'wb') as f:
        pickle.dump(output_dict, f)

    with open(f'./outputs/rcnn_outputs/rcnn_outputs_1778_{split}_3d_v3.pkl', 'wb') as f:
        pickle.dump(output_dict_mesh, f)

def prepare_data_for_evaluation(data_dict, outputs, img, keys, device, split):
    """Postprocessing function"""

    # print(data_dict[0])
    targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]

    labels = {k: v.cpu().detach().numpy() for k, v in targets[0].items()}
    predictions = {k: v.cpu().detach().numpy() for k, v in outputs[0].items()}

    palm = None
    if 'palm' in labels.keys():     #labels.keys() == keys
        palm = labels['palm'][0]

    if split == 'test':     #目前我认为当split=='test'时，这两行代码可以注释掉 #现认为不可注释（注释后其表达意思有变，不过检查代码后发现“注释也无影响”）
        labels = None

    img = img.transpose(1, 2, 0) * 255
    img = np.ascontiguousarray(img, np.uint8) 

    return predictions, img, palm, labels

def project_3D_points(pts3D):

    cam_mat = np.array(
        [[617.343,0,      312.42],
        [0,       617.343,241.42],
        [0,       0,       1]])

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0] / proj_pts[:,2], proj_pts[:,1] / proj_pts[:,2]], axis=1)
    # proj_pts = proj_pts.to(torch.long)
    return proj_pts


def generate_gt_texture(image, mesh3d):
    mesh2d = project_3D_points(mesh3d)

    image = image / 255

    H, W, _ = image.shape

    idx_x = mesh2d[:, 0].clip(min=0, max=W-1).astype(np.int)
    idx_y = mesh2d[:, 1].clip(min=0, max=H-1).astype(np.int)

    texture = image[idx_y, idx_x]
    
    return texture

def calculate_rgb_error(image, mesh3d, p_texture):
    texture = generate_gt_texture(image, mesh3d)
    error = mpjpe(torch.Tensor(texture), torch.Tensor(p_texture))
    return error

#自定义 procrustes
def procrustes(A, B, scaling=True, reflection='best'):
    """ A port of MATLAB's `procrustes` function to Numpy.

    $$ \min_{R, T, S} \sum_i^N || A_i - R B_i + T ||^2. $$
    Use notation from [course note]
    (https://fling.seas.upenn.edu/~cis390/dynamic/slides/CIS390_Lecture11.pdf).

    Args:
        A: Matrices of target coordinates.
        B: Matrices of input coordinates. Must have equal numbers of  points
            (rows), but B may have fewer dimensions (columns) than A.
        scaling: if False, the scaling component of the transformation is forced
            to 1
        reflection:
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

    Returns:
        d: The residual sum of squared errors, normalized according to a measure
            of the scale of A, ((A - A.mean(0))**2).sum().
        Z: The matrix of transformed B-values.
        tform: A dict specifying the rotation, translation and scaling that
            maps A --> B.
    """
    assert A.shape[0] == B.shape[0]
    n, dim_x = A.shape
    _, dim_y = B.shape

    # remove translation
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    A0 = A - A_bar
    B0 = B - B_bar

    # remove scale
    ssX = (A0**2).sum()
    ssY = (B0**2).sum()
    A_norm = np.sqrt(ssX)
    B_norm = np.sqrt(ssY)
    A0 /= A_norm
    B0 /= B_norm

    if dim_y < dim_x:
        B0 = np.concatenate((B0, np.zeros(n, dim_x - dim_y)), 0)

    # optimum rotation matrix of B
    A = np.dot(A0.T, B0)
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    R = np.dot(V, U.T)

    if reflection != 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(R) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            R = np.dot(V, U.T)

    S_trace = s.sum()
    if scaling:
        # optimum scaling of B
        scale = S_trace * A_norm / B_norm

        # standarised distance between A and scale*B*R + c
        d = 1 - S_trace**2

        # transformed coords
        Z = A_norm * S_trace * np.dot(B0, R) + A_bar
    else:
        scale = 1
        d = 1 + ssY / ssX - 2 * S_trace * B_norm / A_norm
        Z = B_norm * np.dot(B0, R) + A_bar

    # transformation matrix
    if dim_y < dim_x:
        R = R[:dim_y, :]
    translation = A_bar - scale * np.dot(B_bar, R)

    # transformation values
    tform = {'rotation': R, 'scale': scale, 'translation': translation}
    return d, Z, tform
