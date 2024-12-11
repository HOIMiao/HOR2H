from __future__ import print_function, unicode_literals

import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import json
import pickle
import re
from tqdm import tqdm
from torch.autograd import Variable

from models.thor_net import create_thor
from utils.dataset import Dataset
from utils.vis_utils import *
from utils.utils import *


def load_obj_pose(data, subset='train'):
    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

    cam_intr = data['camMat']
    obj_pose = data['objCorners3D']

    # Convert to non-OpenGL coordinates
    obj_pose = obj_pose.dot(coordChangeMat.T)

    # Project from 3D world to Camera coordinates using the camera matrix
    obj_pose_proj = cam_intr.dot(obj_pose.transpose()).transpose()
    obj_pose2d = (obj_pose_proj / obj_pose_proj[:, 2:])[:, :2]
    return obj_pose2d


def db_size(set_name, version='v2'):
    """ Hardcoded size of the datasets. """
    if set_name == 'train':
        if version == 'v2':
            return 66034  # number of unique samples (they exists in multiple 'versions')
        elif version == 'v3':
            return 78297
        else:
            raise NotImplementedError
    elif set_name == 'evaluation':
        if version == 'v2':
            return 11524
        elif version == 'v3':
            return 20137
        else:
            raise NotImplementedError
    else:
        assert 0, 'Invalid choice.'


def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.' % (f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data


def create_sequence(rcnn_dict, path, seq_length=1, n_points=21):
    frame_num = int(path.split('/')[-1].split('.')[0])
    point2d_seq = np.zeros((seq_length, n_points, 2))

    missing_frames = False

    for i in range(0, seq_length):
        if frame_num - i < 0:
            missing_frames = True
            break
        new_frame_num = '%04d' % (frame_num - i)
        new_path = re.sub('\d{4}', new_frame_num, path)
        if new_path in rcnn_dict.keys():
            point2d_seq[-i - 1] = rcnn_dict[new_path][:n_points, :2]
            last_pose = -i - 1
        else:  # Replicate the last pose in case of missing information
            point2d_seq[-i - 1] = point2d_seq[last_pose]

    if missing_frames:
        n_missing_frames = seq_length - i
        point2d_seq[0:-i] = np.tile(point2d_seq[-i], (n_missing_frames, 1, 1))

    point2d_seq = point2d_seq.reshape((seq_length * n_points, 2))

    return point2d_seq


def main(base_path, pred_out_path, pred_func, version, model, set_name=None, mesh=False, seq_length=1):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'
    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()

    # read list of evaluation files
    with open(os.path.join(base_path, set_name + '.txt')) as f:
        file_list = f.readlines()
    file_list = [f.strip() for f in file_list]

    # iterate over the dataset once
    xyz = []
    verts = []

    transform_function = transforms.Compose([transforms.ToTensor()])
    testset = Dataset(root="D:\Hand_Object_pose_shape\THOR-Net-ours\datasets\ho3d_v2\data", load_set='test', transform=transform_function, num_kps3d=num_kps3d, num_verts=num_verts)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16, collate_fn=ho3d_collate_fn)

    # for idx in tqdm(range(db_size(set_name, version))):
    for idx, ts_data in enumerate(tqdm(testloader)):
        if idx >= db_size(set_name, version):
            break

        seq_name = file_list[idx].split('/')[0]
        file_id = file_list[idx].split('/')[1]

        rgb_path = os.path.join(base_path, set_name, seq_name, 'rgb', file_id + '.jpg')
        meta_path = os.path.join(base_path, set_name, seq_name, 'meta', file_id + '.pkl')

        aux_info = read_annotation(base_path, seq_name, file_id, set_name)
        # obj_point2d = load_obj_pose(aux_info, subset='test')

        # img = read_RGB_img(base_path, seq_name, file_id, set_name)
        data_dict = ts_data
        inputs = [t['inputs'].to(device) for t in data_dict]
        keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d', 'palm']
        img = inputs[0].cpu().detach().numpy()


        # use some algorithm for prediction
        xyz, verts = pred_func(model, inputs, data_dict, img, keys)

        # simple check if xyz and verts are in opengl coordinate system
        if np.all(xyz[:, 2] > 0) or np.all(verts[:, 2] > 0):
            if np.all(xyz[:, 2] > 0):
                xyz[:, 2] = -xyz[:, 2]
            else:
                verts[:, 2] = -verts[:, 2]
        #     ## continue
        #     print(seq_name, file_id, xyz)
        #     raise Exception(
        #         'It appears the pose estimates are not in OpenGL coordinate system. Please read README.txt in dataset folder. Aborting!')

        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(pred_out_path, xyz_pred_list, verts_pred_list)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def pred_template(model, inputs, data_dict, img, keys):

    outputs = model(inputs)
    # xyz = outputs3d.cpu().detach().numpy()[0][-21:]
    # verts = np.zeros((778, 3))

    predictions, img, palm, labels = prepare_data_for_evaluation(data_dict, outputs, img, keys, device, 'test')

    keypoint3d= predictions['keypoints3d'][0]
    mesh3d= predictions['mesh3d'][0][:, :3]
    xyz = keypoint3d[:21]
    verts = mesh3d[:778]

    xyz = np.array(xyz)
    verts = np.array(verts)

    # OpenGL coordinates and reordering
    order_idx = np.argsort(np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]))
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

    xyz = xyz.dot(coord_change_mat.T)[order_idx] / 1000
    verts = verts.dot(coord_change_mat.T) / 1000
    # xyz = xyz.dot(coord_change_mat.T)[order_idx]
    # verts = verts.dot(coord_change_mat.T)

    return xyz, verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--base_path', type=str, default='D:/Hand_Object_pose_shape/THOR-Net-ours/datasets/ho3d_v2/data',
                        help='Path to where the HO3D dataset is located.')
    parser.add_argument('--out', type=str, default='D:/Hand_Object_pose_shape/THOR-Net-ours/innovation/mypred_output/pred.json', help='File to save the predictions.')
    parser.add_argument('--version', type=str, default='v2', help='version number')
    parser.add_argument('--checkpoint_folder',
                        default='hand-object/ho3d_v2_checkpoints', help='the folder of the pretrained model')
    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=128, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--seq_length', type=int, default=1, help='Sequence length')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    args = parser.parse_args()

    # define model
    num_classes = 2
    num_kps2d, num_kps3d, num_verts = 29, 29, 1778
    graph_input = 'heatmaps'
    photometric = True
    num_features = 2048
    gpu_number = [0]
    device = torch.device(f'cuda:{gpu_number[0]}' if torch.cuda.is_available() else 'cpu')
    model = create_thor(pretrained=False, num_classes=num_classes, device=device,
                                    num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_verts=num_verts,
                                    rpn_post_nms_top_n_test=num_classes-1,
                                    box_score_thresh=0.0,
                                    photometric=photometric, graph_input=graph_input, dataset_name='ho3d_v2',
                                    num_features=num_features, hid_size=args.dim_model)

    # Load pretrained model
    pretrained_model = f'D:/Hand_Object_pose_shape/THOR-Net-ours/checkpoints/{args.checkpoint_folder}/model-50.pkl'
    print("==> Loading checkpoint '{}'".format(pretrained_model))

    if torch.cuda.is_available():
        model = model.cuda(device=gpu_number[0])
        model = nn.DataParallel(model, device_ids=gpu_number)

    model.load_state_dict(torch.load(pretrained_model, map_location='cuda:0'))
    model.eval()

    # call with a predictor function
    main(
        args.base_path,
        args.out,
        pred_func=pred_template,
        set_name='evaluation',
        version=args.version,
        model=model,
        seq_length=args.seq_length
    )