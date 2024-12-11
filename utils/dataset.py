# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
import cv2
import os.path
import io
import torch 
from PIL import Image
from .rcnn_utils import calculate_bounding_box, create_rcnn_data

class Dataset(data.Dataset):
    """# Dataset Class """

    def __init__(self, root='./', load_set='train', transform=None, num_kps3d=21, num_verts=778, hdf5_file=None):

        self.root = root
        self.transform = transform
        self.num_kps3d = num_kps3d
        self.num_verts = num_verts
        self.hdf5 = hdf5_file

        # TODO: add depth transformation
        self.load_set = load_set  # 'train','val','test'
        self.images = np.load(os.path.join(root, 'images-%s.npy' % self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy' % self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy' % self.load_set))

        self.mesh2d = np.load(os.path.join(root, 'mesh2d-%s.npy' % self.load_set))
        self.mesh3d = np.load(os.path.join(root, 'mesh3d-%s.npy' % self.load_set))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """

        image_path = self.images[index]
        palm = self.points3d[index][0]
        point2d = self.points2d[index]
        point3d = self.points3d[index] - palm # Center around palm

        ###-------------------------
        #因为使用make_data_freihand.py为freihand数据集生成的一系列文件中只包含了图片数据在相应rgb文件下的相对路径（而我们想要的是图片数据的绝对路径，借鉴的make_data.py是如此的，拿不到绝对路径将在后续读取图片的相关代码中报错line57，所以使用if语句用来给为freihand数据集生成的文件中的相对路径进行路径拼接已达到获得绝对路径的目的）
        if self.root == r"D:\Hand_Object_pose_shape\THOR-Net-ours\datasets\freihand\data":  #用数据集路径来判断是否是freihand数据集（因为只有freihand数据集才需要做拼接操作，ho3d数据集本身就是绝对路径）
            image_path = os.path.join(self.root, self.load_set, 'rgb', image_path)
        ###-------------------------

        # Load image and apply preprocessing if any
        if self.hdf5 is not None:
            data = np.array(self.hdf5[image_path])
            original_image = np.array(Image.open(io.BytesIO(data)))[..., :3]
        else:
            original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        inputs = self.transform(original_image)  # [:3]

        if self.load_set != 'test':
            # Loading 2D Mesh for bounding box calculation
            if self.num_kps3d == 21: #i.e. hand
                mesh2d = self.mesh2d[index][:778]
                mesh3d = self.mesh3d[index][:778] - palm
            else: # i.e. object
                mesh2d = self.mesh2d[index]
                mesh3d = self.mesh3d[index] - palm
      
            bb = calculate_bounding_box(mesh2d, increase=True)

            if self.num_verts > 0:
                boxes, labels, keypoints, keypoints3d = create_rcnn_data(bb, point2d, point3d, num_keypoints=self.num_kps3d)
                mesh3d = torch.Tensor(mesh3d[:self.num_verts][np.newaxis, ...]).float()
            else:
                boxes, labels, keypoints, keypoints3d = create_rcnn_data(bb, point2d, point3d, num_keypoints=self.num_keypoints)
                mesh3d = torch.tensor([])
        else:
            bb, mesh2d = np.array([]), np.array([])
            boxes, labels, keypoints, keypoints3d, mesh3d = torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([]), torch.Tensor([])

        data = {
            'path': image_path,
            'original_image': original_image,
            'inputs': inputs,
            'point2d': point2d,
            'point3d': point3d,
            'mesh2d': mesh2d,
            'bb': bb,
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints,
            'keypoints3d': keypoints3d,
            'mesh3d': mesh3d,
            'palm': torch.Tensor(palm[np.newaxis, ...]).float()
        }

        return data

    def __len__(self):
        return len(self.images)

########################################################################################################################
import os.path as osp
import copy
from pycocotools.coco import COCO
from FreiHand.utils.mano import MANO
from FreiHand.utils.preprocessing import process_bbox, load_img, augmentation
from FreiHand.utils.transforms import cam2pixel
import json



class FreiHandDataset(data.Dataset):
    def __init__(self, root='./', load_set='train', transform=None, hdf5_file=None):
        self.transform = transform
        self.data_split = load_set
        self.hdf5 = hdf5_file
        self.data_path = root
        self.human_bbox_root_dir = osp.join('D:/Hand_Object_pose_shape/THOR-Net-ours', 'datasets', 'freihand', 'rootnet_output', 'bbox_root_freihand_output.json')

        # MANO joint set
        self.mano = MANO()
        self.face = self.mano.face
        self.joint_regressor = self.mano.joint_regressor
        self.vertex_num = self.mano.vertex_num
        self.joint_num = self.mano.joint_num
        self.joints_name = self.mano.joints_name
        self.skeleton = self.mano.skeleton
        self.root_joint_idx = self.mano.root_joint_idx

        self.datalist = self.load_data()

    def load_data(self):
        if self.data_split == 'train':
            db = COCO(osp.join(self.data_path, 'freihand_train_coco.json'))
            with open(osp.join(self.data_path, 'freihand_train_data.json')) as f:
                data = json.load(f)

        else:
            db = COCO(osp.join(self.data_path, 'freihand_eval_coco.json'))
            with open(osp.join(self.data_path, 'freihand_eval_data.json')) as f:
                data = json.load(f)
            print("Get bounding box and root from " + self.human_bbox_root_dir)
            bbox_root_result = {}
            with open(self.human_bbox_root_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_root_result[str(annot[i]['image_id'])] = {'bbox': np.array(annot[i]['bbox']),
                                                               'root': np.array(annot[i]['root_cam'])}

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.data_path, img['file_name'])
            img_shape = (img['height'], img['width'])
            db_idx = str(img['db_idx'])

            if self.data_split == 'train':
                cam_param, mano_param, joint_cam = data[db_idx]['cam_param'], data[db_idx]['mano_param'], data[db_idx][
                    'joint_3d']
                joint_cam = np.array(joint_cam).reshape(-1, 3)
                bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
                if bbox is None: continue
                root_joint_depth = joint_cam[self.root_joint_idx][2]

            else:
                cam_param, scale = data[db_idx]['cam_param'], data[db_idx]['scale']
                joint_cam = np.ones((self.joint_num, 3), dtype=np.float32)  # dummy
                mano_param = {'pose': np.ones((48), dtype=np.float32), 'shape': np.ones((10), dtype=np.float32)}
                bbox = bbox_root_result[str(image_id)][
                    'bbox']  # bbox should be aspect ratio preserved-extended. It is done in RootNet.
                root_joint_depth = bbox_root_result[str(image_id)]['root'][2]

            datalist.append({
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_cam': joint_cam,
                'cam_param': cam_param,
                'mano_param': mano_param,
                'root_joint_depth': root_joint_depth})

        return datalist

    def get_mano_coord(self, mano_param, cam_param):
        pose, shape, trans = mano_param['pose'], mano_param['shape'], mano_param['trans']
        mano_pose = torch.FloatTensor(pose).view(1, -1);
        mano_shape = torch.FloatTensor(shape).view(1, -1);  # mano parameters (pose: 48 dimension, shape: 10 dimension)
        mano_trans = torch.FloatTensor(trans).view(1, 3)  # translation vector

        # get mesh and joint coordinates
        mano_mesh_coord, mano_joint_coord, _ = self.mano.layer(mano_pose, mano_shape, mano_trans)
        mano_mesh_coord = mano_mesh_coord.numpy().reshape(self.vertex_num, 3);
        mano_joint_coord = mano_joint_coord.numpy().reshape(self.joint_num, 3)

        # milimeter -> meter
        mano_mesh_coord /= 1000;
        mano_joint_coord /= 1000;
        return mano_mesh_coord, mano_joint_coord, mano_pose[0].numpy(), mano_shape[0].numpy()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, joint_cam, cam_param, mano_param = data['img_path'], data['img_shape'], data['bbox'], \
        data['joint_cam'], data['cam_param'], data['mano_param']

        boxes = torch.Tensor(bbox[np.newaxis, ...]).float()
        labels = torch.from_numpy(np.array([1]))

        if self.hdf5 is not None:
            data = np.array(self.hdf5[img_path])
            original_image = np.array(Image.open(io.BytesIO(data)))[..., :3]
        else:
            original_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, _ = augmentation(img, bbox, self.data_split,
                                                               exclude_flip=True)  # FreiHAND dataset only contains right hands. do not perform flip aug.
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split == 'train':
            # mano coordinates
            mano_mesh_cam, mano_joint_cam, mano_pose, mano_shape = self.get_mano_coord(mano_param, cam_param)
            mano_coord_cam = np.concatenate((mano_mesh_cam, mano_joint_cam))
            focal, princpt = cam_param['focal'], cam_param['princpt']
            mano_coord_img = cam2pixel(mano_coord_cam, focal, princpt)  # 2d需
            joints_img_xy1 = np.concatenate((mano_coord_img[:,:2], np.ones_like(mano_coord_img[:,:1])),1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            visibility = np.ones(21).reshape(-1, 1)
            keypoints = np.append(joints_img[:21], visibility, axis=1)
            # normalize to [0,1]
            keypoints[:,0] /= 1000
            keypoints[:,1] /= 1000

            inputs = {'inputs': img}
            targets = {'path': img_path, 'original_image': original_image, 'bb': bbox, 'boxes': boxes, 'labels': labels,
                       'keypoints': keypoints, 'keypoints3d': mano_joint_cam, 'mesh3d': mano_mesh_cam}

        else:
            inputs = {'img': img}
            targets = {}

        dataa = {}
        dataa.update(targets)
        dataa.update(inputs)
        data = {}

        for k, v in dataa.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if k == 'keypoints':
                v = v.float()
            data[k] = v

        return data
