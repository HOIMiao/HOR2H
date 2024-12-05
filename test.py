from posixpath import split
import torch 
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import argparse

# matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

from utils.dataset import Dataset
from utils.vis_utils import *
from tqdm import tqdm
from models.thor_net import create_thor
from utils.utils import *
from utils.options import parse_args_function

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    left_hand_faces, right_hand_faces, obj_faces = load_faces()

    def visualize2d(img, predictions, labels=None, filename=None, palm=None, evaluate=False):

        fig = plt.figure(figsize=(20, 10))

        H = 1
        if evaluate:
            H = 2
        W = 3

        plot_id = 1
        fig_config = (fig, H, W)
        # idx = list(predictions['labels']).index(1) #[0]
        # Plot GT bounding boxes
        if evaluate:
            plot_bb_ax(img, labels, fig_config, plot_id, 'GT BB')
            plot_id += 1

            # Plot GT 2D keypoints
            # plot_pose2d(img, labels, 0, palm, fig_config, plot_id, 'GT 2D pose')
            # plot_id += 1

            # Plot GT 3D Keypoints
            plot_pose3d(labels, fig_config, plot_id, 'GT 3D pose', center=palm)
            plot_id += 1

            # Plot GT 3D mesh
            plot_mesh3d(labels, right_hand_faces, obj_faces, fig_config, plot_id, 'GT 3D mesh', center=palm, left_hand_faces=left_hand_faces)
            plot_id += 1

            # Save textured mesh
            texture = generate_gt_texture(img, labels['mesh3d'][0][:, :3])
            save_mesh(labels, filename, right_hand_faces, obj_faces, texture=texture, shape_dir='mesh_gt', left_hand_faces=left_hand_faces)

        # Plot predicted bounding boxes
        plot_bb_ax(img, predictions, fig_config, plot_id, 'RGB frame and Bounding box')
        plot_id += 1

        # Plot predicted 2D keypoints
        # plot_pose2d(img, predictions, idx, palm, fig_config, plot_id, 'Predicted 2D pose')
        # plot_id += 1

        # plot_pose_heatmap(img, predictions, idx, palm, fig_config, plot_id)
        # plot_id += 1

        # Plot predicted 3D keypoints
        plot_pose3d(predictions, fig_config, plot_id, '3D pose', center=palm)
        plot_id += 1

        # Plot predicted 3D Mesh
        plot_mesh3d(predictions, right_hand_faces, obj_faces, fig_config, plot_id, '3D mesh', center=palm, left_hand_faces=left_hand_faces)
        plot_id += 1

        # Save textured mesh
        predicted_texture = predictions['mesh3d'][0][:, 3:]
        save_mesh(predictions, filename, right_hand_faces, obj_faces, texture=predicted_texture, left_hand_faces=left_hand_faces)

        fig.tight_layout()
        plt.show()
        # plt.savefig(filename)
        # plt.clf()
        plt.close(fig)

    # Input parameters
    args = parse_args_function()

    # Transformer function
    transform_function = transforms.Compose([transforms.ToTensor()])

    num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object)

    # Create Output directory

    # Dataloader

    if args.dataset_name == 'ho3d_v2' or args.dataset_name == 'ho3d_v3' or args.dataset_name == 'freihand':
        testset = Dataset(root=args.root, load_set=args.split, transform=transform_function, num_kps3d=num_kps3d, num_verts=num_verts)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=ho3d_collate_fn)
        num_classes = 2
        graph_input='heatmaps'
    else:
        testset = Dataset(root=args.root, load_set=args.split, transform=transform_function, num_kps3d=num_kps3d, num_verts=num_verts)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=ho3d_collate_fn)
        num_classes = 2
        graph_input='heatmaps'

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True

    # Define device
    device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = create_thor(pretrained=False, num_classes=num_classes, device=device,
                                    num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_verts=num_verts,
                                    rpn_post_nms_top_n_test=num_classes-1,
                                    box_score_thresh=0.0,
                                    photometric=args.photometric, graph_input=graph_input, dataset_name=args.dataset_name,
                                    num_features=args.num_features, hid_size=args.hid_size)

    if torch.cuda.is_available():
        model = model.cuda(device=args.gpu_number[0])
        model = nn.DataParallel(model, device_ids=args.gpu_number)

    ### Load model
    pretrained_model = f'./checkpoints/hand-object/{args.checkpoint_folder}/model-{args.checkpoint_id}.pkl'
    model.load_state_dict(torch.load(pretrained_model, map_location='cuda:0'))
    model = model.eval()
    print(model)
    print('model loaded!')

    keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d', 'palm']
    c = 0
    # supporting_dicts = (pickle.load(open('./rcnn_outputs/rcnn_outputs_778_test_3d.pkl', 'rb')),
    #                     pickle.load(open('./rcnn_outputs_mesh/rcnn_outputs_778_test_3d.pkl', 'rb')))
    supporting_dicts = None
    output_dicts = ({}, {})

    evaluate = False
    if args.dataset_name == 'ho3d_v2' or args.dataset_name == 'ho3d_v3' or args.dataset_name == 'freihand':
        errors = [[], [], [], []]
    else:
        errors = [[], [], [], [], [], []]
    ###----------------------------------------------
    #更多指标
    xyz_err_procrustes_al = []
    verts_err_procrustes_al = []
    xyz_auc_procrustes_al = []
    verts_auc_procrustes_al = []
    F_score_al = []
    PCK = [[], []]  #PCK列表里面应有两个子列表。第一个子列表用来存放procrustes对齐下的pck，第二个子列表用来存在非procrustes对齐的pck
    ###-----------------------------------------------
    if args.split == 'val':
        evaluate = True

    # rgb_errors = []

    for i, ts_data in enumerate(tqdm(testloader)):
        data_dict = ts_data
        path = data_dict[0]['path'].split('\\')[-1]
        if args.seq not in data_dict[0]['path']:
            continue
        if '_' in path:
            path = path.split('_')[-1]
        frame_num = int(path.split('.')[0])

        ### Run inference
        inputs = [t['inputs'].to(device) for t in data_dict]
        outputs = model(inputs)
        img = inputs[0].cpu().detach().numpy()

        predictions, img, palm, labels = prepare_data_for_evaluation(data_dict, outputs, img, keys, device, args.split)

        ### Visualization
        if args.visualize:

            name = path.split('/')[-1]

            if (num_classes == 2 and 1 in predictions['labels']) or (num_classes == 4 and set([1, 2, 3]).issubset(predictions['labels'])):
                visualize2d(img, predictions, labels, filename=f'./outputs/visual_result/{args.seq}/{name}', palm=palm, evaluate=evaluate)
            else:
                cv2.imwrite(f'./visual_results/{args.seq}/{name}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # if (num_classes == 2 and 1 in predictions['labels']) or (num_classes == 4 and set([1, 2, 3]).issubset(predictions['labels'])):
            #     if name == '0988.jpg':
            #         visualize2d(img, predictions, labels, filename=f'./outputs/visual_result/{args.seq}/{name}', palm=palm, evaluate=evaluate)
            # else:
            #     if name == '0988.jpg':
            #         cv2.imwrite(f'./visual_results/{args.seq}/{name}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        ### Evaluation
        if evaluate:
            # c = save_calculate_error(predictions, labels, path, errors, output_dicts, c, num_classes, args.dataset_name, obj=args.object, generate_mesh=True)
            c = save_calculate_error(predictions, labels, path, errors, output_dicts, c, xyz_err_procrustes_al, verts_err_procrustes_al, xyz_auc_procrustes_al, verts_auc_procrustes_al, F_score_al, PCK,
                                     num_classes, args.dataset_name, obj=args.object, generate_mesh=True)

        # if i == 10:
        #     break


    if evaluate:
        if args.dataset_name == 'h2o3d':
            names = ['lh pose', 'lh mesh', 'rh pose', 'rh mesh', 'obj pose', 'obj mesh']
        else:
            names = ['rh pose', 'rh mesh', 'obj pose', 'obj mesh']

        for i in range(len(errors)):
            avg_error = np.average(np.array(errors[i]))
            print(f'{names[i]} average error on test set:', avg_error)

        ###-----------------------------------------------------------
        #打印更多的指标
        F_score_al_5mm = []
        F_score_al_15mm = []
        for f_score_al in F_score_al:
            for f in f_score_al:
                F_score_al_5mm.append(f[0])
                F_score_al_15mm.append(f[1])

        PCK_rh_mesh_al_0mm, PCK_rh_mesh_al_10mm, PCK_rh_mesh_al_20mm, PCK_rh_mesh_al_30mm, PCK_rh_mesh_al_40mm, PCK_rh_mesh_al_50mm= [], [], [], [], [], []
        PCK_rh_joint_al_0mm, PCK_rh_joint_al_10mm, PCK_rh_joint_al_20mm, PCK_rh_joint_al_30mm, PCK_rh_joint_al_40mm, PCK_rh_joint_al_50mm= [], [], [], [], [], []
        for p in PCK[0]:
            PCK_rh_mesh_al_0mm.append(p[0])
            PCK_rh_mesh_al_10mm.append(p[1])
            PCK_rh_mesh_al_20mm.append(p[2])
            PCK_rh_mesh_al_30mm.append(p[3])
            PCK_rh_mesh_al_40mm.append(p[4])
            PCK_rh_mesh_al_50mm.append(p[5])
        for p in PCK[1]:
            PCK_rh_joint_al_0mm.append(p[0])
            PCK_rh_joint_al_10mm.append(p[1])
            PCK_rh_joint_al_20mm.append(p[2])
            PCK_rh_joint_al_30mm.append(p[3])
            PCK_rh_joint_al_40mm.append(p[4])
            PCK_rh_joint_al_50mm.append(p[5])

        print("普氏对齐下的3D手部关节点平均误差: ", np.average(np.array(xyz_err_procrustes_al)))
        print("普氏对齐下的3D手部网格点平均误差: ", np.average(np.array(verts_err_procrustes_al)))

        print("普氏对齐下的3D手部关节点平均AUC: ", np.average(np.array(xyz_auc_procrustes_al)))
        print("普氏对齐下的3D手部网格点平均AUC: ", np.average(np.array(verts_auc_procrustes_al)))

        print("普氏对齐下的3D手部网格点得分（F@5mm）: ", np.average(np.array(F_score_al_5mm)))
        print("普氏对齐下的3D手部网格点得分（F@15mm）: ", np.average(np.array(F_score_al_15mm)))

        print("普氏对齐下的右手3D网格点PCK：")
        print("PCK_rh_mesh_al_0mm：", np.average(PCK_rh_mesh_al_0mm))
        print("PCK_rh_mesh_al_10mm：", np.average(PCK_rh_mesh_al_10mm))
        print("PCK_rh_mesh_al_20mm：", np.average(PCK_rh_mesh_al_20mm))
        print("PCK_rh_mesh_al_30mm：", np.average(PCK_rh_mesh_al_30mm))
        print("PCK_rh_mesh_al_40mm：", np.average(PCK_rh_mesh_al_40mm))
        print("PCK_rh_mesh_al_50mm：", np.average(PCK_rh_mesh_al_50mm))
        print("普氏对齐下的右手3D关节点PCK：")
        print("PCK_rh_joint_al_0mm：", np.average(PCK_rh_joint_al_0mm))
        print("PCK_rh_joint_al_10mm：", np.average(PCK_rh_joint_al_10mm))
        print("PCK_rh_joint_al_20mm：", np.average(PCK_rh_joint_al_20mm))
        print("PCK_rh_joint_al_30mm：", np.average(PCK_rh_joint_al_30mm))
        print("PCK_rh_joint_al_40mm：", np.average(PCK_rh_joint_al_40mm))
        print("PCK_rh_joint_al_50mm：", np.average(PCK_rh_joint_al_50mm))
        ###-----------------------------------------------------------

        # avg_error = np.average(np.array(errors))
        # print('Hand shape average error on validation set:', avg_error)

        # avg_rgb_error = np.average(np.array(rgb_errors))
        # print('Texture average error on validation set:', avg_rgb_error)

    # save_dicts(output_dicts, args.split)