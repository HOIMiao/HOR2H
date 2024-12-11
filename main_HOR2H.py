# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from torch.utils.tensorboard import SummaryWriter

""" import libraries"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os
import math
import time

from utils.options import parse_args_function
from utils.utils import freeze_component, calculate_keypoints, create_loader

from models.thor_net import create_thor

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_args_function()

    # Define device
    device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True

    num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object)

    """ Configure a log """

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    fh = logging.FileHandler(os.path.join(args.output_file[:-6], 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    """ load datasets """

    if args.dataset_name == 'ho3d_v2' or args.dataset_name == 'ho3d_v3' or args.dataset_name == 'freihand':
        trainloader, train_dataset = create_loader(args.dataset_name, args.root, 'train', batch_size=args.batch_size, num_kps3d=num_kps3d, num_verts=num_verts)
        valloader, val_dataset = create_loader(args.dataset_name, args.root, 'val', batch_size=args.batch_size, num_kps3d=num_kps3d, num_verts=num_verts)
        num_classes = 2
        graph_input = 'heatmaps'    #graph_input用来控制是否将热图转换为2D points，graph_input='heatmaps'表示不转换使用热图，graph_input='coords'表示转换使用2D points
        print(f'datasets {args.dataset_name} is loader')
    elif args.dataset_name == 'h2o3d':
        trainloader, train_dataset = create_loader(args.dataset_name, args.root, 'train', batch_size=args.batch_size, num_kps3d=num_kps3d, num_verts=num_verts)
        valloader, val_dataset = create_loader(args.dataset_name, args.root, 'val', batch_size=args.batch_size, num_kps3d=num_kps3d, num_verts=num_verts)
        num_classes = 2 #num_classes应该为4，然而源码用num_class的值设置了条件语句，我们想要执行num_calsses小于2的语句，所以我在这里将num_classes设置为2，但我们要知道num_classes(类型数量)为4
        graph_input = 'heatmaps'    #graph_input用来控制是否将热图转换为2D points，graph_input='heatmaps'表示不转换使用热图，graph_input='coords'表示转换使用2D points
        print(f'datasets {args.dataset_name} is loader')

    """ load model """
    model = create_thor(num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_verts=num_verts, num_classes=num_classes,
                                    rpn_post_nms_top_n_train=num_classes-1,
                                    device=device, num_features=args.num_features, hid_size=args.hid_size,
                                    photometric=args.photometric, graph_input=graph_input, dataset_name=args.dataset_name)
    print('THOR is loaded')

    if torch.cuda.is_available():
        model = model.cuda(args.gpu_number[0])
        model = nn.DataParallel(model, device_ids=args.gpu_number)

    """ load saved model"""

    if args.pretrained_model != '':
        model.load_state_dict(torch.load(args.pretrained_model, map_location=f'cuda:{args.gpu_number[0]}'))
        losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
        start = len(losses)
    else:
        losses = []
        start = 0

    """define optimizer"""

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
    scheduler.last_epoch = start

    keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d', 'palm']

    """ training """

    logging.info('Begin training the network...')

    for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times

        train_loss2d = 0.0
        running_loss2d = 0.0
        running_loss3d = 0.0
        running_mesh_loss3d = 0.0
        running_photometric_loss = 0.0

        for i, tr_data in enumerate(trainloader):

            # get the inputs
            data_dict = tr_data

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
            inputs = [t['inputs'].to(device) for t in data_dict]
            loss_dict = model(inputs, targets)

            # for k in loss_dict.keys():
            #     print(k)

            # Calculate Loss
            loss = sum(loss for _, loss in loss_dict.items())

            # Backpropagate
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss2d += loss_dict['loss_keypoint'].data
            running_loss2d += loss_dict['loss_keypoint'].data
            running_loss3d += loss_dict['loss_keypoint3d'].data
            running_mesh_loss3d += loss_dict['loss_mesh3d'].data
            if 'loss_photometric' in loss_dict.keys():
                running_photometric_loss += loss_dict['loss_photometric'].data

            if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
                logging.info('%s   [Epoch %d/%d,   iter %d/%d] loss 2d: %.4f, loss 3d: %.4f, mesh loss 3d:%.4f, photometric loss: %.4f' %
                (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch + 1, args.num_iterations, i + 1, math.ceil(len(train_dataset)/len(args.gpu_number)/args.batch_size),
                 running_loss2d / args.log_batch, running_loss3d / args.log_batch,
                running_mesh_loss3d / args.log_batch, running_photometric_loss / args.log_batch))       #打印每个batch_size的信息（包含：epoch， batch，各类损失）
                running_mesh_loss3d = 0.0
                running_loss2d = 0.0
                running_loss3d = 0.0
                running_photometric_loss = 0.0

        losses.append((train_loss2d / (i+1)).cpu().numpy())

        if (epoch+1) % args.snapshot_epoch == 0:
            torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
            np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

        if (epoch+1) % args.val_epoch == 0:
            val_loss2d = 0.0
            val_loss3d = 0.0
            val_mesh_loss3d = 0.0
            val_photometric_loss = 0.0

            # model.module.transform.training = False

            for v, val_data in enumerate(valloader):

                # get the inputs
                data_dict = val_data

                # wrap them in Variable
                targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
                inputs = [t['inputs'].to(device) for t in data_dict]
                loss_dict = model(inputs, targets)

                val_loss2d += loss_dict['loss_keypoint'].data
                val_loss3d += loss_dict['loss_keypoint3d'].data
                val_mesh_loss3d += loss_dict['loss_mesh3d'].data
                if 'loss_photometric' in loss_dict.keys():
                    running_photometric_loss += loss_dict['loss_photometric'].data

            # model.module.transform.training = True

            logging.info('%s   [Epoch %d/%d,             ] val loss 2d: %.4f, val loss 3d: %.4f, val mesh loss 3d: %.4f, val photometric loss: %.4f' %
                        (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch + 1, args.num_iterations, val_loss2d / (v+1), val_loss3d / (v+1), val_mesh_loss3d / (v+1), val_photometric_loss / (v+1)))

        if args.freeze and epoch == 0:
            logging.info('Freezing Keypoint RCNN ..')
            freeze_component(model.module.backbone)
            freeze_component(model.module.rpn)
            freeze_component(model.module.roi_heads)

        # Decay Learning Rate
        scheduler.step()

    logging.info('Finished Training')
