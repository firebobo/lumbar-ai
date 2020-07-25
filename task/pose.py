"""
__config__ contains the options for training and testing
Basically all of the variables related to training are put in __config__['train'] 
"""
import torch
import numpy as np
from torch import nn
import os
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from task.loss import KeypointLoss
from utils.misc import make_input, make_output, importNet

__config__ = {
    'data_provider': 'data.dp',
    'network': 'models.pose_higher_hrnet.get_pose_net',
    'inference': {
        'nstack': 2,
        'inp_dim': 256,
        'oup_dim': 11,
        'num_parts': 11,
        'num_class': 9,
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 2958,  ## number of val examples used. entire set is 2958
        'train_num_eval': 150,  ## number of train examples tested at test time
    },

    'train': {
        'batchsize': 20,
        'input_res': 256,
        'output_res': 64,
        'epoch_num': 30,
        'data_num': 150,
        'train_iters': 500,
        'train_valid_iters': 51,
        'valid_iters': 10,
        'valid_train_iters': 8,
        'learning_rate': 1e-3,
        'max_num_people': 1,
        'loss': [1, 1, 100],
        'stack_loss': [1, 1, 1],
        'decay_iters': 10,
        'decay_lr': 0.5,
        'num_workers': 2,
        'use_data_loader': True,
        'train_num_eval': 150,
        'valid_num_eval': 51
    },
}


def build_targets(combined_preds):
    # print(gt.shape)
    nstask = len(combined_preds)
    size = combined_preds[0].shape
    key_num = int(size[1] / 3)
    targets = np.zeros([size[0], nstask, key_num, 4])
    for idx in range(nstask):
        heatmap, label_preds, maskmap = combined_preds[idx][:, key_num:2 * key_num].cpu(), combined_preds[idx][:,
                                                                                           :key_num].cpu(), \
                                        combined_preds[idx][:, 2 * key_num:].cpu()
        for jdx, gg in enumerate(heatmap):
            for kdx in range(key_num):
                a = heatmap[jdx, kdx]
                siz = a.shape
                m = siz[0]
                n = siz[1]
                mask = torch.zeros([m, n])
                mask[maskmap[jdx, kdx] > 0.5] = 1

                a = a * mask
                index = int(a.argmax())
                x = int(index / n)
                y = index % n
                # if a[x,y]>0.9:
                #     plt.imshow(mask)
                #     plt.show()
                targets[jdx, idx, kdx, 0] = a[x, y]
                targets[jdx, idx, kdx, 1:3] = [x, y]
                if kdx % 2 == 0:
                    y_ = label_preds[jdx, 2:7, x, y]

                    if y_.max() < 0.5:
                        ind = y_.argmax() + 1
                    else:
                        ind = 0
                        for inx, la in enumerate(y_):
                            if la > 0.5:
                                ind = ind * 10 + inx + 1

                else:
                    y_ = label_preds[jdx, :2, x, y]
                    ind = y_.argmax() + 1
                targets[jdx, idx, kdx, 3] = ind

    return targets  ## l of dim bsize


def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']

    ## creating new posenet
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(configs["cfg"], True)
    forward_net = DataParallel(poseNet.cuda())
    config['net'] = forward_net
    config['lossLayers'] = KeypointLoss(configs['inference']['num_parts'], 2,
                                        configs['inference']['num_class'])
    ## optimizer, experiment setup
    train_cfg['optimizer'] = torch.optim.AdamW(config['net'].parameters(), train_cfg['learning_rate'])
    train_cfg['scheduler'] = lr_scheduler.StepLR(train_cfg['optimizer'], step_size=train_cfg['decay_iters'],
                                                 gamma=train_cfg[
                                                     'decay_lr'])  # 更新学习率的策略：  每隔step_size个epoch就将学习率降为原来的gamma倍。
    exp_path = os.path.join('exp', configs['opt'].exp)
    if configs['opt'].exp == 'pose' and configs['opt'].continue_exp is not None:
        exp_path = os.path.join('exp', configs['opt'].continue_exp)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')
    config['logger'] = logger


def do_train(epoch, config, loader):
    logger = config['inference']['logger']
    net = config['inference']['net']

    batch_idx = 0
    for inputs in tqdm(loader):
        for i, input in enumerate(inputs):
            try:
                if type(inputs[i]) is list:
                    for ind, inp in enumerate(inputs[i]):
                        inputs[i][ind] = make_input(inp)
                else:
                    inputs[i] = make_input(inputs[i])
            except:
                pass  # for last input, which is a string (id_)

        combined_preds = net(inputs[0])
        num_loss = config['train']['loss']
        combined_loss, labels_loss, masks_loss = config['inference']["lossLayers"](combined_preds,
                                                                                   **{'heatmaps': inputs[1],
                                                                                      'labels': inputs[2],
                                                                                      'masks': inputs[3]})

        toprint = '\n{} {}: '.format(epoch, batch_idx)
        heatmap_loss = torch.sum(combined_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
        label_loss = torch.sum(labels_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
        mask_loss = torch.sum(masks_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))

        loss = num_loss[0] * heatmap_loss + num_loss[1] * label_loss + num_loss[2] * mask_loss
        if batch_idx % 100 == 0:
            toprint += ' \n{}'.format(str(combined_loss.cpu()))
            toprint += ' \n{}'.format(str(labels_loss.cpu()))
            toprint += ' \n{}'.format(str(masks_loss.cpu()))
        else:
            toprint += ' {},{},{},{}'.format(heatmap_loss, label_loss, mask_loss, loss)
        logger.write(toprint)
        logger.flush()
        optimizer = config['train']['optimizer']
        optimizer.zero_grad()

        # loss = heatmap_loss
        loss.backward()
        optimizer.step()
        batch_idx += 1


def build_targets_compute(combined_preds, class_preds, heatmap, label):
    size = combined_preds[0].shape
    key_num = int(size[1] / 3)
    n_correct = 0
    all_correct = 0
    heatmap_preds, label_preds, mask_preds = combined_preds[-1][:, key_num:2 * key_num].cpu(), combined_preds[-1][:,
                                                                                               :key_num].cpu(), \
                                             combined_preds[-1][:, 2 * key_num:].cpu()
    for jdx, gg in enumerate(heatmap_preds):
        for kdx in range(key_num):
            a = heatmap_preds[jdx, kdx]
            siz = a.shape
            m = siz[0]
            n = siz[1]
            mask = torch.zeros([m, n])
            mask[mask_preds[jdx, kdx] > 0.5] = 1
            # plt.imshow(mask)
            # plt.show()
            a = a * mask
            index = int(a.argmax())
            x_pred = int(index / n)
            y_pred = index % n

            aa = heatmap[jdx, kdx]
            index = int(aa.argmax())
            x = int(index / n)
            y = index % n
            y_ = label_preds[jdx, kdx, 2:7, x_pred, y_pred]
            if (x_pred - x) ** 2 + (y - y_pred) ** 2 <= 9:
                if kdx % 2 == 0:
                    for inx, la in enumerate(label[jdx, kdx, 2:7]):
                        if la == 1:
                            all_correct += 1
                            if y_[inx] > 0.5:
                                n_correct += 1
                    if y_.max() < 0.5:
                        ind = y_.argmax()
                        if label[jdx, kdx, 2 + ind] == 1:
                            n_correct += 1
                else:
                    ind = y_.argmax()

                    if label[jdx, kdx, ind] == 1:
                        n_correct += 1
            else:
                all_correct += 1
    return n_correct / all_correct


def do_valid(epoch, config, loader):
    logger = config['inference']['logger']
    net = config['inference']['net']
    run_correct = 0
    batch_idx = 0
    with torch.no_grad():

        for inputs in tqdm(loader):
            for i, input in enumerate(inputs):
                if type(inputs[i]) is list:
                    for ind, inp in enumerate(inputs[i]):
                        inputs[i][ind] = make_input(inp)
                else:
                    inputs[i] = make_input(inputs[i])

            combined_preds = net(inputs[0])
            correct = build_targets_compute(combined_preds, inputs[1][-1], inputs[2])

            combined_loss, labels_loss, masks_loss = config['inference']["lossLayers"](combined_preds,
                                                                                       **{'heatmaps': inputs[1],
                                                                                          'labels': inputs[2],
                                                                                          'masks': inputs[3]})
            toprint = '\n{} {}: '.format(epoch, batch_idx)
            heatmap_loss = torch.sum(combined_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
            label_loss = torch.sum(labels_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
            mask_loss = torch.sum(masks_loss.cpu().mul(torch.Tensor(config['train']['stack_loss'])))
            num_loss = config['train']['loss']
            loss = num_loss[0] * heatmap_loss + num_loss[1] * label_loss + num_loss[2] * mask_loss
            if batch_idx % 100 == 0:
                toprint += ' {},{},{},{},{}'.format(heatmap_loss, label_loss, mask_loss, loss, correct)
                toprint += ' \n{}'.format(str(combined_loss.cpu()))
                toprint += ' \n{}'.format(str(labels_loss.cpu()))
                toprint += ' \n{}'.format(str(masks_loss.cpu()))
            else:
                toprint += ' {},{},{},{},{}'.format(heatmap_loss, label_loss, mask_loss, loss, correct)
            logger.write(toprint)
            logger.flush()
            batch_idx += 1
            run_correct += correct
    return run_correct / batch_idx


def do_test(inputs, config):
    net = config['inference']['net']
    combined_preds = net.eval()(inputs['imgs'])
    result = build_targets(combined_preds)
    return result
