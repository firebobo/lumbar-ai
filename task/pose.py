"""
__config__ contains the options for training and testing
Basically all of the variables related to training are put in __config__['train'] 
"""
import torch
import numpy as np
from torch import nn
import os
from torch.nn import DataParallel

from task.loss import KeypointLoss
from utils.misc import make_input, make_output, importNet

__config__ = {
    'data_provider': 'data.dp',
    'network': 'models.pose_higher_hrnet.get_pose_net',
    'inference': {
        'nstack': 4,
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
        'batchsize': 32,
        'input_res': 256,
        'output_res': 64,
        'epoch_num': 1000000,
        'data_num': 150,
        'train_iters': 10,
        'valid_iters': 10,
        'learning_rate': 1e-3,
        'max_num_people': 1,
        'loss': [
            ['combined_hm_loss', 10],
            ['combined_lb_loss', 1]
        ],
        'stack_loss': [1, 2, 3, 4],
        'decay_iters': 1000,
        'decay_lr': 0.8,
        'num_workers': 2,
        'use_data_loader': True,
        'train_num_eval': 150,
        'valid_num_eval': 51
    },
}


def build_targets(heatmap, labelmap):
    # print(gt.shape)
    size = heatmap.shape
    targets = np.zeros([size[0], size[1], 4])

    m = size[2]
    n = size[3]
    for idx, g in enumerate(heatmap):
        for kdx, ggg in enumerate(g):
            a = heatmap[idx, kdx]
            index = int(a.argmax())
            x = int(index / n)
            y = index % n
            targets[idx, kdx, 0] = a[x, y]
            targets[idx, kdx, 1:3] = [x + labelmap[idx, 7, x, y], y + labelmap[idx, 8, x, y]]
            y_ = labelmap[idx, :, x, y]
            if kdx % 2 == 0:
                y_ = labelmap[idx, 2:7, x, y]
                ind = y_.argmax()
            else:
                y_ = labelmap[idx, :2, x, y]
                ind = y_.argmax()
            targets[idx, kdx, 3] = ind + 1

        return targets  ## l of dim bsize


def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']

    ## creating new posenet
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(configs["cfg"],True)
    forward_net = DataParallel(poseNet.cuda())
    config['net'] = forward_net
    config['lossLayers'] = KeypointLoss(configs['inference']['num_parts'], 2,
                                        configs['inference']['num_class'])
    ## optimizer, experiment setup
    train_cfg['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad, config['net'].parameters()),
                                              train_cfg['learning_rate'])

    exp_path = os.path.join('exp', configs['opt'].exp)
    if configs['opt'].exp == 'pose' and configs['opt'].continue_exp is not None:
        exp_path = os.path.join('exp', configs['opt'].continue_exp)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')

    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            try:
                if type(inputs[i]) is list:
                    for j, input in enumerate(inputs[i]):
                        inputs[i][j] = make_input(input)
                else:
                    inputs[i] = make_input(inputs[i])
            except:
                pass  # for last input, which is a string (id_)

        net = config['inference']['net']
        config['batch_id'] = batch_id
        net = net.train()

        if phase != 'inference':
            combined_preds = net(inputs['imgs'])

            all_loss = config['inference']["lossLayers"](combined_preds,
                                                         **{i: inputs[i] for i in inputs if i != 'imgs'})
            num_loss = len(config['train']['loss'])

            losses = [all_loss[idx].float().cpu()* i[1] for idx, i in enumerate(config['train']['loss'])]

            toprint = '\n{}: '.format(batch_id)
            if batch_id % 100 == 0:
                toprint += ' \n{}\n{}'.format(losses[0],losses[1])
            else:
                toprint += ' {}  {}'.format(format(losses[0].sum(), '.8f'),format(losses[1].sum(), '.8f'))
            logger.write(toprint)
            logger.flush()
            if phase == 'train':
                optimizer = train_cfg['optimizer']
                optimizer.zero_grad()
                loss = losses[0].sum()+losses[1].sum()
                loss.backward()
                optimizer.step()
            elif phase == 'valid':
                loss = (losses[0][:,-1]+losses[1][:,-1]).sum()
                # return float(loss)
                # result = build_targets(combined_hm_preds, combined_lb_preds)
            if batch_id % config['train']['decay_iters'] == 0:
                ## decrease the learning rate after decay # iterations
                for param_group in train_cfg['optimizer'].param_groups:
                    param_group['lr'] = config['train']['decay_lr'] * param_group['lr']
            # return None
        else:
            net = net.eval()
            combined_preds = net(inputs['imgs'])
            combined_preds = make_output(combined_preds)
            result = build_targets(combined_preds[0][:, :config['inference']['num_parts']],
                                   combined_preds[0][:, config['inference']['num_parts']:])
            out = result
            return out

    return make_train
