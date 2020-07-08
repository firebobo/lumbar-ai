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
    'network': 'models.lumbarnet.PoseNet',
    'inference': {
        'nstack': 4,
        'inp_dim': 256,
        'oup_dim': 11,
        'num_parts': 11,
        'num_class': 9,
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 2958, ## number of val examples used. entire set is 2958
        'train_num_eval': 150, ## number of train examples tested at test time
    },

    'train': {
        'batchsize': 32,
        'input_res': 256,
        'output_res': 64,
        'epoch_num': 1000000,
        'data_num': 150,
        'train_iters': 1000,
        'valid_iters': 10,
        'learning_rate': 1e-3,
        'max_num_people' : 1,
        'loss': [
            ['combined_hm_loss', 1],
            ['combined_lb_loss', 1]
        ],
        'stack_loss': [1,2,3,4],
        'decay_iters': 2000,
        'decay_lr': 0.8,
        'num_workers': 2,
        'use_data_loader': True,
        'train_num_eval': 150,
        'valid_num_eval': 51
    },
}


def build_targets(heatmap,labelmap):
    # print(gt.shape)
    size = heatmap.shape
    targets = np.zeros([size[0],size[1],size[2],4])

    m = size[3]
    n = size[4]
    for idx,g in enumerate(heatmap):
        for jdx,gg in enumerate(g):
            for kdx,ggg in enumerate(gg):
                a = heatmap[idx,jdx,kdx]
                index = int(a.argmax())
                x = int(index / n)
                y = index % n
                targets[idx,jdx,kdx,0] = a[x,y]
                targets[idx, jdx,kdx, 1:3] = [x+labelmap[idx, jdx, 7, x, y],y+labelmap[idx, jdx, 8, x, y]]
                y_ = labelmap[idx, jdx, :, x, y]
                if kdx%2==0:
                    y_ = labelmap[idx, jdx, 2:7, x, y]
                    ind = y_.argmax()
                else:
                    y_ = labelmap[idx, jdx, :2, x, y]
                    ind = y_.argmax()
                targets[idx, jdx,kdx, 3] = ind+1

        return targets ## l of dim bsize
def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']

    ## creating new posenet
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(**config)
    forward_net = DataParallel(poseNet.cuda())
    config['net'] = forward_net
    config['lossLayers'] = KeypointLoss(configs['inference']['num_parts'],configs['inference']['nstack'],configs['inference']['num_class'])
    ## optimizer, experiment setup
    train_cfg['optimizer'] = torch.optim.Adam(filter(lambda p: p.requires_grad,config['net'].parameters()), train_cfg['learning_rate'])

    exp_path = os.path.join('exp', configs['opt'].exp)
    if configs['opt'].exp=='pose' and configs['opt'].continue_exp is not None:
        exp_path = os.path.join('exp', configs['opt'].continue_exp)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')

    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            try:
                inputs[i] = make_input(inputs[i])
            except:
                pass #for last input, which is a string (id_)

        net = config['inference']['net']
        config['batch_id'] = batch_id
        net = net.train()

        if phase != 'inference':
            combined_hm_preds, combined_lb_preds = net(inputs['imgs'])
            all_loss= config['inference']["lossLayers"](combined_hm_preds,combined_lb_preds,**{i:inputs[i] for i in inputs if i!='imgs'})
            num_loss = len(config['train']['loss'])

            losses = [all_loss[idx].cpu()*i[1] for idx, i in enumerate(config['train']['loss'])]

            loss = 0
            my_loss=[]
            toprint = '\n{}: '.format(batch_id)
            for i,l in enumerate(losses):
                loss += torch.sum(l.mul(torch.Tensor(config['train']['stack_loss']))).float().cpu()
                my_loss.append(l)


            if len(my_loss) == 1:
                toprint += ' {}: {}'.format(i, format(my_loss, '.8f'))
            else:
                for j in my_loss:
                    if batch_id%100==0:
                        toprint += ' \n{}'.format(format(str(j)))
                    else:
                        toprint += ' {}'.format(format(j.sum(),'.8f'))
            logger.write(toprint)
            logger.flush()
            
            if phase == 'train':
                optimizer = train_cfg['optimizer']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif phase == 'valid':
                loss = losses[-1][-1].sum().cpu()
                # result = build_targets(combined_hm_preds, combined_lb_preds)
            if batch_id%config['train']['decay_iters']==0:
                ## decrease the learning rate after decay # iterations
                for param_group in train_cfg['optimizer'].param_groups:
                    param_group['lr'] = config['train']['decay_lr']*param_group['lr']
            return loss.item()
        else:
            net = net.eval()
            combined_hm_preds, combined_lb_preds = net(inputs['imgs'])
            combined_hm_preds = make_output(combined_hm_preds)
            combined_lb_preds = make_output(combined_lb_preds)
            result= build_targets(combined_hm_preds, combined_lb_preds)
            out = result
            return out
    return make_train
