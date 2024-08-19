"""
Restormer - test script
2024-07-28 Changhyun Yi, chhyyi@gmail.com
I implement new one based on the train, Instead of test.py of restormer
It does not load the model but 'states'! might be not efficient

"""

import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp
from test_ch import *

import sys
sys.path.append("/root/project/restormer")

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed, imwrite)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tifffile as tif

def test_1stack():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    import os
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    if opt['path']['resume_state']:
        # if resume state is given
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
                opt['path']['resume_state'],
                map_location=lambda storage, loc: storage.cuda(device_id))
    else: # if no resume state given in yml file
        resume_state = None
        if len(states) > 0:
            max_state_file = '{}.state'.format(max([int(x[0:-6]) for x in states]))
            resume_state = os.path.join(state_folder_path, max_state_file)
            opt['path']['resume_state'] = resume_state

        # load resume states if necessary
        if opt['path'].get('resume_state'):
            device_id = torch.cuda.current_device()
            resume_state = torch.load(
                opt['path']['resume_state'],
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            resume_state = None 

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    # create model
    if resume_state:  # resume training
        check_resume(opt, resume_state['iter'])
        model = create_model(opt)
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Load training state from epoch: {resume_state['epoch']}, "
                    f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(
        f'Start validation')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    # for epoch in range(start_epoch, total_epochs + 1):

    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch

    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

def save_vis_comparison(exp_path=None, num_iter = 300000, output_pth = "HW_AUGdata_eval"):
    """
    glob prediction results(default: *_300000.png) on the experiments dir, save figure with gt, raw_input. linearly normalized (min-max).
    """
    opt = parse_options(is_train=True)
    exp_path = Path(opt['path']['resume_state']).parent.parent
    pth = Path(exp_path)

    plt.rcParams['font.size']=18
    output_pth = pth.joinpath(output_pth)
    output_pth.mkdir(exist_ok=True)

    mse_list = []
    preds = []

    vis_dirs = [i for i in pth.glob("**/visualization")]
    for vis_dir in vis_dirs: # 1 visualization dirs per 1 experiment...
        for per_patch in vis_dir.iterdir():
            last_pth = per_patch.joinpath(f"{per_patch.stem}_{num_iter}.png")
            if not last_pth.is_file():            
                last_idx = 300000
                raise FileNotFoundError(f"per_patc")
            last = plt.imread(last_pth)
            preds.append(torch.tensor(last))
            gt_pth = per_patch.joinpath(f"{last_pth.stem}_gt.png")
            gt = plt.imread(gt_pth)

            f, axs = plt.subplots(ncols = 2, nrows = 1, figsize=(15, 8.5))
            f.subplots_adjust(left=0.05)
            cbar_ax = f.add_axes([0.93, 0.15, 0.01, 0.7])

            f.suptitle(f"{vis_dir.parent.stem}/{per_patch.stem}, iter {num_iter}")
            #f.tight_layout()
  
            im = axs[0].imshow(last, cmap="cividis", vmin=0, vmax=1)
            #f.colorbar(im, ax=axs[2], shrink=0.8)

            mse = torch.mean((torch.Tensor(gt) -torch.Tensor(last))**2)
            mse_list.append(mse)
            axs[0].axis("off")
            axs[0].set_title(f"prediction, mse:{mse:.06}")
            im = axs[1].imshow(gt, cmap="cividis", vmin=0, vmax=1)
            #f.colorbar(im, ax=axs[3], shrink=0.8)
            f.colorbar(im, cax=cbar_ax)
            axs[1].axis("off")
            axs[1].set_title("GT")
            #plt.savefig(output_pth.joinpath(f"{vis_dir.parent.stem}_{per_patch.stem}.png"))
            plt.close()
    #save ensemble avg
    
    ensemble_avg = torch.mean(torch.stack(preds),dim=0)
    f, axs = plt.subplots(ncols = 2, nrows = 1, figsize=(15, 8.5))
    f.subplots_adjust(left=0.05)
    cbar_ax = f.add_axes([0.93, 0.15, 0.01, 0.7])

    f.suptitle(f"{vis_dir.parent.stem}/ensemble_average, iter {num_iter}")
    #f.tight_layout()

    im = axs[0].imshow(ensemble_avg, cmap="cividis", vmin=0, vmax=1)
    #f.colorbar(im, ax=axs[2], shrink=0.8)

    mse = torch.mean((torch.Tensor(gt) -torch.Tensor(ensemble_avg))**2)
    mse_list.append(mse)
    axs[0].axis("off")
    axs[0].set_title(f"prediction, mse:{mse:.06}")
    im = axs[1].imshow(gt, cmap="cividis", vmin=0, vmax=1)
    #f.colorbar(im, ax=axs[3], shrink=0.8)
    f.colorbar(im, cax=cbar_ax)
    axs[1].axis("off")
    axs[1].set_title("GT")
    plt.savefig(output_pth.joinpath(f"ensemble_average.png"))
    plt.close()

    print(f"mse over whole validation set:{torch.mean(torch.tensor(mse_list))}")

if __name__ == '__main__':
    test_1stack()
    save_vis_comparison(exp_path="/root/project/restormer/experiments", num_iter = 300000,output_pth = "vis_outputs_AugDset")
    