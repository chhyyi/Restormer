import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
try:
    import torchvision.transforms.v2 as transforms
except:
    from torchvision import transforms

from torchvision.transforms.functional import normalize
import tifffile
import sys
sys.path.append("/root/project/restormer/basicsr/data")
"""
import sys
sys.path.insert(0, '/root/project')
from utils import AberratedDataset, ShiftCorrectedAbrrDataset

from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, padding
from basicsr.utils.flow_util import dequantize_flow
"""

class Dataset_SingleStackedTIFF(data.Dataset):
    """
    evaluate trained dataset on mpmneural dataset by JH Park with another dataset,
    HW's mpmnueral data provided on 2024 AUG 02. It comes with pairs of gt (tif) file and 100-stacked tiff measured with random modulation.
    It will just return 100 pairs of (1 stack, gt), to get ensemble average later on 'basicsr/test_stack.py'
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.abrr_tiff = Path(opt['abrr_path'])
        self.gt_tif = Path(opt['gt_path'])
        self.input_stack_num = opt['abrr_inputs']
        self.input_stack_range = (opt['observe_start_stack_idx'], opt['observe_start_stack_idx']+self.input_stack_num)
        self.dtype = torch.float32 if opt['dtype_readas']=="float32" else NotImplementedError
        self.transforms_gt = transforms.Compose([
            transforms.ToTensor(), 
            transforms.ConvertImageDtype(self.dtype)
            ])
        self.transforms_input = transforms.Compose([
            transforms.ToTensor(), 
            transforms.ConvertImageDtype(self.dtype)
            ])
        self.observations = tifffile.imread(self.abrr_tiff).astype(np.float32)[self.input_stack_range[0]:self.input_stack_range[0]+self.input_stack_range[1]]
        self.gt = tifffile.imread(self.gt_tif, key=0).astype((np.float32))

        self.mean_abrr = opt['mean_abrr']
        self.std_abrr = opt['std_abrr']
        self.mean_gt = opt['mean_gt']
        self.std_gt = opt['std_gt']
    def __len__(self):
        return self.input_stack_num
    def __getitem__(self, idx):
        img_lq = self.transforms_input(self.observations[idx])
        img_gt = self.transforms_gt(self.gt)
        
        normalize(img_lq, self.mean_abrr, self.std_abrr, inplace=True)
        normalize(img_gt, self.mean_gt, self.std_gt, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': f"{self.abrr_tiff.parent.joinpath(self.abrr_tiff.stem)}stack{idx}.tiff", 'gt_path': f"stack{idx}_{self.gt_tif}"} # note that these paths are not existing files!


        