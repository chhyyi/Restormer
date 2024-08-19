import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
try:
    import torchvision.transforms.v2 as transforms
except:
    from torchvision import transforms
import tifffile

"""
import sys
sys.path.insert(0, '/root/project')
from utils import AberratedDataset, ShiftCorrectedAbrrDataset

from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, padding
from basicsr.utils.flow_util import dequantize_flow
"""

class LinNorm(torch.nn.Module):
    """
    linearly scale tensor as min = 0.0 and max = 1.0
    """
    def __init__(self):
        super().__init__()
    def forward(self, tensor):
        return (tensor-tensor.min())/(tensor.max()-tensor.min())

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
        self.to_range = 257.0 if opt['dtype']=="uint16" else NotImplementedError #65535/255 = 257
        self.observations = LinNorm()((torch.tensor(tifffile.imread(self.abrr_tiff).astype(np.float32),dtype=self.dtype)[self.input_stack_range[0]:self.input_stack_range[0]+self.input_stack_range[1]]))*(195./255.)+(30./255.)
        self.gt = LinNorm()((torch.tensor(tifffile.imread(self.gt_tif, key=0).astype((np.float32)), dtype=self.dtype)))*(195./255.)+(30./255.)
    def __len__(self):
        return self.input_stack_num
    def __getitem__(self, idx):
        img_lq = self.observations[idx][None,...]
        img_gt = self.gt[None,...]
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': f"{self.abrr_tiff.parent.joinpath(self.abrr_tiff.stem)}stack{idx}.tiff", 'gt_path': f"stack{idx}_{self.gt_tif}"} # note that these paths are not existing files!


        