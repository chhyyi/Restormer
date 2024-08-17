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
    def __init__(self, abrr_tiff, gt_tif, input_stack_num=100, input_stack_range=(1,101), dtype=torch.float32):
        super().__init__()
        self.abrr_tiff = abrr_tiff
        self.gt_tif = gt_tif
        self.input_stack_num = input_stack_num
        self.observations = torch.tensor(tifffile.imread(abrr_tiff),dtype=dtype)[input_stack_range[0]:input_stack_range[1]]
        self.gt = torch.tensor(tifffile.imread(gt_tif, key=0), dtype=dtype)
    def __len__(self):
        return self.input_stack_num
    def __getitem__(self, idx):
        img_lq = self.observations[idx][None,...]
        img_gt = self.gt_tif[None,...]
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': f"{self.abrr_tiff}@stack{idx}", 'gt_path': self.gt_tif}


        