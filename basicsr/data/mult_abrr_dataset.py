import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
try:
    import torchvision.transforms.v2 as transforms
except:
    from torchvision import transforms

import sys
sys.path.insert(0, '/root/project')
from utils import AberratedDataset

from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, padding
from basicsr.utils.flow_util import dequantize_flow

class LinNorm(torch.nn.Module):
    """
    linearly scale tensor as min = 0.0 and max = 1.0
    """
    def __init__(self):
        super().__init__()
    def forward(self, tensor):
        return (tensor-tensor.min())/(tensor.max()-tensor.min())

class Dataset_MultAbrr(AberratedDataset):
    """
    based on the restormer.basicsr.data.paired_image_dataset.Dataset_PairedImage
    and Inherit AberratedDataset
    Currently input_channels should be 1.
    """
    def __init__(self, opt):
        self.opt = opt
        self.gt_size = opt["gt_size"]
        assert opt["dtype"]=="float32", "Not Implemented"
        assert opt['input_channels']==1, "Not Implemented"
        self.dtype = torch.float32 if opt["dtype"]=="float32" else NotImplementedError
        transform_dataset = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(self.dtype), # ToDtype on transforms v1
            LinNorm()
        ])
        gt_pth = Path(opt["dataroot_gt"])
        lq_pth = Path(opt["dataroot_lq"])
        dataset_pth = gt_pth.parent
        assert dataset_pth==lq_pth.parent
        
        dataset_kwargs = {
            "input_channels": opt["input_channels"], 
            "transform": transform_dataset,
            "gt_path": Path(opt['dataroot_gt']).stem,
            "aberrated_path": Path(opt['dataroot_lq']).stem, 
            "max_length": opt['max_length'], 
            "abrr_imgs_number": opt['abrr_imgs_number'], 
            "dataset_mode": opt["dataset_mode"]
        }

        super(Dataset_MultAbrr, self).__init__(dataset_pth, **dataset_kwargs)
        
    def __getitem__(self, idx):
        img_lq, img_gt, input_ls = super().__getitem__(idx)
        img_lq = img_lq.moveaxis(0, 2)
        img_gt = img_gt.moveaxis(0, 2)
    # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, 1,
                                            f"gt of abrr image, {input_ls}")

        # flip, rotation augmentations
        if self.opt["geometric_augs"]:
            img_gt, img_lq = random_augmentation(img_gt, img_lq) #it takse images in [W, H, C] channel order.
            img_gt = torch.from_numpy(img_gt)
            img_lq = torch.from_numpy(img_lq)
            
        img_gt = img_gt.moveaxis(2, 0)
        img_lq = img_lq.moveaxis(2, 0)
    
        # augmentation for training 
        if False: #skip padding because dataset is
            gt_size = self.opt['gt_size']
            # padding
            #img_gt, img_lq = padding(img_gt, img_lq, gt_size)


        return {'lq': img_lq, 'gt': img_gt, 'lq_path': str(input_ls[0]), 'gt_path': str(Path(self.opt["dataroot_gt"]).joinpath(f"{idx}.tif"))}
