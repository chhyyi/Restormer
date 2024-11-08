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

from util_abrr.utils import AberratedDataset, ShiftCorrectedAbrrDataset, parse_transforms_norm

from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, padding
from basicsr.utils.flow_util import dequantize_flow

class Dataset_MultAbrr(AberratedDataset):
    """
    based on the restormer.basicsr.data.paired_image_dataset.Dataset_PairedImage
    and Inherit AberratedDataset
    Currently input_channels should be 1.

    Note: temporal implementation, should be merged with class Dataset_MultAbrr_SC 
    """
    def __init__(self, opt):
        self.opt = opt
        self.gt_size = opt.get("gt_size")
        assert opt["dtype"]=="float32", "Not Implemented"
        #assert opt['input_channels']==1, "Not Implemented"
        self.dtype = torch.float32 if opt["dtype"]=="float32" else NotImplementedError

        gt_pth = Path(opt["dataroot_gt"])
        lq_pth = Path(opt["dataroot_lq"])
        dataset_pth = gt_pth.parent
        assert dataset_pth==lq_pth.parent
        
        kwargs_to_dataset_list=["gt_path", "abrr_path", "maxlen", "abrr_inputs", "dset_mode"] # except for transform and input channels, which will be added from now on.
        ds_kwargs = {i:opt.get(i) for i in kwargs_to_dataset_list}
        ds_kwargs["input_channels"] = opt["input_channels"]

        transform_dataset = [transforms.ToTensor(),
                transforms.ConvertImageDtype(self.dtype),]            
        resize_to = opt.get('resize_to')
        crop_to = opt.get('crop_to')
        if not resize_to==None: transform_dataset.append(transforms.Resize(resize_to))
        transform_dataset.append(transforms.CenterCrop(crop_to))
        ds_kwargs['transform'] = transforms.Compose([*transform_dataset])

        if opt['ds_type']=="AberratedDataset":
            AberratedDataset.__init__(self, dataset_pth, **ds_kwargs)
            self.super_getitem = AberratedDataset.__getitem__
        else:
            raise NotImplementedError
        
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        
    def __getitem__(self, idx):
        img_lq, img_gt, input_ls = self.super_getitem(self, idx)
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
    
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)        

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': str(input_ls[0]), 'gt_path': str(Path(self.opt["dataroot_gt"]).joinpath(f"{idx}.tif"))}


class Dataset_MultAbrr_SC(ShiftCorrectedAbrrDataset):
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

        gt_pth = Path(opt["dataroot_gt"])
        lq_pth = Path(opt["dataroot_lq"])
        dataset_pth = gt_pth.parent
        assert dataset_pth==lq_pth.parent
        
        kwargs_to_dataset_list=["gt_path", "abrr_path", "maxlen", "abrr_inputs", "dset_mode"] # except for transform and input channels, which will be added from now on.
        ds_kwargs = {i:opt[i] for i in kwargs_to_dataset_list}
        ds_kwargs["input_channels"] = opt["input_channels"]
        if self.opt['normalize']=="LinNorm_0.5mean0.5std_inoutsep" or "LinNorm_0mean1std_inoutsep":
            transform_gt, transform_input = parse_transforms_norm(opt, self.dtype) # set scale false, while aberrated inputs are float16.. [0.0, 255.0]
            
            resize_to = opt['resize_to']
            crop_to = opt['crop_to']
            if not resize_to==None: 
                transform_gt.append(transforms.Resize(resize_to))
                transform_input.append(transforms.Resize(resize_to))
            if crop_to!=None and self.opt['ds_type']=="ShiftCorrectedAbrrDataset":
                transform_after_sc=transforms.CenterCrop(crop_to)
                ds_kwargs["transform_after_sc"]=transform_after_sc
            elif self.opt['ds_type']=="AberratedDataset":
                transform_gt.append(transforms.CenterCrop(crop_to))
                transform_input.append(transforms.CenterCrop(crop_to))
                ds_kwargs['transform'] = {"input":transforms.Compose([*transform_input]), "gt":transforms.Compose([*transform_gt])}
            else:
                raise NotImplementedError
        else:
            transform_dataset = parse_transforms_norm(opt, self.dtype) # set scale false, while aberrated inputs are float16.. [0.0, 255.0]
            
            resize_to = opt['resize_to']
            crop_to = opt['crop_to']
            if not resize_to==None: transform_dataset.append(transforms.Resize(resize_to))
            if crop_to!=None and self.opt['ds_type']=="ShiftCorrectedAbrrDataset":
                transform_after_sc=transforms.CenterCrop(crop_to)
                ds_kwargs["transform_after_sc"]=transform_after_sc
            elif self.opt['ds_type']=="AberratedDataset":
                transform_dataset.append(transforms.CenterCrop(crop_to))
            
                ds_kwargs['transform'] = {"gt":transforms.Compose([*transform_dataset])}
            ds_kwargs['transform'] = transforms.Compose([*transform_dataset])

        if opt['ds_type']=="ShiftCorrectedAbrrDataset":
            ShiftCorrectedAbrrDataset.__init__(self, dataset_pth, **ds_kwargs)
            self.super_getitem = ShiftCorrectedAbrrDataset.__getitem__
        else:
            raise NotImplementedError
        
    def __getitem__(self, idx):
        img_lq, img_gt, input_ls = self.super_getitem(self, idx)
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