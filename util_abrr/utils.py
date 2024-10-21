# %% 
# [aberration_sbcho/tree/chyi](https://github.com/chhyyi/aberration_sbcho/blob/ca331d3b291482762a69d5134d2099ed9bb7c7d6/util_abrr/dset_stat.py)

import torch
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import tifffile
try:
    from torchvision.transforms import v2 as transforms
except:
    from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
from pathlib import Path
from torch.fft import fftn, ifftn, rfftn, irfftn, fft2, ifftshift
import torch.nn.functional as F

import sys
sys.path.append(Path(".").resolve().parent)

import datetime

from copy import deepcopy
import logging

class LinNorm(torch.nn.Module):
    """
    linearly scale tensor as min = 0.0 and max = 1.0
    """
    def __init__(self):
        super().__init__()
    def forward(self, tensor):
        return (tensor-tensor.min())/(tensor.max()-tensor.min())
    

class Norm_center_dot5(transforms.Normalize):
    """
    assume tensor range [0,1], perform (tensor-mean)/std,
    move it to mean=0.5 and std=0.5
    """
    def __init__(self, mean, std):
        super().__init__(mean, std)
    def forward(self, tensor):
        return (super().forward(tensor))/2.0+0.5

class Norm_center0(transforms.Normalize):
    """
    assume tensor range [0,1], perform (tensor-mean)/std,
    results in mean=0, std=1 distribution
    """
    def __init__(self, mean, std):
        super().__init__(mean, std)
    def forward(self, tensor):
        return super().forward(tensor)

class Div255(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, tensor):
        return tensor/255.0

class Div65535(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, tensor):
        return tensor/65535.0

def parse_transforms_norm(opt, dtype):
    opt_transforms = opt['normalize']
    if opt_transforms == "LinNorm_0.5mean0.5std":
        return [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                Norm_center_dot5(opt['mean'], opt['std'])]
    elif opt_transforms == "LinNorm_0.5mean0.5std_inoutsep":
        return [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                Norm_center_dot5(opt['mean_gt'], opt['std_gt'])], [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                Norm_center_dot5(opt['mean_input'], opt['std_input'])]
    elif opt_transforms == "LinNorm_0mean1std_inoutsep":
        return [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                Norm_center0(opt['mean_gt'], opt['std_gt'])], [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                Norm_center_dot5(opt['mean_input'], opt['std_input'])]
    elif opt_transforms == "LinNorm_div255":
        return [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                Div255()]
    elif opt_transforms == "LinNorm_div65535":
        return [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                Div65535()]
    elif opt_transforms == "Div65535+0.5mean01":
        return [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                Norm_center_dot5(opt['mean'], opt['std']),
                Div65535()]
    elif opt_transforms == "LinNorm01+0.5mean01":
        return [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                LinNorm(),
                Norm_center_dot5(opt['mean'], opt['std'])]
    elif opt_transforms == "LinNorm01":
        return [transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                LinNorm(),]
    else: raise NotImplementedError

def model_test(iterator_out, preview_model, device=torch.device("cuda")):
    preview_model=preview_model.to(device)
    input_ims, label, _ = iterator_out
    input_ims = input_ims.to(device)

    out = preview_model(input_ims).detach().cpu()
    loss_ = torch.nn.MSELoss()(label, out)
    #print(f"MSELoss = {loss_}")
    return out, loss_


def model_test_sep(input_ims, label, preview_model, device=torch.device("cuda")):
    input_ims = input_ims[:,None,...].to(device)

    out = preview_model(input_ims).detach().cpu()
    loss_ = torch.nn.MSELoss()(label, out)
    #print(f"MSELoss = {loss_}")
    return out, loss_

def gridplot_gt_abb(iterator_out, title, preview_model=None, device=torch.device("cuda"), nrow_ncol=(3,4), figsize_per_tile=2.0, plot=True, return_grid=False, preview_idx=-1, save_fig=None):
    """
    Modification of matplotlib example, [simple imagegrid](https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html)
    preview_idx : choose what to plot in a batch,
        shape of iterator_out: iterator_out = (input_img, label) so its shape is ((*batch_size*, input_channels, W, H),(*batch_size*, 1, W, H))
    """
    
    input_ims, label, input_pth = iterator_out

    figsize = (nrow_ncol[1]*4., nrow_ncol[0]*4.)

    if preview_model:
        out, loss_ = model_test(iterator_out, preview_model, device=device)
            
    preview_idx0 = preview_idx
    if preview_idx0 == -1: # default, plot images over whole batch.
        preview_idx =list(range(len(input_ims)))
    elif type(preview_idx0) is int:
        preview_idx = [preview_idx0]
    else:
        raise NotImplementedError
    
    for idx in preview_idx:

        if preview_model:
            preview = torch.cat((label[idx].cpu(), input_ims[idx].cpu(), out[idx]))
            
        else:
            preview = torch.cat((label[idx], input_ims[idx])).cpu()
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title+"\n(GT:top left, inputs, output:bottom right)")
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(nrow_ncol),
                        axes_pad=0.1,
                        )

        for ax, im in zip(grid, preview):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(im)

        #print(f"np.shape(input_ims) : (batch_size, channel, w, h) = {np.shape(input_ims)}")

        if plot:
            plt.show()
        if save_fig:
            plt.savefig(save_fig.with_stem(f"{save_fig.stem}_{Path(input_pth[0][idx]).stem}"))
        plt.close()
    if return_grid:
        return loss_, grid
    else:
        return loss_
   
def get_ds_kwargs(opt, input_channels):
    """
    return transform for keyword-arguments dictionary for AberratedDataset,
    opt should have these params;
        """
    kwargs_to_dataset_list=["gt_path", "abrr_path", "maxlen", "abrr_inputs", "dset_mode"] # except for transform and input channels, which will be added from now on.
    ds_kwargs = {i:opt[i] for i in kwargs_to_dataset_list}
    ds_kwargs["input_channels"] = input_channels
    # set scale false, while aberrated inputs are float16.. [0.0, 255.0]
    transform_gt = [transforms.ToTensor(),
                    transforms.ToDtype(torch.float32, scale=False),
                    Norm_center_dot5([opt['mean_gt'],], [opt['std_gt'],])]
    transform_input = [transforms.ToTensor(),
                        transforms.ToDtype(torch.float32, scale=False),
                        Norm_center_dot5([opt['mean_input'],], [opt['std_input'],])] 
    
    if not opt['resize_to']==None: 
        transform_gt.append(transforms.Resize(opt['resize_to']))
        transform_input.append(transforms.Resize(opt['resize_to']))
    if opt['crop_to']!=None and opt['ds_type']=="ShiftCorrectedAbrrDataset":
        transform_after_sc=transforms.CenterCrop(opt['crop_to'])
        ds_kwargs["transform_after_sc"]=transform_after_sc
    elif opt['ds_type']=="AberratedDataset":
        transform_gt.append(transforms.CenterCrop(opt['crop_to']))
        transform_input.append(transforms.CenterCrop(opt['crop_to']))
    transform_gt = transforms.Compose([*transform_gt])
    transform_input = transforms.Compose([*transform_input])
    ds_kwargs["transform"] = {"gt": transform_gt, "input": transform_input}
    return ds_kwargs

class AberratedDataset(torch.utils.data.Dataset):
    """
    load aberrated images as dataset.

    shuffle_aberration : If True, shuffle the order of aberrated images.

    file structure in dataset_path:
    
        dataset_path/GT/
        ----0.tif
        ----1.tif
        ----2.tif
        ----...
        dataset_path/Aberrated/
        ----0_0.tif
        ----0_1.tif
        ----...
        ----1_0.tif
        ----1_1.tif
        ----...
    """
    def __init__(self, dataset_path, input_channels = 10, transform=None, gt_path=Path("GT"), abrr_path=Path("Aberrated"), shuffle_abberations=True, maxlen=1000, abrr_inputs=10, dset_mode="fixed"):
        if dset_mode=="strict":
            self.gt_files = [x for x in Path(dataset_path).joinpath(gt_path).glob("*.tif")][:maxlen] 
            #self.ab_files = [[Path(dataset_path).joinpath(aberrated_path).joinpath(f"{gt.stem}_{i}.tif") for i in range(input_channels)] for gt in self.gt_files]
        elif dset_mode=="fixed": #fixed numbers
            self.gt_files = []
            for i in range(maxlen):
                if Path(dataset_path).joinpath(gt_path).joinpath(f"{i}.tif").is_file():
                    self.gt_files.append(Path(dataset_path).joinpath(gt_path).joinpath(f"{i}.tif"))
        else:
            raise NotImplementedError
        self.ds_path = dataset_path
        if type(transform) is dict: # in this case, 'gt' and 'abrr' have other transform
            self.transform_gt = transform['gt']
            self.transform_input = transform['input']
        else:
            self.transform_gt = transform
            self.transform_input = transform
        self.shuffle_abbs = shuffle_abberations
        self.input_channels = input_channels
        self.num_abrr_imgs = abrr_inputs
        self.abrr_path = abrr_path

        def get_inputs_strict():
            def inputs_list(ds_pth, abrr_pth, gt_file, num_abr):
                return [pp for pp in Path(ds_pth).joinpath(abrr_pth).glob(f"{gt_file.stem}_*.tif")]
            return inputs_list
            
        def get_inputs_fixed():
            def inputs_list(ds_pth, abrr_pth, gt_file, num_abr):
                return [Path(ds_pth).joinpath(abrr_pth).joinpath(f"{gt_file.stem}_{i}.tif") for i in range(num_abr)]
            return inputs_list
        
        if dset_mode=="fixed":
            self.get_inputs = get_inputs_fixed() 
        elif dset_mode=="strict":
            self.get_inputs = get_inputs_strict()
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.gt_files)
    
    def __getitem__(self, idx):
        input_ls = self.get_inputs(ds_pth = self.ds_path, abrr_pth=self.abrr_path, gt_file=self.gt_files[idx], num_abr=self.num_abrr_imgs)
        
        if self.shuffle_abbs:
            idxs = torch.randperm(len(input_ls))[:self.input_channels]
            input_ls = [input_ls[i] for i in idxs]
        else:
            input_ls = input_ls[:self.input_channels]

        label = self.gt_files[idx]
        if self.transform_input and self.transform_gt:
            inputs = torch.cat([self.transform_input(tifffile.imread(x).astype(float)) for x in [x for x in input_ls]])
            labels = torch.cat([self.transform_gt(tifffile.imread(label).astype(float)),])
        else:
            raise NotImplementedError("transform should be given")
        return (inputs, labels, [str(input_pth) for input_pth in input_ls])
        #return inputs, labels, input_ls 

class ShiftCorrectedAbrrDataset(AberratedDataset):
    def __init__(self, ds_pth, input_channels = 10, transform=None, transform_after_sc=None, gt_path=Path("GT"), abrr_path=Path("Aberrated"), shuffle_abberations=True, maxlen=1000, abrr_inputs=10, dset_mode="fixed"):
        super().__init__(ds_pth, input_channels=input_channels, transform=transform, gt_path=gt_path, abrr_path=abrr_path, shuffle_abberations=shuffle_abberations, maxlen=maxlen, abrr_inputs=abrr_inputs, dset_mode=dset_mode)
        self.transform_after_sc = transform_after_sc

    def __getitem__(self, idx):
        inputs, labels, inputs_list = super().__getitem__(idx)
        sc_inputs, _, _ = shift_correction(recon=inputs[None,...], ref=labels[None, ...])
        if not self.transform_after_sc == None:
            cat_out = self.transform_after_sc(torch.cat((sc_inputs.squeeze(0), labels)))
            sc_inputs = cat_out[:-1]
            labels = cat_out[-1:]
        return (sc_inputs, labels, inputs_list)
    
def shift_correction(recon:torch.tensor, ref, pad=False):
    """
    recon : aberrated image in dim (B, C, H, W)  
    ref: reference image in dim (B, C, H, W)  
    shift correction implementation by SW Cho(LAIT, UNIST, Ulsan, Korea), 2024.
    Which is based on the paper Hwang, Byungjae, Taeseong Woo, Cheolwoo Ahn, and Jung‐Hoon Park. “Imaging through Random Media Using Coherent Averaging.” Laser & Photonics Reviews 17, no. 3 (March 2023): 2200673. https://doi.org/10.1002/lpor.202200673

    """
    B, C, H, W = recon.shape
    
    # Ensure inputs are float tensors
    recon = recon.float()
    ref = ref.float()
    
    # Remove mean
    recon_zeromean = recon - recon.mean(dim=(-2, -1), keepdim=True)
    ref_zeromean = ref - ref.mean(dim=(-2, -1), keepdim=True)
    
    if pad:
        # Compute FFT
        recon_fr = rfftn(recon_zeromean, dim=[-2, -1], s=[2*H, 2*W])
        ref_fr = rfftn(ref_zeromean, dim=[-2, -1], s=[2*H, 2*W])
    
        # Compute cross-correlation (with conjugate)
        cross_correlation_fr = complex_matmul(torch.conj(recon_fr), ref_fr)
        # cross_correlation_fr = complex_matmul(torch.conj(ref_fr), recon_fr)
    
        # Inverse FFT to get spatial cross-correlation
        cross_correlation = irfftn(cross_correlation_fr, dim=[-2, -1], s=[2*H, 2*W])
    
        # Find the maximum correlation for each image in the batch
        _, idx = torch.max(cross_correlation.view(B, -1), 1)
    
        # Calculate the shift
        max_row = torch.div(idx, (2*W), rounding_mode='floor')
        max_col = torch.remainder(idx, (2*W))
    
        # Calculate the required shift
        shift_row = torch.where(max_row > H, max_row - 2*H, max_row)
        shift_col = torch.where(max_col > W, max_col - 2*W, max_col)
    
        # Apply the shift
        grid_x = torch.arange(W, device=recon.device).repeat(H, 1)
        grid_y = torch.arange(H, device=recon.device).unsqueeze(1).repeat(1, W)
    
        grid_x = (grid_x.unsqueeze(0) - shift_col.view(B, 1, 1)) % W
        grid_y = (grid_y.unsqueeze(0) - shift_row.view(B, 1, 1)) % H
    
        grid = torch.stack((grid_x / (W - 1) * 2 - 1, grid_y / (H - 1) * 2 - 1), dim=-1)
    
        x_aligned = F.grid_sample(recon, grid.repeat(C, 1, 1, 1), mode='bilinear', padding_mode='border', align_corners=True)
        
        return x_aligned, shift_row, shift_col
    else:
        recon_fr = rfftn(recon_zeromean, dim=[-2, -1])
        ref_fr = rfftn(ref_zeromean, dim=[-2, -1])
        
        cross_correlation_fr = (recon_fr) * torch.conj(ref_fr)
        cross_correlation = irfftn(cross_correlation_fr, dim=[-2, -1], s=[-1, -1])
        cross_correlation = ifftshift(cross_correlation, dim=[-2, -1])
        cross_correlation = torch.abs(cross_correlation)
        
        # x_aligned, delM1, delM2 = batchwise_align_images(recon, ref)
        
        # Find the maximum correlation for each image in the batch
        _, idx = torch.max(cross_correlation.view(B,C,-1), 2)
        
        B, _, H, W = cross_correlation.shape
        max_row = torch.div(idx, W, rounding_mode='floor')
        max_col = torch.remainder(idx, W)
    
        delM1 = (max_row - H // 2) % H
        delM2 = (max_col - W // 2) % W
    
        # Apply the shift using roll for each image in the batch
        x_aligned_channel = []
        for i in range(B):
            x_aligned = []
            for j in range(C):
                x_aligned.append(torch.roll(recon[i][j], shifts=(-delM1[i][j].item(), -delM2[i][j].item()), dims=(-2, -1)))
            x_aligned_channel.append(torch.stack(x_aligned))
        x_aligned_batch = torch.stack(x_aligned_channel)
    
        return x_aligned_batch, delM1, delM2
    # %%
    
def complex_matmul(a, b, groups = 1):
    """Multiplies two complex-valued tensors."""
    # https://github.com/fkodom/fft-conv-pytorch
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])

