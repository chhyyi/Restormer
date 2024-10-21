# %% [markdown]
# # New dataset class for Restormer
# 14th Oct, I started to implement new features;
# * on-the-fly synthetic aberration: it will synthesize aberration when loading data.
# * Now I runs independently (Not as a sub-repo)
#
# To do
# %%

from basicsr.utils.zernike import zern_polynomial, is_in_circle, ifftn_with_shift, fftn_with_shift
from basicsr.data.transforms import paired_random_crop, random_augmentation
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor



import torch
from pathlib import Path
import tifffile
import numpy as np

class AbrrDataset3D(torch.utils.data.Dataset):
    """
    From mpmneuron dataset by prof JH Park, load subvolume of aberrated <-> GT pair. The dataset is provided as 550 files for 550 stacks. each file is 2048x2048 16bit tif image.
    """
    def __init__(self, stack_path=None, lateral_size=None, axial_size=None, psf_size=None, len_stack=550, section=9, section_size=600, stride_z = "overlap0", fused_size = 2048, axialc = 0.5, axialc_std = 0.5, device=torch.device("cuda")):
        self.stack_tifs = [Path(stack_path).joinpath(f"AOon_Stack_{i}.tif") for i in range(1, 1+len_stack)]
        self.lateral_size = lateral_size
        self.axial_size = axial_size
        assert section==9, NotImplementedError
        self.section=section
        self.section_rt=int(np.sqrt(section)) # as this dataset is stitched from 9 sub-patches, we will divide it into 9 600x600 patches again.
        self.section_size = section_size
        if stride_z=="overlap0":
            self.stride_z = self.axial_size # because overlap is 0. (stride = size - overlap)
        elif type(stride_z) is int:
            self.stride_z = stride_z
    
        self.psf_size = psf_size # pixel size of ideal psf, given by NA size and others...
        self.len_stack = len_stack
        self.len_stack_per_section=(self.len_stack-self.axial_size)//self.stride_z+1
        assert fused_size == 2048, NotImplementedError # JH Park's MPM dataset size
        self.section_stride = fused_size//self.section_rt
        self.axialc = axialc
        self.axialc_std = axialc_std
        self.device = device
    
    def __len__(self):
        return self.len_stack_per_section*self.section
    
    def RandAbrr(self, vol_gt, size_xy, axial_size, psf_size, axialc_mean=3.6264, axialc_std = 0.2,
                 abrr_modes = [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 
                 abrr_tot_amp=4.0, device=torch.device("cuda")):
        """
        for details, see test/synth_aberration.py 4. Exanole of 3d volume.
        abrr_modes : abrr_modes to be included, in OSA/ANSI index. In default, 4(defocus) is excluded
        """
        abrr_amps = torch.rand(len(abrr_modes)).to(device=device)*2.0-1.0
        abrr_amps = (abrr_amps/abrr_amps.abs().sum())*abrr_tot_amp # normalize sum of amplitudes

        abrr_phase = torch.zeros((size_xy, size_xy), dtype=torch.float64).to(device=device)
        for mode, amp in zip(abrr_modes, abrr_amps):
            abrr_phase = abrr_phase + amp*2.0*torch.pi*zern_polynomial(psf_size, mode, (size_xy-psf_size)//2) # generate phasemap from amplitudes
        
        ctf_ideal = is_in_circle(size_xy, ctf_size=psf_size/size_xy).to(device=device)
        psf_stack_abrr = torch.zeros((size_xy, size_xy, axial_size))
        axialc = (axialc_mean+(axialc_std*torch.randn(1,))).to(device=device)
        for k in range(axial_size):
            defocus = (k-axial_size//2)/axialc
            ctf = defocus*zern_polynomial(psf_size, 4, (size_xy-psf_size)//2)
            ctf_field = ctf_ideal*torch.exp(1j*ctf)*torch.exp(1j*abrr_phase)
            psf_stack_abrr[:,:,k] = torch.pow(torch.abs(torch.fft.fftshift(torch.fft.fft2(ctf_field))), 4)

         # normalize maximum of PSF as unity...
        otf = fftn_with_shift(psf_stack_abrr).to(torch.complex128).to(device=device)
        otf = otf/torch.max(torch.abs(otf))
        vol_abrr = torch.abs(ifftn_with_shift(torch.mul(otf,fftn_with_shift(vol_gt)))) # instead of convolution of psf and gt it 
        return vol_abrr


    def __getitem__(self, idx):
        section = idx//self.len_stack_per_section
        section_xidx = section//self.section_rt
        section_yidx = section%self.section_rt
        depth0 = (idx%self.len_stack_per_section)*self.stride_z
        randcrop_x0, randcrop_y0 = torch.randint(0, self.section_size-self.lateral_size,(2,)) if self.lateral_size!=self.section_size else (0, 0)
        gt = torch.zeros(self.lateral_size, self.lateral_size, self.axial_size, dtype=torch.float64).to(device = self.device)
        for i in range(self.axial_size):
            x0 = section_xidx*self.section_stride+randcrop_x0
            y0 = section_yidx*self.section_stride+randcrop_y0
            gt[:,:,i] = torch.from_numpy(tifffile.imread(self.stack_tifs[depth0+i])[x0:x0+self.lateral_size, y0:y0+self.lateral_size].astype(np.float64)).to(device = self.device)

        abrr = self.RandAbrr(gt, self.lateral_size,self.axial_size, self.psf_size, axialc_mean=self.axialc, axialc_std=self.axialc_std, device=self.device)
        return {"gt":gt.moveaxis(2, 0)[None,...], "abrr":abrr.moveaxis(2, 0)[None,...], "pos0":[section_xidx, section_yidx, depth0]}
    
# %%
class AbrrDataset2D_3dPSF_SaveDataset(AbrrDataset3D):
    def __init__(self, opt):
        super().__init__(**opt['AbrrDataset3D'])
        self.opt = opt
        self.mean = opt['mean']
        self.std = opt['std']
        self.gt_folder = opt['dataroot_gt']
        train_test_split = opt.get('train_test_split')
        
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
            self.dset_split = train_test_split
            self.idx_skip = 0
        elif self.opt['phase'] == 'val':
            self.dset_split = 1 - train_test_split
            self.idx_skip = round(super().__len__()*train_test_split)
        

    def __len__(self):
        return round(super().__len__() * self.dset_split)

    def __getitem__(self, index):
        super_out = super().__getitem__(index + self.idx_skip)
        center_z = self.opt['AbrrDataset3D']['axial_size']//2
        img_gt = super_out['gt'][0,center_z][...,None].cpu().numpy()
        img_lq = super_out['abrr'][0,center_z][...,None].cpu().numpy()
        gt_path = super_out['pos0'].append(f"center_z: {center_z}")
        gt_path = super_out['pos0']

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': 'synthesized from gt',
            'gt_path': gt_path
        }


# %%

class AbrrDataset2D_3dPSF(AbrrDataset3D):
    def __init__(self, opt):
        super().__init__(**opt['AbrrDataset3D'])
        self.opt = opt
        self.mean = opt['mean']
        self.std = opt['std']
        self.gt_folder = opt['dataroot_gt']
        
        train_test_split = opt.get('train_test_split')
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']
            self.dset_split = train_test_split
            self.idx_skip = 0
        elif self.opt['phase'] == 'val':
            self.dset_split = 1 - train_test_split
            self.idx_skip = round(super().__len__()*train_test_split)
        

    def __len__(self):
        return round(super().__len__() * self.dset_split)

    def __getitem__(self, index):
        super_out = super().__getitem__(index + self.idx_skip)
        center_z = self.opt['AbrrDataset3D']['axial_size']//2
        img_gt = super_out['gt'][0,center_z][...,None].cpu().numpy()
        img_lq = super_out['abrr'][0,center_z][...,None].cpu().numpy()
        gt_path = super_out['pos0'].append(f"center_z: {center_z}")
        gt_path = super_out['pos0']

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, 1.0, gt_path)
        
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32 = True)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)        

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': 'synthesized from gt',
            'gt_path': gt_path
        }

# %%
#import matplotlib.pyplot as plt
if __name__ == "__main__":
    dset = AbrrDataset3D("/app/mpmneuron_original/aoon_stack/", 600, 12, 512)
    #dset = AbrrDataset2D_3dPSF("/app/mpmneuron_original/aoon_stack/", 600, 12, 512)
    #print(dset[1]['abrr'][0,3].shape)
    pass
# %%
