# %% [markdown]
# # AbrrDataset2D_3dPSF
# For 3D(include depth) input output, I made a new dataset. It was implemented with simple fouriernets on [aberration_sbcho/tree/chyi](https://github.com/chhyyi/aberration_sbcho/tree/chyi) (October, 2024)  
#  
# **new features**:
# * on-the-fly aberration -> was discarded cuz too slow for restormer... Instead, I'm using these dataset to process new aberrated dataset.
# * synthetic aberration with 3d PSF (So 3d volume is required even if output is a 2D image)  
#
# **Config files**:  
# 
# These config files are not for restormer but process dataset only.
# * ```Denoising/Options/synabrr_mpmneural_2dpsf_createdset.yml``` : config to compare with basic 3dpsf...
# * ```Denoising/Options/synabrr_mpmneural_3dpsf_example.yml``` : basic 3dpsf config   

# %%
from basicsr.utils.options import parse # this function parse yaml file and process some options on parsed dictionary
config = parse("shared/mpmneuron_3dpsf_1v10_noDefocus/synabrr_mpmneural_3dpsf_example.yml", is_train=True)

# for example, config['datasets']['train']['phase'] is a new option from parse function.
print(config['datasets']['train']['phase'], config['datasets']['val']['phase'])
# %% 
from basicsr.data.onthefly_abrr3dpsf_dataset import AbrrDataset2D_3dPSF, AbrrDataset2D_3dPSF_SaveDataset
train_dset = AbrrDataset2D_3dPSF(config['datasets']['train'])
val_dset = AbrrDataset2D_3dPSF(config['datasets']['val'])
# let's check out some features. length of train_dset and val_dset is determined by the train_test_split on config yaml file ('train_test_split' was 0.6666667)
print(train_dset[55]['lq_path'], train_dset[55]['gt_path'], len(train_dset), len(val_dset))
print(f"the last image position of train dataset: {train_dset[1079]['gt_path']}, the first image position of validation dataset: {val_dset[0]['gt_path']}")
# Note that axial size=11 and stride_z=3 in config file and it have 550 z-stack. so [1, 2, 537] load [537,...,548] z-stacks. Next one will be [1,2,540] if we have 552 or more z-stack.
# %% save 1gt - ? abrr paired dataset.
from basicsr.data.onthefly_abrr3dpsf_dataset import AbrrDataset2D_3dPSF_SaveDataset
from pathlib import Path

train_dataset = AbrrDataset2D_3dPSF_SaveDataset(config['datasets']['train'])
val_dataset = AbrrDataset2D_3dPSF_SaveDataset(config['datasets']['val'])

processed_dataset_root = Path("shared/mpmneuron_3dpsf_1v10_noDefocus")
#%% Let's save these for old multiabrr dataset... (It was too slow to do on-the-fly)
train_path = processed_dataset_root.joinpath("train")
train_path.mkdir(parents=True, exist_ok=True)
train_gt_path = train_path.joinpath("gt")
train_gt_path.mkdir()
train_path_abrr = train_path.joinpath("abrr")
train_path_abrr.mkdir()

abrr_per_gt = 10
train_files = []
# %%
import yaml
with open(processed_dataset_root.joinpath('dataset_config.yaml'), 'w') as f:
    yaml.dump(config['datasets'], f)

# %%

import tifffile

for i in range(len(train_dataset)):
    save_path = []
    tif_gt_path = train_gt_path.joinpath(f"{i}.tif")
    save_path.append(str(tif_gt_path))
    tifffile.imwrite(tif_gt_path, train_dataset[i]['gt'][:,:,0])
    for j in range(abrr_per_gt):
        tif_abrr_path = train_path_abrr.joinpath(f"{i}_{j}.tif")
        tifffile.imwrite(tif_abrr_path, train_dataset[i]['lq'][:,:,0])
        save_path.append(str(tif_abrr_path))
    train_files.append(save_path)
    
# %%
import csv

with open(processed_dataset_root.joinpath('train_files.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(train_files)
# %% save val dataswet
test_path = processed_dataset_root.joinpath("test/")
test_path.mkdir(parents=True, exist_ok=True)
test_gt_path = test_path.joinpath("gt")
test_gt_path.mkdir()
test_path_abrr = test_path.joinpath("abrr")
test_path_abrr.mkdir()

test_files = []
for i in range(len(val_dataset)):
    save_path = []
    tif_gt_path = test_gt_path.joinpath(f"{i}.tif")
    save_path.append(str(tif_gt_path))
    tifffile.imwrite(tif_gt_path, val_dataset[i]['gt'][:,:,0])
    for j in range(abrr_per_gt):
        tif_abrr_path = test_path_abrr.joinpath(f"{i}_{j}.tif")
        tifffile.imwrite(tif_abrr_path, val_dataset[i]['lq'][:,:,0])
        save_path.append(str(tif_abrr_path))
    test_files.append(save_path)
    
# %%
with open(processed_dataset_root.joinpath('test_files.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(test_files)
# %%
