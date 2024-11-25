#%%
"""
examine dataset, get some statistics
originally from [aberration_sbcho/tree/chyi](https://github.com/chhyyi/aberration_sbcho/blob/ca331d3b291482762a69d5134d2099ed9bb7c7d6/util_abrr/dset_stat.py)
"""
# %%
import tifffile
import numpy as np
from pathlib import Path

dset_root = Path("/root/project/shared/mpmneuron_3dpsf_1v10_noDefocus")
sub_dir_list=[
    "test/gt",
    "train/gt"
]
dir_lists = [Path(dset_root).joinpath(dir) for dir in sub_dir_list]

tifs_mean = []
dset_max = 0
dset_min = 65535

data_len = 0
# %%
for dir in dir_lists:
    for tif_pth in dir.glob("*.tif"):

        tif = tifffile.imread(tif_pth)
        tifs_mean.append(np.mean(tif))

        tif_max = np.max(tif)
        tif_min = np.min(tif)
        if tif_max>dset_max:
            dset_max=tif_max
        if tif_min<dset_min:
            dset_min=tif_min
        
        data_len=data_len+1

dset_mean = np.mean(tifs_mean)

tifs_std = []
dset_mean = np.mean(tifs_mean)

tif_mse_list = []
for dir in dir_lists:
    for tif_pth in dir.glob("*.tif"):
        tif = tifffile.imread(tif_pth) 
        # In some cases, not standard tif stacks, you may try something like this;
        # tif = tifffile.imread(tif_pth, key=0)
        tifs_std.append(np.mean(tif))
        tif_mse_list.append(np.mean(np.square((tif-dset_mean))))

dset_std = np.sqrt(np.mean(tif_mse_list))

print(f"statistics of tifs in {sub_dir_list}\n len:{data_len}, (min, max):({dset_min}, {dset_max}, mean:{dset_mean}, std:{dset_std}")
# %%

try: 
    file = open(dset_root.joinpath("statistics.dat"),"a")
except:
    file = open(dset_root.joinpath("statistics.dat"),"w")
file.write(f"\n\n statistics of tifs in {sub_dir_list}\n len:{data_len}, (min, max):({dset_min}, {dset_max}), mean:{dset_mean}, std:{dset_std}")
file.close()
# %%