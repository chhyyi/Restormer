# %% [markdown]
# # Introduction  
# * Research notebook on Restormer, for First meeting on NOVEMBER 14, 2024.  
# * **Overview**: I keep trying to train resotration model that compensate bio-tissue induced aberration. As the follow-up study of the [paper about coherent averaging by Hwang et al.](https://onlinelibrary.wiley.com/doi/10.1002/lpor.202200673), there is another approach by [SB Cho](https://github.com/Shaerit) developing a model based on CNN(U-Net), while I am trying to figure out how much restormer capable of this task because it was considered as a SOTA model!
# * **Synthetic Aberration Dataset**: we are employing MPM Neuron dataset (Not public) by [JH Park](https://github.com/bio-optics) as GT and synthetic aberrated images on it as inputs. Previously, input of synthetic aberration was a slice but I introduced 3d psf and synthesized aberration from 3d input.
# * **LATEST report & discussion**: I have reported that restormer trained with '3D-PSF aberration dataset' could not exhibit any improvement.
#   * Prof. Yoo pointed out that comparison with control group (trained on 2D-PSF abrr dataset) was not fair, as its input of validation was not 3D-PSF abrr dataset but 2D-PSF abrr dataset.
#   * Prof. Park commented that introducing dummy volume / shallow sample(than 5 micrometer) to verify....
#   * I have said that "Further verification is needed on the way of inputting experimental data into the trained model"
# # To Do (in this order)
# * [ ] Fair comparison of 3D-PSF <-> 2D-PSF (See latest comment of prof. Yoo)  
# * [ ] train restormer with multichannel input, aberration should includes defocus. (let gt 1-vs-10 aberrated)
# * [ ] final patch size of progressive learning : I have trained up to patch size 384 on K-BDS server, which is not capable on local server (using 344 instead). Let's compare this.

# %% [markdown]
# # Fair comparison of 3D-PSF <-> 2D-PSF 
# resulting in ```2024-11-06 08:56:07,230 INFO: Validation ValSet,		 # psnr: 54.9144	 # l1loss: 0.2787	 # mse: 0.2699``` slightly worse than 3D-PSF case, coincides with expectations.... I think 2D-PSF model's results are more blurry and noisy texture is well restored.
# 
# # Train Restormer for multi-channel aberrated input   
# Including Defocus, I have processed 3D-PSF aberrated dataset, at ```shared/mpmneuron_3dpsf_1v10_incDefocus```  
# ## Checkout preprocessed dataset & dataset class  
# %%
from basicsr.utils.options import parse
from basicsr.data.mult_abrr_dataset import Dataset_MultAbrr

config = parse("/root/project/shared/mpmneuron_3dpsf_1v10_incDefocus/multabrr_mpmneural.yml", is_train=True)
train_dset = Dataset_MultAbrr(config['datasets']['train'])

print(f"length of train_dset: {len(train_dset)}\n keys of train_dset[-1] is: {train_dset[len(train_dset)-1].keys()}\n last size {train_dset[-1]['lq'].size()}") 
# %% Let's preview gt-abrr pair...
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 4, figsize=(48, 36))
i = 0
train_data = train_dset[-10]
for axrows in axs:
    for ax in axrows:
        ax.imshow(train_data['lq'][i])
        ax.axis("off")
        i+=1
        if i==10:
            break
axs[2, 2].imshow(train_data['gt'][0])
axs[2, 2].axis("off")
axs[2, 2].set_title("gt")
# %% [markdown]
# ## Debug & Train restormer  
# 
# You may use 
# ```bash
# python basicsr/train.py
# ```
# ## Results  
# %%
