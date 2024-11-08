
# Restormer for de-aberration task

I am testing restormer on the project [aberration_sbcho](https://github.com/chhyyi/aberration_sbcho) (private)
Mostly, it follows 'real-denoising task' of orginal repo. I am not plan to run it on the multi-gpu system. I hope nvidia rtx 4090 is enough for my purpose.  
  
See original [README.md](https://github.com/swz30/Restormer/blob/main/README.md) for further informations

## Changes
* It was submodule but now it is runnable independently
* Made new dataset class for 3D-PSF abrr dataset for generation of dataset instead of on-the-fly use... cuz it was too slow
## To do 
- [x] test on-the-fly synthetic aberration (new dataset class for this)
- [ ] prepare to run on remote server, via bash script file : Trained one model with patch size 384...(which causes oom in RTX 4090) It should be merged with local server version...

## Usage
option file should be passed to the train/test scripts ```-opt``` (option)
```bash
python basicsr\train.py -opt Denoising\Options\multabrr_mpmneural.yml # for train
python basicsr\test_ch.py -opt Denoising\Options\multabrr_mpmneural_test.yml # for test. should modify config file to pass desired 'resume state'
```

## config files
All the config files at ```Denoising/Options``` just same as original repo. See ```README.md``` for usage...

* ```multabrr_mpmneural.yml```: Default, train. should be passed to ```basicsr/train.py```
* ```multabrr_mpmneural_test.yml``` should be ```basicsr/test.py```
* ```multabrr_mpmneural_test_AUGdata.yml``` should be passed to ```basicsr/test_1stack.py``` as option. rename original ```vis_outputs```, ```visualization``` directory on outputs(experiments) directory.
* **To process dataset** ```synabrr_mpmneural_3dpsf_example.yml```, ```synabrr_mpmneural_2dpsf_createdset.yml```