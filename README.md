
# Restormer for de-aberration task

I am testing restormer on the project [aberration_sbcho](https://github.com/chhyyi/aberration_sbcho) (private)
Mostly, it follows 'real-denoising task' of orginal repo. I am not plan to run it on the multi-gpu system. I hope nvidia rtx 4090 is enough for my purpose.  
  
See original [README.md](https://github.com/swz30/Restormer/blob/main/README.md) for further informations

## Changes
* It was submodule but now it is runnable independently

## To do 
- [ ] make it independantly runable
- [ ] test on-the-fly synthetic aberration (new dataset class for this)
- [ ] prepare to run on remote server, via bash script file

## Usage

```bash
python basicsr\train.py -opt Denoising\Options\multabrr_mpmneural.yml # for train
python basicsr\test_ch.py -opt Denoising\Options\multabrr_mpmneural_test.yml # for test. should modify config file to pass desired 'resume state'
```