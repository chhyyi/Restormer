
# Restormer for de-aberration task

I am testing restormer on the project [aberration_sbcho](https://github.com/chhyyi/aberration_sbcho) (private), as submodule.  
Mostly, it follows 'real-denoising task' of orginal repo. I am not plan to run it on the multi-gpu system. I hope nvidia rtx 4090 is enough for my purpose.  
  
See original [README.md](https://github.com/swz30/Restormer/blob/main/README.md) for further informations


### To do 
- [ ] add multi - abrreation dataset, configuration file, make test-script...
- [ ] implement [coherent averaging](https://onlinelibrary.wiley.com/doi/10.1002/lpor.202200673)

### Usage

```bash
python basicsr\train.py -opt Denoising\Options\multabrr_mpmneural.yml # for train
python basicsr\test_ch.py -opt Denoising\Options\multabrr_mpmneural_test.yml # for test. should modify config file to pass desired 'resume state'
```