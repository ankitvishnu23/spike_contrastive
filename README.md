### Modification of SimCLR torch implementation (https://github.com/sthalles/SimCLR) for neural waveform templates. 

## Installation

```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
$ python run.py
```

## Config file

Before running the spike trainer, make sure you choose the correct running configurations. You can change the running configurations by passing keyword arguments to the ```run.py``` file.

```python

$ python run.py -data ./datasets --dataset-name wfs --log-every-n-steps 100 --epochs 100 

```
