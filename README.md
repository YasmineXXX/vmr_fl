FedVMR
=====
PyTorch implementation of Federated Video Moment Retrieval (FedVMR). 

## Getting started
### Prerequisites

1. Prepare feature files

Download [feature.tar.gz](https://drive.google.com/file/d/1j4mVkXjKCgafW3ReNjZ2Rk6CKx0Fk_n5/view?usp=sharing) (33GB). 
After downloading the feature file, extract it to the `feature` directory under `fedvmr` directory:
```
tar zxvf feature.tar.gz
```
It contains I3D video features of Charades-STA (charades_i3d_rgb.hdf5), C3D video features of ActivityNet-Captions (sub_activitynet_v1-3.c3d.hdf5) and text features (6B.300d.npy). 

2. Install dependencies.
- Python 3.7
- PyTorch 1.8.0
- Cuda 11.1
- nltk
- tqdm
- h5py
- easydict

### Running Code

```
python baseline_train_fvmr.py
```
