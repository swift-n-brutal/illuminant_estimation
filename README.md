[//]: <links>
[proj]: http://mmlab.ie.cuhk.edu.hk/projects/illuminant_estimation.html
[gs568]: http://www.cs.sfu.ca/~colour/data/shi_gehler/
[data]: ./data/
[models]: ./models/
[preds]: ./preds/ 
[pretrained_3fd]: https://mycuhk-my.sharepoint.com/:u:/g/personal/1155067722_link_cuhk_edu_hk/Ee4oEhBopxxMka2XXHjkjsoBTvqAGL6-oBqOzwVtSD4whg?e=27AiVW
[pretrained_all]: https://mycuhk-my.sharepoint.com/:u:/g/personal/1155067722_link_cuhk_edu_hk/EX-CcOPFjwJOvMoWaNzxrFcBmuigT-UHOgnBrH3tZ6aeZw?e=aDth61

# Illuminant Estimation

This project implements the illuminant estimation method which is presented in the paper "Deep Specialized Network for Illuminant Estimation" [Project][proj]. The implementation is based on Python and TensorFlow.

## Prerequisites
* python=3.6
* tensorflow=1.14
* pyzmq=19.0.1 (optional)

## Training from scratch

We take the training procedure on [Gehler-Shi][gs568] dataset as example. Please first follow the instructions in [data][data] to preprocess the data. Then run the following commands to train three models, as the performance should be evaluated by 3-fold cross validation.
```
CUDA_VISIBLE_DEVICES=0 python solver.py --gs-has-loc --gs-test-set 0 &
CUDA_VISIBLE_DEVICES=1 python solver.py --gs-has-loc --gs-test-set 1 &
CUDA_VISIBLE_DEVICES=2 python solver.py --gs-has-loc --gs-test-set 2 &
```

NOTE: *ZeroMQ is recommended for efficient training.* The training for each model takes roughly 12 hours on a single GeForce GTX TITAN X gpu.

If default parameters are used during training, the model parameters will be stored in [models][models] finally and the file names look like:
```
--- gs568-0_bs128_lr0.02
 |   |- hypnet_4000000.npz
 |   |- selnet_4000000.npz
 |
 |- gs568-1_bs128_lr0.02
 |   |- hypnet_4000000.npz
 |   |- selnet_4000000.npz
 |
 |- gs568-2_bs128_lr0.02
     |- hypnet_4000000.npz
     |- selnet_4000000.npz
```

## Test

Then run the following commands to test on the three sets:
```
CUDA_VISIBLE_DEVICES=0 python solver.py --gs-has-loc --gs-test-set 0 --test-only &
CUDA_VISIBLE_DEVICES=1 python solver.py --gs-has-loc --gs-test-set 1 --test-only &
CUDA_VISIBLE_DEVICES=2 python solver.py --gs-has-loc --gs-test-set 2 --test-only &
```
### Pre-trained models
Pre-trained models can be downloaded from the following links. Please unzip the files inside [models][models].
|Link|Description|
|---|---|
|[OneDrive][pretrained_3fd]|Trained for 3-fold cross validation|
|[OneDrive][pretrained_all]|Trained on all images|

The estimated illuminants for local patches of each image will be stored in [preds][preds]. Finally run the following command to get global predictions for each image:
```
python test_preds.py --weighted-median
```
