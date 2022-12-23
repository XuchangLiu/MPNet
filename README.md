
## Prerequisites
- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)
- MATLAB for computing [evaluation metrics](statistic/)


## Datasets

MPNet are evaluated on four datasets*: 
Rain100H [1], Rain100L [1], Rain12 [2] and Rain1400 [3]. 
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), 
and place the unzipped folders into `./datasets/test/`.

To train the models, please download training datasets: 
RainTrainH [1], RainTrainL [1] and Rain12600 [3] from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), 
and place the unzipped folders into `./datasets/train/`. 

*_We note that:_

_(i) The datasets in the website of [1] seem to be modified. 
    But the models and results in recent papers are all based on the previous version, 
    and thus we upload the original training and testing datasets 
    to [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg) 
    and [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g)._ 

_(ii) For RainTrainH, we strictly exclude 546 rainy images that have the same background contents with testing images.
    All our models are trained on remaining 1,254 training samples._


## Getting Started

### 1) Testing

We have placed our pre-trained models into `./logs/`. 

Run shell scripts to test the models:
```bash
bash test_Rain100H.sh   # test models on Rain100H
bash test_Rain100L.sh   # test models on Rain100L
bash test_Rain12.sh     # test models on Rain12
bash test_Rain1400.sh   # test models on Rain1400 
bash test_real.sh       # test PReNet on real rainy images
```

### 2) Evaluation metrics

We also provide the MATLAB scripts to compute the average PSNR and SSIM values reported in the paper.
 

```Matlab
 cd ./statistic
 run statistic_Rain100H.m
 run statistic_Rain100L.m
 run statistic_Rain12.m
 run statistic_Rain1400.m
 run statistic_Ablation.m  # compute the metrics in Ablation Study
```
###


### 3) Training

Run shell scripts to train the models:
```bash
bash train_PReNet.sh      
bash train_PReNet_r.sh    

```
You can use `tensorboard --logdir ./logs/your_model_path` to check the training procedures. 

### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 18            | Training batch size
recurrent_iter         | 6             | Number of recursive stages
epochs                 | 100           | Number of training epochs
milestone              | [30,50,80]    | When to decay learning rate
lr                     | 1e-3          | Initial learning rate
save_freq              | 1             | save intermediate model
use_GPU                | True          | use GPU or not
gpu_id                 | 0             | GPU id
data_path              | N/A           | path to training images
save_path              | N/A           | path to save models and status           

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0                | GPU id
recurrent_iter         | 6                | Number of recursive stages
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results

