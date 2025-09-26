## HSFL

A base code of paper:
> Performance Optimization of Split Federated Learning in Heterogeneous Edge Computing Environments

## General Guidelines

We provide this codebase which contains:

* Training on 3 datasets: **CIFAR10**, **CIFAR100** and **Tiny-imagenet** dataset.
* For hyperparameters, see `train_params.py`
* We prepare the model of **Resnet**, **MobilVit** and **Unet** for image classification tasks. If you want to add
  your custom model, add it in the `models` folder and modify the code in `utils/model_utils.py`.

* This codebase is easy to extend to other FL algorithms, models and datasets.

## Preparations

### Dataset generation

To prepare CIFAR10 and CIFAR100 for training, you can just run the training script to allow for self downloading.

For Tiny-ImageNet, please **download** the dataset to respective folder (
i.e., download Tiny-ImageNet to `data/tiny-imagenet-200`).

The statistics of real federated datasets are summarized as follows.
<center>

| Dataset       | Devices | Training Samples | Num classes <br> |
|---------------|---------|------------------|------------------|
| CIFAR10       | 20      | 50000            | 10               | 
| CIFAR100      | 20      | 50000            | 100              |
| Tiny-ImageNet | 20      | 100000           | 200              |

</center>

### Downloading dependencies

```
pytorch==1.11.0
python==3.8
numpy==1.22.4
pandas==1.5.3
opencv-python==4.8.1.78
```

## Run

(1) Non-iid for CIFAR--10 , CIFAR-100 and Tiny-ImageNet

Please add listed for non iid

```
--noniid="dirichlet"
--dirichlet==0.5
```

(2) Client sampling

If `frac != 1.0`, we recommend not using the learning rate scheduler, or modifying the code for updating from
`XXClientFedXX.py` to `XXServerFedXX.py` .

(3) Split Federated Learning

Modify the parameter of **batch_sizes** and **train_layer** in `SFLServerFedAvg.py`
