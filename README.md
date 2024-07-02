# Expand and Merge: Continual Learning with the Guidance of Fixed Text Embedding Space

This is the official implementation of the IJCNN2024 paper: Expand and Merge: Continual Learning with the Guidance of Pretrained Text Embedding Space. The codebase is modified from my continual learning framework `CL_Pytorch` (https://github.com/GiantJun/CL_Pytorch). You can find more baselines on CL_Pytorch.

## How to run the code

### Prepare environment

```bash
pip3 install pyyaml tensorboard tensorboard wandb scikit-learn timm quadprog tensorboardX
```

### Run experiments
Let's take reproducing results on CIFAR100 as an example.

1. Edit the hyperparameters in the corresponding `options\multi_steps\expand_merge\cifar100.yaml` file

2. Train models:

```bash
python main.py --config options\multi_steps\expand_merge\cifar100.yaml
```

3. Test models with checkpoint (ensure save_model option is True before training)

```bash
python main.py --checkpoint_dir logs/XXX/XXX.pkl
```

If you want to temporary change GPU device in the experiment, you can type `--device #GPU_ID` without changing 'device' in `.yaml` config file.

### Add datasets and your method

Add corresponding dataset .py file to `datasets/`. It is done! The programme can automatically import the newly added datasets.

we put continual learning methods inplementations in `/methods/multi_steps` folder, pretrain methods in `/methods/pretrain` folder and normal one step training methods in `/methods/singel_steps`.

Supported Datasets:

- Natural image datasets: CIFAR-10, CIFAR-100, ImageNet1K, ImageNet100, TinyImageNet

- Medical image datasets: MedMNIST, SD-198

More information about the supported datasets can be found in `datasets/`
