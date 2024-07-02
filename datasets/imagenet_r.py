from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np
from utils.toolkit import split_images_labels

class ImageNet_R(iData):
    '''
    Dataset Name:   ImageNet_R dataset
    Source:         A subset of the Tiny Images dataset.
    Task:           Classification Task
    Data Format:    32x32 color images.
    Data Amount:    60000 (500 training images and 100 testing images per class)
    Class Num:      100 (grouped into 20 superclass).
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size if img_size != None else 224
        self.train_trsf = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63/255)
        ]
        self.strong_trsf = [
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
        ]
        self.test_trsf = []
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.480, 0.448, 0.398], std=[0.230, 0.227, 0.226]),
        ]
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        # train_dataset = datasets.ImageFolder(os.path.join(os.environ['DATA'], 'imagenet_r', 'train'))
        # test_dataset = datasets.ImageFolder(os.path.join(os.environ['DATA'], 'imagenet_r', 'test'))
        
        # self.train_data, self.train_targets = split_images_labels(train_dataset.imgs)
        # self.test_data, self.test_targets = split_images_labels(test_dataset.imgs)
        
        root_dir = os.path.join(os.environ['DATA'], 'imagenet_r')

        id_to_name = {}
        with open(os.path.join(root_dir, 'README.txt')) as f:
            for i in range(13):
                f.readline()
            for i in range(200):
                text = f.readline().split(' ')
                id_to_name[text[0]] = text[1].replace('\n', '')

        self.class_to_idx = {}
        train_data, train_targets = [], []
        # sorted function is important to get the same class order on different devices!
        for target_id, class_id in enumerate(sorted(os.listdir(os.path.join(root_dir, 'train')))):
            class_name = id_to_name[class_id]
            self.class_to_idx[class_name] = target_id
            
            # sorted function is important to get the same class order on different devices!
            for img_id in sorted(os.listdir(os.path.join(root_dir, 'train', class_id))):
                train_data.append(os.path.join(root_dir, 'train', class_id, img_id))
                train_targets.append(target_id)
        self.train_data, self.train_targets = np.array(train_data), np.array(train_targets, dtype=int)

        test_data, test_targets = [], []
        # sorted function is important to get the same class order on different devices!
        for target_id, class_id in enumerate(sorted(os.listdir(os.path.join(root_dir, 'test')))):
            
            # sorted function is important to get the same class order on different devices!
            for img_id in sorted(os.listdir(os.path.join(root_dir, 'test', class_id))):
                test_data.append(os.path.join(root_dir, 'test', class_id, img_id))
                test_targets.append(target_id)
        self.test_data, self.test_targets = np.array(test_data), np.array(test_targets, dtype=int)
    
