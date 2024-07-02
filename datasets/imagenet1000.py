from torchvision import datasets, transforms
from datasets.idata import iData
import os
import numpy as np
from utils.toolkit import split_images_labels

class ImageNet1000(iData):
    '''
    Dataset Name:   ImageNet1K (ILSVRC2012)
    Source:         Organized according to the WordNet hierarchy
    Task:           Classification Task
    Data Format:    224x224 color images.
    Data Amount:    1281167 for training and 50,000 for validation
    Class Num:      1000
    Label:          Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).

    Reference: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        self.use_path = True
        self.img_size = img_size if img_size != None else 224
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63/255)
        ]
        self.test_trsf = [
            transforms.Resize((256,256)), # 256
            transforms.CenterCrop(224), # 224
        ]
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0,"You should specify the folder of your dataset"
        root_dir = os.path.join(os.environ['DATA'], 'ilsvrc2012')

        self.class_to_idx = {}
        train_data, train_targets = [], []
        # sorted function is important to get the same class order on different devices!
        for target_id, class_id in enumerate(sorted(os.listdir(os.path.join(root_dir, 'train')))):
            # class_name = id_to_name[class_id]
            class_name = class_id
            self.class_to_idx[class_name] = target_id
            
            # sorted function is important to get the same class order on different devices!
            for img_id in sorted(os.listdir(os.path.join(root_dir, 'train', class_id))):
                train_data.append(os.path.join(root_dir, 'train', class_id, img_id))
                train_targets.append(target_id)
        self.train_data, self.train_targets = np.array(train_data), np.array(train_targets, dtype=int)

        test_data, test_targets = [], []
        # sorted function is important to get the same class order on different devices!
        for target_id, class_id in enumerate(sorted(os.listdir(os.path.join(root_dir, 'val')))):
            
            # sorted function is important to get the same class order on different devices!
            for img_id in sorted(os.listdir(os.path.join(root_dir, 'val', class_id))):
                test_data.append(os.path.join(root_dir, 'val', class_id, img_id))
                test_targets.append(target_id)
        self.test_data, self.test_targets = np.array(test_data), np.array(test_targets, dtype=int)
