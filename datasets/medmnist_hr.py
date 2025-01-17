from torchvision import transforms
from datasets.idata import iData
import os
import numpy as np
import random

class MedMNIST_HR(iData):
    '''
    Dataset Name:   Path16
    Task:           Diverse classification task (binary/multi-class)
    Data Format:    224x224 color images.
    Data Amount:    800 each class for training , 100 each class for test.
    
    Reference: 
    '''
    def __init__(self, img_size=None) -> None:
        super().__init__()
        # 以下表示中，字符为子数据集名字, 数字为子数据集中包含的类别数
        self.has_valid = True

        # self._dataset_info = [('blood',8), ('derma',7), ('breast',3), ('pne',2), ('path',9), ('oct',4)] # order1
        self._dataset_info = [('path',9), ('oct',4), ('blood',8), ('derma',7), ('pne',2), ('breast',3)] # order2

        self.use_path = True
        self.img_size = img_size if img_size != None else 224 # original img size various
        self.train_trsf = [
            transforms.RandomHorizontalFlip(p=0.5),
            ]
        
        self.test_trsf = []
        self.common_trsf = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]

        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]
        self.class_order = list(range(sum(self._dataset_inc)))
    
    def shuffle_order(self, seed):
        random.seed(seed)
        random.shuffle(self._dataset_info)
        self._dataset_inc = [data_flag[1] for data_flag in self._dataset_info]

    def getdata(self, src_dir, mode):
        assert mode == 'train' or mode == 'test' or mode== 'val', 'Unkown mode: {}'.format(mode)
        known_class = 0
        data, targets = [], []
        for sub_dataset_name, class_num in self._dataset_info:
            sub_dataset_dir = os.path.join(src_dir, sub_dataset_name, mode)
            
            # sorted function is important to get the same class order on different devices!
            for class_id, class_name in enumerate(sorted(os.listdir(sub_dataset_dir))):
                class_dir = os.path.join(sub_dataset_dir, class_name)

                # sorted function is important to get the same class order on different devices!
                for img_name in sorted(os.listdir(class_dir)):
                    data.append(os.path.join(class_dir, img_name))
                    targets.append(known_class+class_id)
            known_class += class_num
        
        return np.array(data), np.array(targets, dtype=int)

    def download_data(self):
        src_dir = os.path.join(os.environ["DATA"], "medmnist_HR")

        self.train_data, self.train_targets = self.getdata(src_dir, 'train')
        self.valid_data, self.valid_targets = self.getdata(src_dir, 'val')
        self.test_data, self.test_targets = self.getdata(src_dir, 'test')