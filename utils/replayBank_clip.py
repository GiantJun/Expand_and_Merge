import numpy as np

from utils.replayBank import ReplayBank
from utils.toolkit import DummyDataset
from torchvision import transforms

EPSILON = 1e-8

class ReplayBank_CLIP(ReplayBank):

    def get_replay_dataset(self, batch_size:int, dataset_size:int, transform, use_path:bool):
        sample_idx_list = []
        for i in range(dataset_size // batch_size):
            sample_idx_list.append(np.random.choice(len(self._targets_memory), size=batch_size, replace=False))
        
        if dataset_size % batch_size != 0:
            sample_idx_list.append(np.random.choice(len(self._targets_memory), size=dataset_size%batch_size, replace=False))

        sample_idx_list = np.concatenate(sample_idx_list)

        replay_data, replay_targets = self._data_memory[sample_idx_list], self._targets_memory[sample_idx_list]

        return DummyDataset(replay_data, replay_targets, transform, use_path)
    
    def get_unified_replay_dataset(self, new_task_dataset: DummyDataset, train_dataset_size:int):
        balanced_data = []
        balanced_targets = []
        
        per_class = self._memory_per_class
        balanced_data.append(self._data_memory)
        balanced_targets.append(self._targets_memory)

        new_task_sampler_idx = np.random.choice(len(new_task_dataset), size=per_class, replace=False)
        balanced_data.append(new_task_dataset.data[new_task_sampler_idx])
        balanced_targets.append(new_task_dataset.targets[new_task_sampler_idx])

        balanced_data, balanced_targets = np.concatenate(balanced_data), np.concatenate(balanced_targets)

        repeated_data = []
        repeated_targets = []
        for i in range(train_dataset_size//len(balanced_targets)):
            repeated_data.append(balanced_data)
            repeated_targets.append(balanced_targets)

        repeated_data, repeated_targets = np.concatenate(repeated_data), np.concatenate(repeated_targets)
        
        return DummyDataset(repeated_data, repeated_targets, new_task_dataset.transform, new_task_dataset.use_path)
    