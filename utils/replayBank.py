import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch
from torchvision import transforms
from utils.toolkit import DummyDataset, pil_loader
from PIL import Image
from torch.cuda.amp import autocast

EPSILON = 1e-8

def cat_with_broadcast(data_list:list):
    max_dim = 0
    max_len = 0
    for item in data_list:
        max_len += item.shape[0]
        if max_dim < item.shape[1]:
            max_dim = item.shape[1]
    
    result = np.zeros((max_len, max_dim))
    idx = 0
    for item in data_list:
        result[idx:idx+item.shape[0],:item.shape[1]] = item
        idx += item.shape[0]
    
    return result

class ReplayBank:

    def __init__(self, config, logger):
        self._logger = logger
        self._apply_nme = config.apply_nme
        self._batch_size = config.batch_size
        self._num_workers = config.num_workers
        self._total_class_num = config.total_class_num
        self._increment_steps = config.increment_steps

        self._memory_size = config.memory_size
        # 有两种样本存储形式, 但都固定存储空间。一种是固定每一类数据存储样本的数量(为True时)
        # 另一种在固定存储空间中，平均分配每一类允许存储的样本数量
        self._fixed_memory = config.fixed_memory
        self._sampling_method = config.sampling_method # 采样的方式
        if self._fixed_memory:
            if config.memory_per_class is not None:
                self._memory_per_class = config.memory_per_class # 预期每类保存的样本数量
            elif self._memory_size is not None:
                self._memory_per_class = self._memory_size // config.total_class_num
            else:
                raise ValueError('Value error in setting memory per class!')

        self._data_memory = np.array([])
        self._targets_memory = np.array([])
        self._soft_targets_memory = np.array([])
        self._class_sampler_info = [] # 列表中保存了每个类实际保存的样本数

        self._class_means = []
        self._num_seen_examples = 0
    
    @property
    def sample_per_class(self):
        return self._memory_per_class

    def is_empty(self):
        return len(self._data_memory) == 0

    def get_class_means(self):
        return self._class_means
    
    def set_class_means(self, class_means):
        self._class_means = class_means

    def store_samples(self, dataset:DummyDataset, model):
        """dataset 's transform should be in test mode!"""
        class_range = np.unique(dataset.targets)
        assert min(class_range)+1 > len(self._class_sampler_info), "Store_samples's dataset should not overlap with buffer"
        if self._fixed_memory:
            per_class = self._memory_per_class
        else:
            self._memory_per_class = per_class = self._memory_size // (len(self._class_sampler_info) + len(class_range))
            if len(self._class_sampler_info) > 0:
                self.reduce_memory(per_class)

        # to reduce calculation when applying replayBank (expecially for some methods do not apply nme)
        if self._apply_nme:
            class_means = []
            memory_dataset = DummyDataset(self._data_memory, self._targets_memory, dataset.transform, dataset.use_path)
            stored_data_means = self.cal_class_means(model, memory_dataset)
            if stored_data_means is not None:
                class_means.append(stored_data_means)

        data_mamory, targets_memory = [], []
        if len(self._class_sampler_info) > 0:
            data_mamory.append(self._data_memory)
            targets_memory.append(self._targets_memory)

        self._logger.info('Constructing exemplars for the sequence of {} new classes...'.format(len(class_range)))
        for class_idx in class_range:
            class_data_idx = np.where(dataset.targets == class_idx)[0]
            idx_data, idx_targets = dataset.data[class_data_idx], dataset.targets[class_data_idx]
            idx_dataset = Subset(dataset, class_data_idx)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            idx_vectors, idx_logits = self._extract_vectors(model, idx_loader)
            selected_idx = self.select_sample_indices(self._sampling_method, idx_vectors, idx_logits, class_idx, per_class)
            self._logger.info("New Class {} instance will be stored: {} => {}".format(class_idx, len(idx_targets), len(selected_idx)))
            
            # to reduce calculation when applying replayBank (expecially for some methods do not apply nme)
            if self._apply_nme:
                # 计算类中心
                idx_vectors = F.normalize(idx_vectors[selected_idx], dim=1)# 对特征向量做归一化
                mean = torch.mean(idx_vectors, dim=0)
                mean = F.normalize(mean, dim=0)
                class_means.append(mean.unsqueeze(0))
                self._logger.info('calculated class mean of class {}'.format(class_idx))

            data_mamory.append(idx_data[selected_idx])
            targets_memory.append(idx_targets[selected_idx])
            
            self._class_sampler_info.append(len(selected_idx))
        
        self._logger.info('Replay Bank stored {} classes, {} samples ({} samples for each class)'.format(
                len(self._class_sampler_info), sum(self._class_sampler_info), per_class))
        
        if self._apply_nme:
            self._class_means = torch.cat(class_means, dim=0)
        
        self._data_memory = np.concatenate(data_mamory)
        self._targets_memory = np.concatenate(targets_memory)
    
    def cal_class_means(self, model, dataset:DummyDataset):
        class_means = []
        self._logger.info('Re-calculating class means for stored classes...')
        # for class_idx, class_samples in enumerate(self._data_memory):
        for class_idx in np.unique(dataset.targets):
            mask = np.where(dataset.targets == class_idx)[0]
            idx_dataset = DummyDataset(dataset.data[mask], dataset.targets[mask], dataset.transform, dataset.use_path)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

            idx_vectors, _ = self._extract_vectors(model, idx_loader)
            idx_vectors = F.normalize(idx_vectors, dim=1)# 对特征向量做归一化
            mean = torch.mean(idx_vectors, dim=0)
            mean = F.normalize(mean, dim=0)
            class_means.append(mean)
            self._logger.info('calculated class mean of class {}'.format(class_idx))
        return torch.stack(class_means, dim=0) if len(class_means) > 0 else None

    def reduce_memory(self, m):
        data_mamory, targets_memory, soft_targets_memory = [], [], []
        for i in range(len(self._class_sampler_info)):
            if self._class_sampler_info[i] > m:
                store_sample_size = m
            else:
                self._logger.info('The whole class samples are less than the allocated memory size!')
                store_sample_size = self._class_sampler_info[i]
            
            self._logger.info("Old class {} storage will be reduced: {} => {}".format(i, self._class_sampler_info[i], store_sample_size))
            
            mask = np.where(self._targets_memory == i)[0]
            data_mamory.append(self._data_memory[mask[:store_sample_size]])
            targets_memory.append(self._targets_memory[mask[:store_sample_size]])
            if len(self._soft_targets_memory) > 0:
                soft_targets_memory.append(self._soft_targets_memory[mask[:store_sample_size]])
            
            self._class_sampler_info[i] = store_sample_size
            # self._logger.info("类别 {} 存储样本数为: {}".format(i, len(self._data_memory[i])))
        self._data_memory = np.concatenate(data_mamory)
        self._targets_memory = np.concatenate(targets_memory)
        if len(soft_targets_memory) > 0:
            self._soft_targets_memory = np.concatenate(soft_targets_memory)

    def KNN_classify(self, task_begin, task_end, vectors=None, model=None, loader=None, ret_logits=False):
        assert self._apply_nme, 'if apply_nme=False, you should not apply KNN_classify!'
        if model != None and loader != None:
            vectors, _ = self._extract_vectors(model, loader)
        
        vectors = F.normalize(vectors, dim=1)# 对特征向量做归一化

        dists = torch.cdist(vectors.float(), self._class_means[task_begin:task_end].float(), p=2)

        min_scores, nme_predicts = torch.min(dists, dim=1)
        nme_predicts += task_begin
        if ret_logits:
            return nme_predicts, dists
        else:
            return nme_predicts, 1-min_scores
    
    def KNN_classify_split(self, task_id_predicts, vectors, cur_task_id):
        assert self._apply_nme, 'if apply_nme=False, you should not apply KNN_classify!'

        nme_predicts = torch.zeros(vectors.shape[0], dtype=torch.long).cuda()
        nme_min_scores = torch.zeros(vectors.shape[0], dtype=torch.float).cuda()
        known_class_num, total_class_num = 0, 0
        task_feature_begin, task_feature_end = 0, 0
        feature_dim = vectors.shape[1] // (cur_task_id+1)
        for id, cur_class_num in enumerate(self._increment_steps[:cur_task_id+1]):
            total_class_num += cur_class_num
            task_feature_end += feature_dim

            task_data_idxs = torch.argwhere(task_id_predicts == id).squeeze(-1)
            if len(task_data_idxs) > 0:
                task_features = vectors[task_data_idxs][:, task_feature_begin:task_feature_end]
                task_features = F.normalize(task_features, dim=1)

                dists = torch.cdist(task_features, self._class_means[known_class_num:total_class_num], p=2)
                min_scores, nme_pred = torch.min(dists, dim=1)
                nme_pred += known_class_num

                nme_predicts[task_data_idxs] = nme_pred
                nme_min_scores[task_data_idxs] = min_scores

            task_feature_begin = task_feature_end
            known_class_num = total_class_num

        return nme_predicts, 1-min_scores

    def get_memory(self, indices=None):
        replay_data, replay_targets = [], []
        if sum(self._class_sampler_info) <= 0:
            self._logger.info('Replay nothing or Nothing have been stored')
            return None
        elif indices is None: # default replay all stored data
            indices = range(len(self._class_sampler_info))

        for idx in indices:
            mask = np.where(self._targets_memory == idx)[0]
            replay_data.append(self._data_memory[mask])
            replay_targets.append(self._targets_memory[mask])
        
        return np.concatenate(replay_data), np.concatenate(replay_targets)
    
    def get_unified_sample_dataset(self, new_task_dataset:DummyDataset, model):
        """dataset 's transform should be in train mode!"""
        balanced_data = []
        balanced_targets = []
        class_range = np.unique(new_task_dataset.targets)
        if len(self._data_memory) > 0:
            per_class = self._memory_per_class
            balanced_data.append(self._data_memory)
            balanced_targets.append(self._targets_memory)
        else:
            per_class = self._memory_size // len(class_range)
        self._logger.info('Getting unified samples from old and new classes, {} samples for each class (replay {} old classes)'.format(per_class, len(self._class_sampler_info)))

        # balanced new task data and targets
        for class_idx in class_range:
            if class_idx < len(self._class_sampler_info):
                continue
            class_data_idx = np.where(np.logical_and(new_task_dataset.targets >=class_idx, new_task_dataset.targets < class_idx + 1))[0]
            idx_data, idx_targets = new_task_dataset.data[class_data_idx], new_task_dataset.targets[class_data_idx]
            idx_dataset = Subset(new_task_dataset, class_data_idx)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            idx_vectors, idx_logits = self._extract_vectors(model, idx_loader)
            selected_idx = self.select_sample_indices(self._sampling_method, idx_vectors, idx_logits, class_idx, per_class)
            self._logger.info("New Class {} instance will be down-sample: {} => {}".format(class_idx, len(idx_targets), len(selected_idx)))

            balanced_data.append(idx_data[selected_idx])
            balanced_targets.append(idx_targets[selected_idx])
        
        balanced_data, balanced_targets = np.concatenate(balanced_data), np.concatenate(balanced_targets)
        return DummyDataset(balanced_data, balanced_targets, new_task_dataset.transform, new_task_dataset.use_path)
    
    ########### Reservoir memory (used in DarkER, X-DER, ...) begin ###########
    def store_samples_reservoir(self, examples, logits, labels):
        """ This function is for DarkER and DarkER++ """
        init_size = 0
        if len(self._data_memory) == 0:
            init_size = min(len(examples), self._memory_size)
            self._data_memory = examples[:init_size]
            self._targets_memory = labels[:init_size]
            self._soft_targets_memory = logits[:init_size]
            self._num_seen_examples += init_size
        elif len(self._data_memory) < self._memory_size:
            init_size = min(len(examples), self._memory_size - len(self._data_memory))
            self._data_memory = np.concatenate([self._data_memory, examples[:init_size]])
            self._targets_memory = np.concatenate([self._targets_memory, labels[:init_size]])
            self._soft_targets_memory = np.concatenate([self._soft_targets_memory, logits[:init_size]])
            self._num_seen_examples += init_size

        for i in range(init_size, len(examples)):
            index = np.random.randint(0, self._num_seen_examples + 1)
            self._num_seen_examples += 1
            if index < self._memory_size:
                self._data_memory[index] = examples[i]
                self._targets_memory[index] = labels[i]
                self._soft_targets_memory[index] = logits[i]

    def get_memory_reservoir(self, size, use_path, transform=None, ret_idx=False):
        if size > min(self._num_seen_examples, self._memory_size):
            size = min(self._num_seen_examples, self._memory_size)

        choice = np.random.choice(min(self._num_seen_examples, self._memory_size), size=size, replace=False)
        
        data_all = []
        for sample in self._data_memory[choice]:
            if transform is None:
                data_all.append(torch.from_numpy(sample)) # [h, w, c]
            elif use_path:
                data_all.append(transform(pil_loader(sample))) # [c, h, w]
            else:
                data_all.append(transform(Image.fromarray(sample))) # [c, h, w]
        data_all = torch.stack(data_all)
        
        targets_all = torch.from_numpy(self._targets_memory[choice])
        soft_targets_all = torch.from_numpy(self._soft_targets_memory[choice])

        ret = (data_all, targets_all, soft_targets_all)
        if ret_idx:
            ret = (torch.tensor(choice),) + ret

        return ret

    def update_memory_reservoir(self, new_logits, new_idx, task_begin, gamma):
        # future logits for current task
        transplant = new_logits[:, task_begin:]

        gt_values = self._soft_targets_memory[new_idx, self._targets_memory[new_idx]]
        max_values = transplant.max(1)
        coeff = gamma * gt_values / max_values
        coeff = np.repeat(np.expand_dims(coeff, 1), new_logits.shape[1]-task_begin, 1)
        mask = np.repeat(np.expand_dims(max_values > gt_values, 1), new_logits.shape[1]-task_begin, 1)
        transplant[mask] *= coeff[mask]

        self._soft_targets_memory[new_idx][:, task_begin:] = transplant
    
    def store_samples_reservoir_v2(self, dataset:DummyDataset, model, gamma):
        """dataset 's transform should be in train mode!"""
        # Reduce buffer
        class_range = np.unique(dataset.targets)
        assert min(class_range)+1 > len(self._class_sampler_info), "Store_samples's dataset should not overlap with buffer"
        self._memory_per_class = per_class = self._memory_size // (len(self._class_sampler_info) + len(class_range))
        if len(self._data_memory) > 0:
            self.reduce_memory(per_class)
        
        is_first_task = len(self._class_sampler_info) == 0
        task_begin = min(class_range)

        if not is_first_task:
            # update future past
            idx_dataset = DummyDataset(self._data_memory, self._targets_memory, dataset.transform, dataset.use_path)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            _, idx_logits = self._extract_vectors(model, idx_loader)
            idx_logits = idx_logits.cpu().numpy()
            self.update_memory_reservoir(idx_logits, np.arange(len(self._targets_memory)), task_begin, gamma)

        data_mamory, targets_memory, soft_targets_memory = [], [], []
        if len(self._class_sampler_info) > 0:
            data_mamory.append(self._data_memory)
            targets_memory.append(self._targets_memory)
            soft_targets_memory.append(self._soft_targets_memory)

        self._logger.info('Constructing exemplars for the sequence of {} new classes...'.format(len(class_range)))
        new_task_memory_size = self._memory_size - len(self._class_sampler_info)*per_class
        new_task_per_class = new_task_memory_size // len(class_range)
        addition_num = np.zeros(len(class_range), dtype=int)
        remainder_num = new_task_memory_size % len(class_range)
        if remainder_num > 0:
            addition_num[np.random.permutation(len(class_range))][:remainder_num] += 1
        for class_idx in class_range:
            class_data_idx = np.where(dataset.targets == class_idx)[0]
            idx_data, idx_targets = dataset.data[class_data_idx], dataset.targets[class_data_idx]
            idx_dataset = Subset(dataset, class_data_idx)
            idx_loader = DataLoader(idx_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
            _, idx_logits = self._extract_vectors(model, idx_loader)
            idx_logits = idx_logits.cpu().numpy()

            # future past scaling
            if not is_first_task:
                transplant = idx_logits[:, :task_begin]
                
                gt_values = idx_logits[np.arange(len(idx_logits)), idx_targets]
                max_values = transplant.max(1)
                coeff = gamma * gt_values / max_values
                coeff = np.repeat(np.expand_dims(coeff, 1), task_begin, 1)
                mask = np.repeat(np.expand_dims(max_values > gt_values, 1), task_begin, 1)
                transplant[mask] *= coeff[mask]
                idx_logits[:, :task_begin] = transplant
            
            idx_per_class = new_task_per_class + addition_num[class_idx-task_begin]
            data_mamory.append(idx_data[:idx_per_class])
            targets_memory.append(idx_targets[:idx_per_class])
            soft_targets_memory.append(idx_logits[:idx_per_class])

            self._class_sampler_info.append(idx_per_class)
            self._logger.info("New Class {} instance will be stored: {} => {}".format(class_idx, len(idx_targets), idx_per_class))
        
        self._logger.info('Replay Bank stored {} classes, {} samples (more than {} samples per class)'.format(
                len(self._class_sampler_info), sum(self._class_sampler_info), per_class))

        self._data_memory = np.concatenate(data_mamory)
        self._targets_memory = np.concatenate(targets_memory)
        self._soft_targets_memory = np.concatenate(soft_targets_memory)
        self._num_seen_examples += len(dataset)

    def reset_update_counter(self):
        self._update_counter = np.zeros(self._memory_size)
    
    ########### Reservoir memory (used in DarkER, X-DER, ...) end ###########

    ##################### Sampler Methods #####################
    def select_sample_indices(self, sampling_method, vectors, logits, class_id, m):
        if sampling_method == 'herding':
            selected_idx = self.herding_select(vectors, m)
        elif sampling_method == 'random':
            selected_idx = self.random_select(vectors, m)
        elif sampling_method == 'closest_to_mean':
            selected_idx = self.closest_to_mean_select(vectors, m)
        else:
            raise ValueError('Unknown sample select strategy: {}'.format(sampling_method))
        return selected_idx

    def random_select(self, vectors, m):
        idxes = np.arange(vectors.shape[0])
        np.random.shuffle(idxes)# 防止类别数过少的情况
        
        # 防止类别数过少的情况
        if vectors.shape[0] > m:
            store_sample_size = m
        else:
            self._logger.info('The whole class samples are less than the allocated memory size!')
            store_sample_size = vectors.shape[0]
        return idxes[:store_sample_size]
    
    def closest_to_mean_select(self, vectors, m):
        normalized_vector = F.normalize(vectors, dim=1) # 对特征向量做归一化
        class_mean = torch.mean(normalized_vector, dim=0)
        class_mean = F.normalize(class_mean, dim=0).unsqueeze(0)
        distences = torch.cdist(normalized_vector, class_mean).squeeze()

        # 防止类别数过少的情况
        if vectors.shape[0] > m:
            store_sample_size = m
        else:
            self._logger.info('The whole class samples are less than the allocated memory size!')
            store_sample_size = vectors.shape[0]
        return torch.argsort(distences)[:store_sample_size].cpu()

    def herding_select(self, vectors, m):
        selected_idx = []
        all_idxs = list(range(vectors.shape[0]))
        nomalized_vector = F.normalize(vectors, dim=1) # 对特征向量做归一化
        class_mean = torch.mean(nomalized_vector, dim=0)
        # class_mean = F.normalize(class_mean, dim=0)

        # 防止类别数过少的情况
        if vectors.shape[0] > m:
            store_sample_size = m
        else:
            self._logger.info('The whole class samples are less than the allocated memory size!')
            store_sample_size = vectors.shape[0]
            
        for k in range(1, store_sample_size+1):
            sub_vectors = nomalized_vector[all_idxs]
            S = torch.sum(nomalized_vector[selected_idx], dim=0)
            mu_p = (sub_vectors + S) / k
            i = torch.argmin(torch.norm(class_mean-mu_p, p=2, dim=1))
            selected_idx.append(all_idxs.pop(i))
        return selected_idx
    ###########################################################

    def _extract_vectors(self, model, loader, ret_data=False, ret_add_info=False):
        model.eval()
        vectors = []
        logits = []
        data = []
        addition_info = []
        with torch.no_grad():
            for _add_info, _inputs, _targets in loader:
                with autocast():
                    out, output_features = model(_inputs.cuda())
                vectors.append(output_features['features'])
                logits.append(out)
                if ret_data:
                    data.append(_inputs)
                if ret_add_info:
                    addition_info.append(_add_info)
        
        ret = (torch.cat(vectors), torch.cat(logits))
        if ret_data:
            ret = ret + (torch.cat(data),)
        if ret_add_info:
            ret = ret + (torch.cat(addition_info),)
        return ret
    