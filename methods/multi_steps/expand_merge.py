
import numpy as np
import torch
from argparse import ArgumentParser
from torch.cuda.amp import autocast
from torch.nn.functional import cross_entropy, binary_cross_entropy, binary_cross_entropy_with_logits
from torch.utils.data import DataLoader
from torch import optim

from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import tensor2numpy, count_parameters, tensor2numpy, target2onehot
from utils.replayBank_clip import ReplayBank_CLIP

EPSILON = 1e-8

def add_special_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--context', type=str, default=None, help="context for continual clip testing. e.g. A photo of a {}.")
    parser.add_argument('--layer_names', nargs='+', type=str, default=None, help='layers to apply adapter, e.t. [layer1, layer2]')
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--T', type=float, default=None, help='tempreture apply to the output logits befor softmax')
    return parser

class Expand_merge(Finetune_IL):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._mode = self._config.mode.split('|') if self._config.mode is not None else []
        self._context = config.context
        self._hidden_dim = config.hidden_dim
        self._T = config.T if config.T is not None else 0.01
        self._img_backbone, self._text_backbone = config.backbone.split('-')

        self._memory_bank = ReplayBank_CLIP(self._config, logger)

        self._class_to_idx = None
        self._layer_names = []
        for i in range(12):
            for layer_name in config.layer_names:
                self._layer_names.append(layer_name.format(i))
    
    def prepare_task_data(self, data_manager):
        self._cur_task += 1
        self._cur_classes = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + self._cur_classes

        self._train_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='train')
        
        self._test_dataset = data_manager.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')

        if self._cur_task > 0:
            self._replay_datset = self._memory_bank.get_replay_dataset(self._batch_size, len(self._train_dataset),
                                                                self._train_dataset.transform,
                                                                self._train_dataset.use_path)
            self._replay_dataloader = DataLoader(self._replay_datset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._openset_test_dataset = data_manager.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._logger.info('Train dataset size: {}'.format(len(self._train_dataset)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))

        self._train_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._test_loader = DataLoader(self._test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._sampler_dataset = data_manager.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                    source='train', mode='test')
        
        if self._class_to_idx is None:
            self._class_to_idx = data_manager.class_to_idx
            self._idx_to_class = dict((value, key) for key, value in self._class_to_idx.items())

    def prepare_model(self, checkpoint=None):
        from backbone.expand_merge_net import ExpandMergeNet

        if self._network == None:
            self._network = ExpandMergeNet(self._logger, self._img_backbone, self._text_backbone, self._context,
                                      self._layer_names, self._hidden_dim, self._mode)
            self._network.freeze_FE()

        new_class_names = [self._idx_to_class[class_id] for class_id in range(self._known_classes, self._total_classes)]
        new_class_names = [name.replace("_", " ") for name in new_class_names]
        self._network.update_new_class_name(new_class_names)

        if checkpoint is not None:
            self._network.load_state_dict(checkpoint['state_dict'])
            self._logger.info("Loaded checkpoint model's state_dict !")
        
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                self._logger.info('{} requre grad!'.format(name))

        self._logger.info('All params: {}'.format(count_parameters(self._network)))
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        self._network = self._network.cuda()
    
    def _epoch_train(self, model, train_loader, optimizer, scheduler, task_begin=None, task_end=None, task_id=None):
        losses = 0.
        ce_losses, replay_losses = 0., 0.
        correct, total = 0, 0

        if task_id > 0:
            replay_iter = iter(self._replay_dataloader)

        model.new_adapters_train()
        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            with autocast():
                logits, feature_outputs = model(inputs, self._T)

                # ce loss version implementation
                ce_loss = cross_entropy(logits, targets)

            ce_losses += ce_loss.item()
            loss = ce_loss
            
            if task_id > 0:
                _, replay_inputs, replay_targets = replay_iter.next()
                replay_inputs, replay_targets = replay_inputs.cuda(), replay_targets.cuda()

                with autocast():
                    replay_logits, replay_feature_outputs = model(replay_inputs, self._T)

                    replay_loss = cross_entropy(replay_logits, replay_targets)
                    replay_losses += replay_loss.item()
                    loss += replay_loss

            preds = torch.max(logits, dim=1)[1]
    
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            losses += loss.item()

            correct += preds.eq(targets).cpu().sum()
            total += len(targets)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        train_loss = ['Loss', losses/len(train_loader), 'Loss_ce', ce_losses/len(train_loader), 'Loss_replay', replay_losses/len(train_loader)]
        return model, train_acc, train_loss
