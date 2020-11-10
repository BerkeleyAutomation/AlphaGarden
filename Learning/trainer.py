from collections import OrderedDict
from datetime import datetime
import logging
from math import ceil
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import SubsetRandomSampler
from logger import Logger
from constants import TrainingConstants


class Trainer(object):
    """One-time-use utility for training a network.
    """
    def __init__(self,
                 net,
                 dataset,
                 num_epochs=TrainingConstants.NUM_EPOCHS,
                 total_size=TrainingConstants.TOTAL_SIZE,
                 val_size=TrainingConstants.VAL_SIZE,
                 bsz=TrainingConstants.BSZ,
                 base_lr=TrainingConstants.BASE_LR,
                 lr_step_size=TrainingConstants.LR_STEP_SIZE,
                 lr_decay_rate=TrainingConstants.LR_DECAY_RATE,
                 log_interval=TrainingConstants.LOG_INTERVAL,
                 device=TrainingConstants.DEVICE,
                 output_dir=TrainingConstants.OUTPUT_DIR):
        self._net = net
        self._dataset = dataset
        self._num_epochs = num_epochs
        self._total_size = total_size
        self._val_size = val_size
        self._bsz = bsz
        self._base_lr = base_lr
        self._lr_step_size = lr_step_size
        self._lr_decay_rate = lr_decay_rate
        self._device = device
        self._log_interval = log_interval
        self._output_dir = output_dir

        self._native_logger = logging.getLogger(self.__class__.__name__)

    def _setup(self):
        date_time = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
        self._output_dir = os.path.join(
            self._output_dir,
            '{}_{}'.format(self._net.name, str(date_time))
        )
        os.makedirs(self._output_dir)

        self._logger = Logger(
            os.path.join(self._output_dir, TrainingConstants.LOG_DIR)
        )

        ind = np.arange(len(self._dataset))
        np.random.shuffle(ind)
        ind = ind[:ceil(self._total_size*len(ind))]
        train_ind = ind[:ceil((1-self._val_size)*len(ind))]
        val_ind = ind[ceil((1-self._val_size)*len(ind)):]

        train_sampler = SubsetRandomSampler(train_ind)
        val_sampler = SubsetRandomSampler(val_ind)
        self._train_data_loader = torch.utils.data.DataLoader(
                                    self._dataset,
                                    batch_size=self._bsz,
                                    num_workers=1,
                                    pin_memory=True,
                                    sampler=train_sampler
                                 )
        self._val_data_loader = torch.utils.data.DataLoader(
                                    self._dataset,
                                    batch_size=self._bsz,
                                    num_workers=1,
                                    pin_memory=True,
                                    sampler=val_sampler
                               )

        self._device = torch.device(self._device)
        # self._net = torch.nn.DataParallel(self._net, device_ids=[0, 1, 2, 3])
        self._net = self._net.to(self._device)

        self._optimizer = optim.Adadelta(self._net.parameters(),
                                         lr=self._base_lr)
        self._scheduler = StepLR(self._optimizer,
                                step_size=self._lr_step_size,
                                gamma=self._lr_decay_rate)

    def _log_metric(self, epoch, metric_name, data):
        self._native_logger.info('Logging {} ...'.format(metric_name))

        if not isinstance(data, (list, np.ndarray)):
            data = [data]
        data = np.asarray(data)

        logs = OrderedDict()
        logs['{}_average'.format(metric_name)] = np.mean(data)
        logs['{}_stddev'.format(metric_name)] = np.std(data)
        logs['{}_max'.format(metric_name)] = np.max(data)
        logs['{}_min'.format(metric_name)] = np.min(data)

        # Write TensorFlow summaries.
        for key, value in logs.items():
            self._native_logger.info('\t{} : {}'.format(key, value))
            self._logger.log_scalar(value, key, epoch)
        self._logger.flush()


    def _train(self, epoch):
        self._net.train()

        num_batches = len(self._train_data_loader)
        train_losses = []
        for batch_idx, (cc_sector, water_plants_health, global_cc, target) in enumerate(self._train_data_loader):
            cc_sector = cc_sector.to(self._device)
            water_plants_health = water_plants_health.to(self._device)
            global_cc = global_cc.to(self._device)
            target = target.to(self._device)
            self._optimizer.zero_grad()
            
            ''' Classification '''
            # output = self._net((cc_sector, water_plants_health, global_cc))
            # criterion = torch.nn.CrossEntropyLoss()
            
            ''' Regression '''
            output = self._net((cc_sector, water_plants_health, global_cc)).double()
            criterion = torch.nn.MSELoss()
            target = target.unsqueeze(1).double()
            # print(target, output)

            loss = criterion(output, target)
            loss.backward()
            self._optimizer.step()
            if batch_idx % self._log_interval == 0:
                self._native_logger.info(
                    'Train Epoch: {} [Batch {}/{} ({:.0f}%)]\tLoss: {:.6f}\t'
                    'LR: {:.6f}'.format(
                        epoch,
                        batch_idx+1,
                        num_batches,
                        100 * (batch_idx+1) / num_batches,
                        loss.item(),
                        self._optimizer.param_groups[0]['lr']
                    )
                )
                train_losses.append(loss.item())

        self._log_metric(epoch, 'train/epoch_ce_loss', train_losses)

    def _eval(self, epoch):
        self._net.eval()

        eval_loss = 0
        eval_losses = []
        with torch.no_grad():
            i = 0
            for batch_idx, (cc_sector, water_plants_health, global_cc, target) in enumerate(self._val_data_loader):
                i += 1
                cc_sector = cc_sector.to(self._device)
                water_plants_health = water_plants_health.to(self._device)
                global_cc = global_cc.to(self._device)
                target = target.to(self._device) 
                output = self._net((cc_sector, water_plants_health, global_cc)).double()

                ''' Classification '''
                # criterion = torch.nn.CrossEntropyLoss()

                ''' Regression '''
                criterion = torch.nn.MSELoss()
                target = target.unsqueeze(1).double()
            
                loss = criterion(output, target)
                eval_loss += loss.item()
                eval_losses.append(loss.item())

        num_batches = len(self._val_data_loader)
        eval_loss /= num_batches
        self._log_metric(epoch, 'eval/epoch_ce_loss', eval_losses)

    def train(self):
        self._setup()
        for epoch in range(1, self._num_epochs+1):
            self._train(epoch)
            self._eval(epoch)
            self._scheduler.step()
            self._native_logger.info('')
            if epoch % 5 == 0:
                self._net.save(self._output_dir, TrainingConstants.NET_SAVE_FNAME, str(epoch) + '_')
        self._net.save(self._output_dir, TrainingConstants.NET_SAVE_FNAME, 'final_')
