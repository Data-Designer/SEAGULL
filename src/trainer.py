#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 16:00
# @Author  : Anonymous
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm

# #Desc: trainer
import os
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from itertools import cycle
from logging import getLogger
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils import *
from time import time

from recbole.trainer import Trainer
from recbole.config import Config


class SeagullTrainer():
    def __init__(self, config, model):
        super(SeagullTrainer, self).__init__()
        self.config = config
        self.model = model
        self.num_m_step = config['num_m_step']
        # assert self.num_m_step is not None
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric']
        self.test_batch_size = config['eval_batch']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        self.checkpoint_dir = '/dfs/data/ORec/' + 'log/' + config['checkpoint_dir'] + '/log' + config['num_log'] + '/checkpoint/'
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0

        self.train_loss_dict_src = dict()
        self.train_loss_dict_tgt = dict() # 
        
        self.train_loss_dict_src_batch = []
        self.train_loss_dict_tgt_batch = [] # save loss
        
        
        
        self.optimizer = self._build_optimizer(self.model.parameters())
        self._best_init()
        
        self.max_item_size_A = config['max_item_size_A']
        self.max_item_size_B = config['max_item_size_B']

    def _best_init(self):
        """init best metric"""
        self.best_scores = {}
        for metric in self.valid_metric:
            self.best_scores["src_"+metric] = -np.inf
            self.best_scores["src_epoch_"+metric] = -1
            self.best_scores["tgt_"+metric] = -np.inf
            self.best_scores["tgt_epoch_"+metric] = -1



    def _check_nan(self, loss):
        """loss==nan"""
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        """str train loss input"""
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        """tensorboard ->> loss visual"""
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _build_optimizer(self, params):
        r"""Init the Optimizer        """
        if self.config['reg_weight'] and self.weight_decay and self.weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=self.learning_rate)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information."""
        state = {
            'config': self.config,
            'epoch': epoch,
            'best_scores': self.best_scores,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=None, callback_fn=None):
        r"""Train the model based on the train data and the valid data."""
        if saved and self.start_epoch >= self.epochs: 
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            if epoch_idx % self.num_m_step == 0:
                self.logger.info("Running E-step ! ")
                self.model.e_step()
            # train
            training_start_time = time()
            train_loss_src, train_loss_tgt = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict_src[epoch_idx] = sum(train_loss_src) if isinstance(train_loss_src, tuple) else train_loss_src
            self.train_loss_dict_tgt[epoch_idx] = sum(train_loss_tgt) if isinstance(train_loss_tgt, tuple) else train_loss_tgt
            training_end_time = time()
            train_loss_output_src = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss_src)
            train_loss_output_tgt = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss_tgt)
            if verbose:
                self.logger.info(train_loss_output_src)
                self.logger.info(train_loss_output_tgt)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss_src, tag='Loss/Train-src')
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss_tgt, tag='Loss/Train-tgt')
            self.model.sp_loss.increase_threshold()  # progressive
            

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_result_src, valid_result_tgt = self._valid_epoch(valid_data,show_progress=show_progress)

                update_flag = False
                for metric in self.valid_metric:
                    if valid_result_src[metric] > self.best_scores['src_'+metric]:
                        self.best_scores['src_'+metric] = valid_result_src[metric]
                        self.best_scores['src_epoch_'+ metric] = epoch_idx
                        update_flag = True
                    if valid_result_tgt[metric] > self.best_scores['tgt_' + metric]:
                        self.best_scores['tgt_' + metric] = valid_result_tgt[metric]
                        self.best_scores['tgt_epoch_' + metric] = epoch_idx
                        update_flag = True


                valid_end_time = time()
                valid_score_output_best = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("best_scores", 'blue') + ": %s]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, dict2str(self.best_scores))
                valid_result_output_src = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result_src)
                valid_result_output_tgt = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result_tgt)

                if verbose:
                    self.logger.info(valid_score_output_best)
                    self.logger.info(valid_result_output_src)
                    self.logger.info(valid_result_output_tgt)
                # self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
        
        # 存个epochcsv
#         df_src = pd.DataFrame.from_dict(self.train_loss_dict_src, orient='index',columns=['loss'])
#         df_tgt = pd.DataFrame.from_dict(self.train_loss_dict_tgt, orient='index',columns=['loss'])
#         df_src = df_src.reset_index().rename(columns = {'index':'epoch'})
#         df_tgt = df_tgt.reset_index().rename(columns = {'index':'epoch'}) 
#         df_src.to_csv(os.path.join(self.checkpoint_dir,'loss_src.csv'))
#         df_tgt.to_csv(os.path.join(self.checkpoint_dir,'loss_tgt.csv'))
        
        # batch
#         df_src = pd.DataFrame(self.train_loss_dict_src_batch)
#         df_tgt = pd.DataFrame(self.train_loss_dict_tgt_batch)
#         df_src.to_csv(os.path.join(self.checkpoint_dir,'loss_src_batch.csv'))
#         df_tgt.to_csv(os.path.join(self.checkpoint_dir,'loss_tgt_batch.csv'))
        
        

        

        stop_output = 'Finished training~'
        if verbose:
            self.logger.info(stop_output)
        return self.best_scores

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        loss_src_batch = []
        loss_tgt_batch = []
        
        r"""Train the model in an epoch"""
        train_data_src, train_data_tgt = train_data
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss_src, total_loss_tgt = None, None
        iter_data_src = (
            tqdm(
                train_data_src,
                total=len(train_data_src),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data_src
        )
        iter_data_tgt = (
            tqdm(
                train_data_tgt,
                total=len(train_data_tgt),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data_tgt
        )
        stop_point = min(len(iter_data_src),len(iter_data_tgt)) 
#         print(len(iter_data_src),len(iter_data_tgt)) 2206 batch
        for batch_idx, (batch_src, batch_tgt) in enumerate(zip(iter_data_src, cycle(iter_data_tgt))):  # small latter
            self.optimizer.zero_grad()
            losses_src, losses_tgt = loss_func(batch_src, batch_type='src'), loss_func(batch_tgt, batch_type='tgt')
            if isinstance(losses_src, tuple):
                if epoch_idx < self.config['warm_up_step']:
                    losses_src = losses_src[:-1] 
                loss_src = sum(losses_src)
                loss_tuple_src = tuple(per_loss.item() for per_loss in losses_src)
                total_loss_src = loss_tuple_src if total_loss_src is None else tuple(map(sum, zip(total_loss_src, loss_tuple_src)))
            else:
                loss_src = losses_src
                total_loss_src = losses_src.item() if total_loss_src is None else total_loss_src + losses_src.item()
            self._check_nan(loss_src)

            if isinstance(losses_tgt, tuple):
                if epoch_idx < self.config['warm_up_step']:
                    losses_tgt = losses_tgt[:-1] 
                loss_tgt = sum(losses_tgt)
                loss_tuple_tgt = tuple(per_loss.item() for per_loss in losses_tgt)
                total_loss_tgt = loss_tuple_tgt if total_loss_tgt is None else tuple(map(sum, zip(total_loss_tgt, loss_tuple_tgt)))
            else:
                loss_tgt = losses_tgt
                total_loss_tgt = losses_tgt.item() if total_loss_tgt is None else total_loss_tgt + losses_tgt.item()
            self._check_nan(loss_tgt)
            
            loss = loss_src + loss_tgt
            
            
#             if batch_idx >=stop_point:
#                 loss = loss_src
#                 loss_src_batch.append(loss_src.item())
#                 self.train_loss_dict_src_batch.append(np.mean(loss_src_batch)) # 可删掉
#             else:
#                 loss = loss_src + loss_tgt
#                 loss_src_batch.append(loss_src.item())
#                 loss_tgt_batch.append(loss_tgt.item())
#                 self.train_loss_dict_src_batch.append(np.mean(loss_src_batch)) # 可删掉
#                 self.train_loss_dict_tgt_batch.append(np.mean(loss_tgt_batch)) # 可删掉


            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data_src.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
                iter_data_tgt.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return total_loss_src, total_loss_tgt # tuple


    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data"""
        valid_result_src,  valid_result_tgt = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        return valid_result_src, valid_result_tgt

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data. 
        """
        if not eval_data:
            return

        eval_data_src, eval_data_tgt = eval_data

        metric_dict_src,metric_dict_tgt = defaultdict(list), defaultdict(list)

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        iter_data_src = (
            tqdm(
                eval_data_src,
                total=len(eval_data_src),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data_src
        )
        iter_data_tgt = (
            tqdm(
                eval_data_tgt,
                total=len(eval_data_tgt),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data_tgt
        )

        for batch_idx, batched_data in enumerate(iter_data_src):
            scores = self.model.predict(batched_data, batch_type='src') # ,B, 100
#             scores = self.model.full_sort_predict(batched_data, batch_type='src') # ,B, 100

            # calculate metric
            batch_results = calculate_metrics(scores, batched_data) # B, dict=={hit:[], ndcg:[]}
#             batch_results = calculate_metrics_full(scores, batched_data,self.max_item_size_A) # B, dict=={hit:[], ndcg:[]}

            if self.gpu_available and show_progress:
                iter_data_src.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            for metric in self.valid_metric:
                metric_dict_src[metric].extend(batch_results[metric])


        for batch_idx, batched_data in enumerate(iter_data_tgt):
            scores = self.model.predict(batched_data, batch_type='tgt')
#             scores = self.model.full_sort_predict(batched_data, batch_type='tgt')

            # calculate metric
            # targets = batched_data[1][:, 0] 
            batch_results = calculate_metrics(scores,batched_data)
#             batch_results = calculate_metrics_full(scores,batched_data,self.max_item_size_B)

            if self.gpu_available and show_progress:
                iter_data_tgt.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            for metric in self.valid_metric:
                metric_dict_tgt[metric].extend(batch_results[metric])

        for k, v in metric_dict_src.items():
            metric_dict_src[k] = np.mean(v)

        for k, v in metric_dict_tgt.items():
            metric_dict_tgt[k] = np.mean(v)

        return metric_dict_src, metric_dict_tgt


if __name__ == '__main__':
    print("This is seagull algorithms !")

