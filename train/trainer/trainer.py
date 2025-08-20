import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from tensorboardX.writer import SummaryWriter
from datetime import datetime
from collections import defaultdict

import os
import json
import logging
import pandas as pd

from .PinkModule.logging import *

def log_audio(self, audio_data):
    pass  # Implement logging logic here if necessary

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_accuracy(pred, target):
    pred = torch.max(pred, 1)[1]
    corrects = torch.sum(pred == target).float()
    return corrects / pred.size(0)


class Trainer:
    experiment_name = None

    def __init__(
       self,
        net,
        criterion=None,
        metric=cal_accuracy,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        tensorboard_dir="./pinkblack_tb/",
        ckpt="./ckpt/ckpt.pth",
        experiment_id=None,
        clip_gradient_norm=False,
        is_data_dict=False,
        resume=False,
        ckpt_path=None,
        num_step=100000
    ):
        """
        :param net: nn.Module Network
        :param criterion: loss function. __call__(prediction, *batch_y)
        :param metric: metric function __call__(prediction, *batch_y).
                        *note* : bigger is better. (Early Stopping할 때 metric이 더 큰 값을 선택한다)

        :param train_dataloader:
        :param val_dataloader:
        :param test_dataloader:

        :param optimizer: torch.optim
        :param lr_scheduler:
        :param tensorboard_dir: tensorboard log
        :param ckpt:
        :param experiment_id: be shown on tensorboard
        :param clip_gradient_norm: False or Scalar value (숫자를 입력하면 gradient clipping한다.)
        :param is_data_dict: whether dataloaders return dict. (dataloader에서 주는 데이터가 dict인지)
        """

        self.net = net
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.metric = metric

        self.dataloader = dict()
        if train_dataloader is not None:
            self.dataloader["train"] = train_dataloader
        if val_dataloader is not None:
            self.dataloader["val"] = val_dataloader
        if test_dataloader is not None:
            self.dataloader["test"] = test_dataloader

        if train_dataloader is None or val_dataloader is None:
            logging.warning("Init Trainer :: Two dataloaders are needed!")

        self.optimizer = (
            Adam(filter(lambda p: p.requires_grad, self.net.parameters()))
            if optimizer is None
            else optimizer
        )
        self.lr_scheduler = lr_scheduler

        self.ckpt = ckpt

        #self.config = defaultdict(float)
        self.config = defaultdict(lambda: 0.0)
        self.config["max_train_metric"] = -1e8
        self.config["max_val_metric"] = -1e8
        self.config["max_test_metric"] = -1e8
        self.config["tensorboard_dir"] = tensorboard_dir
        self.config["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config["clip_gradient_norm"] = clip_gradient_norm
        self.config["is_data_dict"] = is_data_dict

        if experiment_id is None:
            self.config["experiment_id"] = self.config["timestamp"]
        else:
            self.config["experiment_id"] = experiment_id

        self.dataframe = pd.DataFrame()

        self.device = Trainer.get_model_device(self.net)
        if self.device == torch.device("cpu"):
            logging.warning(
                "Init Trainer :: Do you really want to train the network on CPU instead of GPU?"
            )

        if self.config["tensorboard_dir"] is not None:
            self.tensorboard = SummaryWriter(self.config["tensorboard_dir"])
        else:
            self.tensorboard = None

        self.callbacks = defaultdict(list)
        self.resume = resume
        self.ckpt_path = ckpt_path
        self.num_step = num_step

        if self.resume and self.ckpt_path:
            self._load_checkpoint(self.ckpt_path)

    def register_callback(self, func, phase="val"):
        self.callbacks[phase].append(func)

    def save(self, f=None):
        if f is None:
            f = self.ckpt
        os.makedirs(os.path.dirname(f), exist_ok=True)
        
        # Save all training state
        checkpoint = {
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': dict(self.config),
            'step': self.config.get('step', 0)
        }
        
        if self.lr_scheduler is not None:
            checkpoint['scheduler'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, f)
        print(f"Saved checkpoint to {f} (step {self.config['step']})")
        
        # # Clean config for serialization
        # config_clean = {}
        # for k, v in self.config.items():
        #     try:
        #         json.dumps(v)
        #         config_clean[k] = v
        #     except (TypeError, OverflowError):
        #         if hasattr(v, '__dict__'):
        #             config_clean[k] = str(v.__dict__)
        #         else:
        #             config_clean[k] = str(v)
        
   
    def plot_loss(self, save_path):
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        if 'train_loss' in self.dataframe:
            plt.plot(self.dataframe['train_loss'], label='Train Loss')
        
        # Plot validation loss
        if 'val_loss' in self.dataframe:
            plt.plot(self.dataframe['val_loss'], label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Step' if self.config.get('step') else 'Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Create plots directory
        plot_dir = os.path.join(os.path.dirname(save_path), 'loss_plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Save plot
        plot_path = os.path.join(plot_dir, f"loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        
    def load(self, f=None):
        if f is None:
            f = self.ckpt
        
        if os.path.exists(f):
            checkpoint = torch.load(f, map_location=self.device)
            
            # Check for new format (dictionary with 'model' key)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                # New format
                self.net.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.config.update(checkpoint['config'])
                print(f"Resumed from step {checkpoint['config'].get('step', 0)}")
            else:
                # Old format (model weights only)
                self.net.load_state_dict(checkpoint)
                print("Loaded legacy checkpoint (model weights only)")
                
        else:
            raise FileNotFoundError(f"Checkpoint file {f} not found!")
            
            # # Backward compatibility check
            # if 'model' not in checkpoint:  # Old format
            #     print("Loading legacy checkpoint format...")
            #     self.net.load_state_dict(checkpoint)
            #     return
                
            # # Load model
            # if isinstance(self.net, nn.DataParallel):
            #     self.net.module.load_state_dict(checkpoint['model'])
            # else:
            #     self.net.load_state_dict(checkpoint['model'])
            
            # # Load optimizer
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # # Load training state
            # self.config.update(checkpoint['config'])
            # self.config['max_val_metric'] = checkpoint['metric']
            
            
            # New format loading
        #     self.net.load_state_dict(checkpoint['model'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        #     self.config.update(checkpoint['config'])
            
            
            
        #     # Load scheduler if exists
        #     if 'scheduler' in checkpoint and self.lr_scheduler is not None:
        #         self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            
        #     print(f"Resumed training from step {checkpoint['config'].get('step', 0)}")
        # else:
        #     raise FileNotFoundError(f"Checkpoint file {f} not found!")

    def train(
        self, epoch=None, phases=None, step=None, validation_interval=1, save_every_validation=False
    ):
        """
        :param epoch: train dataloader를 순회할 횟수
        :param phases: ['train', 'val', 'test'] 중 필요하지 않은 phase를 뺄 수 있다.
        >> trainer.train(1, phases=['val'])

        :param step: epoch이 아닌 step을 훈련단위로 할 때의 총 step 수.
        :param validation_interval: validation 간격
        :param save_every_validation: True이면, validation마다 checkpoint를 저장한다.
        :return: None
        """
        if phases is None:
            phases = list(self.dataloader.keys())

        if epoch is None and step is None:
            raise ValueError("PinkBlack.trainer :: epoch or step should be specified.")

        train_unit = "epoch" if step is None else "step"
        initial_value = self.config.get(train_unit, 0)
        self.config[train_unit] = int(initial_value)

        num_unit = epoch if step is None else step
        validation_interval = max(1, validation_interval)
        
        
        if train_unit == "step" and self.config.get("total_steps", 0) > 0:
            remaining_steps = self.config["total_steps"] - self.config[train_unit]
            num_unit = remaining_steps if remaining_steps > 0 else num_unit
            
        # try:
        #     self.config[train_unit] = int(self.config[train_unit])
        # except (TypeError, ValueError):
        #     self.config[train_unit] = 0  # مقدار پیش‌فرض اگر تبدیل ممکن نبود
            
        kwarg_list = [train_unit]
        for phase in phases:
            kwarg_list += [f"{phase}_loss", f"{phase}_metric"]
        kwarg_list += ["lr", "time"]

        print_row(kwarg_list=[""] * len(kwarg_list), pad="-")
        print_row(kwarg_list=kwarg_list, pad=" ")
        print_row(kwarg_list=[""] * len(kwarg_list), pad="-")

        start = self.config[train_unit]
        
        if train_unit == "step":
            self.config["total_steps"] = start + num_unit

        
        for i in range(start, start + num_unit, validation_interval):
            start_time = time()
            if train_unit == "epoch":
                for phase in phases:
                    self.config[f"{phase}_loss"], self.config[f"{phase}_metric"] = self._train(
                        phase, num_steps=len(self.dataloader[phase])
                    )
                    for func in self.callbacks[phase]:
                        func()
                self.config[train_unit] += 1
            elif train_unit == "step":
                for phase in phases:
                    # Initialize num_steps with a default value
                    num_steps = 0  
            
                    if phase == "train":
                        # Calculate steps for training phase
                        num_steps = min((start + num_unit - i), validation_interval)
                        self.config[train_unit] += num_steps
                    else:
                        # Handle validation/test phases
                        if phase in self.dataloader and self.dataloader[phase] is not None:
                            num_steps = len(self.dataloader[phase])
                        else:
                            raise ValueError(f"No dataloader found for phase: {phase}")
        
                    # Now num_steps is guaranteed to be initialized
                    self.config[f"{phase}_loss"], self.config[f"{phase}_metric"] = self._train(
                        phase, num_steps=num_steps
                    )
                    for func in self.callbacks[phase]:
                        func()
            # [MODIFIED] Persist training state periodically
            if self.config[train_unit] % validation_interval == 0:
                self.save(self.ckpt)  # Save with current step    
            else:
                raise NotImplementedError

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(self.config["val_metric"])
                else:
                    self.lr_scheduler.step()

            i_str = str(self.config[train_unit])
            is_best = self.config["max_val_metric"] < self.config["val_metric"]
            if is_best:
                for phase in phases:
                    self.config[f"max_{phase}_metric"] = max(
                        self.config[f"max_{phase}_metric"], self.config[f"{phase}_metric"]
                    )
                i_str = (str(self.config[train_unit])) + "-best"

            elapsed_time = time() - start_time
            if self.tensorboard is not None:
                _loss, _metric = {}, {}
                for phase in phases:
                    _loss[phase] = self.config[f"{phase}_loss"]
                    _metric[phase] = self.config[f"{phase}_metric"]

                self.tensorboard.add_scalars(
                    f"{self.config['experiment_id']}/loss", _loss, self.config[train_unit]
                )
                self.tensorboard.add_scalars(
                    f"{self.config['experiment_id']}/metric", _metric, self.config[train_unit]
                )
                self.tensorboard.add_scalar(
                    f"{self.config['experiment_id']}/time", elapsed_time, self.config[train_unit]
                )
                self.tensorboard.add_scalar(
                    f"{self.config['experiment_id']}/lr",
                    self.optimizer.param_groups[0]["lr"],
                    self.config[train_unit],
                )

            print_kwarg = [i_str]
            for phase in phases:
                print_kwarg += [self.config[f"{phase}_loss"], self.config[f"{phase}_metric"]]
            print_kwarg += [self.optimizer.param_groups[0]["lr"], elapsed_time]

            print_row(kwarg_list=print_kwarg, pad=" ")
            print_row(kwarg_list=[""] * len(kwarg_list), pad="-")
           # self.dataframe = self.dataframe.append(
           #    dict(zip(kwarg_list, print_kwarg)), ignore_index=True
           #  )
            #new
            self.dataframe = pd.concat([self.dataframe, pd.DataFrame([dict(zip(kwarg_list, print_kwarg))])], ignore_index=True)
            #end
            if is_best:
                self.save(self.ckpt)
                if Trainer.experiment_name is not None:
                    self.update_experiment()

            if save_every_validation:
                self.save(self.ckpt + f"-{self.config[train_unit]}")

    def _step(self, phase, iterator, only_inference=False):

        if self.config["is_data_dict"]:
            batch_dict = next(iterator)
            batch_size = batch_dict[list(batch_dict.keys())[0]].size(0)
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(self.device)
        else:
            batch_x, batch_y = next(iterator)
            if isinstance(batch_x, list):
                batch_x = [x.to(self.device) for x in batch_x]
            else:
                batch_x = [batch_x.to(self.device)]

            if isinstance(batch_y, list):
                batch_y = [y.to(self.device) for y in batch_y]
            else:
                batch_y = [batch_y.to(self.device)]

            batch_size = batch_x[0].size(0)

        self.optimizer.zero_grad()
        with torch.set_grad_enabled(phase == "train"):
            if self.config["is_data_dict"]:
                outputs = self.net(batch_dict)
                if not only_inference:
                    loss = self.criterion(outputs, batch_dict)
            else:
                outputs = self.net(*batch_x)
                if not only_inference:
                    loss = self.criterion(outputs, *batch_y)

            if only_inference:
                return outputs

            if phase == "train":
                loss.backward()
                if self.config["clip_gradient_norm"]:
                    clip_grad_norm_(self.net.parameters(), self.config["clip_gradient_norm"])
                self.optimizer.step()

        with torch.no_grad():
            if self.config["is_data_dict"]:
                metric = self.metric(outputs, batch_dict)
            else:
                metric = self.metric(outputs, *batch_y)

        return {"loss": loss.item(), "batch_size": batch_size, "metric": metric.item()}

    def _train(self, phase, num_steps=0):
        running_loss = AverageMeter()
        running_metric = AverageMeter()

        if phase == "train":
            self.net.train()
        else:
            self.net.eval()

        dataloader = self.dataloader[phase]
        step_iterator = iter(dataloader)
        tq = tqdm(range(num_steps), leave=False)
        for st in tq:
            if (st + 1) % len(dataloader) == 0:
                step_iterator = iter(dataloader)
            results = self._step(phase=phase, iterator=step_iterator)
            tq.set_description(f"Loss:{results['loss']:.4f}, Metric:{results['metric']:.4f}")
            running_loss.update(results["loss"], results["batch_size"])
            running_metric.update(results["metric"], results["batch_size"])

        return running_loss.avg, running_metric.avg

    def eval(self, dataloader=None):
        self.net.eval()
        if dataloader is None:
            dataloader = self.dataloader["val"]
            phase = "val"

        output_list = []
        step_iterator = iter(dataloader)
        num_steps = len(dataloader)
        for st in tqdm(range(num_steps), leave=False):
            results = self._step(phase="val", iterator=step_iterator, only_inference=True)
            output_list.append(results)

        output_cat = torch.cat(output_list)
        return output_cat

    def add_external_config(self, args):
        """
        args : a dict-like object which contains key-value configurations.
        """
        if isinstance(args, dict):
            new_d = defaultdict(float)
            for k, v in args.items():
                new_d[f"config_{k}"] = v
            self.config.update(new_d)
        else:
            new_d = defaultdict(float)
            for k, v in args.__dict__.items():
                new_d[f"config_{k}"] = v
            self.config.update(new_d)

    def update_experiment(self):
        assert Trainer.experiment_name is not None
        df_config = pd.DataFrame([self.config]).set_index("experiment_id")
        
        if os.path.exists(Trainer.experiment_name + ".csv"):
            df_ex = pd.read_csv(Trainer.experiment_name + ".csv", index_col=0)
            df_ex = pd.concat([df_ex, df_config], sort=False)
        else:
            df_ex = df_config
        
        df_ex.to_csv(Trainer.experiment_name + ".csv")
        return df_ex

    @staticmethod
    def get_model_device(net):
        device = torch.device("cpu")
        for param in net.parameters():
            device = param.device
            break
        return device

    @staticmethod
    def set_experiment_name(name):
        Trainer.experiment_name = name
