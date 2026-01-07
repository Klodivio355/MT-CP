#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict, total_loss=None):
        super(MultiTaskLoss, self).__init__()
        #assert(set(tasks) == set(loss_ft.keys()))
        #assert(set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.total_loss = total_loss
        self.loss_weights.requires_grad = False
        self.eta = 1e-6
    def forward(self, pred, gt):
        #out = {task: torch.log(self.loss_ft[task](pred[task], gt[task])+1+self.eta) for task in self.tasks}
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        out['total'] = torch.sum(torch.stack([out[t] for t in self.tasks]))
        return out

class MultiTaskLoss2(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss2, self).__init__()
        #assert(set(tasks) == set(loss_ft.keys()))
        #assert(set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        self.eta = 1e-6

    def forward(self, pred, gt, batch, epoch, previous_task_values=None, previous_global_loss=None, previous_ratios=None):
        out = {task: torch.log(self.loss_ft[task](pred[task], gt[task])+1+self.eta) for task in self.tasks}
        if epoch == 0 and batch == 0:
            out['total'] = torch.sum(torch.stack([out[t] for t in self.tasks]))
            total = out['total']
            ratios = {key: total / value for key, value in out.items() if key != 'total'}
            previous_loss_values = {key: value for key, value in out.items() if key != 'total'}
            return out, total, previous_loss_values, ratios
            
        elif epoch==0 and batch == 1:
            out['total'] = torch.sum(torch.stack([out[t] * previous_ratios[t].detach() for t in self.tasks]))
            total2 = out['total']
            previous_loss_values = {key: value for key, value in out.items() if key != 'total'}

            # Calculate task-wise ratios
            task_ratios = {key: out[key] / previous_task_values[key] for key in out if key != 'total'}

            # Calculate overall loss ratio
            overall_loss_ratio = total2 / previous_global_loss

            # Calculate the final ratios
            ratios2 = {key: task_ratios[key] / overall_loss_ratio for key in task_ratios}

            return out, total2, previous_loss_values, ratios2
        else:
            # Do the weighting
            with torch.no_grad():
                out['total'] = torch.sum(torch.stack([out[t] * previous_ratios[t] for t in self.tasks]))
                total2 = out['total']
                previous_loss_values = {key: value for key, value in out.items() if key != 'total'}

                # Calculate task-wise ratios
                task_ratios = {key: out[key] / previous_task_values[key] for key in out if key != 'total'}
                # Calculate overall loss ratio
                overall_loss_ratio = total2 / previous_global_loss
                # Calculate the final ratios
                ratios2 = {key: task_ratios[key] / overall_loss_ratio for key in task_ratios}
            return out, overall_loss_ratio, previous_loss_values, ratios2

class MultiTaskLoss_uncertainty(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss_uncertainty, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt, log_vars):
        loss_weight_puls = {}
        loss_weight_mul = {}
        count = 0
        for task in self.tasks:
            if 'semseg' in task:
                precision = torch.exp(-log_vars[count])
            elif 'depth' in task:
                precision = torch.exp(-log_vars[count])
            elif 'human_parts' in task:
                precision = torch.exp(-log_vars[count])
            elif 'sal' in task:
                precision = torch.exp(-log_vars[count])
            elif 'reconstruction' in task:
                precision = torch.exp(-log_vars[count])
            loss_weight_mul[task] = precision
            loss_weight_puls[task] = log_vars[count]
            count += 1

        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([self.loss_weights[t] * loss_weight_mul[t] * 10 * out[t] + loss_weight_puls[t] for t in self.tasks])) ### original
        return out

