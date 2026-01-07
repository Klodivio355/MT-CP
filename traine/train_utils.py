#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from losses.loss_functions import SoftMaxwithLoss, DepthLoss, NormalsLoss, BalancedCrossEntropyLoss
from utils.utils import AverageMeter, ProgressMeter, get_output
from diffusers import UNet2DModel, AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
import numpy as np
import torch
import random
import wandb

def get_loss_meters(p):
    """ Return dictionary with loss meters to monitor training """
    tasks = p.TASKS.NAMES
    losses = {task: AverageMeter('Loss %s' % (task), ':.4e') for task in tasks}
    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses


def train_vanilla(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}

        # Measure loss and performance
        if p['loss_kwargs']['loss_scheme'] == 'baseline_uncertainty':
            output, log_var_list = model(images)
            loss_dict = criterion(output, targets, log_var_list)
        else:
            output = model(images)
            loss_dict = criterion(output, targets)
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES}, 
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results

def print_grad_stats(model):
    grad_mins = []
    grad_maxs = []
    grad_means = []
    
    for param in model.parameters():
        if param.grad is not None:
            grad_means.append(param.grad.mean().item())
            
    print("Gradient Mean: ", sum(grad_means) / len(grad_means))


def train_nyud(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    semseg_loss = SoftMaxwithLoss()
    depth_loss = DepthLoss()
    normal_loss = NormalsLoss()
    losses['con'] = AverageMeter('Loss con', ':.4e')
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
                             [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))
    model.train()
    total_running_loss = []
    loss1_running_loss = []
    loss2_running_loss = []
    loss3_running_loss = []

    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        index = batch['index'].cuda(non_blocking=True)

        # Measure loss and performance
        if p['loss_kwargs']['loss_scheme'] == 'baseline_uncertainty':
            output, log_var_list, loss_c = model(images, texts, depth_texts, index=index, inference=False)
            #output, log_var_list, loss_c = model(images, index=index, inference=False)
            loss_dict = criterion(output, targets, log_var_list)
            loss_c = torch.mean(loss_c) * 0.01
            loss_dict['con'] = loss_c 
            loss_dict['total'] += loss_c
        else:
            output, loss_c = model(images, index=index, inference=False)
            loss_dict = criterion(output, targets)
        
            intermediate_losses = 0
            intermediate_losses_2 = 0
            intermediate_losses_3 = 0
            sem_losses_list = []
            dep_losses_list = []
            normal_losses_list = []
            running_loss = 0

            for pred in output:
                if pred.startswith('alpha'):
                    intermediate_loss = semseg_loss(output[pred], targets['semseg'])
                    intermediate_losses += intermediate_loss
                    #sem_losses_list.append(intermediate_loss)
                if pred.startswith('beta'):
                    intermediate_loss_2 = depth_loss(output[pred], targets['depth'])
                    intermediate_losses_2 += intermediate_loss_2
                if pred.startswith('charlie'):
                    intermediate_loss_3 = depth_loss(output[pred], targets['normals'])
                    intermediate_losses_3 += intermediate_loss_3 

            loss_dict['total'] += (intermediate_losses) / 2
            loss_dict['total'] += (intermediate_losses_2) / 2
            loss_dict['total'] += (intermediate_losses_3) / 2

            loss_1 = loss_dict['semseg'].item()
            loss_2 = loss_dict['depth'].item()
            loss_3 = loss_dict['normals'].item()

            loss1_running_loss.append(loss_1)
            loss2_running_loss.append(loss_2)
            loss3_running_loss.append(loss_3)
            total_running_loss.append(loss_1+loss_2+loss_3)

        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES},
                                {t: targets[t] for t in p.TASKS.NAMES})
        
       
        optimizer.zero_grad()

        loss_dict['total'].backward()
        optimizer.step()
        #torch.nn.utils.clip_grad_norm_(model.module.parameters(), 15, norm_type=2.0)



    eval_results = performance_meter.get_score(verbose=True)

    loss1_running_loss = np.array(loss1_running_loss)
    loss2_running_loss = np.array(loss2_running_loss)
    loss3_running_loss = np.array(loss3_running_loss)
    total_running_loss = np.array(total_running_loss)
    
    return eval_results, np.mean(loss1_running_loss), np.mean(loss2_running_loss), np.mean(loss3_running_loss), np.mean(total_running_loss) 

def train_pascal(p, train_loader, model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    losses = get_loss_meters(p)
    semseg_loss = SoftMaxwithLoss()
    human_parts_loss = SoftMaxwithLoss()
    sal_loss = BalancedCrossEntropyLoss(size_average=True)
    losses['con'] = AverageMeter('Loss con', ':.4e')
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
                             [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))
    model.train()
    for i, batch in enumerate(train_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in p.ALL_TASKS.NAMES}
        index = batch['index'].cuda(non_blocking=True)
        #texts = batch['description'].cuda(non_blocking=True)
        #human_texts = batch['human_description'].cuda(non_blocking=True)
        #saliency_texts = batch['saliency_description'].cuda(non_blocking=True)

        # Measure loss and performance
        if p['loss_kwargs']['loss_scheme'] == 'baseline_uncertainty':
            output, log_var_list, loss_c = model(images, index=index, inference=False)
            loss_dict = criterion(output, targets, log_var_list)
            loss_c = torch.mean(loss_c) * 0.01
            loss_dict['con'] = loss_c
            loss_dict['total'] += loss_c
        else:
            output, loss_c = model(images, index=index, inference=False)
            loss_dict = criterion(output, targets)
            intermediate_losses = 0
            intermediate_losses_2 = 0
            intermediate_losses_3 = 0
            for pred in output:
                if pred.startswith('alpha'):
                    intermediate_loss = semseg_loss(output[pred], targets['semseg'])
                    intermediate_losses += intermediate_loss
                if pred.startswith('beta'):
                    intermediate_loss_2 = human_parts_loss(output[pred], targets['human_parts'])
                    intermediate_losses_2 += intermediate_loss_2
                if pred.startswith('charlie'):
                    intermediate_loss_3 = sal_loss(output[pred], targets['sal'])
                    intermediate_losses_3 += intermediate_loss_3
            loss_dict['total'] += intermediate_losses/2
            loss_dict['total'] += intermediate_loss_2/2
            loss_dict['total'] += intermediate_losses_3/2
            #loss_c = torch.mean(loss_c) * 0.01
            #loss_dict['con'] = loss_c
            #loss_dict['total'] += loss_c
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        performance_meter.update({t: get_output(output[t], t) for t in p.TASKS.NAMES},
                                 {t: targets[t] for t in p.TASKS.NAMES})
        
        
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 9, norm_type=2.0)

        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose=True)

    return eval_results