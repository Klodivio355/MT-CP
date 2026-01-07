# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import warnings
import cv2
import os.path
import numpy as np
import glob
import torch
import json
import scipy.io as sio
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import torchvision.transforms as transforms 


def eval_reconstruction(loader, folder):

    total_fids = 0.0
    transform_PIL = transforms.Compose([transforms.PILToTensor()])
    transform_ND = transforms.Compose([transforms.ToTensor()])
    fid = FrechetInceptionDistance(feature=64)

    pred_init = np.zeros((len(loader), 3, 480, 640), dtype=np.uint8) 
    label_init = np.zeros((len(loader), 3, 480, 640), dtype=np.uint8) 
    pred_fid, label_fid = torch.tensor(pred_init, dtype=torch.uint8), torch.tensor(label_init, dtype=torch.uint8) 

    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating reconstruction: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['meta']['image'] + '.png')
        pred_image = Image.open(filename)
        pred = transform_PIL(pred_image) 
        pred = torch.unsqueeze(pred, 0)
        pred_fid[i] = pred

        label_sample = sample['reconstruction']
        label = transform_ND(label_sample)
        label = label.to(torch.uint8)
        label = torch.unsqueeze(label, 0)
        label_fid[i] = label
        
        if pred.shape != label.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')
            pred = cv2.resize(pred, label.shape[::-1], interpolation=cv2.INTER_LINEAR)
        
    fid.update(label_fid, real=True)
    fid.update(pred_fid, real=False)

    eval_result = dict()
    eval_result['fid'] = float(fid.compute())
    fid.reset()

    return eval_result
        

class ReconstructionMeter(object):
    def __init__(self):
        self.fid_score = 0.0
        self.pred_fid = []
        self.label_fid = []
        self.fid = FrechetInceptionDistance(feature=64).cuda()
        
    @torch.no_grad()
    def update(self, pred, gt):
        #pred, gt = pred.squeeze(), gt.squeeze()
        self.pred_fid.append(pred)
        self.label_fid.append(gt)

    def reset(self):
        self.fid_score = 0.0
        self.pred_fid = []
        self.label_fid = []
        self.fid.reset()
        
    def get_score(self, verbose=True):
        label_fid = torch.tensor(torch.stack(self.label_fid), dtype=torch.uint8)
        pred_fid = torch.tensor(torch.stack(self.pred_fid), dtype=torch.uint8)
        label_fid, pred_fid = label_fid.squeeze().cuda(), pred_fid.squeeze()
        pred_fid = pred_fid.transpose(-1, 1)
        pred_fid = pred_fid.transpose(-1, 2).cuda()
        # print(label_fid.size())
        # print(pred_fid.size())
        self.fid.update(label_fid, real=True)
        self.fid.update(pred_fid, real=False)

        eval_result = dict()
        eval_result['fid'] = float(self.fid.compute())

        if verbose:
            print('Results for Reconstruction')
            for x in eval_result:
                spaces = ''
                for j in range(0, 15 - len(x)):
                    spaces += ' '
                print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_result[x]))

        return eval_result


def eval_reconstruction_predictions(database, save_dir, overfit=False):

    # Dataloaders
    if database == 'NYUD':
        from data.nyud2 import NYUD_MT 
        gt_set = 'val'
        db = NYUD_MT(split=gt_set, do_reconstruction=True, overfit=overfit)
    else:
        raise NotImplementedError

    base_name = database + '_' + 'test' + '_reconstruction'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (reconstruction)')
    eval_results = eval_reconstruction(db, os.path.join(save_dir, 'reconstruction'))
    with open(fname, 'w') as f:
        json.dump(eval_results, f)

    # Print results
    print('Results for Reconstruction Estimation')
    for x in eval_results:
        spaces = ''
        for j in range(0, 15 - len(x)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(x, spaces, eval_results[x]))

    return eval_results
