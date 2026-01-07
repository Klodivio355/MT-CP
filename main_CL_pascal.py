#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import pathlib
import torch
import yaml
#import wandb
import huggingface_hub

from traine.train_utils import train_pascal

from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion

from utils.logger import Logger

from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions3,\
                                    eval_all_results 
from termcolor import colored

from models.utils import spread_transformation

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def ddp_setup(rank, world_size):
        """
        Args:
            rank: Unique identifier of each process
            world_size: Total number of processes
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def load_image_captioner(rank):
    print(colored('Retrieve Image Captioner', 'blue'))
    os.system("/users/k21220263/.conda/envs/minigptv2/bin/pip install --quiet fschat==0.1.10 gdown")
    local_cache_dir = "hf_cache" # Change 
    default_cache_dir = pathlib.Path(local_cache_dir)
    load_vicuna_weights(cache_dir=local_cache_dir)
    patch_tokenizer_config(default_cache_dir)
    apply_delta(local_cache_dir)
    #load_blip_checkpoint()

    eval_config_path = pathlib.Path("MiniGPT-4/eval_configs/minigpt4_eval.yaml")
    with open(eval_config_path, "r") as f:
        eval_config_dict = yaml.safe_load(f)
        eval_config_dict["model"]["ckpt"] = "./pretrained_minigpt4.pth"
        eval_config_dict["model"]["prompt_path"] = "./MiniGPT-4/prompts/alignment.txt"
        
    with open(eval_config_path, "w") as f:
        yaml.dump(eval_config_dict, f)

    minigpt4_config_path = pathlib.Path("MiniGPT-4/minigpt4/configs/models/minigpt_v2.yaml")
    with open(minigpt4_config_path, "r") as f:
        minigpt4_config_dict = yaml.safe_load(f)
        minigpt4_config_dict["model"]["llama_model"] = "./vicuna-7b-v0"
        
    with open(minigpt4_config_path, "w") as f:
        yaml.dump(minigpt4_config_dict, f)

    parser2 = argparse.ArgumentParser(description="")
    parser2.add_argument('--cfg-path', help='')
    parser2.add_argument('--options', nargs="+",help='')
    #parser2.add_argument('--gpu-id', default=0, help='')
    args2 = parser2.parse_args(" --cfg-path ./MiniGPT-4/eval_configs/minigpt4_eval.yaml".split())

    cfg = Config(args2)

    model_config = cfg.model_cfg
    model_config.device_8bit = rank
    model_cls = registry.get_model_class(model_config.arch)
    model_config.image_size = (480, 480)
    model_minigpt = model_cls.from_config(model_config).to('cuda:{}'.format(rank))
    model_minigpt = model_cls.from_config(model_config)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    return model_minigpt, vis_processor

def main(rank, world_size):
    ddp_setup(rank, world_size)

    parser = argparse.ArgumentParser(description='Vanilla Training')
    parser.add_argument('--config_env',
                        help='Config file for the environment')
    parser.add_argument('--config_exp',
                        help='Config file for the experiment')
    args = parser.parse_args()
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
    print(colored(p, 'red'))
    
    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = DDP(model.cuda(), device_ids=[rank], find_unused_parameters=True)
    
    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    p['train_db_name'] = 'PASCALContext2'
    train_dataset = get_train_dataset(p, train_transforms)
    p['train_db_name'] = 'PASCALContext'

    val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset) 
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)

    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.module.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        val_dataloader.sampler.set_epoch(start_epoch)
        #save_model_predictions3(p, val_dataloader, model) # DONT FORGET TO CHANGE IT BACK TO RUN A FULL TRAINING!!!
        #best_result = eval_all_results(p) # SAME!
        best_result = 0

    
    # Main loop
    print(colored('Starting main loop', 'blue'))
    if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']:
        print('Will only evaluate 10 epochs')

    #if rank==0:
        #.init(project='Multi-Saliency')
        #wandb.config = {"learning_rate": 0.00005, 
        #                "epochs": 100, 
        #                "batch_size": 8
        #                }
        #wandb.watch(model.module)
    #import pdb; pdb.set_trace()
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))
        
        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        train_dataloader.sampler.set_epoch(epoch)

        # Train 
        print('Train ...')
        print('Weights: ', criterion.loss_weights)
        multi_task = False
        if multi_task:
            eval_train, l1, l2, l3, lT = train_pascal(p, train_dataloader, model, criterion, optimizer, epoch)
        else:
            eval_train = train_pascal(p, train_dataloader, model, criterion, optimizer, epoch)

        data_spread = True

        if multi_task:
            if epoch == 0: 
                task_losses = {
                    'semseg': l1,
                    'human_parts': l2,
                    'sal': l3
                }
                previous_task_losses = task_losses 
                ratios = {task: lT / loss for task, loss in task_losses.items()}
                criterion.loss_weights = ratios
                criterion.total_loss = lT
                previous_global_loss = lT
            else:
                task_losses = {
                    'semseg': l1,
                    'human_parts': l2,
                    'sal': l3
                }
                task_specific_ratios = {task: loss / previous_task_losses[task] for task, loss in task_losses.items()}
                previous_task_losses = task_losses 
                previous_global_loss = criterion.total_loss
                overall_loss_ratio = lT / previous_global_loss
                previous_global_loss = lT
                criterion.total_loss = previous_global_loss
                ratios = {key: task_specific_ratios[key] / overall_loss_ratio for key in task_specific_ratios}
                if data_spread:
                    ratios = spread_transformation(ratios, 1.6)
                    print('Spread weights :', ratios)
                criterion.loss_weights = ratios

        # Evaluate
        # Check if need to perform eval first

        #if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']: # To speed up -> Avoid eval every epoch, and only test during final 10 epochs.
        #    if epoch + 1 > p['epochs'] - 10:
        #        eval_bool = True
        #    else:
        #        eval_bool = False
        if (epoch+1) % 25 == 0:
            eval_bool = True
        elif epoch + 1 > p['epochs'] - 10: 
            eval_bool = True
        else:                               
            eval_bool = False

        print('Will evaluate? ', eval_bool)

        # Perform evaluation
        if eval_bool:
            print('Evaluate ...')
            save_model_predictions3(p, val_dataloader, model)
            val_dataloader.sampler.set_epoch(epoch)
            curr_result = eval_all_results(p)
            improves, best_result = validate_results(p, curr_result, best_result)
            if improves and rank==0:
                print('Save new best model')
                torch.save(model.module.state_dict(), p['best_model'])

        # Checkpoint
        if rank == 0:
            #print('WANDB LOGGNIG')
            #import pdb; pdb.set_trace()
            #wandb.log({"epoch": epoch, "learning_rate": lr, "skip_gate": int(model.module.skip_gate[0]),
            #"semseg_miou": eval_train['semseg']['mIoU'], "rmse_depth": eval_train['depth']['rmse'], "log_rmse_depth": eval_train['depth']['log_rmse']}, step=epoch) 
            print('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.module.state_dict(), 
                    'epoch': epoch + 1, 'best_result': best_result}, p['checkpoint'])

    # Evaluate best model at the end
    print(colored('Evaluating best model at the end', 'blue'))
    model.module.load_state_dict(torch.load(p['best_model']))#['model'])
    val_dataloader.sampler.set_epoch(epoch)
    save_model_predictions3(p, val_dataloader, model)
    eval_stats = eval_all_results(p)
    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")
    print('World size:', world_size)
    if world_size != 1 :
        mp.spawn(main, args=(world_size,), nprocs=world_size)
    else:
        main(0,1)