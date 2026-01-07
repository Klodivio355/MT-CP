#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
#from diffusers import StableDiffusionImg2ImgPipeline
#from diffusers import UNet2DModel, AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
from utils.utils import blockPrint, enablePrint
from utils.common_config import fill_list_to_length
from torch.nn.utils.rnn import pad_sequence
import copy 
import matplotlib.pyplot as plt

import numpy as np

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): sscaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


from models.NCEAverage import NCEAverage
from models.NCECriterion import NCECriterion

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class DynamicFPNFusionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DynamicFPNFusionModule, self).__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels
        # Create lateral convolutions based on provided input channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, output_channels, kernel_size=1) for in_channels in input_channels
        ])
        
        # Create FPN convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1) for _ in input_channels
        ])

    def forward(self, *inputs):
        # Apply lateral convolutions
        latents = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]
        
        # Top-down pathway
        fused = latents[-1]
        outputs = [self.fpn_convs[-1](fused)]
        for i in range(len(latents) - 2, -1, -1):
            fused = F.interpolate(fused, size=latents[i].shape[2:], mode='nearest') + latents[i]
            outputs.insert(0, self.fpn_convs[i](fused))
        
        return outputs[0]  # Return the highest resolution output


class MultiTaskModel_NYUD(nn.Module):
    def __init__(self, backbone: nn.Module, decoders: dict, tasks: list, input_backbone_channels):
        super(MultiTaskModel_NYUD, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.batch = 2

        num_task = len(tasks)
        self.log_var_list = nn.Parameter(torch.zeros((num_task,), requires_grad=True))
        self.skip_gate = nn.Parameter(torch.ones((1,), requires_grad=True))
        
        from models.swim_transformer2 import SwinTransformer as SwinTransformer2

        backbone_channels = input_backbone_channels 
        backbone_channels_reduce = 256 
        self.backbone_channels_reduce = backbone_channels_reduce
        in_chans = backbone_channels_reduce
        embed_dim = backbone_channels_reduce
        in_chans2 = in_chans
        embed_dim2 = embed_dim

        self.transformer1 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(1,2,1), num_heads=(2,2,2),
                                             in_chans=in_chans2, embed_dim=embed_dim2, drop_path_rate=0.0,
                                             out_indices=(0,1,2))
        self.transformer2 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(1,2,1), num_heads=(2,2,2),
                                             in_chans=in_chans2, embed_dim=embed_dim2, drop_path_rate=0.0,
                                             out_indices=(0,1,2))
        self.transformer3 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(1,2,1), num_heads=(2,2,2),
                                             in_chans=in_chans2, embed_dim=embed_dim2, drop_path_rate=0.0,
                                             out_indices=(0,1,2))

        self.lambda_param = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))  # Learnable scalar for L2-norm regularization
        self.alpha_param = torch.nn.Parameter(torch.tensor([0.01], requires_grad=True))
        
        size_list = [256, 512, 1024]
        self.fpn_semseg = DynamicFPNFusionModule(size_list, backbone_channels_reduce).cuda()
        self.fpn_depth = DynamicFPNFusionModule(size_list, backbone_channels_reduce).cuda()
        self.fpn_normal = DynamicFPNFusionModule(size_list, backbone_channels_reduce).cuda()

        self.output_channel1 = decoders['semseg']
        self.output_channel2 = decoders['depth']
        self.output_channel3 = decoders['normals']

        self.project_semseg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, self.output_channel1, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.project_depth = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=1, stride=1, padding=0),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256 // 2, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, self.output_channel2, kernel_size=1, stride=1, padding=0),
        )

        self.project_normals = nn.Sequential(
                nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(backbone_channels_reduce),
                nn.ReLU(True),
                nn.Conv2d(backbone_channels_reduce, self.output_channel3, kernel_size=1),
        ) 

        intermediate_predictions = 2
        self.sem_pred_layers = nn.ModuleList([SemSegPred(self.backbone_channels_reduce, self.output_channel1) for i in range(intermediate_predictions)])
        self.depth_pred_layers = nn.ModuleList([DepthPred(self.backbone_channels_reduce, self.output_channel2) for i in range(intermediate_predictions)])
        self.normals_pred_layers = nn.ModuleList([NormalPred(self.backbone_channels_reduce, self.output_channel3) for i in range(intermediate_predictions)])

        from .utils import TraceBack, GAAF_module2
        self.trace_back = TraceBack()
        self.trace_back2 = TraceBack()
        self.trace_back3 = TraceBack()
        self.gaaf = GAAF_module2(dim=self.backbone_channels_reduce)
        self.gaaf2 = GAAF_module2(dim=self.backbone_channels_reduce)
        self.gaaf3 = GAAF_module2(dim=self.backbone_channels_reduce)

        from transformers import PretrainedConfig, BertTokenizer, Mask2FormerForUniversalSegmentation, BertModel
        self.backbon = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic", cache_dir='hf_cache')       

    def forward(self, x, index=0, inference=True):
        out_size = x.size()[2:]
        outputs = self.backbon(x, output_hidden_states=True)
        shared_representation = outputs.pixel_decoder_last_hidden_state
        class_queries_logits = outputs.class_queries_logits # [:,:,:41]  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        shared_representation = torch.einsum("bnhw, bdhw -> bdhw", masks_queries_logits, shared_representation)

        _, sem_reps = self.transformer1.forward2(shared_representation)
        _, depth_reps = self.transformer2.forward2(shared_representation)    
        _, normals_reps = self.transformer3.forward2(shared_representation)

        sem_embeds = self.fpn_semseg(*sem_reps)
        depth_embeds = self.fpn_depth(*depth_reps)
        normal_embeds = self.fpn_normal(*normals_reps)

        auxiliary_embeds1 = torch.cat((depth_embeds, normal_embeds), dim=1)
        auxiliary_embeds2 = torch.cat((sem_embeds, normal_embeds), dim=1)
        auxiliary_embeds3 = torch.cat((sem_embeds, depth_embeds), dim=1)

        sem_aux = self.gaaf(sem_embeds, auxiliary_embeds1)
        depth_aux = self.gaaf2(depth_embeds, auxiliary_embeds2)
        normal_aux = self.gaaf3(normal_embeds, auxiliary_embeds3)

        output = {}    
        if not (inference):
            loss_con = torch.zeros(0)

        pred_list = self.trace_back(sem_reps[::-1], initial_pred=sem_aux)
        segmentation_div_4 = self.sem_pred_layers[0](pred_list[0])
        segmentation_div_2 = self.sem_pred_layers[1](pred_list[1])

        pred_list2 = self.trace_back2(depth_reps[::-1], initial_pred=depth_aux)
        depth_div_4 = self.depth_pred_layers[0](pred_list2[0])
        depth_div_2 = self.depth_pred_layers[1](pred_list2[1])

        pred_list3 = self.trace_back3(normals_reps[::-1], initial_pred=normal_aux)
        normal_div_4 = self.normals_pred_layers[0](pred_list3[0])
        normal_div_2 = self.normals_pred_layers[1](pred_list3[1])

        output['alpha_semseg_4'] = F.interpolate(segmentation_div_4, out_size, mode='bilinear')
        output['alpha_semseg_2'] = F.interpolate(segmentation_div_2, out_size, mode='bilinear')

        output['beta_depth_8'] = F.interpolate(depth_div_4, out_size, mode='bilinear')
        output['beta_depth_4'] = F.interpolate(depth_div_2, out_size, mode='bilinear')

        output['charlie_depth_8'] = F.interpolate(normal_div_4, out_size, mode='bilinear')
        output['charlie_depth_4'] = F.interpolate(normal_div_2, out_size, mode='bilinear')

        segmentation = self.project_semseg(pred_list[-1])
        depth = self.project_depth(pred_list2[-1])
        normals = self.project_normals(pred_list3[-1])
        
        output['semseg'] = F.interpolate(segmentation, out_size, mode='bilinear')
        output['depth'] = F.interpolate(depth, out_size, mode='bilinear')
        output['normals'] = F.interpolate(normals, out_size, mode='bilinear')

        if inference:
            return output   
        else:
            return output, loss_con    


class SingleTaskModelSemSeg(nn.Module):
    def __init__(self, backbone: nn.Module, decoders: dict, tasks: list, input_backbone_channels):
        super(SingleTaskModelSemSeg, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.batch = 2

        num_task = len(tasks)
        #self.log_var_list = nn.Parameter(torch.zeros((num_task,), requires_grad=True))
        #self.skip_gate = nn.Parameter(torch.ones((1,), requires_grad=True))
        #breakpoint()
        from models.swim_transformer2 import SwinTransformer as SwinTransformer2

        backbone_channels = input_backbone_channels 
        backbone_channels_reduce = 256
        self.backbone_channels_reduce = backbone_channels_reduce
        in_chans = backbone_channels_reduce
        embed_dim = backbone_channels_reduce
        in_chans2 = in_chans
        embed_dim2 = embed_dim

        self.transformer1 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(1,2,1), num_heads=(2,2,2),
                                             in_chans=in_chans2, embed_dim=embed_dim2, drop_path_rate=0.0,
                                             out_indices=(0,1,2))

        self.batch_size = 2

        self.output_channel1 = decoders['semseg']

        self.project1 = nn.Sequential(
            nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(backbone_channels_reduce),
            nn.ReLU(True),
            nn.Conv2d(backbone_channels_reduce, self.output_channel1, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
    
        from transformers import PretrainedConfig, BertTokenizer, Mask2FormerForUniversalSegmentation, BertModel
        self.backbon = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic", cache_dir='hf_cache')


    def forward(self, x, text, depth_text, index=0, inference=True):
        out_size = x.size()[2:]
        outputs = self.backbon(x)
        shared_representation = outputs.pixel_decoder_last_hidden_state
        class_queries_logits = outputs.class_queries_logits # [:,:,:41]  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        shared_representation = torch.einsum("bnhw, bdhw -> bdhw", masks_queries_logits, shared_representation)

        _, feature_T_task1_new = self.transformer1.forward2(shared_representation)

        output = {}
        if not (inference):
            loss_con = torch.zeros(0)
        #breakpoint()
        segmentation = self.project1(feature_T_task1_new[0])

        output['semseg'] = F.interpolate(segmentation, out_size, mode='bilinear')
        if inference:
            return output
        else:
            return output, loss_con


class SingleTaskModelDepth(nn.Module):
    def __init__(self, backbone: nn.Module, decoders: dict, tasks: list, input_backbone_channels):
        super(SingleTaskModelDepth, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.batch = 2

        num_task = len(tasks)
        #self.log_var_list = nn.Parameter(torch.zeros((num_task,), requires_grad=True))
        #self.skip_gate = nn.Parameter(torch.ones((1,), requires_grad=True))

        from models.swim_transformer2 import SwinTransformer as SwinTransformer2

        backbone_channels = input_backbone_channels 
        backbone_channels_reduce = 256
        self.backbone_channels_reduce = backbone_channels_reduce
        in_chans = backbone_channels_reduce
        embed_dim = backbone_channels_reduce
        in_chans2 = in_chans
        embed_dim2 = embed_dim

        self.transformer1 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(1,2,1), num_heads=(2,2,2),
                                             in_chans=in_chans2, embed_dim=embed_dim2, drop_path_rate=0.0,
                                             out_indices=(0,1,2))

        self.batch_size = 2

        self.output_channel1 = decoders['depth']

        self.project1 = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=1, stride=1, padding=0),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256 // 2, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, self.output_channel1, kernel_size=1, stride=1, padding=0),
            #nn.ReLU(True)
        )
    
        from transformers import PretrainedConfig, BertTokenizer, Mask2FormerForUniversalSegmentation, BertModel
        self.backbon = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic", cache_dir='hf_cache')


    def forward(self, x, text, depth_text, index=0, inference=True):
        out_size = x.size()[2:]
        outputs = self.backbon(x)
        shared_representation = outputs.pixel_decoder_last_hidden_state
        class_queries_logits = outputs.class_queries_logits # [:,:,:41]  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        shared_representation = torch.einsum("bnhw, bdhw -> bdhw", masks_queries_logits, shared_representation)

        _, feature_T_task1_new = self.transformer1.forward2(shared_representation)

        output = {}
        if not (inference):
            loss_con = torch.zeros(0)
        #breakpoint()
        depth = self.project1(feature_T_task1_new[0])

        output['depth'] = F.interpolate(depth, out_size, mode='bilinear')
        if inference:
            return output
        else:
            return output, loss_con

class SingleTaskModelNormal(nn.Module):
    def __init__(self, backbone: nn.Module, decoders: dict, tasks: list, input_backbone_channels):
        super(SingleTaskModelNormal, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.batch = 2

        num_task = len(tasks)
        #self.log_var_list = nn.Parameter(torch.zeros((num_task,), requires_grad=True))
        #self.skip_gate = nn.Parameter(torch.ones((1,), requires_grad=True))

        from models.swim_transformer2 import SwinTransformer as SwinTransformer2

        backbone_channels = input_backbone_channels 
        backbone_channels_reduce = 256
        self.backbone_channels_reduce = backbone_channels_reduce
        in_chans = backbone_channels_reduce
        embed_dim = backbone_channels_reduce
        in_chans2 = in_chans
        embed_dim2 = embed_dim

        self.transformer1 = SwinTransformer2(pretrain_img_size=224, window_size=7, depths=(1,2,1), num_heads=(2,2,2),
                                             in_chans=in_chans2, embed_dim=embed_dim2, drop_path_rate=0.0,
                                             out_indices=(0,1,2))

        self.batch_size = 2

        self.output_channel1 = decoders['normals']

        self.project1 = nn.Sequential(
                nn.Conv2d(backbone_channels_reduce, backbone_channels_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(backbone_channels_reduce),
                nn.ReLU(True),
                nn.Conv2d(backbone_channels_reduce, self.output_channel1, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
    
        from transformers import PretrainedConfig, BertTokenizer, Mask2FormerForUniversalSegmentation, BertModel
        self.backbon = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic", cache_dir='hf_cache')


    def forward(self, x, text, depth_text, index=0, inference=True):
        out_size = x.size()[2:]
        outputs = self.backbon(x)
        shared_representation = outputs.pixel_decoder_last_hidden_state
        class_queries_logits = outputs.class_queries_logits # [:,:,:41]  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        shared_representation = torch.einsum("bnhw, bdhw -> bdhw", masks_queries_logits, shared_representation)

        _, feature_T_task1_new = self.transformer1.forward2(shared_representation)

        output = {}
        if not (inference):
            loss_con = torch.zeros(0)

        normals = self.project1(feature_T_task1_new[0])

        output['normals'] = F.interpolate(normals, out_size, mode='bilinear')
        if inference:
            return output
        else:
            return output, loss_con

class SemSegPred(nn.Module):
    def __init__(self, dim, output_channel):
        super(SemSegPred, self).__init__()
        self.output_channel = output_channel
        self.dim = dim
        self.module = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(True),
                nn.Conv2d(self.dim, self.output_channel, kernel_size=1),
                #Interpolate(scale_factor=4, mode="bilinear", align_corners=True),
        )
    
    def forward(self, x):
        return self.module(x)


class DepthPred(nn.Module):
    def __init__(self, dim, output_channel):
        super(DepthPred, self).__init__()
        self.output_channel = output_channel
        self.dim = dim
        self.module = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(True),
                nn.Conv2d(self.dim, self.output_channel, kernel_size=1),
                #Interpolate(scale_factor=4, mode="bilinear", align_corners=True),
        )
    
    def forward(self, x):
        return self.module(x)


class NormalPred(nn.Module):
    def __init__(self, dim, output_channel):
        super(NormalPred, self).__init__()
        self.output_channel = output_channel
        self.dim = dim
        self.module = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(True),
                nn.Conv2d(self.dim, self.output_channel, kernel_size=1),
                #Interpolate(scale_factor=4, mode="bilinear", align_corners=True),
        )
    
    def forward(self, x):
        return self.module(x)


