# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Train and eval functions used in main.py
"""
from typing import Iterable, Optional
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
import json
import numpy as np
from pathlib import Path
# import clip

# clip_model, preprocess = clip.load("ViT-B/32", device='cuda')

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, target):
#         euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
#         loss_contrastive = torch.mean((1-target) * torch.pow(euclidean_distance, 2) +
#                                       (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         return loss_contrastive
    


def train_one_epoch(model: torch.nn.Module,model_effnet: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, num_cilps:int, optimizer: torch.optim.Optimizer, optimizer_effnet: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_scaler_effnet, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = True, amp=True,
                    contrastive_nomixup=False, hard_contrastive=False,
                    finetune=False
                    ):
    # TODO fix this for finetuning
    if finetune:
        model.train(not finetune)
    else:
        model.train()
        model_effnet.train()
    #criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    # criterion2 = ContrastiveLoss(1)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        batch_size = targets.size(0)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # images = preprocess(samples.unsqueeze(0).to(device))    #Load data for CLIP
        # with torch.no_grad():
        #     embd = clip_model.encode_image(images)                   #Generates the embeddings

        if mixup_fn is not None:
            # batch size has to be an even number
            if batch_size == 1:
                continue
            if batch_size % 2 != 0:
                    samples, targets = samples[:-1], targets[:-1]
            samples, targets = mixup_fn(samples, targets)


        # contrastive_loss = criterion2(embd, targets)    #Calculating loss

        with torch.cuda.amp.autocast(enabled=amp):
            
            outputs = model(samples)
            outputs_effnet = model_effnet(samples)
            # print('Outputs shape: {}'.format(outputs.shape))
            outputs = outputs.reshape(batch_size, num_cilps, -1).mean(dim=1) 
            outputs_effnet = outputs_effnet.reshape(batch_size, num_cilps * 4, -1).mean(dim=1) 
            # print('Outputs shape after reshape: {}'.format(outputs.shape))
            # print('Target shape: {}'.format(targets.shape))
            loss = criterion(outputs, targets) #+ contrastive_loss
            loss_effnet = criterion(outputs_effnet, targets)
            average_loss = loss*0.5 + loss_effnet*0.5


        loss_value = loss.item()
        loss_value_effnet = loss_effnet.item()
        avergae_loss_value = average_loss.item()
        optimizer.zero_grad()
        optimizer_effnet.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        if amp:
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
        else:
            average_loss.backward(create_graph=is_second_order)
            if max_norm is not None and max_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(model_effnet.parameters(), max_norm)
            optimizer.step()
            optimizer_effnet.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=avergae_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print('SwinTransformer loss: {} , EfficientNet loss: {} , Average loss: {}'.format(loss_value, loss_value_effnet, avergae_loss_value))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, model_effnet, device, world_size, distributed=True, amp=False, num_crops=1, num_clips=1, output_dir = Path('/home/yakul/code/output/test')):
    criterion = torch.nn.CrossEntropyLoss()
    to_np = lambda x: x.data.cpu().numpy()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    model_effnet.eval()

    outputs = []
    targets = []
    logits = []
    binary_label = []
    for images, target in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast(enabled=amp):

            output = model(images)
            output_effnet = model_effnet(images)

        output = output.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)
        output_effnet = output_effnet.reshape(batch_size, num_clips * 4, -1).mean(dim=1) 
        output = (output + output_effnet) / 2
        output_np = to_np(output[:,1])

        
        if distributed:
            outputs.append(concat_all_gather(output))
            targets.append(concat_all_gather(target))
            output_ = concat_all_gather(output)
            target_ = concat_all_gather(target)
            output_np_ = to_np(output_[:,1])
            logits.append(output_np_)
            binary_label.append(target_.detach().cpu())
        else:
            outputs.append(output)
            targets.append(target)
            logits.append(output_np)
            binary_label.append(target.detach().cpu())
        batch_size = images.shape[0]

        acc1 = accuracy(output, target, topk=(1,))[0]
        metric_logger.meters['acc1'].update(acc1.item(), images.size(0))

    # import pdb;pdb.set_trace()

    acc_outputs = numpy.stack(logits,0).reshape(-1,1)
    acc_label = numpy.stack(binary_label,0).reshape(-1,1)

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    _,preds = torch.max(outputs,1)
    preds = preds.detach().cpu().numpy()
    targets_numpy = targets.detach().cpu().numpy()

    metrics = precision_recall_fscore_support(y_true=targets_numpy, y_pred=preds)
    metrics = np.array(metrics)
    real = metrics[:,0]
    fake = metrics[:,1]

    auc_score = roc_auc_score(acc_label, acc_outputs)    
    fpr, tpr, thresholds = roc_curve(acc_label, acc_outputs)
    curve = np.vstack((fpr, tpr, thresholds))

    real_loss = criterion(outputs, targets)
    metric_logger.update(loss=real_loss.item())

    print('* Acc@1 {top1.global_avg:.3f} AUC {auc} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1,auc=auc_score,losses=metric_logger.loss))
    
    log_stats = {'Real video': {'Precision': real[0] , 'Recall': real[1], 'F1 score': real[2] , 'Support': real[3]}, 
                 'Fake video' : {'Precision': fake[0] , 'Recall': fake[1], 'F1 score': fake[2] , 'Support': fake[3]}}
    with (output_dir / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")
    with (output_dir / 'curve.npy').open('wb') as f:
        np.save(f, curve)
        
    
    
    print('Real videos: Precision: {} , Recall: {} , F1 score: {} , Support: {}'.format(real[0], real[1], real[2], real[3]))
    print('Fake videos: Precision: {} , Recall: {} , F1 score: {} , Support: {}'.format(fake[0], fake[1], fake[2], fake[3]))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor.contiguous(), async_op=False)

    #output = torch.cat(tensors_gather, dim=0)
    if tensor.dim() == 1:
        output = rearrange(tensors_gather, 'n b -> (b n)')
    else:
        output = rearrange(tensors_gather, 'n b c -> (b n) c')

    return output
