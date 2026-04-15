#!/usr/bin/env python3
"""Train Medical SAM Adapter network using PyTorch."""

import os
import time
import random

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import cfg
from dataset import OCTADataset, FireflyDataset, CHASEDataset, HRFDataset, DCA1Dataset, ARCADEDataset
import function
import function_tpp
import function_psm

from conf import settings
from utils import *
import utils_hq.misc as misc
import function_tpp


# ── 1. Config & reproducibility ───────────────────────────────────────────────
args = cfg.parse_args()

# Use a different seed per process in distributed training to avoid identical augmentations
seed = args.seed + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ── 2. Model & optimiser ──────────────────────────────────────────────────────
GPUdevice = torch.device('cuda', args.gpu_device)

net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)



optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)  # decay LR every 10 epochs


# ── 3. Resume from checkpoint (optional) ──────────────────────────────────────
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights), f'Checkpoint not found: {args.weights}'
    checkpoint = torch.load(args.weights, map_location=f'cuda:{args.gpu_device}')
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if args.tpp_weights != 0:
        tpp_checkpoint = torch.load(args.tpp_weights, map_location=f'cuda:{args.gpu_device}')
        net.text_predictor.load_state_dict(tpp_checkpoint['state_dict'], strict=True)
        del tpp_checkpoint

    del checkpoint  


# ── 4. Logger ─────────────────────────────────────────────────────────────────
args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


# ── 5. Data transforms ────────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_train_seg = transforms.Compose([
    transforms.Resize((args.out_size, args.out_size)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
transform_test_seg = transforms.Compose([
    transforms.Resize((args.out_size, args.out_size)),
])


# ── 6. Dataset & DataLoader ───────────────────────────────────────────────────
# Maps dataset name -> factory that returns (train_dataset, test_dataset)
DATASET_MAP = {
    'firefly': lambda: (
        FireflyDataset(args, dataset_path=args.data_path, transform=transform_train, transform_msk=transform_train_seg, split='train'),
        FireflyDataset(args, dataset_path=args.data_path, transform=transform_test,  transform_msk=transform_test_seg,  split='val'),
    ),
    'chase': lambda: (
        CHASEDataset(args, dataset_path=args.data_path, transform=transform_train, transform_msk=transform_train_seg, split='train'),
        CHASEDataset(args, dataset_path=args.data_path, transform=transform_test,  transform_msk=transform_test_seg,  split='val'),
    ),
    'hrf': lambda: (
        HRFDataset(args, dataset_path=args.data_path, transform=transform_train, transform_msk=transform_train_seg, split='train'),
        HRFDataset(args, dataset_path=args.data_path, transform=transform_test,  transform_msk=transform_test_seg,  split='val'),
    ),
    'octa': lambda: (
        OCTADataset(args, dataset_path=args.data_path, transform=transform_train, transform_msk=transform_train_seg, split='train'),
        OCTADataset(args, dataset_path=args.data_path, transform=transform_test,  transform_msk=transform_test_seg,  split='val'),
    ),
    'dca1': lambda: (
        DCA1Dataset(args, dataset_path=args.data_path, transform=transform_train, transform_msk=transform_train_seg, split='train'),
        DCA1Dataset(args, dataset_path=args.data_path, transform=transform_test,  transform_msk=transform_test_seg,  split='val'),
    ),
    'arcade': lambda: (
        ARCADEDataset(args, dataset_path=args.data_path, transform=transform_train, transform_msk=transform_train_seg, split='seg_train'),
        ARCADEDataset(args, dataset_path=args.data_path, transform=transform_test,  transform_msk=transform_test_seg,  split='seg_val'),
    ),
}

assert args.dataset in DATASET_MAP, f'Unknown dataset: {args.dataset}'
train_dataset, test_dataset = DATASET_MAP[args.dataset]()


nice_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True,  num_workers=8, pin_memory=True)
nice_test_loader  = DataLoader(test_dataset,  batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)


# ── 7. TensorBoard writer & checkpoint directory ───────────────────────────────
os.makedirs(settings.LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
os.makedirs(checkpoint_path, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


# ── 8. Training loop ──────────────────────────────────────────────────────────
best_tol = 1000.0

for epoch in range(settings.EPOCH):
    # --- Train ---
    net.train()
    if args.train_stage == 'train_backbone':
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis=args.vis)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        print(f'time_for_training {time.time() - time_start:.2f}s')

    if args.train_stage == 'train_tpp':
        time_start = time.time()
        train_loss, train_acc, train_cls, train_ml = function_tpp.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {train_loss}, Train_acc: {train_acc}, Cls_loss: {train_cls}, Mask_loss: {train_ml} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
    
    if args.train_stage == 'train_psm':
        time_start = time.time()
        tol, (eiou, edice), cldice = function_psm.train_sam(args, net, optimizer, nice_train_loader, epoch, writer, vis = args.vis)
        logger.info(f'Train loss: {tol}, IoU: {eiou}, Dice: {edice}, clDice: {cldice} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)
    

    # --- Validate every val_freq epochs and on the final epoch ---
    net.eval()
    if (epoch and epoch % args.val_freq == 0) or epoch == settings.EPOCH - 1:
        if args.train_stage == 'train_backbone':
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Val loss: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            # Unwrap DDP wrapper before extracting state_dict in distributed training
            sd = net.module.state_dict() if args.distributed != 'none' else net.state_dict()
            if tol < best_tol:
                best_tol = tol
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': args.net,
                        'state_dict': sd,
                        'optimizer': optimizer.state_dict(),
                        'best_tol': best_tol,
                        'path_helper': args.path_helper,
                    },
                    output_dir=args.path_helper['ckpt_path'],
                )
        
        if args.train_stage == 'train_tpp':
            tol, val_acc, val_cls, val_ml = function_tpp.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Val loss: {tol}, Val acc: {val_acc}, Cls_loss: {val_cls}, Mask loss: {val_ml}  || @ epoch {epoch}.')
            # Only save the text_predictor sub-module for tpp stage
            sd = net.module.text_predictor.state_dict() if args.distributed != 'none' else net.text_predictor.state_dict()
            if tol < best_tol:
                best_tol = tol
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': sd,
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, output_dir=args.path_helper['ckpt_path'])


        if args.train_stage == 'train_psm':
            tol,  (eiou, edice), cldice = function_psm.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Val loss: {tol}, IOU: {eiou}, Dice: {edice}, clDice: {cldice}|| @ epoch {epoch}.')

            # Unwrap DDP wrapper before extracting state_dict in distributed training
            sd = net.module.state_dict() if args.distributed != 'none' else net.state_dict()
            if tol < best_tol:
                best_tol = tol
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'model': args.net,
                        'state_dict': sd,
                        'optimizer': optimizer.state_dict(),
                        'best_tol': best_tol,
                        'path_helper': args.path_helper,
                    },
                    output_dir=args.path_helper['ckpt_path'],
                )

writer.close()
