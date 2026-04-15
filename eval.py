"""Train Medical SAM Adapter network using PyTorch."""

import os
import random

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import cfg
from dataset import Test_OCTADataset, Test_FireflyDataset, Test_CHASEDataset, Test_HRFDataset, Test_DCA1Dataset, Test_ARCADEDataset

import function_psm

from conf import settings
from utils import *
import utils_hq.misc as misc



# ── 1. Config & reproducibility ───────────────────────────────────────────────
args = cfg.parse_args()

# Set different seeds for each process in distributed training
seed = args.seed + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ── 2. Model & optimizer ──────────────────────────────────────────────────────
GPUdevice = torch.device('cuda', args.gpu_device)

# Initialize network
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

# Optimizer & learning rate scheduler
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)  # Decay LR every 10 epochs


# Load pretrained weights if provided
if args.weights != 0:
    print(f'=> resuming from {args.weights}')
    assert os.path.exists(args.weights)
    checkpoint_file = os.path.join(args.weights)
    assert os.path.exists(checkpoint_file)
    loc = 'cuda:{}'.format(args.gpu_device)
    checkpoint = torch.load(checkpoint_file, map_location=loc)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint  # Free memory


# ── 4. Logger ─────────────────────────────────────────────────────────────────
args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


# ── 5. Data transforms ────────────────────────────────────────────────────────
# Define data preprocessing pipelines
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
# Mapping dataset names to corresponding dataset constructors
DATASET_MAP = {
    'firefly': lambda:
        Test_FireflyDataset(args, dataset_path=args.data_path, transform=transform_test, transform_msk=transform_test_seg, split='val'),

    'chase': lambda:
        Test_CHASEDataset(args, dataset_path=args.data_path, transform=transform_test, transform_msk=transform_test_seg, split='val'),

    'hrf': lambda:
        Test_HRFDataset(args, dataset_path=args.data_path, transform=transform_test, transform_msk=transform_test_seg, split='val'),

    'octa': lambda:
        Test_OCTADataset(args, dataset_path=args.data_path, transform=transform_test, transform_msk=transform_test_seg, split='test'),

    'dca1': lambda:
        Test_DCA1Dataset(args, dataset_path=args.data_path, transform=transform_test, transform_msk=transform_test_seg, split='test'),

    'arcade': lambda:
        Test_ARCADEDataset(args, dataset_path=args.data_path, transform=transform_test, transform_msk=transform_test_seg, split='seg_val'),
}

# Ensure dataset name is valid
assert args.dataset in DATASET_MAP, f'Unknown dataset: {args.dataset}'

# Initialize dataset and dataloader
test_dataset = DATASET_MAP[args.dataset]()
nice_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)


# ── 7. TensorBoard writer & checkpoint directory ───────────────────────────────
# Create logging directory
os.makedirs(settings.LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))

# Create checkpoint directory
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
os.makedirs(checkpoint_path, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


# ── 8. Evaluation ─────────────────────────────────────────────────────────────
# Set model to evaluation mode
net.eval()

# Run validation
tol, (eiou, edice), cldice = function_psm.validation_sam(args, nice_test_loader, 1, net, writer)

# Log results
logger.info(f'Val loss: {tol}, IOU: {eiou}, Dice: {edice}, clDice: {cldice}.')

# Close TensorBoard writer
writer.close()