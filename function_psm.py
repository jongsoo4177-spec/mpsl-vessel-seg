import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import loss
import cfg

from conf import settings
from utils import *


# ── Global setup ─────────────────────────────────────────────────────────────
args = cfg.parse_args()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1, 11, (args.b, 7))

torch.backends.cudnn.benchmark = True

loss_function = nn.CrossEntropyLoss()
max_iterations = settings.EPOCH
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

instance_loss = loss.NormalizedFocalLoss(alpha=0.5, gamma=2)
instance_loss_weight = 1.0


# ── Training ──────────────────────────────────────────────────────────────────

def train_sam(args, net: nn.Module, optimizer, train_loader,
              epoch, writer, schedulers=None, vis=50):
    """Run one training epoch with iterative click-based prompting.

    For each batch:
      1. Encode the image once (no grad).
      2. Run `num_clicks * 2` interactive rounds:
         - Alternating between text-conditioned and text-inference stages.
         - In text-conditioned rounds, segmentation is performed using available text guidance.
         - In text-inference rounds, the model first predicts a text prompt from the image.
      3. Back-propagate the mask loss and update point prompts.
    """
    hard = 0
    total_loss = 0
    mix_res = (0,) * args.multimask_output * 2
    epoch_ml = 0
    ind = 0
    threshold = 0.5
    n_train = len(train_loader)
    total_cl = 0


    net.train()
    optimizer.zero_grad()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    num_clicks = args.num_clicks

    feat_size = (64, 64)

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = nn.CrossEntropyLoss()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['gt'].to(dtype=torch.float32, device=GPUdevice)

            prev_masks = pack['mask'].to(dtype=torch.float32, device=GPUdevice) \
                if 'mask' in pack else None

            # Generate or load click prompts
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']

            text_list = []

            pt_labels = [int(point_labels[0])]

            showp = pt
            ind += 1
            b_size, c, w, h = imgs.size()

            # Convert point coordinates to torch tensors
            if point_labels[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch = coords_torch[:, None, :]
                labels_torch = labels_torch[:, None]
                pt = (coords_torch, labels_torch)

            pt_shape = pt

            # Freeze / unfreeze parameters according to training mode
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = True
            elif args.mod in ('sam_lora', 'sam_adalora'):
                from models.common import loralib as lora
                lora.mark_only_lora_as_trainable(net.image_encoder)
                if args.mod == 'sam_adalora':
                    rankallocator = lora.RankAllocator(
                        net.image_encoder, lora_r=4, target_rank=8,
                        init_warmup=500, final_warmup=1500, mask_interval=10,
                        total_step=3000, beta1=0.85, beta2=0.85,
                    )
            else:
                for n, value in net.image_encoder.named_parameters():
                    value.requires_grad = False

            # Always train the HQ decoder
            for n, value in net.hq_decoder.named_parameters():
                value.requires_grad = True

            # Resize inputs to model-expected dimensions
            masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
            imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)

            # Encode image features once (no gradient needed)
            with torch.no_grad():
                imge, interm_embeddings = net.image_encoder(imgs)


            # Optionally load text prompt from dataset
            if 'text' in pack:
                text = pack['text']
                #text_list.append(text[0])
            else:
                text = None

            np_masks = np.array(masks[0][0].cpu())

            # Iterative click loop: alternates text prediction and mask decoding
            for click_indx in range(num_clicks * 2):

                if prev_masks is not None:
                    show_prev = prev_masks.clone()
                    prev_masks = torchvision.transforms.Resize(
                        (args.image_size // 4, args.image_size // 4)
                    )(prev_masks)

                with torch.no_grad():
                    text_embedding = net.text_encoder(tuple(text), device=GPUdevice) \
                        if text is not None else None

                    se, de = net.prompt_encoder(
                        points=pt,
                        text_embedding=text_embedding,
                        boxes=None,
                        masks=prev_masks,
                    )

                if text is None:
                    # Round without text: predict which edit command to apply
                    with torch.no_grad():
                        _, pred_cls = net.text_predictor(
                            image_embeddings=imge,
                            image_pe=net.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                            hq_token_only=True,
                            interm_embeddings=interm_embeddings,
                        )

                        _, pred_cls = torch.max(pred_cls, 1)
                        pred_cls = pred_cls.cpu().numpy()
                        point_label = int(pt[-1][0][-1])

                        text = return_text(pred_cls)

                        text = refine_text_prediction(text, point_label)

                        #text_list.append(text)

                else:
                    # Round with text: decode segmentation mask
                    masks_hq = net.hq_decoder(
                        image_embeddings=imge,
                        text_embedding=text_embedding,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                        hq_token_only=True,
                        interm_embeddings=interm_embeddings,
                    )

                    pred_mask = F.interpolate(masks_hq, size=(args.out_size, args.out_size))
                    loss = criterion_G(pred_mask, masks)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    total_loss += loss.item()

                    # Backward pass
                    if args.mod == 'sam_adalora':
                        (loss + lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                        optimizer.step()
                        rankallocator.update_and_mask(net, ind)
                    else:
                        loss.backward()
                        optimizer.step()

                    optimizer.zero_grad()

                    # Visualize periodically
                    if vis and ind % vis == 0:
                        namecat = 'Train'
                        for na in name[:2]:
                            namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                        visualization(imgs, pred_mask, masks, show_prev, 
                                            os.path.join(args.path_helper['sample_path'], 
                                            'epoch+' +str(epoch) + namecat+ str(int(click_indx/2))+ '.jpg'), 
                                            mode='psm', reverse=False, points=showp,text=text, pt_labels= pt_labels)

                    # Reset text and update mask / point for next iteration
                    text = None
                    prev_masks = torch.sigmoid(pred_mask)
                    np_prev_masks = np.array(prev_masks[0][0].cpu().detach())

                    new_pt = get_next_click(np_masks, np_prev_masks >= 0.5, pt_shape)
                    points = torch.cat((pt[0], new_pt[0]), dim=1)
                    label = torch.cat((pt[1], new_pt[1]), dim=1)
                    pt = (points, label)
                    showp = torch.cat((showp, new_pt[0][:, 0].cpu()))
                    pt_labels.append(int(new_pt[1][0][0]))


            temp = eval_seg(torch.sigmoid(pred_mask), masks, threshold)
            mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            cl= eval_cl(torch.sigmoid(pred_mask), masks, threshold)
            total_cl += cl

            pbar.update()



    return (total_loss / (n_train * b_size * num_clicks), 
            tuple([a / (n_train * b_size) for a in mix_res]),
            total_cl / (n_train * b_size), 
    )
            


# ── Validation ────────────────────────────────────────────────────────────────

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    """Run validation with the same iterative click-based prompting as training.

    Returns averaged metrics: (loss, seg_metrics, iou, rate_dict, cl_dice,
                                nn_rate_dict, avg_inference_time_ms).
    """
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)
    mix_res = (0,) * args.multimask_output * 2
    total_ml = 0
    threshold = 0.5
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice
    num_clicks = args.num_clicks
    feat_size = (64, 64)
    total_cl = 0


    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = nn.CrossEntropyLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['gt'].to(dtype=torch.float32, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            prev_masksw = pack['mask'].to(dtype=torch.float32, device=GPUdevice) \
                if 'mask' in pack else None

            text_list = []

            # Generate or load click prompts
            if 'pt' not in pack or args.thd:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']

            pt_labels = [int(point_labels[0])]

            buoy = 0
            evl_ch = int(args.evl_chunk) if args.evl_chunk else int(imgsw.size(-1))

            while (buoy + evl_ch) <= imgsw.size(-1):
                # Slice chunk for 3-D (THD) evaluation
                if args.thd:
                    pt = ptw[:, :, buoy:buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                if prev_masksw is not None:
                    prev_masksw = prev_masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                # Reshape volumetric data for 2-D model
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))
                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp = pt

                mask_type = torch.float32
                ind += 1
                b_size, c, w, h = imgs.size()
                longsize = w if w >= h else h

                # Convert point coordinates to torch tensors
                if point_labels.clone().flatten()[0] != -1:
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch = coords_torch[:, None, :]
                    labels_torch = labels_torch[:, None]
                    pt = (coords_torch, labels_torch)

                pt_shape = pt

                imgs = imgs.to(dtype=mask_type, device=GPUdevice)
                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
                with torch.no_grad():
                    imge, interm_embeddings = net.image_encoder(imgs)

                # Optionally load text prompt from dataset
                if 'text' in pack:
                    text = pack['text']
                    #text_list.append(text[0])
                else:
                    text = None

                np_masks = np.array(masks[0][0].cpu())

                # Iterative click loop (same structure as training)
                for click_indx in range(num_clicks * 2):

                    if prev_masksw is not None:
                        show_prev = prev_masksw.clone()
                        prev_masks = torchvision.transforms.Resize(
                            (args.image_size // 4, args.image_size // 4)
                        )(prev_masksw)

                    with torch.no_grad():
                        text_embedding = net.text_encoder(tuple(text), device=GPUdevice) \
                            if text is not None else None

                        se, de = net.prompt_encoder(
                            points=pt,
                            text_embedding=text_embedding,
                            boxes=None,
                            masks=prev_masks,
                        )

                        if text is None:
                            # Predict edit command from current state
                            _, pred_cls = net.text_predictor(
                                image_embeddings=imge,
                                image_pe=net.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de,
                                multimask_output=False,
                                hq_token_only=True,
                                interm_embeddings=interm_embeddings,
                            )

                            _, pred_cls = torch.max(pred_cls, 1)
                            pred_cls = pred_cls.cpu().numpy()
                            text = return_text(pred_cls)

                            point_label = int(pt[-1][0][-1])

                            text = refine_text_prediction(text, point_label)
                            #text_list.append(text)

                        else:
                            # Decode segmentation mask with text guidance
                            masks_hq = net.hq_decoder(
                                image_embeddings=imge,
                                text_embedding=text_embedding,
                                image_pe=net.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=se,
                                dense_prompt_embeddings=de,
                                multimask_output=False,
                                hq_token_only=True,
                                interm_embeddings=interm_embeddings,
                            )


                            pred_mask = F.interpolate(masks_hq, size=(args.out_size, args.out_size))
                            loss = criterion_G(pred_mask, masks)
                            total_ml += loss

                            # Visualize periodically
                            if ind % args.vis_val == 0:
                                namecat = 'Test'
                                for na in name[:2]:
                                    img_name = na.split('/')[-1].split('.')[0]
                                    namecat = namecat + img_name + '+'
                                visualization(
                                    imgs, pred_mask, masks, show_prev,
                                    os.path.join(args.path_helper['sample_path'],
                                                 f'epoch+{epoch}{namecat}{int(click_indx / 2)}.jpg'),
                                    mode='psm', reverse=False, points=showp, text=text, pt_labels=pt_labels,
                                )

                            # Reset text and update mask / point for next iteration
                            text = None
                            prev_masksw = torch.sigmoid(pred_mask)
                            np_prev_masks = np.array(prev_masksw[0][0].cpu().detach())

                            new_pt = get_next_click(np_masks, np_prev_masks >= 0.5, pt_shape)
                            points = torch.cat((pt[0], new_pt[0]), dim=1)
                            label = torch.cat((pt[1], new_pt[1]), dim=1)
                            pt = (points, label)
                            showp = torch.cat((showp, new_pt[0][:, 0].cpu()))
                            pt_labels.append(int(new_pt[1][0][0]))

                # Compute per-sample metrics
                temp = eval_seg(torch.sigmoid(pred_mask), masks, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

                cl = eval_cl(torch.sigmoid(pred_mask), masks, threshold)
                total_cl += cl



                pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return (
        total_ml / (n_val * b_size * num_clicks),
        tuple([a / (n_val * b_size) for a in mix_res]),
        total_cl / (n_val * b_size),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def transform_prompt(coord, label, h, w):
    """Reshape and rescale point prompts to match the decoder's expected format.

    Pads or truncates to `decoder_max_num_input_points` (6) along the point
    dimension and rescales coordinates to the 1024-pixel model input space.
    """
    coord = coord.transpose(0, 1)
    label = label.transpose(0, 1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[:, :, :decoder_max_num_input_points, :]
        label = label[:, :, :decoder_max_num_input_points]
    elif num_pts < decoder_max_num_input_points:
        pad = decoder_max_num_input_points - num_pts
        rescaled_batched_points = F.pad(rescaled_batched_points, (0, 0, 0, pad), value=-1.0)
        label = F.pad(label, (0, pad), value=-1.0)

    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(batch_size * max_num_queries, decoder_max_num_input_points)

    return rescaled_batched_points, label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
    """Rescale (x, y) point coordinates to the 1024-pixel SAM input space.

    Padding points (coord < 0) are preserved as -1.
    """
    return torch.stack(
        [
            torch.where(batched_points[..., 0] >= 0,
                        batched_points[..., 0] * 1024 / input_w, -1.0),
            torch.where(batched_points[..., 1] >= 0,
                        batched_points[..., 1] * 1024 / input_h, -1.0),
        ],
        dim=-1,
    )

def refine_text_prediction(text, point_label):
        # Ensure predicted text is consistent with click polarity
    if point_label == 0 and text[0] == 'Make thicker':
        text[0] = 'Make thinner'
    if point_label == 0 and text[0] in ['Extend', 'Make a connection']:
        text[0] = 'Remove'
    if point_label == 1 and text[0] == 'Make thinner':
        text[0] = 'Make thicker'
    if point_label == 1 and text[0] == 'Remove':
        text[0] = random.choice(['Extend', 'Make a connection'])
    
    return text
