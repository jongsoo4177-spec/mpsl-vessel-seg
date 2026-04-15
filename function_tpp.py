import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from tqdm import tqdm

import cfg
from conf import settings
from utils import *

args = cfg.parse_args()

# Set device
GPUdevice = torch.device('cuda', args.gpu_device)

# Loss for binary segmentation (with class imbalance handling)
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

torch.backends.cudnn.benchmark = True

# Post-processing utilities (not heavily used in current code)
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


def train_sam(args, net: nn.Module, optimizer, train_loader,
              epoch, writer, schedulers=None, vis=50):


    epoch_loss = 0
    ind = 0
    n_train = len(train_loader)

    net.train()
    optimizer.zero_grad()

    epoch_ml = 0
    epoch_cl = 0
    epoch_acc = 0

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # Loss function selection
    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = nn.CrossEntropyLoss()

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:

            # -------------------- Load batch --------------------
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            labels = pack['label'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['gt'].to(dtype=torch.float32, device=GPUdevice)

            # Previous prediction mask (optional)
            if 'mask' in pack:
                prev_masks = pack['mask'].to(dtype=torch.float32, device=GPUdevice)
            else:
                prev_masks = None

            # Generate click prompt if not provided
            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt = pack['pt']
                point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']

            # -------------------- 3D mode handling --------------------
            if args.thd:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
                pt = rearrange(pt, 'b n d -> (b d) n')
                imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
                masks = rearrange(masks, 'b c h w d -> (b d) c h w')
                imgs = imgs.repeat(1, 3, 1, 1)
                point_labels = torch.ones(imgs.size(0))
                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

            showp = pt
            ind += 1
            b_size, c, w, h = imgs.size()

            # -------------------- Prompt encoding --------------------
            if point_labels[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)

                coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
                pt = (coords_torch, labels_torch)

            # -------------------- Freeze image encoder --------------------
            if args.mod == 'sam_adpt':
                for n, value in net.image_encoder.named_parameters():
                    if "Adapter" not in n:
                        value.requires_grad = False
                    else:
                        value.requires_grad = False

            elif args.mod in ['sam_lora', 'sam_adalora']:
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

            # Resize masks/images
            masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
            imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)

            # Image encoder forward pass
            imge, interm_embeddings = net.image_encoder(imgs)

            # Downsample previous mask for prompt encoder
            show_prev = None
            if prev_masks is not None:
                show_prev = prev_masks.clone()
                prev_masks = torchvision.transforms.Resize(
                    (args.image_size // 4, args.image_size // 4)
                )(prev_masks)

            # -------------------- Prompt encoder --------------------
            with torch.no_grad():
                if args.net in ['sam', 'mobile_sam']:
                    se, de = net.prompt_encoder(
                        points=pt,
                        text_embedding=None,
                        boxes=None,
                        masks=prev_masks,
                    )
                elif args.net == 'efficient_sam':
                    coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )

            # -------------------- Decoder (text + mask prediction) --------------------
            pred_mask, pred_cls = net.text_predictor(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
                hq_token_only=True,
                interm_embeddings=interm_embeddings,
            )

            pred_mask = F.interpolate(pred_mask, size=(args.out_size, args.out_size))

            # -------------------- Loss computation --------------------
            cls_loss = lossfunc(pred_cls, labels.to(torch.int64))
            mask_loss = criterion_G(pred_mask, masks)
            loss = cls_loss + mask_loss

            pbar.set_postfix(**{'loss (batch)': loss.item()})

            epoch_loss += loss.item()
            epoch_cl += cls_loss
            epoch_ml += mask_loss

            # Accuracy
            _, pred_cls = torch.max(pred_cls, 1)
            epoch_acc += (pred_cls == labels).sum().item()

            pred_text = return_text(pred_cls.cpu().numpy())
            gt_text = return_text(labels.cpu().numpy())

            # -------------------- Backprop --------------------
            if args.mod == 'sam_adalora':
                (loss + lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            # -------------------- Visualization --------------------
            if vis and ind % vis == 0:
                namecat = 'Train'
                for na in name[:2]:
                    namecat = namecat + na.split('/')[-1].split('.')[0] + '+'

                visualization(
                    imgs, pred_mask, masks, show_prev,
                    os.path.join(args.path_helper['sample_path'],
                                 'epoch+' + str(epoch) + namecat + '.jpg'),
                    mode='tpp', reverse=False,
                    points=showp, pt_labels=point_labels,
                    pred_text=pred_text, gt_text=gt_text
                )

            pbar.update()

    return (epoch_loss / (n_train * b_size),
            epoch_acc / (n_train * b_size),
            epoch_cl / (n_train * b_size),
            epoch_ml / (n_train * b_size))


def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
    """Run one validation epoch and return average loss, accuracy, cls_loss, mask_loss."""
    net.eval()

    n_val = len(val_loader)
    epoch_loss = 0
    epoch_ml = 0
    epoch_cl = 0
    epoch_acc = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = nn.CrossEntropyLoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw = pack['gt'].to(dtype=torch.float32, device=GPUdevice)
            labels = pack['label'].to(dtype=torch.float32, device=GPUdevice)

            if 'mask' in pack:
                prev_masksw = pack['mask'].to(dtype=torch.float32, device=GPUdevice)
            else:
                prev_masksw = None

            if 'pt' not in pack or args.thd:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw = pack['pt']
                point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']

            # Chunk-based evaluation: slide through depth dimension in steps of evl_ch
            # This avoids OOM for large 3D volumes by processing one chunk at a time
            buoy = 0
            if args.evl_chunk:
                evl_ch = int(args.evl_chunk)
            else:
                evl_ch = int(imgsw.size(-1))  # process entire volume at once

            while (buoy + evl_ch) <= imgsw.size(-1):
                if args.thd:
                    pt = ptw[:, :, buoy:buoy + evl_ch]
                else:
                    pt = ptw

                imgs = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                if prev_masksw is not None:
                    prev_masksw = prev_masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                # 3D (thd) mode: rearrange volumetric data into per-slice 2D batches
                if args.thd:
                    pt = rearrange(pt, 'b n d -> (b d) n')
                    imgs = rearrange(imgs, 'b c h w d -> (b d) c h w')
                    masks = rearrange(masks, 'b c h w d -> (b d) c h w')
                    imgs = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))
                    imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp = pt
                ind += 1
                b_size, c, w, h = imgs.size()

                # Convert point prompts to torch tensors with batch and point dimensions
                if point_labels.clone().flatten()[0] != -1:
                    point_coords = pt
                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch, labels_torch = coords_torch[:, None, :], labels_torch[:, None]
                    pt = (coords_torch, labels_torch)

                imgs = imgs.to(dtype=torch.float32, device=GPUdevice)
                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                # Downsample previous mask to match prompt encoder's dense input resolution
                show_prev = None
                prev_masks = None
                if prev_masksw is not None:
                    show_prev = prev_masksw.clone()
                    prev_masks = torchvision.transforms.Resize(
                        (args.image_size // 4, args.image_size // 4)
                    )(prev_masksw)

                with torch.no_grad():
                    imge, interm_embeddings = net.image_encoder(imgs)

                    if args.net == 'sam' or args.net == 'mobile_sam':
                        se, de = net.prompt_encoder(
                            points=pt,
                            text_embedding=None,
                            boxes=None,
                            masks=prev_masks,
                        )
                    elif args.net == 'efficient_sam':
                        coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

                    pred_mask, pred_cls = net.text_predictor(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                        hq_token_only=True,
                        interm_embeddings=interm_embeddings,
                    )

                    # Resize prediction to target output size
                    pred_mask = F.interpolate(pred_mask, size=(args.out_size, args.out_size))

                    cls_loss = lossfunc(pred_cls, labels.to(torch.int64))
                    mask_loss = criterion_G(pred_mask, masks)
                    loss = cls_loss + mask_loss

                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    epoch_loss += loss.item()
                    epoch_cl += cls_loss
                    epoch_ml += mask_loss

                    _, pred_cls = torch.max(pred_cls, 1)
                    epoch_acc += (pred_cls == labels).sum().item()

                    pred_text = return_text(pred_cls.cpu().numpy())
                    gt_text = return_text(labels.cpu().numpy())

                    if args.vis and ind % args.vis_val == 0:
                        namecat = 'Test'
                        for na in name[:2]:
                            namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                        visualization(imgs, pred_mask, masks, show_prev,
                                os.path.join(args.path_helper['sample_path'],
                                            'epoch+' + str(epoch) + namecat + '.jpg'),
                                mode='tpp', reverse=False, points=showp  ,pt_labels=point_labels,
                                pred_text=pred_text, gt_text=gt_text)

                pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return (epoch_loss / (n_val * b_size), epoch_acc / (n_val * b_size),
            epoch_cl / (n_val * b_size), epoch_ml / (n_val * b_size))


def transform_prompt(coord, label, h, w):
    coord = coord.transpose(0, 1)
    label = label.transpose(0, 1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, :decoder_max_num_input_points, :
        ]
        label = label[:, :, :decoder_max_num_input_points]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )

    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points, label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
    return torch.stack(
        [
            torch.where(
                batched_points[..., 0] >= 0,
                batched_points[..., 0] * 1024 / input_w,
                -1.0,
            ),
            torch.where(
                batched_points[..., 1] >= 0,
                batched_points[..., 1] * 1024 / input_h,
                -1.0,
            ),
        ],
        dim=-1,
    )