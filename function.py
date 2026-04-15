"""Training and validation routines for Medical SAM Adapter."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from tqdm import tqdm

import cfg
from utils import *

# ── Module-level setup ────────────────────────────────────────────────────────
args = cfg.parse_args()

GPUdevice  = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

torch.backends.cudnn.benchmark = True


def train_sam(args, net: nn.Module, optimizer, train_loader,
              epoch, writer, vis=50):
    """Run one training epoch and return the average batch loss."""
    net.train()
    optimizer.zero_grad()

    n_train    = len(train_loader)
    epoch_loss = 0
    ind        = 0

    GPUdevice = torch.device('cuda', args.gpu_device)

    with tqdm(total=n_train, desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            imgs     = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks    = pack['gt'].to(dtype=torch.float32, device=GPUdevice)

            prev_masks = (pack['mask'].to(dtype=torch.float32, device=GPUdevice)
                          if 'mask' in pack else None)

            if 'pt' not in pack:
                imgs, pt, masks = generate_click_prompt(imgs, masks)
            else:
                pt           = pack['pt']
                point_labels = pack['p_label']

            name = pack['image_meta_dict']['filename_or_obj']

            # Volumetric mode: rearrange depth slices into the batch dimension
            if args.thd:
                pt           = rearrange(pt,    'b n d -> (b d) n')
                imgs         = rearrange(imgs,  'b c h w d -> (b d) c h w')
                masks        = rearrange(masks, 'b c h w d -> (b d) c h w')
                imgs         = imgs.repeat(1, 3, 1, 1)
                point_labels = torch.ones(imgs.size(0))
                imgs         = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks        = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

            showp    = pt
            ind     += 1
            _, _, w, h = imgs.size()

            # Pack point coordinates and labels into SAM prompt format
            if point_labels[0] != -1:
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch = coords_torch[:, None, :]
                labels_torch = labels_torch[:, None]
                pt = (coords_torch, labels_torch)

            # In sam_adpt mode, freeze all image encoder weights except Adapter blocks
            train_all = (args.mod != 'sam_adpt')
            for n, value in net.image_encoder.named_parameters():
                #value.requires_grad = train_all or ('Adapter' in n)
                value.requires_grad = False

            for _, value in net.hq_decoder.named_parameters():
                value.requires_grad = True

            imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
            masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
            imge, interm_embeddings = net.image_encoder(imgs)

            text = pack.get('text', None)

            if prev_masks is not None:
                show_prev  = prev_masks.clone()
                prev_masks = torchvision.transforms.Resize(
                    (args.image_size // 4, args.image_size // 4))(prev_masks)

            text_embedding = (net.text_encoder(tuple(text), device=GPUdevice)
                              if text is not None else None)

            with torch.no_grad():
                if args.net in ('sam', 'mobile_sam'):
                    se, de = net.prompt_encoder(
                        points=pt,
                        text_embedding=text_embedding,
                        boxes=None,
                        masks=prev_masks,
                    )
                elif args.net == 'efficient_sam':
                    coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                    se = net.prompt_encoder(
                        coords=coords_torch,
                        labels=labels_torch,
                    )

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

            masks_hq   = F.interpolate(masks_hq, size=(args.out_size, args.out_size))
            loss       = criterion_G(masks_hq, masks)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            if args.mod == 'sam_adalora':
                (loss + lora.compute_orth_regu(net, regu_weight=0.1)).backward()
                optimizer.step()
                rankallocator.update_and_mask(net, ind)
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            # Periodically save visualisation images to disk
            if vis and ind % vis == 0:
                namecat = 'Train'
                for na in name:
                    namecat += na.split('/')[-1].split('.')[0] + '+'
                visualization(
                    imgs, masks_hq, masks, show_prev,
                    os.path.join(args.path_helper['sample_path'],
                                 f'epoch+{epoch}{namecat}.jpg'),
                    mode='backbone', reverse=False, points=showp, text=text, pt_labels=point_labels,
                )

            pbar.update()

    return epoch_loss / n_train


def validation_sam(args, val_loader, epoch, net: nn.Module, writer):
    """Run validation and return (total_loss, (mean_iou, mean_dice))."""
    net.eval()

    n_val     = len(val_loader)
    mix_res   = (0,) * args.multimask_output * 2
    tot       = 0
    threshold = 0.5

    GPUdevice = torch.device('cuda', args.gpu_device)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            imgsw       = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masksw      = pack['gt'].to(dtype=torch.float32, device=GPUdevice)
            prev_masksw = (pack['mask'].to(dtype=torch.float32, device=GPUdevice)
                           if 'mask' in pack else None)

            if 'pt' not in pack:
                imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)
            else:
                ptw          = pack['pt']
                point_labels = pack['p_label']

            name     = pack['image_meta_dict']['filename_or_obj']


            # Chunk-based evaluation for 3-D volumes; evl_ch equals full depth for 2-D inputs
            buoy   = 0
            evl_ch = int(args.evl_chunk) if args.evl_chunk else int(imgsw.size(-1))

            while buoy + evl_ch <= imgsw.size(-1):
                pt    = ptw[:, :, buoy:buoy + evl_ch] if args.thd else ptw
                imgs  = imgsw[..., buoy:buoy + evl_ch]
                masks = masksw[..., buoy:buoy + evl_ch]
                if prev_masksw is not None:
                    prev_masksw = prev_masksw[..., buoy:buoy + evl_ch]
                buoy += evl_ch

                # Volumetric mode: rearrange depth slices into the batch dimension
                if args.thd:
                    pt           = rearrange(pt,    'b n d -> (b d) n')
                    imgs         = rearrange(imgs,  'b c h w d -> (b d) c h w')
                    masks        = rearrange(masks, 'b c h w d -> (b d) c h w')
                    imgs         = imgs.repeat(1, 3, 1, 1)
                    point_labels = torch.ones(imgs.size(0))
                    imgs         = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                    masks        = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)

                showp    = pt
                ind     += 1
                _, _, w, h = imgs.size()

                # Pack point coordinates and labels into SAM prompt format
                if point_labels[0] != -1:
                    coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                    labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                    coords_torch = coords_torch[:, None, :]
                    labels_torch = labels_torch[:, None]
                    pt = (coords_torch, labels_torch)

                imgs = imgs.to(dtype=torch.float32, device=GPUdevice)
                imgs = torchvision.transforms.Resize((args.image_size, args.image_size))(imgs)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(masks)
                text = pack.get('text', None)

                if prev_masksw is not None:
                    show_prev   = prev_masksw.clone()
                    prev_masksw = torchvision.transforms.Resize(
                        (args.image_size // 4, args.image_size // 4))(prev_masksw)

                with torch.no_grad():
                    text_embedding = (net.text_encoder(tuple(text), device=GPUdevice)
                                      if text is not None else None)

                    imge, interm_embeddings = net.image_encoder(imgs)

                    if args.net in ('sam', 'mobile_sam'):
                        se, de = net.prompt_encoder(
                            points=pt,
                            text_embedding=text_embedding,
                            boxes=None,
                            masks=prev_masksw,
                        )
                    elif args.net == 'efficient_sam':
                        coords_torch, labels_torch = transform_prompt(coords_torch, labels_torch, h, w)
                        se = net.prompt_encoder(
                            coords=coords_torch,
                            labels=labels_torch,
                        )

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

                    masks_hq = F.interpolate(masks_hq, size=(args.out_size, args.out_size))
                    loss     = criterion_G(masks_hq, masks)
                    tot     += loss

                    # Periodically save visualisation images to disk
                    if args.vis and ind % args.vis_val == 0:
                        namecat = 'Test'
                        for na in name:
                            namecat += na.split('/')[-1].split('.')[0] + '+'
                        visualization(
                            imgs, masks_hq, masks, show_prev,
                            os.path.join(args.path_helper['sample_path'],
                                         f'epoch+{epoch}{namecat}.jpg'),
                            mode='backbone', reverse=False, points=showp, text=text, pt_labels=point_labels,
                        )

                    temp    = eval_seg(masks_hq, masks, threshold)
                    mix_res = tuple(sum(a) for a in zip(mix_res, temp))

            pbar.update()

    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot / n_val, tuple(a / n_val for a in mix_res)


def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
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

    return rescaled_batched_points,label


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