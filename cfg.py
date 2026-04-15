import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training configuration for segmentation / multimodal models"
    )

    # ===================== Basic Settings =====================
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--net', type=str, default='sam',
                        help='Main network type (e.g., sam)')

    parser.add_argument('--baseline', type=str, default='unet',
                        help='Baseline model architecture')

    parser.add_argument('--encoder', type=str, default='vit_l',
                        help='Encoder backbone type (e.g., vit_h, vit_l, vit_b)')

    parser.add_argument('--seg_net', type=str, default='transunet',
                        help='Segmentation network type')

    parser.add_argument('--mod', type=str, default='',
                        help='Mode of operation: seg, cls, val_ad, etc.')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='Experiment name (used for logging/saving)')

    parser.add_argument('--type', type=str, default='map',
                        help='Condition type: ave, rand, rand_map')

    # ===================== Visualization =====================
    parser.add_argument('--vis', type=int, default=0,
                        help='Visualization frequency during training')

    parser.add_argument('--vis_val', type=int, default=0,
                        help='Visualization frequency during validation')

    # ===================== Flags =====================
    parser.add_argument('--reverse', action='store_true',
                        help='Enable adversarial reverse training')

    parser.add_argument('--gpu', type=bool, default=True, 
                        help='use gpu or not')

    parser.add_argument('--s', type=bool, default=True, 
                        help='whether shuffle the dataset')

    parser.add_argument('--thd', action='store_true',
                        help='Enable 3D processing mode')
    
    parser.add_argument('--eval', action='store_true',
                        help='Enable evaluation stage')

    # ===================== Training =====================
    parser.add_argument('--train_stage', type=str, default='',
                        help='Training step or pipeline stage')
    
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency (in epochs)')

    parser.add_argument('--gpu_device', type=int, default=0,
                        help='GPU device index')

    parser.add_argument('--sim_gpu', type=int, default=0,
                        help='GPU index for similarity computation')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Total number of training epochs')

    parser.add_argument('--epoch_ini', type=int, default=0,
                        help='Starting epoch (for resume training)')

    parser.add_argument('--b', type=int, default=1,
                        help='Batch size')

    parser.add_argument('--w', type=int, default=4,
                        help='Number of data loading workers')

    parser.add_argument('--warm', type=int, default=1,
                        help='Warm-up epochs')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Initial learning rate')

    parser.add_argument('--imp_lr', type=float, default=3e-4,
                        help='Implicit learning rate')

    # ===================== Model Parameters =====================
    parser.add_argument('--num_clicks', type=int, default=10,
                        help='Number of interaction clicks (for interactive segmentation)')

    parser.add_argument('--image_size', type=int, default=1024,
                        help='Input image size')

    parser.add_argument('--out_size', type=int, default=256,
                        help='Output resolution')

    parser.add_argument('--patch_size', type=int, default=2,
                        help='Patch size for transformer')

    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')

    parser.add_argument('--depth', type=int, default=1,
                        help='Transformer depth')

    parser.add_argument('--heads', type=int, default=16,
                        help='Number of attention heads')

    parser.add_argument('--mlp_dim', type=int, default=1024,
                        help='MLP hidden dimension')

    parser.add_argument('--uinch', type=int, default=1,
                        help='Number of input channels')

    parser.add_argument('--num_cls', type=int, default=5,
                        help='Number of classes predicted by the text predict model')

    parser.add_argument('--multimask_output', type=int, default=1,
                        help='Number of output masks (multi-class segmentation)')

    parser.add_argument('--mid_dim', type=int, default=None,
                        help='Middle dimension for adapter or LoRA rank')
    
    parser.add_argument('--model-type', type=str, default='vit_l',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='Type of SAM model')

    # ===================== Dataset =====================
    parser.add_argument('--data_path', type=str,
                        default= '',
                        help='Path to dataset directory')

    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset name')


    # ===================== Pretrained Weights =====================
    parser.add_argument('--weights', type=str,
                        default=0,
                        help='Path to main model weights')


    parser.add_argument('--tpp_weights', type=str,
                        default=0,
                        help='Path to text predictor weights')

    parser.add_argument('--base_weights', type=str, default=0,
                        help='Path to baseline model weights')

    parser.add_argument('--sim_weights', type=str, default=0,
                        help='Path to similarity model weights')

    parser.add_argument('--sam_ckpt', type=str,
                        default='',
                        help='Path to SAM checkpoint')

    parser.add_argument('--decoder_path', type=str,
                        default='',
                        help='Path to HQ decoder checkpoint')

    parser.add_argument('--clip_ckpt', type=str,
                        default='config/clip',
                        help='Path to CLIP checkpoint')

    # ===================== 3D Settings =====================
    parser.add_argument('--chunk', type=int, default=96,
                        help='Depth of cropped volume')

    parser.add_argument('--num_sample', type=int, default=4,
                        help='Number of positive/negative samples')

    parser.add_argument('--roi_size', type=int, default=96,
                        help='ROI resolution')

    parser.add_argument('--evl_chunk', type=int, default=None,
                        help='Evaluation chunk size')

    # ===================== Distributed =====================
    parser.add_argument('--distributed', type=str, default='none',
                        help='Distributed training mode (e.g., none, ddp)')



    return parser.parse_args()

