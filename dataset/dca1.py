"""dataset_dca1.py — DCA1 관상동맥 조영술 혈관 분할 데이터셋."""

import random

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .dataset_utils import *


class DCA1Dataset(Dataset):
    """
    DCA1(Digital Coronary Angiography) 관상동맥 혈관 분할 데이터셋.

    텍스트 가이드 augmentation을 위해 5가지 case를 랜덤 선택한다:
        case 0  Make thinner      두꺼운 혈관 → 얇게  (dis_thick 마스크)
        case 1  Make thicker      얇은 혈관 → 두껍게  (thin 마스크)
        case 2  Extend            혈관 연장            (origin 마스크)
        case 3  Remove            과다 분할 제거       (over_seg 마스크)
        case 4  Make a connection 분기점 연결          (branching 마스크)

    디렉토리 구조:
        dataset_path/{split}/
            Image_png/            # 원본 이미지 (PNG)
            Mask_png/             # GT 마스크 (PNG, 파일명: {stem}_gt.*)
            Thick/                # 두꺼운 혈관 마스크
            Thin/                 # 얇은 혈관 마스크
            Branching/            # 혈관 분기점 마스크
            disconn_thick/        # 일부 세그먼트가 제거된 두꺼운 혈관 마스크
            disconn_thin/         # 일부 세그먼트가 제거된 얇은 혈관 마스크
            inference_results_pp/ # 과다 분할 예측 마스크
    """

    def __init__(self, args, dataset_path, split='train',
                 transform=None, transform_msk=None, prompt='click'):
        self.dataset_path  = Path(dataset_path)
        self._split_path   = self.dataset_path / split

        self._images_path    = self._split_path / 'Image_png'
        self._origin_path    = self._split_path / 'Mask_png'
        self._thick_path     = self._split_path / 'Thick'
        self._thin_path      = self._split_path / 'Thin'
        self._branching_path = self._split_path / 'Branching'
        self._dis_thick_path = self._split_path / 'disconn_thick'
        self._dis_thin_path  = self._split_path / 'disconn_thin'
        self._over_seg_path  = self._split_path / 'inference_results_pp'

        # stem → Path 딕셔너리: 확장자 무관 빠른 조회
        self.dataset_samples   = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._origin_paths     = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._thick_paths      = {x.stem: x for x in self._thick_path.glob('*.*')}
        self._thin_paths       = {x.stem: x for x in self._thin_path.glob('*.*')}
        self._branching_paths  = {x.stem: x for x in self._branching_path.glob('*.*')}
        self._dis_thick_paths  = {x.stem: x for x in self._dis_thick_path.glob('*.*')}
        self._dis_thin_paths   = {x.stem: x for x in self._dis_thin_path.glob('*.*')}
        self._over_seg_paths   = {x.stem: x for x in self._over_seg_path.glob('*.*')}

        self.transform     = transform
        self.transform_msk = transform_msk
        self.prompt        = prompt
        self.img_size      = args.image_size
        self.out_size      = args.out_size
        self.threshold     = 0.49
        self.train_stage    = args.train_stage

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        image_name     = self.dataset_samples[index]
        stem           = image_name.split('.')[0]
        image_path     = str(self._images_path / image_name)
        # DCA1 마스크 파일명 규칙: {stem}_gt.*
        origin_path    = str(self._origin_paths[stem + '_gt'])
        branching_path = str(self._branching_paths[stem])

        # ── 텍스트 가이드 케이스 랜덤 선택 ──────────────────────────────────
        case_num = random.choice([0, 1, 2, 3, 4])

        if case_num == 0:
            mask_path = str(self._dis_thick_paths[stem])
            text      = 'Make thinner'
        elif case_num == 1:
            mask_path = str(self._thin_paths[stem])
            text      = 'Make thicker'
        elif case_num == 2:
            mask_path = origin_path
            text      = 'Extend'
        elif case_num == 3:
            mask_path = str(self._over_seg_paths[stem])
            text      = 'Remove'
        else:  # case 4
            mask_path = branching_path
            text      = 'Make a connection'

        # ── 이미지 및 마스크 로드 ────────────────────────────────────────────
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 1, 1, cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)

        modify_mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32)
        if modify_mask.max() > 100:
            modify_mask /= 255.0

        branching_mask = cv2.imread(branching_path)[:, :, 0].astype(np.float32)
        branching_mask /= 255.0

        # ── 랜덤 크롭 (out_size × out_size) ─────────────────────────────────
        ch = cw = self.out_size
        h, w    = image.shape[:2]
        top     = np.random.randint(0, h - ch + 1)
        left    = np.random.randint(0, w - cw + 1)

        image          = image[top:top+ch, left:left+cw]
        gt_mask        = gt_mask[top:top+ch, left:left+cw]
        modify_mask    = modify_mask[top:top+ch, left:left+cw]
        branching_mask = branching_mask[top:top+ch, left:left+cw]

        point_label = 1

        # modify_mask 이진화 복사본 (connected component 분석용)
        modify_copy_mask = (modify_mask >= 0.5).astype(np.float32)

        # ── Case별 마스크 augmentation ───────────────────────────────────────
        if case_num == 1:
            # 얇게 → 두껍게:
            # 분기점을 경계로 세그먼트를 분리한 뒤 일부를 modify_mask 세그먼트로 교체
            #   dis_gt:      gt에서 분기점을 제외한 세그먼트 영역
            #   dis_modify:  modify_mask에서 분기점을 제외한 세그먼트 영역
            #   branching_gt: gt와 분기점의 교집합 (보존해야 할 분기 영역)
            branching_mask = (branching_mask >= 0.5).astype(np.float32)

            dis_gt       = np.clip(gt_mask     - branching_mask, 0, None)
            dis_modify   = np.clip(modify_mask - branching_mask, 0, None)
            branching_gt = np.bitwise_and(
                gt_mask.astype(np.uint8), branching_mask.astype(np.uint8)
            )

            gt_retvals, gt_labels, _, _ = cv2.connectedComponentsWithStats(
                dis_gt.astype(np.uint8)
            )
            sampling_list = list(range(1, gt_retvals))
            sampling_num  = random.randint(1, gt_retvals)

            if sampling_num != gt_retvals:
                for num in random.sample(sampling_list, sampling_num):
                    dis_gt[gt_labels == num] = dis_modify[gt_labels == num]
                modify_mask = np.clip(dis_gt + branching_gt, 0, 1.0)

        elif case_num == 2:
            # 연장: 분기점 제거 후 최대 2개 세그먼트만 남겨 연장 대상 선택
            # (OCTA와 달리 sampling_num 최대값을 2로 제한)
            branching_mask = (branching_mask >= 0.5).astype(np.float32)
            modify_mask    = np.clip(modify_mask - branching_mask, 0, None)
            modify_copy_mask = modify_mask

            retvals, labels, _, _ = cv2.connectedComponentsWithStats(
                modify_copy_mask.astype(np.uint8)
            )
            sampling_list = list(range(1, retvals))
            sampling_num  = random.randint(1, 2)      # DCA1 전용: 최대 2개 세그먼트

            if len(sampling_list) >= sampling_num:
                for num in random.sample(sampling_list, sampling_num):
                    labels[labels == num] = retvals + 1
                modify_mask[labels < retvals + 1] = 0

        elif case_num in (0, 4):
            # 얇게(0) / 연결(4): 일부 세그먼트만 남기고 나머지 제거
            retvals, labels, _, _ = cv2.connectedComponentsWithStats(
                modify_copy_mask.astype(np.uint8)
            )
            sampling_list = list(range(1, retvals))
            sampling_num  = random.randint(1, retvals)

            if sampling_num != retvals:
                for num in random.sample(sampling_list, sampling_num):
                    labels[labels == num] = retvals + 1
                modify_mask[labels < retvals + 1] = 0
            elif case_num == 0:
                # 제거할 세그먼트가 없으면 thick 마스크 전체를 대신 사용
                modify_mask = cv2.imread(
                    str(self._thick_paths[stem])
                )[:, :, 0].astype(np.float32) / 255.0
                modify_mask = modify_mask[top:top+ch, left:left+cw]

        # ── prev_mask / click_mask 설정 ──────────────────────────────────────
        # prev_mask:  모델에 입력되는 "편집 전 현재 상태" 마스크
        # click_mask: 클릭 프롬프트를 생성할 영역 (편집이 필요한 곳)
        if case_num == 0:
            prev_mask  = np.clip(gt_mask + modify_mask, 0, 1.0)
            prev_mask[prev_mask > 0] = 1.0
            click_mask = np.clip(modify_mask - gt_mask, 0, None)
        elif case_num == 1:
            prev_mask  = modify_mask
            click_mask = np.clip(gt_mask - modify_mask, 0, None)
        elif case_num == 2:
            prev_mask  = np.clip(gt_mask - modify_mask, 0, None)
            click_mask = modify_mask
        elif case_num == 3:
            prev_mask  = np.clip(gt_mask + modify_mask, 0, 1.0)
            modify_mask = np.clip(
                (prev_mask >= 0.3).astype(np.float32) - gt_mask, 0, None
            )
            click_mask = modify_mask
        else:  # case 4
            prev_mask  = np.clip(gt_mask - modify_mask, 0, None)
            click_mask = modify_mask

        gt_mask   = torch.as_tensor(gt_mask)
        prev_mask = torch.as_tensor(prev_mask)

        # ── 클릭 프롬프트 생성 ───────────────────────────────────────────────
        if self.prompt == 'click':
            point_label, pt = random_click(click_mask, point_label)
            pt = (pt / self.out_size) * self.img_size
            # Make thinner(0) / Remove(3)는 제거 방향 → negative click
            if case_num in (0, 3):
                point_label = 0

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(case_num)

        image_meta_dict = {'filename_or_obj': image_name}

        if self.train_stage == 'train_tpp':
            return {
                'image':           image,
                'mask':            prev_mask.unsqueeze(0),
                "gt":              gt_mask.unsqueeze(0),
                'label':           label,
                'p_label':         point_label,
                'pt':              pt,
                'image_meta_dict': image_meta_dict,
            }
        else:
            return {
                'image':           image,
                'mask':            prev_mask.unsqueeze(0),
                'gt':              gt_mask.unsqueeze(0),
                'p_label':         point_label,
                'pt':              pt,
                'text':            text,
                'image_meta_dict': image_meta_dict,
            }


class Test_DCA1Dataset(Dataset):
    """Test dataset for DCA1 coronary artery segmentation."""

    def __init__(self, args, dataset_path, split='train',
                 transform=None, transform_msk=None, prompt='click'):
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'Image_png'
        self._origin_path = self._split_path / 'Mask_png'
        self._pred_path   = self._split_path / 'inference_results_pp'

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._origin_paths = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._pred_paths   = {x.stem: x for x in self._pred_path.glob('*.*')}

        self.transform = transform
        self.transform_msk = transform_msk
        self.prompt = prompt
        self.img_size = args.image_size
        self.out_size = args.out_size
        self.threshold = 0.49

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        image_name = self.dataset_samples[index]
        stem = image_name.split('.')[0]

        image = cv2.imread(str(self._images_path / image_name))
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        gt_mask  = cv2.imread(str(self._origin_paths[stem + '_gt']), cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 1, 1, cv2.THRESH_BINARY)
        gt_mask  = gt_mask.astype(np.float32)

        prev_mask = cv2.imread(str(self._pred_paths[stem]))[:, :, 0].astype(np.float32)
        if np.max(prev_mask) > 200:
            prev_mask /= 255.0

        point_label, pt = get_next_click(gt_mask, prev_mask > self.threshold)

        h, w = image.shape[:2]
        crop_size = self.out_size[0]
        half_size = crop_size // 2

        x1 = max(0, min(pt[1] - half_size, w - crop_size))
        y1 = max(0, min(pt[0] - half_size, h - crop_size))
        x2, y2 = x1 + crop_size, y1 + crop_size

        image     = image[y1:y2, x1:x2]
        prev_mask = prev_mask[y1:y2, x1:x2]
        gt_mask   = gt_mask[y1:y2, x1:x2]

        pt = np.array(pt, dtype=np.float32)
        pt[0] -= y1
        pt[1] -= x1
        pt = (pt / np.array(self.out_size)) * self.img_size

        if self.transform:
            image = self.transform(image)

        gt_mask   = torch.as_tensor(gt_mask)
        prev_mask = torch.as_tensor(prev_mask)

        return {
            "image": image,
            "mask": prev_mask.unsqueeze(0),
            "gt": gt_mask.unsqueeze(0),
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": {"filename_or_obj": image_name},
        }