"""dataset_serval.py — Serval 혈관 분할 데이터셋."""

import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset

from .dataset_utils import *


class FireflyDataset(Dataset):
    """
    Firefly 혈관 분할 데이터셋.

    텍스트 가이드 augmentation을 위해 5가지 case와 2가지 range를 조합한다:
        case 0  Make thinner      두꺼운 혈관 → 얇게
        case 1  Make thicker      얇은 혈관 → 두껍게
        case 2  Extend            혈관 연장
        case 3  Remove            과다 분할 제거
        case 4  Make a connection 분기점 연결

        range_num 1  로컬  연속 세그먼트 범위 선택
        range_num 2  전역  전체 마스크 사용 (case 0) 또는 연장 전체 처리 (case 2)
                          ※ case 2에서 range_num 2는 0 또는 1로 재샘플링됨
                            → range_num 0은 case 2에서만 도달 가능

    파일명 규칙 (다른 데이터셋과 다름):
        이미지:     {stem}.*         (data_GT/)
        GT 마스크:  {stem}_mask.*    (mask/)
        분기점:     {stem}_branching.* (branching/)
        두꺼운:     {stem}_thick.*   (thick/)
        얇은:       {stem}_new_thin.* (new_thin/)
        dis_thick:  {stem}_disconn_thick.* (disconn_thick/)
        over_seg:   {stem}_over.*   (over_seg/)

    디렉토리 구조:
        dataset_path/{split}/
            data_GT/     # 원본 이미지
            mask/        # GT 마스크
            thick/       # 두꺼운 혈관 마스크
            new_thin/    # 얇은 혈관 마스크
            branching/   # 혈관 분기점 마스크
            disconn_thick/ # 일부 세그먼트가 제거된 두꺼운 혈관 마스크
            disconn_thin/  # 일부 세그먼트가 제거된 얇은 혈관 마스크
            over_seg/    # 과다 분할 예측 마스크
    """


    def __init__(self, args, dataset_path, split='train',
                 transform=None, transform_msk=None, prompt='click'):
        self.dataset_path  = Path(dataset_path)
        self._split_path   = self.dataset_path / split

        self._images_path    = self._split_path / 'data_GT'
        self._origin_path    = self._split_path / 'mask'
        self._thick_path     = self._split_path / 'thick'
        self._thin_path      = self._split_path / 'new_thin'
        self._branching_path = self._split_path / 'branching'
        self._dis_thick_path = self._split_path / 'disconn_thick'
        self._dis_thin_path  = self._split_path / 'disconn_thin'
        self._over_seg_path  = self._split_path / 'over_seg'

        # stem → Path 딕셔너리: 확장자 무관 빠른 조회
        self.dataset_samples  = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._origin_paths    = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._thick_paths     = {x.stem: x for x in self._thick_path.glob('*.*')}
        self._thin_paths      = {x.stem: x for x in self._thin_path.glob('*.*')}
        self._branching_paths = {x.stem: x for x in self._branching_path.glob('*.*')}
        self._dis_thick_paths = {x.stem: x for x in self._dis_thick_path.glob('*.*')}
        self._dis_thin_paths  = {x.stem: x for x in self._dis_thin_path.glob('*.*')}
        self._over_seg_paths  = {x.stem: x for x in self._over_seg_path.glob('*.*')}

        self.transform     = transform
        self.transform_msk = transform_msk
        self.prompt        = prompt
        self.img_size      = args.image_size
        self.out_size      = args.out_size
        self.train_stage    = args.train_stage

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        image_name     = self.dataset_samples[index]
        stem           = image_name.split('.')[0]
        image_path     = str(self._images_path / image_name)
        # Serval 파일명 규칙: 각 마스크 파일에 _mask, _branching 등의 접미사가 붙음
        origin_path    = str(self._origin_paths[stem + '_mask'])
        branching_path = str(self._branching_paths[stem + '_branching'])

        # ── 텍스트 가이드 케이스 + 편집 범위 랜덤 선택 ──────────────────────
        case_num  = random.choice([0, 1, 2, 3, 4])
        # range_num ∈ {1, 2}: Serval은 0이 없으나
        # case 2에서 range_num 2 → {0, 1}로 재샘플링하면 range_num 0도 가능
        range_num = random.choice([1, 2])

        if case_num == 0:
            mask_path = (str(self._thick_paths[stem + '_thick']) if range_num == 2
                         else str(self._dis_thick_paths[stem + '_disconn_thick']))
            text = 'Make thinner'
        elif case_num == 1:
            mask_path = str(self._thin_paths[stem + '_new_thin'])
            text      = 'Make thicker'
        elif case_num == 2:
            if range_num == 2:
                range_num = random.choice([0, 1])
            mask_path = origin_path
            text      = 'Extend'
        elif case_num == 3:
            mask_path = str(self._over_seg_paths[stem + '_over'])
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

        modify_mask    = cv2.imread(mask_path)[:, :, 0].astype(np.float32) / 255.0
        branching_mask = cv2.imread(branching_path)[:, :, 0].astype(np.float32) / 255.0

        point_label = 1
        crop_size   = (self.out_size, self.out_size)

        # ── 랜덤 크롭 (transforms.RandomCrop) ───────────────────────────────
        if self.transform:
            image = self.transform(image)

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        gt_mask        = torch.as_tensor(gt_mask)
        modify_mask    = torch.as_tensor(modify_mask)
        branching_mask = torch.as_tensor(branching_mask)

        image          = transforms.functional.crop(image, i, j, h, w)
        gt_mask        = transforms.functional.crop(gt_mask, i, j, h, w)
        modify_mask    = transforms.functional.crop(modify_mask, i, j, h, w)
        branching_mask = transforms.functional.crop(branching_mask, i, j, h, w)

        # Tensor → numpy (cv2 기반 연산을 위해)
        gt_mask          = np.array(gt_mask)
        modify_mask      = (np.array(modify_mask) * 255.0).astype(np.uint8)
        modify_copy_mask = modify_mask.copy()
        modify_copy_mask[modify_copy_mask >= 50] = 255
        modify_copy_mask[modify_copy_mask <  50] = 0

        # ── range_num별 세그먼트 augmentation ───────────────────────────────
        # range_num 0: case 2에서 range_num 2 → 재샘플링으로만 도달 가능
        #              단일 세그먼트 선택 (포인트 수준)
        if range_num == 0:
            if case_num == 1:
                gt_u8, branching_u8 = self._to_uint8_binary(gt_mask, branching_mask)
                dis_gt, dis_modify, branching_gt = self._split_by_branching(
                    gt_u8, modify_mask, branching_u8
                )
                gt_retvals, gt_labels, _, _ = cv2.connectedComponentsWithStats(dis_gt)
                sampling_num = 1 if gt_retvals == 1 else random.randint(1, gt_retvals - 1)

                for i in range(1, gt_retvals):
                    if i == sampling_num:
                        dis_gt[gt_labels == i] = dis_modify[gt_labels == i]

                modify_mask = np.clip(
                    dis_gt.astype(np.uint16) + branching_gt.astype(np.uint16) * 255,
                    0, 255
                ).astype(np.uint8)
                gt_mask = gt_u8.astype(np.float32) / 255.0

            elif case_num == 2:
                branching_u8 = self._branching_to_uint8(branching_mask)
                modify_mask  = np.clip(modify_mask - branching_u8, 0, None)
                retvals, labels, _, _ = cv2.connectedComponentsWithStats(modify_mask)
                sampling_num = 1 if retvals == 1 else random.randint(1, retvals - 1)
                modify_mask[labels != sampling_num] = 0

            elif case_num == 3:
                pass

            else:
                retvals, labels, _, _ = cv2.connectedComponentsWithStats(modify_copy_mask)
                sampling_num = 1 if retvals == 1 else random.randint(1, retvals - 1)
                modify_mask[labels != sampling_num] = 0

        elif range_num == 1:
            # 로컬: 연속된 세그먼트 범위를 편집
            if case_num == 1:
                gt_u8, branching_u8 = self._to_uint8_binary(gt_mask, branching_mask)
                dis_gt, dis_modify, branching_gt = self._split_by_branching(
                    gt_u8, modify_mask, branching_u8
                )
                gt_retvals, gt_labels, _, _ = cv2.connectedComponentsWithStats(dis_gt)
                sampling_num = random.randint(1, gt_retvals)
                random_range = random.choice([1, 2])

                for i in range(1, gt_retvals):
                    if sampling_num - random_range <= i <= sampling_num + random_range:
                        dis_gt[gt_labels == i] = dis_modify[gt_labels == i]

                modify_mask = np.clip(
                    dis_gt.astype(np.uint16) + branching_gt.astype(np.uint16) * 255,
                    0, 255
                ).astype(np.uint8)
                gt_mask = gt_u8.astype(np.float32) / 255.0

            elif case_num == 3:
                pass

            else:
                if case_num == 2:
                    branching_u8 = self._branching_to_uint8(branching_mask)
                    modify_mask  = np.clip(modify_mask - branching_u8, 0, None)
                    modify_copy_mask = modify_mask

                retvals, labels, _, _ = cv2.connectedComponentsWithStats(modify_copy_mask)
                if retvals >= 6:
                    sampling_num = random.randint(3, retvals - 3)
                    for i in range(sampling_num - 2, sampling_num + 3):
                        labels[labels == i] = retvals + 1
                else:
                    for i in range(1, retvals):
                        labels[labels == i] = retvals + 1
                modify_mask[labels < retvals + 1] = 0

        else:
            # range_num == 2: 전역
            # case 0: 이미 thick 전체 마스크를 로드했으므로 추가 처리 없음
            # case 2: 위에서 range_num이 0 또는 1로 재샘플링되어 도달 불가
            # 그 외 (3, 4): 마스크 그대로 사용
            pass

        modify_mask = modify_mask.astype(np.float32) / 255.0

        # ── prev_mask / click_mask 설정 ──────────────────────────────────────
        if case_num == 0:
            prev_mask  = np.clip(gt_mask + modify_mask, 0, 1.0)
            prev_mask[prev_mask > 0] = 1.0
            click_mask = np.clip(modify_mask - gt_mask, 0, None)
        elif case_num == 1:
            prev_mask  = modify_mask
            modify_mask = np.clip(gt_mask - modify_mask, 0, None)
            click_mask = modify_mask
        elif case_num == 2:
            prev_mask  = np.clip(gt_mask - modify_mask, 0, None)
            click_mask = modify_mask
        elif case_num == 3:
            prev_mask  = np.clip(gt_mask + modify_mask, 0, 1.0)
            prev_mask[prev_mask > 0] = 1.0
            modify_mask = np.clip(prev_mask - gt_mask, 0, None)
            click_mask  = modify_mask
        else:  # case 4
            prev_mask  = np.clip(gt_mask - modify_mask, 0, None)
            click_mask = modify_mask

        gt_mask   = torch.as_tensor(gt_mask)
        prev_mask = torch.as_tensor(prev_mask)

        # ── 클릭 프롬프트 생성 ───────────────────────────────────────────────
        if self.prompt == 'click':
            point_label, pt = random_click(click_mask, point_label)
            pt = (pt / self.out_size) * self.img_size
            if case_num in (0, 3):
                point_label = 0

        # case 2: 1/3 확률로 prev_mask를 zero로 초기화 (초기 마스크 없이 연장 시작)
        if case_num == 2:
            if random.choice([0, 1, 2]) == 0:
                prev_mask = torch.zeros_like(gt_mask)

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
        

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_uint8_binary(gt_mask_f32, branching_tensor):
        gt_u8     = (gt_mask_f32 * 255.0).astype(np.uint8)
        branch_u8 = (np.array(branching_tensor) * 255.0).astype(np.uint8)
        branch_u8[branch_u8 >= 50] = 255
        branch_u8[branch_u8 <  50] = 0
        return gt_u8, branch_u8

    @staticmethod
    def _branching_to_uint8(branching_tensor):
        branch_u8 = (np.array(branching_tensor) * 255.0).astype(np.uint8)
        branch_u8[branch_u8 >= 50] = 255
        branch_u8[branch_u8 <  50] = 0
        return branch_u8

    @staticmethod
    def _split_by_branching(gt_u8, modify_u8, branching_u8):
        """분기점을 경계로 gt와 modify_mask를 세그먼트 단위로 분리."""
        dis_gt     = np.clip(gt_u8.astype(np.int16)     - branching_u8, 0, None).astype(np.uint8)
        dis_modify = np.clip(modify_u8.astype(np.int16) - branching_u8, 0, None).astype(np.uint8)
        branching_gt = np.bitwise_and(gt_u8, branching_u8)
        return dis_gt, dis_modify, branching_gt

class Test_FireflyDataset(Dataset):
    """Test dataset for SERVAL retinal segmentation; returns full-image context alongside the crop."""

    def __init__(self, args, dataset_path, split='train',
                 transform=None, transform_msk=None, prompt='click'):
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path  = self._split_path / 'data_GT'
        self._origin_path  = self._split_path / 'new_mask'
        self._pred_path    = self._split_path / 'inference_results_pp'
        self._optic_path   = self._split_path / 'optic'

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._origin_paths = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._pred_paths   = {x.stem: x for x in self._pred_path.glob('*.*')}
        self._optic_paths  = {x.stem: x for x in self._optic_path.glob('*.*')}

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(str(self._origin_paths[stem]), cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 1, 1, cv2.THRESH_BINARY)
        gt_mask  = gt_mask.astype(np.float32)

        optic_mask = cv2.imread(str(self._optic_paths[stem]))[:, :, 0].astype(np.float32)
        if np.max(optic_mask) > 100:
            optic_mask /= 255.0

        prev_mask  = cv2.imread(str(self._pred_paths[stem]))[:, :, 0].astype(np.float32)


        gt_optic   = np.clip(gt_mask - optic_mask, 0, None)
        prev_optic = np.clip(prev_mask - optic_mask, 0, None)

        point_label, pt = val_get_next_click(gt_optic, prev_optic > self.threshold)


        image     = crop_with_padding(image, pt, (self.out_size,self.out_size))
        gt_mask   = crop_with_padding(gt_mask, pt,  (self.out_size,self.out_size))
        prev_mask = crop_with_padding(prev_mask, pt,  (self.out_size,self.out_size))

        pt = np.array([(self.out_size / 2) - 1, (self.out_size / 2) - 1],
                      dtype=np.float32)
        pt = (pt / np.array(self.out_size)) * self.img_size

        if self.transform:
            image        = self.transform(image)

        gt_mask    = torch.as_tensor(gt_mask)
        prev_mask  = torch.as_tensor(prev_mask)

        return {
            "image": image,
            "mask": prev_mask.unsqueeze(0),
            "gt": gt_mask.unsqueeze(0),
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": {"filename_or_obj": image_name},
        }

