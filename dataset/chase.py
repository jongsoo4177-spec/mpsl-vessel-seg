"""dataset_chase.py — CHASE 망막 혈관 분할 데이터셋."""

import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset

from .dataset_utils import *

class CHASEDataset(Dataset):
    """
    CHASE 망막 혈관 분할 데이터셋.

    텍스트 가이드 augmentation을 위해 5가지 case와 3가지 range를 조합한다:
        case 0  Make thinner      두꺼운 혈관 → 얇게  (dis_thick 또는 thick)
        case 1  Make thicker      얇은 혈관 → 두껍게  (thin 마스크)
        case 2  Extend            혈관 연장            (origin 마스크)
        case 3  Remove            과다 분할 제거       (over_seg 마스크)
        case 4  Make a connection 분기점 연결          (branching 마스크)

        range_num 0  포인트 수준  단일 세그먼트를 편집 대상으로 선택
        range_num 1  로컬         연속된 세그먼트 범위를 편집 대상으로 선택
        range_num 2  전역         세그먼트 전체 사용 (case 0에서 thick 마스크 활용)
                                  ※ case 2에서는 range_num 2가 0 또는 1로 재샘플링됨

    디렉토리 구조:
        dataset_path/{split}/
            Image/        # 원본 이미지
            Mask/         # GT 마스크
            Thick/        # 두꺼운 혈관 마스크
            Thin/         # 얇은 혈관 마스크
            Branching/    # 혈관 분기점 마스크
            disconn_thick/# 일부 세그먼트가 제거된 두꺼운 혈관 마스크
            disconn_thin/ # 일부 세그먼트가 제거된 얇은 혈관 마스크
            res_pred/     # 과다 분할 예측 마스크
    """


    def __init__(self, args, dataset_path, split='train',
                 transform=None, transform_msk=None, prompt='click'):
        self.dataset_path  = Path(dataset_path)
        self._split_path   = self.dataset_path / split

        self._images_path    = self._split_path / 'Image'
        self._origin_path    = self._split_path / 'Mask'
        self._thick_path     = self._split_path / 'Thick'
        self._thin_path      = self._split_path / 'Thin'
        self._branching_path = self._split_path / 'Branching'
        self._dis_thick_path = self._split_path / 'disconn_thick'
        self._dis_thin_path  = self._split_path / 'disconn_thin'
        self._over_seg_path  = self._split_path / 'res_pred'

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
        self.threshold     = 0.49
        self.train_stage    = args.train_stage
    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        image_name     = self.dataset_samples[index]
        stem           = image_name.split('.')[0]
        image_path     = str(self._images_path / image_name)
        origin_path    = str(self._origin_paths[stem])
        branching_path = str(self._branching_paths[stem])

        # ── 텍스트 가이드 케이스 + 편집 범위 랜덤 선택 ──────────────────────
        case_num  = random.choice([0, 1, 2, 3, 4])
        range_num = random.choice([0, 1, 2])

        if case_num == 0:
            # range 2: thick 전체 사용, 나머지: dis_thick(일부 제거 버전)
            mask_path = (str(self._thick_paths[stem]) if range_num == 2
                         else str(self._dis_thick_paths[stem]))
            text = 'Make thinner'
        elif case_num == 1:
            mask_path = str(self._thin_paths[stem])
            text      = 'Make thicker'
        elif case_num == 2:
            # range 2는 "연장" 케이스에서 의미가 없으므로 0 또는 1로 재샘플링
            if range_num == 2:
                range_num = random.choice([0, 1])
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

        modify_mask    = cv2.imread(mask_path)[:, :, 0].astype(np.float32) / 255.0
        branching_mask = cv2.imread(branching_path)[:, :, 0].astype(np.float32) / 255.0

        point_label = 1
        crop_size   = (self.out_size, self.out_size)

        # ── 크롭 전략 ────────────────────────────────────────────────────────
        # case 1 (Make thicker): gt에만 있는 영역(추가 필요 부분)에 클릭을 구한 뒤
        #                        해당 좌표 중심으로 crop_with_padding 적용
        # 나머지 case: transforms.RandomCrop 적용
        if case_num == 1:
            click_mask = np.clip(gt_mask - (modify_mask > self.threshold), 0, None)
            point_label, pt = random_click(click_mask, point_label)
            pt = tuple(pt)

            image          = crop_with_padding(image, pt, crop_size)
            gt_mask        = crop_with_padding(gt_mask, pt, crop_size)
            modify_mask    = crop_with_padding(modify_mask, pt, crop_size)
            branching_mask = crop_with_padding(branching_mask, pt, crop_size)

            if self.transform:
                image = self.transform(image)

            gt_mask        = torch.as_tensor(gt_mask)
            modify_mask    = torch.as_tensor(modify_mask)
            branching_mask = torch.as_tensor(branching_mask)
        else:
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
        if range_num == 0:
            # 포인트 수준: 단일 세그먼트만 편집
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
                pass  # Remove: 마스크 그대로 사용

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
                pass  # Remove: 마스크 그대로 사용

            else:
                if case_num == 2:
                    branching_u8 = self._branching_to_uint8(branching_mask)
                    modify_mask  = np.clip(modify_mask - branching_u8, 0, None)
                    modify_copy_mask = modify_mask

                retvals, labels, _, _ = cv2.connectedComponentsWithStats(modify_copy_mask)
                # 컴포넌트가 6개 이상이면 중간 범위, 아니면 전체 선택
                if retvals >= 6:
                    sampling_num = random.randint(3, retvals - 3)
                    for i in range(sampling_num - 2, sampling_num + 3):
                        labels[labels == i] = retvals + 1
                else:
                    for i in range(1, retvals):
                        labels[labels == i] = retvals + 1
                modify_mask[labels < retvals + 1] = 0

        else:
            # range_num == 2: 전역 — case 2에서는 도달 불가 (위에서 재샘플링됨)
            # case 0: 이미 mask_path에서 thick 전체 마스크를 로드했으므로 추가 처리 없음
            # 그 외 case (3, 4): 마스크 그대로 사용
            pass

        modify_mask = modify_mask.astype(np.float32) / 255.0

        # ── prev_mask / click_mask 설정 ──────────────────────────────────────
        # prev_mask:  모델에 입력되는 "편집 전 현재 상태" 마스크
        # click_mask: 클릭 프롬프트를 생성할 영역 (편집이 필요한 곳)
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
        """float32 gt_mask와 branching Tensor를 uint8 이진 마스크로 변환."""
        gt_u8       = (gt_mask_f32 * 255.0).astype(np.uint8)
        branch_u8   = (np.array(branching_tensor) * 255.0).astype(np.uint8)
        branch_u8[branch_u8 >= 50] = 255
        branch_u8[branch_u8 <  50] = 0
        return gt_u8, branch_u8

    @staticmethod
    def _branching_to_uint8(branching_tensor):
        """branching Tensor를 uint8 이진 마스크로 변환."""
        branch_u8 = (np.array(branching_tensor) * 255.0).astype(np.uint8)
        branch_u8[branch_u8 >= 50] = 255
        branch_u8[branch_u8 <  50] = 0
        return branch_u8

    @staticmethod
    def _split_by_branching(gt_u8, modify_u8, branching_u8):
        """
        분기점을 경계로 세그먼트를 분리한다.

        Returns:
            dis_gt:      gt에서 분기점을 제외한 세그먼트
            dis_modify:  modify_mask에서 분기점을 제외한 세그먼트
            branching_gt: gt와 분기점의 교집합 (보존할 분기 영역)
        """
        dis_gt     = np.clip(gt_u8.astype(np.int16)     - branching_u8, 0, None).astype(np.uint8)
        dis_modify = np.clip(modify_u8.astype(np.int16) - branching_u8, 0, None).astype(np.uint8)
        branching_gt = np.bitwise_and(gt_u8, branching_u8)
        return dis_gt, dis_modify, branching_gt


class Test_CHASEDataset(Dataset):
    """Test dataset for CHASE retinal vessel segmentation with optic-disc masking."""

    def __init__(self, args, dataset_path, split='train',
                 transform=None, transform_msk=None, prompt='click'):
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path  = self._split_path / 'Image'
        self._origin_path  = self._split_path / 'Mask'
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
        gt_mask = gt_mask.astype(np.float32)

        optic_mask = cv2.imread(str(self._optic_paths[stem]))[:, :, 0].astype(np.float32)
        if np.max(optic_mask) > 100:
            optic_mask /= 255.0

        prev_mask = cv2.imread(str(self._pred_paths[stem]))[:, :, 0].astype(np.float32)

        # Exclude optic-disc region from click selection
        gt_optic   = np.clip(gt_mask - optic_mask, 0, None)
        prev_optic = np.clip(prev_mask - optic_mask, 0, None)

        point_label, pt = val_get_next_click(gt_optic, prev_optic > self.threshold)

        image     = crop_with_padding(image, pt,  (self.out_size,self.out_size))
        gt_mask   = crop_with_padding(gt_mask, pt,  (self.out_size,self.out_size))
        prev_mask = crop_with_padding(prev_mask, pt,  (self.out_size,self.out_size))

        # After center-crop the click lands at the middle of the crop
        pt = np.array([(self.out_size / 2) - 1, (self.out_size / 2) - 1],
                      dtype=np.float32)
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

