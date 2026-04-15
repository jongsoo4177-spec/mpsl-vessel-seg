"""dataset_arcade.py — ARCADE 관상동맥 혈관 분할 데이터셋."""

import json
import random

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .dataset_utils import *


class ARCADEDataset(Dataset):
    """
    ARCADE(Automatic Region-based Coronary Artery Disease diagnostics) 관상동맥 분할 데이터셋.

    JSON 어노테이션 파일에서 혈관 세그먼트 폴리곤 좌표를 읽어 마스크를 생성한다.
    4가지 case를 랜덤 선택한다 (OCTA/DCA1의 5가지와 달리):
        case 0  Make thinner  두꺼운 혈관 → 얇게  (thick 마스크)
        case 1  Make thicker  얇은 혈관 → 두껍게  (thin 마스크)
        case 2  Extend        혈관 연장            (JSON 폴리곤으로 세그먼트 선택)
        case 3  Remove        노이즈/과다 분할 제거 (JSON 폴리곤으로 세그먼트 선택)

    디렉토리 구조:
        dataset_path/{split}/
            images/                  # 원본 이미지
            mask/                    # GT 마스크
            thick/                   # 두꺼운 혈관 마스크 ({stem}_thick.*)
            thin/                    # 얇은 혈관 마스크 ({stem}_thin.*)
            connection/              # 혈관 연결 마스크 ({stem}_connection.*)
            inference_results_pp/    # 과다 분할 예측 마스크
            annotations/
                info.json            # 폴리곤 좌표 {filename → [coords, ...]}
                name.json            # 이미지 파일명 → 어노테이션 키 매핑
    """

    def __init__(self, args, dataset_path, split='train',
                 transform=None, transform_msk=None, prompt='click'):
        self.dataset_path  = Path(dataset_path)
        self._split_path   = self.dataset_path / split

        self._images_path    = self._split_path / 'images'
        self._origin_path    = self._split_path / 'mask'
        self._thick_path     = self._split_path / 'thick'
        self._thin_path      = self._split_path / 'thin'
        self._connection_path = self._split_path / 'connection'
        self._pred_path      = self._split_path / 'inference_results_pp'

        self.json_path       = self._split_path / 'annotations/info.json'
        self.file_name_path  = self._split_path / 'annotations/name.json'

        # stem → Path 딕셔너리: 확장자 무관 빠른 조회
        self.dataset_samples   = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._origin_paths     = {x.stem: x for x in self._origin_path.glob('*.*')}
        self._thick_paths      = {x.stem: x for x in self._thick_path.glob('*.*')}
        self._thin_paths       = {x.stem: x for x in self._thin_path.glob('*.*')}
        self._connection_paths = {x.stem: x for x in self._connection_path.glob('*.*')}
        self._pred_paths       = {x.stem: x for x in self._pred_path.glob('*.*')}

        self.transform     = transform
        self.transform_msk = transform_msk
        self.prompt        = prompt
        self.img_size      = args.image_size
        self.out_size      = args.out_size
        self.train_stage    = args.train_stage

        # JSON 어노테이션 로드
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)   # {png_filename → [[x,y,...], ...]}

        with open(self.file_name_path, 'r', encoding='utf-8') as f:
            self.name_data = json.load(f)   # {image_filename → annotation_key (파일명 기반)}

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        image_name  = self.dataset_samples[index]
        stem        = image_name.split('.')[0]
        image_path  = str(self._images_path / image_name)
        origin_path = str(self._origin_paths[stem])

        # ── 텍스트 가이드 케이스 랜덤 선택 ──────────────────────────────────
        case_num = random.choice([0, 1, 2, 3])

        if case_num == 0:
            mask_path = str(self._thick_paths[stem + '_thick'])
            text      = 'Make thinner'
        elif case_num == 1:
            mask_path = str(self._thin_paths[stem + '_thin'])
            text      = 'Make thicker'
        elif case_num == 2:
            mask_path = str(self._pred_paths[stem])
            text      = 'Extend'
        else:  # case 3
            mask_path = str(self._pred_paths[stem])
            text      = 'Remove'

        # ── 이미지 및 마스크 로드 ────────────────────────────────────────────
        image = cv2.imread(image_path)
        # 그레이스케일 이미지인 경우 3채널로 변환
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        gt_mask = cv2.imread(origin_path, cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 1, 1, cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)

        modify_mask = cv2.imread(mask_path)[:, :, 0].astype(np.float32)
        if np.max(modify_mask) > 200:
            modify_mask /= 255.0

        point_label = 1
        crop_size   = (self.out_size, self.out_size)

        # ── Case별 마스크 augmentation ───────────────────────────────────────
        if case_num == 0:
            # Make thinner: thick 마스크에서 gt를 빼 두께 차이 영역을 클릭 대상으로
            prev_mask   = modify_mask.copy()
            modify_mask = np.clip(modify_mask - gt_mask, 0, None)

        elif case_num == 1:
            # Make thicker: thin 마스크가 현재 상태 (gt가 목표)
            prev_mask = modify_mask.copy()

        elif case_num in (2, 3):
            # Extend / Remove: JSON 폴리곤에서 임의 세그먼트 하나를 선택
            # modify: 해당 세그먼트 영역 (폴리곤 fill)
            # prev_mask: gt에서 해당 세그먼트를 뺀 상태
            modify      = np.zeros_like(gt_mask)
            ann_key     = self.name_data[image_name] + '.png'
            rnum        = random.choice(range(len(self.json_data[ann_key])))
            coords      = self.json_data[ann_key][rnum]
            seg_coords  = np.array(coords).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(modify, [seg_coords], color=(1,))

            prev_mask   = np.clip(gt_mask - modify, 0, None)
            modify_mask = modify

        # ── 클릭 프롬프트 생성 및 크롭 ──────────────────────────────────────
        if self.prompt == 'click':
            point_label, pt = random_click(modify_mask, point_label)
            pt = tuple(pt)

        # Make thinner(0) / Remove(3)는 제거 방향 → negative click
        if case_num in (0, 3):
            point_label = 0

        # case 2, 3: 폴리곤 세그먼트 기반이므로 원본 해상도 유지 (크롭 없음)
        # case 0, 1: 클릭 좌표 중심으로 크롭 (경계에서도 패딩 유지)
        if case_num in (2, 3):
            origin_prev = prev_mask
        else:
            origin_prev = gt_mask.copy()
            prev_mask, xy = crop_with_padding_no_padding(prev_mask, pt, crop_size)
            origin_prev[xy[0]:xy[1], xy[2]:xy[3]] = prev_mask

        gt_mask   = torch.as_tensor(gt_mask)
        prev_mask = torch.as_tensor(origin_prev)

        pt = (np.array(pt) / self.out_size) * self.img_size

    

        # case 3 (Remove): gt와 prev_mask 역할 교환 (제거 전 상태를 gt로)
        if case_num == 3:
            gt_mask, prev_mask = prev_mask.clone(), gt_mask.clone()

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

class Test_ARCADEDataset(Dataset):
    """Test dataset for ARCADE coronary artery segmentation."""

    def __init__(self, args, dataset_path, split='train',
                 transform=None, transform_msk=None, prompt='click'):
        self.dataset_path = Path(dataset_path)
        self._split_path = self.dataset_path / split
        self._images_path = self._split_path / 'images'
        self._origin_path = self._split_path / 'mask'
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

        gt_mask = cv2.imread(str(self._origin_paths[stem]), cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 1, 1, cv2.THRESH_BINARY)
        gt_mask = gt_mask.astype(np.float32)

        prev_mask = cv2.imread(str(self._pred_paths[stem]))[:, :, 0].astype(np.float32)
        if np.max(prev_mask) > 200:
            prev_mask /= 255.0

        point_label, pt = val_get_next_click(gt_mask, prev_mask > self.threshold)

        pt = np.array(pt, dtype=np.float32)
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
