import cv2
import numpy as np



def val_get_next_click(gt_mask, pred_mask):
    """Select the next click point using connected-component centroids.

    Finds the largest FN and FP connected components, then clicks at the
    centroid of whichever region has the greater area.
    Returns (is_positive, (y, x)).
    """
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask)).astype(np.int8)
    fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask).astype(np.int8)

    # Analyze connected components for false-negative region
    _, fn_labels, fn_stats, fn_centroids = cv2.connectedComponentsWithStats(fn_mask)
    fn_areas = fn_stats[1:, cv2.CC_STAT_AREA]  # exclude background label
    largest_fn_idx = np.argmax(fn_areas) + 1
    fn_max = np.max(fn_areas)
    fn_centroid = fn_centroids[largest_fn_idx]

    # Analyze connected components for false-positive region
    _, fp_labels, fp_stats, fp_centroids = cv2.connectedComponentsWithStats(fp_mask)
    fp_areas = fp_stats[1:, cv2.CC_STAT_AREA]  # exclude background label
    largest_fp_idx = np.argmax(fp_areas) + 1
    fp_max = np.max(fp_areas)
    fp_centroid = fp_centroids[largest_fp_idx]

    is_positive = fn_max > fp_max
    if is_positive:
        coords_x, coords_y = map(int, fn_centroid)
    else:
        coords_x, coords_y = map(int, fp_centroid)

    return is_positive, (coords_y, coords_x)


def get_next_click(gt_mask, pred_mask, padding=True):
    """Select the next click point using distance transform on FN/FP regions.

    Picks a positive click if the largest false-negative region is bigger than
    the largest false-positive region, and a negative click otherwise.
    Returns (is_positive, (y, x)).
    """
    fn_mask = np.logical_and(gt_mask, np.logical_not(pred_mask))
    fp_mask = np.logical_and(np.logical_not(gt_mask), pred_mask)

    if padding:
        fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
        fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    if padding:
        fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    is_positive = fn_max_dist > fp_max_dist
    if is_positive:
        coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
    else:
        coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)

    return is_positive, (coords_y[0], coords_x[0])


def crop_with_padding_no_padding(image, center, crop_size):
    """
    이미지를 주어진 센터 포인트를 기준으로 주어진 크기로 자르되, 이미지 경계를 벗어나는 경우 자동으로 패딩을 추가하는 함수
    :param image: 원본 이미지(numpy 배열)
    :param center: 자를 영역의 중심 좌표 (x, y)
    :param crop_size: 자를 크기 (width, height)
    :return: 자른 이미지(numpy 배열)
    """
    height, width = image.shape[:2]
    crop_width, crop_height = crop_size
    center_y, center_x = center

    # 자를 영역의 좌상단과 우하단 좌표 계산
    start_x = max(center_x - crop_width // 2, 0)
    start_y = max(center_y - crop_height // 2, 0)
    end_x = min(center_x + crop_width // 2, width)
    end_y = min(center_y + crop_height // 2, height)

    # # 자를 영역이 이미지 경계를 벗어나는지 확인하고 패딩 추가
    # pad_left = max(0 - (center_x - crop_width // 2), 0)
    # pad_top = max(0 - (center_y - crop_height // 2), 0)
    # pad_right = max((center_x + crop_width // 2)- width, 0)
    # pad_bottom = max((center_y + crop_height // 2) - height, 0)


    # 이미지 자르기
    cropped_image = image[start_y:end_y, start_x:end_x]

    # 패딩 추가
    #cropped_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return cropped_image, (start_y,end_y,start_x,end_x)


def crop_with_padding(image, center, crop_size):
    """
    이미지를 주어진 센터 포인트를 기준으로 주어진 크기로 자르되, 이미지 경계를 벗어나는 경우 자동으로 패딩을 추가하는 함수
    :param image: 원본 이미지(numpy 배열)
    :param center: 자를 영역의 중심 좌표 (x, y)
    :param crop_size: 자를 크기 (width, height)
    :return: 자른 이미지(numpy 배열)
    """
    height, width = image.shape[:2]
    crop_width, crop_height = crop_size
    center_y, center_x = center

    # 자를 영역의 좌상단과 우하단 좌표 계산
    start_x = max(center_x - crop_width // 2, 0)
    start_y = max(center_y - crop_height // 2, 0)
    end_x = min(center_x + crop_width // 2, width)
    end_y = min(center_y + crop_height // 2, height)

    # 자를 영역이 이미지 경계를 벗어나는지 확인하고 패딩 추가
    pad_left = max(0 - (center_x - crop_width // 2), 0)
    pad_top = max(0 - (center_y - crop_height // 2), 0)
    pad_right = max((center_x + crop_width // 2)- width, 0)
    pad_bottom = max((center_y + crop_height // 2) - height, 0)


    # 이미지 자르기
    cropped_image = image[start_y:end_y, start_x:end_x]

    # 패딩 추가
    cropped_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

    return cropped_image


def random_click(mask, point_labels = 1):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
        indices = np.argwhere(mask == max_label) 
    # max agreement position
    else:
        indices = np.argwhere(mask > 0) 
    return point_labels, indices[np.random.randint(len(indices))]