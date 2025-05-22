"""
이미지 변환 관련 모듈

이 모듈은 이미지의 원근 변환 및 관련 기하학적 처리 기능을 제공합니다.
"""

import cv2
import numpy as np


def extract_road_corners_direct(road_lane_mask, roi_mask=None):
    """
    도로 마스크에서 코너 포인트를 직접 추출하는 함수

    Process:
    1. 마스크 유효성 검사
    2. ROI 마스크가 있으면 도로 마스크에 적용
    3. 마스크의 픽셀 좌표 추출
    4. y값 기준으로 상위 1%와 하위 1%의 픽셀 선택 
    5. 선택된 픽셀 중 x 좌표 기준으로 좌/우 지점 결정
    6. 이상치 제거 및 코너 포인트 유효성 검사
    7. 최종 코너 포인트 반환

    Parameters:
        - road_lane_mask (numpy.ndarray) : 차선 내 도로 마스크
        - roi_mask (numpy.ndarray) : ROI 마스크 (선택적)

    Returns:
        - numpy.ndarray : 변환에 사용할 4개의 포인트 (좌상, 우상, 좌하, 우하)
    """
    height, width = road_lane_mask.shape
    
    # ROI 정보 추출 (있는 경우)
    roi_bounds = None
    if roi_mask is not None and np.count_nonzero(roi_mask) > 0:
        roi_y, roi_x = np.where(roi_mask > 0)
        roi_bounds = {
            'min_x': np.min(roi_x),
            'max_x': np.max(roi_x),
            'min_y': np.min(roi_y),
            'max_y': np.max(roi_y)
        }
    
    # ROI 마스크를 도로 마스크에 적용 (ROI 내부 영역만 고려)
    masked_road = road_lane_mask.copy()
    if roi_mask is not None:
        masked_road = cv2.bitwise_and(masked_road, roi_mask)
    
    # 1. 마스크가 비어있는지 확인
    if np.count_nonzero(masked_road) < 10:  # 최소 10픽셀 이상은 필요
        # 기본 사각형 반환 (ROI가 있다면 ROI 기반, 없다면 이미지 전체)
        if roi_bounds is not None:
            return np.float32([
                [roi_bounds['min_x'], roi_bounds['min_y']],  # 좌상단
                [roi_bounds['max_x'], roi_bounds['min_y']],  # 우상단
                [roi_bounds['min_x'], roi_bounds['max_y']],  # 좌하단
                [roi_bounds['max_x'], roi_bounds['max_y']]   # 우하단
            ])
        else:
            return np.float32([
                [0, 0],               # 좌상단
                [width-1, 0],         # 우상단
                [0, height-1],        # 좌하단
                [width-1, height-1]   # 우하단
            ])
    
    # 2. 마스크 픽셀 좌표 가져오기 (이미 ROI로 제한됨)
    y_indices, x_indices = np.where(masked_road > 0)
    
    if len(y_indices) == 0:
        # 마스크가 비어있으면 기본 사각형 반환
        if roi_bounds is not None:
            return np.float32([
                [roi_bounds['min_x'], roi_bounds['min_y']],
                [roi_bounds['max_x'], roi_bounds['min_y']],
                [roi_bounds['min_x'], roi_bounds['max_y']],
                [roi_bounds['max_x'], roi_bounds['max_y']]
            ])
        else:
            return np.float32([
                [0, 0],
                [width-1, 0],
                [0, height-1],
                [width-1, height-1]
            ])
    
    # 3. y값으로 상위 1%와 하위 1% 지점 선택
    y_sorted_indices = np.argsort(y_indices)
    
    # 상단 1% 인덱스
    top_percent = 0.01
    top_count = max(1, int(len(y_indices) * top_percent))
    top_indices = y_sorted_indices[:top_count]
    
    # 하단 1% 인덱스
    bottom_indices = y_sorted_indices[-top_count:]
    
    # 4. 상위 지점들 중에서 x 좌표로 좌/우 결정
    top_x_values = x_indices[top_indices]
    top_y_values = y_indices[top_indices]
    
    # 상단 점들의 중간 y값 계산 (이상치 제거를 위해)
    median_top_y = np.median(top_y_values)
    
    # 상단 점들 중 y값이 중간값에서 너무 떨어진 점 제외
    valid_top_indices = np.abs(top_y_values - median_top_y) < (height * 0.1)  # 10% 차이 이내
    filtered_top_x = top_x_values[valid_top_indices]
    filtered_top_y = top_y_values[valid_top_indices]
    
    if len(filtered_top_x) > 0:
        # 상단 지점 중 가장 왼쪽 지점
        top_left_idx = np.argmin(filtered_top_x)
        left_top = np.array([filtered_top_x[top_left_idx], filtered_top_y[top_left_idx]], dtype=np.float32)
        
        # 상단 지점 중 가장 오른쪽 지점
        top_right_idx = np.argmax(filtered_top_x)
        right_top = np.array([filtered_top_x[top_right_idx], filtered_top_y[top_right_idx]], dtype=np.float32)
    else:
        # ROI가 있으면 ROI 상단 가장자리 사용
        if roi_bounds is not None:
            left_top = np.array([roi_bounds['min_x'], roi_bounds['min_y']], dtype=np.float32)
            right_top = np.array([roi_bounds['max_x'], roi_bounds['min_y']], dtype=np.float32)
        else:
            left_top = np.array([0, 0], dtype=np.float32)
            right_top = np.array([width-1, 0], dtype=np.float32)
    
    # 5. 하위 지점들 중에서 x 좌표로 좌/우 결정
    bottom_x_values = x_indices[bottom_indices]
    bottom_y_values = y_indices[bottom_indices]
    
    # 하단 점들의 중간 y값 계산 (이상치 제거를 위해)
    median_bottom_y = np.median(bottom_y_values)
    
    # 하단 점들 중 y값이 중간값에서 너무 떨어진 점 제외
    valid_bottom_indices = np.abs(bottom_y_values - median_bottom_y) < (height * 0.1)  # 10% 차이 이내
    filtered_bottom_x = bottom_x_values[valid_bottom_indices]
    filtered_bottom_y = bottom_y_values[valid_bottom_indices]
    
    if len(filtered_bottom_x) > 0:
        # 하단 지점 중 가장 왼쪽 지점
        bottom_left_idx = np.argmin(filtered_bottom_x)
        left_bottom = np.array([filtered_bottom_x[bottom_left_idx], filtered_bottom_y[bottom_left_idx]], dtype=np.float32)
        
        # 하단 지점 중 가장 오른쪽 지점
        bottom_right_idx = np.argmax(filtered_bottom_x)
        right_bottom = np.array([filtered_bottom_x[bottom_right_idx], filtered_bottom_y[bottom_right_idx]], dtype=np.float32)
    else:
        # ROI가 있으면 ROI 하단 가장자리 사용
        if roi_bounds is not None:
            left_bottom = np.array([roi_bounds['min_x'], roi_bounds['max_y']], dtype=np.float32)
            right_bottom = np.array([roi_bounds['max_x'], roi_bounds['max_y']], dtype=np.float32)
        else:
            left_bottom = np.array([0, height-1], dtype=np.float32)
            right_bottom = np.array([width-1, height-1], dtype=np.float32)
    
    # 6. 코너 포인트가 ROI 내에 있는지 강제로 확인 및 조정
    if roi_bounds is not None:
        # 코너 포인트가 ROI 내에 있도록 조정
        left_top[0] = max(roi_bounds['min_x'], min(roi_bounds['max_x'], left_top[0]))
        left_top[1] = max(roi_bounds['min_y'], min(roi_bounds['max_y'], left_top[1]))
        right_top[0] = max(roi_bounds['min_x'], min(roi_bounds['max_x'], right_top[0]))
        right_top[1] = max(roi_bounds['min_y'], min(roi_bounds['max_y'], right_top[1]))
        left_bottom[0] = max(roi_bounds['min_x'], min(roi_bounds['max_x'], left_bottom[0]))
        left_bottom[1] = max(roi_bounds['min_y'], min(roi_bounds['max_y'], left_bottom[1]))
        right_bottom[0] = max(roi_bounds['min_x'], min(roi_bounds['max_x'], right_bottom[0]))
        right_bottom[1] = max(roi_bounds['min_y'], min(roi_bounds['max_y'], right_bottom[1]))
    
    # 7. 코너 포인트가 이상하게 교차하지 않는지 확인
    # 왼쪽 상단은 오른쪽 상단보다 x좌표가 작아야 함
    if left_top[0] >= right_top[0]:
        mid_x = (left_top[0] + right_top[0]) / 2
        left_top[0] = mid_x - (width * 0.1)
        right_top[0] = mid_x + (width * 0.1)
    
    # 왼쪽 하단은 오른쪽 하단보다 x좌표가 작아야 함
    if left_bottom[0] >= right_bottom[0]:
        mid_x = (left_bottom[0] + right_bottom[0]) / 2
        left_bottom[0] = mid_x - (width * 0.1)
        right_bottom[0] = mid_x + (width * 0.1)
    
    # 상단 y값은 하단 y값보다 작아야 함
    if left_top[1] >= left_bottom[1]:
        left_top[1] = min(left_top[1], left_bottom[1] - (height * 0.1))
    
    if right_top[1] >= right_bottom[1]:
        right_top[1] = min(right_top[1], right_bottom[1] - (height * 0.1))
    
    # 최종 코너 포인트 반환
    return np.float32([left_top, right_top, left_bottom, right_bottom])


def perspective_transform_direct_masked(img, road_lane_mask, roi_mask=None, road_width_m=3.0, road_length_m=12.0):
    """
    도로 마스크에서 추출한 코너 포인트로 원근 변환을 수행하고, 
    코너 포인트 외부 영역을 마스킹하는 함수
    
    Parameters:
        - img (numpy.ndarray) : 원본 이미지
        - road_lane_mask (numpy.ndarray) : 차선 내 도로 마스크
        - roi_mask (numpy.ndarray) : ROI 마스크 (선택적)
        - road_width_m (float) : 도로 너비의 실제 미터 값 (기본값: 3.0m)
        - road_length_m (float) : 도로 길이의 실제 미터 값 (기본값: 30.0m)

    Returns:
        - warped_img (numpy.ndarray) : 원근 변환된 이미지 (코너 포인트 외부는 검은색)
        - warped_mask (numpy.ndarray) : 원근 변환된 마스크
        - M (numpy.ndarray) : 원근 변환 행렬
        - src_points (numpy.ndarray) : 원본 이미지에서의 소스 포인트
    """
    height, width = road_lane_mask.shape
    
    # 도로 마스크에서 직접 코너 포인트 추출
    src_points = extract_road_corners_direct(road_lane_mask, roi_mask)
    
    # 도로의 물리적 비율 계산
    aspect_ratio = road_length_m / road_width_m  # 길이/너비 비율 (예: 30m/3m = 10)
    
    # 출력 이미지의 크기를 유지하면서 비율 적용
    # 이미지 너비에 맞춰 도로 너비를 정하고, 길이는 비율대로 계산
    road_width_pixels = width / 2  # 이미지 너비의 절반을 도로 너비로 사용
    road_length_pixels = road_width_pixels * aspect_ratio  # 비율에 맞게 길이 계산
    
    # 만약 계산된 길이가 이미지 높이를 초과하면 조정
    if road_length_pixels > height:
        # 이미지 높이에 맞추고 너비를 비율대로 줄임
        road_length_pixels = height
        road_width_pixels = road_length_pixels / aspect_ratio
    
    # 이미지 중앙에 도로가 위치하도록 설정
    left_margin = (width - road_width_pixels) / 2
    right_margin = left_margin + road_width_pixels
    
    # 변환 대상 포인트 설정 (물리적 비율에 맞는 직사각형)
    dst_points = np.float32([
        [left_margin, 0],                  # 왼쪽 상단
        [right_margin, 0],                 # 오른쪽 상단
        [left_margin, road_length_pixels], # 왼쪽 하단
        [right_margin, road_length_pixels] # 오른쪽 하단
    ])
    
    # 원근 변환 행렬 계산 및 적용
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(img, M, (width, height))
    warped_mask = cv2.warpPerspective(road_lane_mask, M, (width, height))
    
    # 코너 포인트 외부 영역 마스킹
    # 1. 원근 변환된 코너 포인트 영역 마스크 생성
    transformed_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 변환된 사각형 영역 채우기
    transformed_rect = np.int32([
        [left_margin, 0],                  # 왼쪽 상단
        [right_margin, 0],                 # 오른쪽 상단
        [right_margin, road_length_pixels], # 오른쪽 하단
        [left_margin, road_length_pixels]  # 왼쪽 하단
    ])
    cv2.fillPoly(transformed_mask, [transformed_rect], 255)
    
    # 2. 원근 변환된 이미지에 마스크 적용
    masked_warped_img = cv2.bitwise_and(warped_img, warped_img, mask=transformed_mask)
    
    # 3. 원근 변환된 마스크도 동일하게 마스킹
    warped_mask = cv2.bitwise_and(warped_mask, transformed_mask)
    
    return masked_warped_img, warped_mask, M, src_points
