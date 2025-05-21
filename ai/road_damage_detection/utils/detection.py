"""
도로 손상 감지 모듈

이 모듈은 YOLO 모델을 사용하여 도로 손상(균열, 포트홀)을 감지하는 기능을 제공합니다.
"""

import cv2
import numpy as np


def apply_yolo_on_masked_area(model, img, mask):
    """
    마스크된 영역에서 YOLO 모델로 도로 손상을 감지하는 함수

    Process:
    1. 마스크를 3채널 이미지로 변환
    2. 원본 이미지에 마스크 적용
    3. YOLO 모델로 객체 감지 수행
    4. 감지된 객체별로 마스크 생성 (균열, 포트홀)
    5. 클래스별 마스크 반환

    Parameters:
        - model : 로드된 YOLO 모델
        - img (numpy.ndarray) : 원본 이미지
        - mask (numpy.ndarray) : 도로 영역 마스크

    Returns:
        - mask_crack (numpy.ndarray) : 균열 마스크
        - mask_pothole (numpy.ndarray) : 포트홀 마스크
    """
    # 마스크를 3채널로 확장
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # 마스크 적용된 이미지 생성
    masked_img = cv2.bitwise_and(img, mask_3ch)
    
    # YOLO 추론
    results = model(masked_img)
    res = results[0]
    
    # 클래스별 마스크 생성
    mask_crack = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_pothole = np.zeros(img.shape[:2], dtype=np.uint8)
    
    for i, box in enumerate(res.boxes):
        cls = int(box.cls[0])
        if i < len(res.masks):  # 안전하게 인덱스 확인
            single_mask = res.masks.data[i].cpu().numpy()
            binary_mask = (single_mask > 0.5).astype(np.uint8) * 255
            binary_mask_resized = cv2.resize(binary_mask, (img.shape[1], img.shape[0]))
            
            if cls == 0:  # crack
                mask_crack = cv2.bitwise_or(mask_crack, binary_mask_resized)
            elif cls == 1:  # pothole
                mask_pothole = cv2.bitwise_or(mask_pothole, binary_mask_resized)
    
    return mask_crack, mask_pothole


def analyze_crack_severity(crack_mask, pixel_to_cm_ratio=1.0):
    """
    균열의 심각도를 분석하는 함수

    Parameters:
        - crack_mask (numpy.ndarray) : 균열 마스크
        - pixel_to_cm_ratio (float) : 픽셀 대 센티미터 비율

    Returns:
        - crack_severity (float) : 0~1 사이의 균열 심각도
        - crack_type (str) : 균열 유형
        - crack_info (dict) : 균열 상세 정보
    """
    # 균열이 없는 경우
    if np.count_nonzero(crack_mask) == 0:
        return 0.0, "없음", {"crack_area_pixels": 0, "crack_length_cm": 0, "avg_width_cm": 0}

    # 균열 면적 계산
    crack_area = np.count_nonzero(crack_mask)
    
    # 균열 윤곽선 검출
    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 균열 총 길이와 너비 계산
    total_length_pixels = 0
    avg_width_pixels = 0
    
    if contours:
        # 각 균열 윤곽선에 대해
        for contour in contours:
            # 윤곽선을 감싸는 최소 사각형 찾기
            rect = cv2.minAreaRect(contour)
            width = min(rect[1][0], rect[1][1])  # 짧은 쪽이 너비
            length = max(rect[1][0], rect[1][1])  # 긴 쪽이 길이
            
            total_length_pixels += length
            avg_width_pixels += width
        
        # 평균 너비 계산
        avg_width_pixels /= len(contours)
    
    # 픽셀 -> 실제 단위 변환
    crack_length_cm = total_length_pixels * pixel_to_cm_ratio
    avg_width_cm = avg_width_pixels * pixel_to_cm_ratio
    
    # 균열 유형 결정
    crack_type = "선형 균열"  # 기본값
    
    # 균열 패턴 분석을 위한 모멘트 계산
    m = cv2.moments(crack_mask)
    if m["m00"] != 0:
        # 형태 특성 분석
        # 원형도: 4*pi*면적/둘레^2 (1에 가까울수록 원형)
        perimeters = sum([cv2.arcLength(contour, True) for contour in contours])
        if perimeters > 0:
            circularity = 4 * np.pi * crack_area / (perimeters ** 2)
            
            # 형태에 따른 균열 유형 분류
            if circularity > 0.6:  # 원형에 가까움
                crack_type = "포트홀 시작점"
            elif 0.3 < circularity <= 0.6:  # 중간 형태
                crack_type = "블록형 균열"
            elif len(contours) > 5:  # 다수의 연결된 균열
                crack_type = "악어 균열"
    
    # 심각도 계산 (0~1 사이 값)
    # 1cm 이상의 폭은 심각한 균열로 간주
    width_severity = min(1.0, avg_width_cm / 1.0)
    
    # 길이가 50cm 이상이면 심각한 균열로 간주
    length_severity = min(1.0, crack_length_cm / 50.0)
    
    # 패턴 심각도
    pattern_severity = 0.5  # 선형 균열 기본값
    if crack_type == "블록형 균열":
        pattern_severity = 0.7
    elif crack_type == "악어 균열":
        pattern_severity = 1.0
    elif crack_type == "포트홀 시작점":
        pattern_severity = 0.9
    
    # 최종 심각도 계산 (너비, 길이, 패턴 가중 평균)
    crack_severity = (width_severity * 0.4) + (length_severity * 0.3) + (pattern_severity * 0.3)
    
    # 균열 정보 사전 생성
    crack_info = {
        "crack_area_pixels": crack_area,
        "crack_length_cm": round(crack_length_cm, 2),
        "avg_width_cm": round(avg_width_cm, 2),
        "pattern": crack_type
    }
    
    return crack_severity, crack_type, crack_info


def check_pothole_emergency(pothole_mask, pixel_to_cm_ratio=1.0):
    """
    포트홀의 응급 수준을 확인하는 함수

    Parameters:
        - pothole_mask (numpy.ndarray) : 포트홀 마스크
        - pixel_to_cm_ratio (float) : 픽셀 대 센티미터 비율

    Returns:
        - is_emergency (bool) : 응급 상황 여부
        - pothole_info (dict) : 포트홀 상세 정보
    """
    # 포트홀이 없는 경우
    if np.count_nonzero(pothole_mask) == 0:
        return False, {"potholes_count": 0, "max_diameter_cm": 0, "total_area_cm2": 0}
    
    # 포트홀 윤곽선 검출
    contours, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pothole_count = len(contours)
    max_diameter_pixels = 0
    total_area_pixels = np.count_nonzero(pothole_mask)
    
    # 각 포트홀의 직경 계산
    for contour in contours:
        # 윤곽선을 감싸는 최소 원 찾기
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter = radius * 2
        
        # 최대 직경 갱신
        max_diameter_pixels = max(max_diameter_pixels, diameter)
    
    # 픽셀 -> 실제 단위 변환
    max_diameter_cm = max_diameter_pixels * pixel_to_cm_ratio
    total_area_cm2 = total_area_pixels * (pixel_to_cm_ratio ** 2)
    
    # 응급 상황 판단 (직경 10cm 이상을 응급으로 간주)
    is_emergency = max_diameter_cm >= 10.0
    
    # 포트홀 정보 사전 생성
    pothole_info = {
        "potholes_count": pothole_count,
        "max_diameter_cm": round(max_diameter_cm, 2),
        "total_area_cm2": round(total_area_cm2, 2),
        "is_emergency": is_emergency
    }
    
    return is_emergency, pothole_info
