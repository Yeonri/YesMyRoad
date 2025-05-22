"""
도로 영역 추출 관련 모듈

이 모듈은 이미지에서 도로 영역을 추출하는 기능과 관련 유틸리티 함수를 제공합니다.
"""

import cv2
import numpy as np


def create_roi_mask(img, use_roi=True, roi_type="trapezoid"):
    """
    관심 영역(ROI) 마스크를 생성하는 함수

    Process:
    1. 이미지 크기 확인
    2. use_roi 매개변수 확인
    3. roi_type에 따라 다른 형태의 마스크 생성
        - trapezoid: 사다리꼴 모양 (기본값)
        - middle_strip: 이미지 중간 영역만 선택

    Parameters:
        - img (numpy.ndarray) : 원본 이미지
        - use_roi (bool) : ROI 사용 여부
        - roi_type (str) : ROI 마스크 타입 ("trapezoid", "middle_strip")

    Returns:
        - roi_mask (numpy.ndarray) : 생성된 ROI 마스크
    """
    height, width = img.shape[:2]
    roi_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    if not use_roi:
        return roi_mask
    
    if roi_type == "trapezoid":
        # 사다리꼴 ROI - 일반적인 도로 영역
        roi_points = np.array([
            [width * 0, height * 0.85],        # 좌하단
            [width * 1, height * 0.85],        # 우하단
            [width * 0.75, height * 0.45],     # 우상단
            [width * 0.25, height * 0.45]      # 좌상단
        ], dtype=np.int32)
        
        # 빈 마스크 생성 및 ROI 영역만 255로 채우기
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_points], 255)
    
    elif roi_type == "middle_strip":
        # 상단 n%와 하단 m%를 삭제하고 중간 (100 - m - n)%만 사용
        top = 0.45
        bottom = 0.15
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        top_boundary = int(height * top)
        bottom_boundary = int(height * (1 - bottom))
        roi_mask[top_boundary:bottom_boundary, :] = 255
    
    return roi_mask


def extract_road_mask(img, use_roi=True, roi_mask=None):
    """
    이미지에서 도로 영역을 추출하는 함수

    Process:
    1. ROI 마스크 확인 및 생성
    2. 가우시안 블러로 노이즈 제거
    3. HSV 색상 공간으로 변환 (색상(Hue), 채도(Saturation), 명도(Value)) : 색상 분할에 효과적
    4. 다양한 도로 색상 범위에 대한 마스크 생성 (회색, 어두운, 밝은 도로)
    5. 모든 마스크 결합 및 ROI 적용
    6. 모폴로지 연산으로 노이즈 제거 및 도로 영역 연결
    7. 유의미한 크기의 연결 요소만 선택
    8. 홍수 채우기 알고리즘으로 가장 가능성 높은 도로 영역 선택

    Parameters:
        - img (numpy.ndarray) : 원본 이미지
        - use_roi (bool) : ROI 사용 여부
        - roi_mask (numpy.ndarray) : 사전에 정의된 ROI 마스크

    Returns:
        - road_mask (numpy.ndarray) : 도로 영역 마스크
    """
    # 이미지 복사 및 크기 가져오기
    height, width = img.shape[:2]
    
    # ROI 마스크가 제공되지 않았다면 생성
    if roi_mask is None and use_roi:
        roi_mask = create_roi_mask(img, use_roi=True)
    elif not use_roi:
        roi_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # 이미지 전처리: 가우시안 블러로 노이즈 제거
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # HSV 변환
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 도로 색상 범위
    # 일반 범위의 도로 색상 감지
    lower_gray_road = np.array([0, 0, 70])
    upper_gray_road = np.array([180, 30, 170])
    
    # 아스팔트 도로 (어두운 색상)
    lower_dark_road = np.array([0, 0, 40])
    upper_dark_road = np.array([180, 25, 100])
    
    # 콘크리트 도로 (밝은 색상)
    lower_light_road = np.array([0, 0, 140])
    upper_light_road = np.array([180, 35, 220])
    
    # 각 범위에 대한 마스크 생성
    gray_mask = cv2.inRange(hsv, lower_gray_road, upper_gray_road)
    dark_mask = cv2.inRange(hsv, lower_dark_road, upper_dark_road)
    light_mask = cv2.inRange(hsv, lower_light_road, upper_light_road)
    
    # 모든 마스크 결합
    road_mask = cv2.bitwise_or(gray_mask, dark_mask)
    road_mask = cv2.bitwise_or(road_mask, light_mask)
    
    # ROI 적용
    if use_roi:
        road_mask = cv2.bitwise_and(road_mask, roi_mask)
    
    # 작은 노이즈 제거
    kernel_open = np.ones((3, 3), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_open)
    
    # 도로 영역 연결
    kernel_close = np.ones((7, 7), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 넓은 영역 닫기 연산으로 도로 틈새 메우기
    kernel_close_large = np.ones((15, 15), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close_large)
    
    # 연결 요소 찾기
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 면적 기준으로 정렬된 상위 3개 연결 요소 선택
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        refined_mask = np.zeros_like(road_mask)
        
        # 이미지 전체 면적
        total_area = height * width
        
        # 상위 연결 요소 중 유의미한 크기인 것만 선택
        for contour in sorted_contours[:3]:
            area = cv2.contourArea(contour)
            if area > (total_area * 0.05):  # 전체 이미지의 5% 이상인 연결 요소만 고려
                cv2.drawContours(refined_mask, [contour], 0, 255, -1)
        
        # 최종 마스크가 비어있지 않은 경우에만 갱신
        if np.any(refined_mask):
            road_mask = refined_mask
            
        # 마스크의 하단 중앙에서 시작하여 홍수 채우기(Flood Fill)로 가장 가능성 높은 도로 영역 선택
        seed_point = (width // 2, int(height * 0.9))
        flood_mask = np.zeros((height+2, width+2), np.uint8)
        cv2.floodFill(road_mask.copy(), flood_mask, seed_point, 255, 0, 0, cv2.FLOODFILL_FIXED_RANGE)
        flood_mask = flood_mask[1:-1, 1:-1]  # 경계 제거
        
        # 가장 큰 연결 요소와 홍수 채우기 결과를 결합
        combined_mask = cv2.bitwise_or(road_mask, flood_mask)
        road_mask = combined_mask
    
    return road_mask


def extract_road_between_lanes(img, road_mask, left_lane_mask, right_lane_mask):
    """
    차선 내부의 도로 영역을 추출하는 함수

    Process:
    1. 왼쪽과 오른쪽 차선의 윤곽선 검출
    2. 차선 마스크 유효성 검사
    3. 각 차선의 외곽점 계산 (상단/하단)
    4. 도로 폴리곤 생성 (차선 내부 + 차선 아래 영역)
    5. 폴리곤 기반 도로 영역 마스크 생성
    6. 원래 도로 마스크와 교차 연산

    Parameters:
        - img (numpy.ndarray) : 원본 이미지
        - road_mask (numpy.ndarray) : 도로 마스크
        - left_lane_mask (numpy.ndarray) : 왼쪽 차선 마스크
        - right_lane_mask (numpy.ndarray) : 오른쪽 차선 마스크

    Returns:
        - final_road_mask (numpy.ndarray) : 차선 내부의 도로 마스크
        - road_polygon (numpy.ndarray) : 도로 폴리곤 좌표 배열
    """
    height, width = road_mask.shape
    
    # 왼쪽과 오른쪽 차선 윤곽선 찾기
    left_contours, _ = cv2.findContours(left_lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_contours, _ = cv2.findContours(right_lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 차선 마스크가 충분하지 않으면 기본 도로 마스크 반환
    if not left_contours or not right_contours:
        return road_mask, None
    
    # 가장 큰 왼쪽/오른쪽 차선 윤곽선 선택
    left_contour = max(left_contours, key=cv2.contourArea) if left_contours else None
    right_contour = max(right_contours, key=cv2.contourArea) if right_contours else None
    
    if left_contour is None or right_contour is None:
        return road_mask, None
    
    # 왼쪽/오른쪽 차선의 외곽점 찾기
    left_points = left_contour.reshape(-1, 2)
    right_points = right_contour.reshape(-1, 2)
    
    # 각 차선에서 가장 아래/위 지점 찾기
    left_bottom = left_points[np.argmax(left_points[:, 1])]
    right_bottom = right_points[np.argmax(right_points[:, 1])]
    left_top = left_points[np.argmin(left_points[:, 1])]
    right_top = right_points[np.argmin(right_points[:, 1])]
    
    # 차선이 충분히 감지되지 않은 경우 대체 포인트 설정
    if np.linalg.norm(left_top - left_bottom) < 50:
        left_top[1] = max(0, height - height//3)
    
    if np.linalg.norm(right_top - right_bottom) < 50:
        right_top[1] = max(0, height - height//3)
        
    # 이미지 하단 좌표 추가
    bottom_left = [0, height-1]
    bottom_right = [width-1, height-1]
    
    # 도로 폴리곤 생성 (차선 내부 + 차선 아래 영역)
    road_polygon = np.array([
        bottom_left, left_bottom, left_top, right_top, right_bottom, bottom_right
    ], dtype=np.int32)
    
    # 도로 영역 마스크 생성
    lane_road_mask = np.zeros_like(road_mask)
    cv2.fillPoly(lane_road_mask, [road_polygon], 255)
    
    # 원래 도로 마스크와 교차
    final_road_mask = cv2.bitwise_and(road_mask, lane_road_mask)
    
    return final_road_mask, road_polygon
