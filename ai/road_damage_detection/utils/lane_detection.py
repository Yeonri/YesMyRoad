"""
차선 감지 관련 모듈

이 모듈은 이미지에서 차선을 감지하고 마스크를 생성하는 기능을 제공합니다.
"""

import cv2
import numpy as np
from .road_extraction import create_roi_mask


def detect_lane_lines(img, use_roi=True, roi_mask=None):
    """
    이미지에서 차선을 감지하고 마스크를 생성하는 함수

    Process:
    1. 이미지 전처리 (그레이스케일 변환, 블러, 캐니 엣지 검출)
    2. ROI 마스크 적용
    3. 색상 기반 차선 추출 (흰색/노란색)
    4. 엣지와 색상 마스크 결합
    5. 호프 변환으로 직선 검출
    6. 기울기 기반으로 왼쪽/오른쪽 차선 분류
    7. 각 차선의 가장 긴 선분 선택 및 연장
    8. 차선 마스크 생성 및 병합
    9. 교차되는 차선을 처리하여 교차점 아래 부분만 유지

    Parameters:
        - img (numpy.ndarray) : 원본 이미지
        - use_roi (bool) : ROI 사용 여부
        - roi_mask (numpy.ndarray) : 사전에 정의된 ROI 마스크

    Returns:
        - extended_lane_mask (numpy.ndarray) : 연장된 차선 마스크
        - left_line_mask (numpy.ndarray) : 왼쪽 차선 마스크
        - right_line_mask (numpy.ndarray) : 오른쪽 차선 마스크
        - original_lane_mask (numpy.ndarray) : 원본 차선 마스크 (연장 전)
        - masked_edges (numpy.ndarray) : 엣지 마스크 (ROI 적용)
        - color_mask (numpy.ndarray) : 색상 기반 마스크
        - final_mask (numpy.ndarray) : 엣지와 색상 마스크 결합
    """
    # 기존 전처리 과정 유지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # ROI 설정
    height, width = edges.shape
    
    # ROI 마스크가 제공되지 않았다면 생성
    if roi_mask is None and use_roi:
        roi_mask = create_roi_mask(img, use_roi=True)
    elif not use_roi:
        roi_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # ROI 마스크가 비어있는지 확인
    if use_roi and np.count_nonzero(roi_mask) == 0:
        print("경고: ROI 마스크가 비어 있습니다. 기본 사다리꼴 ROI를 사용합니다.")
        roi_mask = create_roi_mask(img, use_roi=True, roi_type="trapezoid")
    
    masked_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    
    # 색상 기반 차선 추출 (흰색/노란색)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 흰색 및 노란색 범위 설정
    # 흰색 범위는 그대로 유지
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([10, 60, 80])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 색상 마스크 통합 및 ROI 적용
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    if use_roi:
        color_mask = cv2.bitwise_and(color_mask, roi_mask)
    
    # 최종 마스크 (엣지 + 색상)
    final_mask = cv2.bitwise_or(masked_edges, color_mask)
    
    # 여기서 모폴로지 연산을 적용하여 비워진 부분 채우기 (도로 감지와 유사한 방식)
    # 1. 작은 노이즈 제거
    kernel_open = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    
    # 2. 차선 영역 연결 (가까운 간격 메우기)
    kernel_close = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. 주행 방향으로 차선을 연결하기 위한 수직방향 모폴로지 연산 (선택적)
    # 수직 방향으로 길쭉한 커널 생성
    # vertical_kernel = np.ones((7, 1), np.uint8)
    # final_mask_vertical = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, vertical_kernel)
    # final_mask = cv2.bitwise_or(final_mask, final_mask_vertical)

    # 호프 변환으로 직선 검출 - "완화된 호프 파라미터" 설정 적용
    lines = cv2.HoughLinesP(
        final_mask, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20,     
        minLineLength=10, 
        maxLineGap=1      
    )
    
    # 차선 마스크 초기화
    left_line_mask = np.zeros_like(gray)
    right_line_mask = np.zeros_like(gray)
    original_lane_mask = np.zeros_like(gray)
    
    # 왼쪽/오른쪽 차선 정보 저장할 리스트
    left_lines = []
    right_lines = []
    
    # 차선 필터링 및 분류
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 선의 길이 계산
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 선의 길이가 너무 짧으면 무시 - 완화된 길이 적용
            if line_length < 20:  # 30에서 20으로 낮춤
                continue
                
            # 기울기 계산
            if x2 - x1 == 0:  # 수직선 방지
                continue
            else:
                slope = (y2 - y1) / (x2 - x1)
            
            # 적절한 기울기 범위만 선택 - 유지
            if 0.3 < abs(slope) < 2:  
                # 원본 마스크에 모든 유효한 선분 추가
                cv2.line(original_lane_mask, (x1, y1), (x2, y2), 255, 2)
                
                # 왼쪽/오른쪽 차선 구분
                mid_x = width // 2
                line_mid_x = (x1 + x2) // 2
                
                if slope < 0 and line_mid_x < mid_x:  # 왼쪽 차선 (음수 기울기)
                    # 직접 마스크에 그리지 않고 리스트에만 저장
                    left_lines.append((x1, y1, x2, y2, slope, line_length))
                    
                elif slope > 0 and line_mid_x > mid_x:  # 오른쪽 차선 (양수 기울기)
                    # 직접 마스크에 그리지 않고 리스트에만 저장
                    right_lines.append((x1, y1, x2, y2, slope, line_length))
    
    # 확장된 차선 마스크 생성 (이 마스크만 최종 결과로 사용)
    extended_lane_mask = np.zeros_like(gray)
    
    # ROI 영역의 y 좌표 범위 계산
    if use_roi:
        roi_indices = np.where(roi_mask > 0)
        if len(roi_indices[0]) > 0:  # ROI 영역이 비어있지 않은 경우
            min_y_roi = np.min(roi_indices[0])
            max_y_roi = np.max(roi_indices[0])
        else:  # ROI 영역이 비어있는 경우
            min_y_roi = 0
            max_y_roi = height - 1
    else:
        min_y_roi = 0
        max_y_roi = height - 1
    
    # 왼쪽 차선의 직선 파라미터 (기본값은 None)
    left_line_params = None
    
    # 왼쪽 차선 처리 - 가장 긴 선분 선택
    if left_lines:
        # 길이 기준으로 정렬하여 가장 긴 선분 선택
        longest_left = sorted(left_lines, key=lambda x: x[5], reverse=True)[0]
        x1, y1, x2, y2, slope, _ = longest_left
        
        # 차선 연장을 위한 직선 방정식 계산
        # y = m*x + b, b = y - m*x
        b = y1 - slope * x1
        left_line_params = (slope, b)  # 차선 파라미터 저장
        
        # 차선 연장 - ROI 영역 내에서만
        top_y = min_y_roi
        bottom_y = max_y_roi
        
        # x 좌표 계산
        if slope != 0:
            top_x = int((top_y - b) / slope)
            bottom_x = int((bottom_y - b) / slope)
        else:
            top_x = x1
            bottom_x = x1
        
        # x 좌표가 화면 범위를 벗어나는지 확인
        if top_x < 0:
            # x가 0일 때의 y 계산
            top_y = int(slope * 0 + b)
            top_x = 0
        elif top_x >= width:
            # x가 width-1일 때의 y 계산
            top_y = int(slope * (width-1) + b)
            top_x = width - 1
            
        if bottom_x < 0:
            bottom_y = int(slope * 0 + b)
            bottom_x = 0
        elif bottom_x >= width:
            bottom_y = int(slope * (width-1) + b)
            bottom_x = width - 1
        
        # 연장된 왼쪽 차선 그리기
        temp_mask = np.zeros_like(gray)
        cv2.line(temp_mask, (top_x, top_y), (bottom_x, bottom_y), 255, 5)
        
        # ROI 내에서만 차선 표시
        if use_roi:
            left_line_mask_roi = cv2.bitwise_and(temp_mask, roi_mask)
            left_line_mask = left_line_mask_roi
        else:
            left_line_mask = temp_mask
    
    # 오른쪽 차선의 직선 파라미터 (기본값은 None)
    right_line_params = None
    
    # 오른쪽 차선 처리 - 가장 긴 선분 선택
    if right_lines:
        # 길이 기준으로 정렬하여 가장 긴 선분 선택
        longest_right = sorted(right_lines, key=lambda x: x[5], reverse=True)[0]
        x1, y1, x2, y2, slope, _ = longest_right
        
        # 차선 연장을 위한 직선 방정식 계산
        b = y1 - slope * x1
        right_line_params = (slope, b)  # 차선 파라미터 저장
        
        # 차선 연장 - ROI 영역 내에서만
        top_y = min_y_roi
        bottom_y = max_y_roi
        
        # x 좌표 계산
        if slope != 0:
            top_x = int((top_y - b) / slope)
            bottom_x = int((bottom_y - b) / slope)
        else:
            top_x = x1
            bottom_x = x1
        
        # x 좌표가 화면 범위를 벗어나는지 확인
        if top_x < 0:
            # x가 0일 때의 y 계산
            top_y = int(slope * 0 + b)
            top_x = 0
        elif top_x >= width:
            # x가 width-1일 때의 y 계산
            top_y = int(slope * (width-1) + b)
            top_x = width - 1
            
        if bottom_x < 0:
            bottom_y = int(slope * 0 + b)
            bottom_x = 0
        elif bottom_x >= width:
            bottom_y = int(slope * (width-1) + b)
            bottom_x = width - 1
        
        # 연장된 오른쪽 차선 그리기
        temp_mask = np.zeros_like(gray)
        cv2.line(temp_mask, (top_x, top_y), (bottom_x, bottom_y), 255, 5)
        
        # ROI 내에서만 차선 표시
        if use_roi:
            right_line_mask_roi = cv2.bitwise_and(temp_mask, roi_mask)
            right_line_mask = right_line_mask_roi
        else:
            right_line_mask = temp_mask
    
    # 차선 교차점 계산 및 처리
    intersection_point = None
    if left_line_params is not None and right_line_params is not None:
        left_slope, left_b = left_line_params
        right_slope, right_b = right_line_params
        
        # 기울기가 서로 충분히 다른 경우에만 교차점 계산
        if abs(left_slope - right_slope) > 0.1:
            # 두 직선의 교차점 계산
            # y = m1*x + b1 = m2*x + b2
            # x = (b2 - b1) / (m1 - m2)
            intersect_x = (right_b - left_b) / (left_slope - right_slope)
            intersect_y = left_slope * intersect_x + left_b
            
            # 교차점이 이미지 범위 내에 있는지 확인
            if 0 <= intersect_x < width and min_y_roi <= intersect_y < max_y_roi:
                intersection_point = (int(intersect_x), int(intersect_y))
                
                # 교차점 아래 부분만 유지하도록 마스크 수정
                if intersection_point:
                    x_intersect, y_intersect = intersection_point
                    
                    # 새로운 마스크 생성 (교차점 아래 부분만)
                    below_intersection_mask = np.zeros_like(gray)
                    below_intersection_mask[y_intersect:, :] = 255
                    
                    # 각 차선 마스크에 적용
                    left_line_mask = cv2.bitwise_and(left_line_mask, below_intersection_mask)
                    right_line_mask = cv2.bitwise_and(right_line_mask, below_intersection_mask)
    
    # 전체 차선 마스크 통합
    extended_lane_mask = cv2.bitwise_or(left_line_mask, right_line_mask)
    
    # 확장된 차선 마스크 확장
    kernel = np.ones((3, 3), np.uint8)
    extended_lane_mask = cv2.dilate(extended_lane_mask, kernel, iterations=1)
    
    # ROI 내에서만 원본 차선 마스크 적용
    if use_roi:
        original_lane_mask = cv2.bitwise_and(original_lane_mask, roi_mask)
    
    # 연장된 가장 긴 선분만을 차선으로 인식하여 반환
    return extended_lane_mask, left_line_mask, right_line_mask, original_lane_mask, masked_edges, color_mask, final_mask
