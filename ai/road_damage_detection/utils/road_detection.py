import cv2
import numpy as np

def extract_road_mask(img):
    """도로 영역 추출 함수"""
    # HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 도로 색상 범위 (회색 계열)
    lower_road = np.array([0, 0, 60])
    upper_road = np.array([180, 30, 160])
    road_mask = cv2.inRange(hsv, lower_road, upper_road)
    
    # 이미지 높이의 하단 75%만 고려 (상단 25% 제외)
    height = img.shape[0]
    road_mask[:int(height * 0.25), :] = 0
    
    # 노이즈 제거를 위한 모폴로지 연산
    kernel = np.ones((15, 15), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # 가장 큰 연결 요소만 유지
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > (height * img.shape[1] * 0.1):
            refined_mask = np.zeros_like(road_mask)
            cv2.drawContours(refined_mask, [max_contour], 0, 255, -1)
            road_mask = refined_mask
    
    return road_mask

def detect_lane_lines(img):
    """차선 감지 및 마스크 생성 함수"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러 및 엣지 감지
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # ROI 설정 - 상단 25%를 제외한 영역만 고려
    height, width = edges.shape
    roi_mask = np.ones_like(edges)
    roi_mask[:int(height * 0.25), :] = 0
    masked_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    
    # 색상 기반 차선 추출 (흰색/노란색)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 흰색 및 노란색 범위 설정
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    lower_yellow = np.array([10, 60, 80])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 색상 마스크 통합 및 ROI 적용
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=roi_mask)
    
    # 최종 마스크 (엣지 + 색상)
    final_mask = cv2.bitwise_or(masked_edges, color_mask)
    
    # 호프 변환으로 직선 검출
    lines = cv2.HoughLinesP(
        final_mask, 
        rho=1, 
        theta=np.pi/180, 
        threshold=40,
        minLineLength=30,
        maxLineGap=5
    )
    
    # 차선 마스크 초기화
    lane_mask = np.zeros_like(gray)
    left_line_mask = np.zeros_like(gray)
    right_line_mask = np.zeros_like(gray)
    
    # 차선 필터링 및 분류
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 선의 길이가 너무 짧으면 무시
            if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < 30:
                continue
                
            # 기울기 계산
            if x2 - x1 == 0:  # 수직선 방지
                continue
            else:
                slope = (y2 - y1) / (x2 - x1)
            
            # 적절한 기울기 범위만 선택 (약 15도 ~ 65도 범위)
            if 0.3 < abs(slope) < 2:  
                cv2.line(lane_mask, (x1, y1), (x2, y2), 255, 3)
                
                # 왼쪽/오른쪽 차선 구분
                mid_x = width // 2
                line_mid_x = (x1 + x2) // 2
                
                if slope < 0 and line_mid_x < mid_x:  # 왼쪽 차선 (음수 기울기)
                    cv2.line(left_line_mask, (x1, y1), (x2, y2), 255, 3)
                elif slope > 0 and line_mid_x > mid_x:  # 오른쪽 차선 (양수 기울기)
                    cv2.line(right_line_mask, (x1, y1), (x2, y2), 255, 3)
    
    # 차선 마스크 확장
    kernel = np.ones((3, 3), np.uint8)
    lane_mask = cv2.dilate(lane_mask, kernel, iterations=1)
    
    return lane_mask, left_line_mask, right_line_mask, edges, masked_edges

def extract_road_between_lanes(img, road_mask, left_lane_mask, right_lane_mask):
    """차선 내부의 도로 영역 추출 함수"""
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
