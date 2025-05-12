import cv2
import numpy as np

def get_default_perspective_points(road_polygon, height, width):
    """road_polygon이 있을 경우 사용하고, 없으면 기본값 설정"""
    if road_polygon is not None and len(road_polygon) >= 6:
        # road_polygon에서 필요한 포인트 추출
        left_bottom = road_polygon[1]  # 왼쪽 차선 하단
        right_bottom = road_polygon[4] # 오른쪽 차선 하단
        left_top = road_polygon[2]     # 왼쪽 차선 상단
        right_top = road_polygon[3]    # 오른쪽 차선 상단
        
        # 사용할 코너 포인트
        return np.float32([left_bottom, right_bottom, left_top, right_top])
    else:
        # 기본값 설정
        bottom_left = np.array([0, height-1])
        bottom_right = np.array([width-1, height-1])
        top_left = np.array([width//4, height//3])
        top_right = np.array([width*3//4, height//3])
        
        return np.float32([bottom_left, bottom_right, top_left, top_right])

def perspective_transform(img, mask, road_polygon=None):
    """
    원근 변환 함수 - 상단 포인트는 코너 특징점으로, 하단 포인트는 기존 방식으로 찾기
    """
    height, width = mask.shape
    
    # 상단 포인트용 코너 검출에 적합한 마스크 준비
    mask_smooth = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Harris 코너 검출
    corners = cv2.cornerHarris(mask_smooth, blockSize=5, ksize=3, k=0.04)
    
    # 코너 검출 결과를 이진화하고 팽창 (더 뚜렷하게)
    corners_dilated = cv2.dilate(corners, None)
    _, corners_binary = cv2.threshold(corners_dilated, 0.01 * corners_dilated.max(), 255, 0)
    corners_binary = np.uint8(corners_binary)

    # 코너 포인트 좌표 추출
    corner_coords = np.where(corners_binary > 0)
    corner_y = corner_coords[0]
    corner_x = corner_coords[1]
    
    # 모든 코너 좌표를 리스트로 변환
    all_corners = []
    for i in range(len(corner_y)):
        all_corners.append([corner_x[i], corner_y[i]])
    
    # 코너가 충분히 많지 않으면 대체 방식 사용
    if len(all_corners) < 10:
        return fallback_perspective_transform(img, mask, road_polygon)
    
    # 이미지를 4분할하여 상단 영역에서 코너점 찾기
    h_mid = height // 2
    w_mid = width // 2
    
    # 상단 영역의 코너 포인트
    top_left_corners = [p for p in all_corners if p[0] < w_mid and p[1] < h_mid]
    top_right_corners = [p for p in all_corners if p[0] >= w_mid and p[1] < h_mid]
    
    # 상단 영역에 코너가 없는 경우 대체 방법 사용
    if not (top_left_corners and top_right_corners):
        return fallback_perspective_transform(img, mask, road_polygon)
    
    # 좌상단: 원점에서 가장 가까운 포인트
    top_left = min(top_left_corners, key=lambda p: (p[0]-w_mid)**2 + p[1]**2)
    
    # 우상단: (width, 0)에서 가장 가까운 포인트
    top_right = min(top_right_corners, key=lambda p: (p[0]-w_mid)**2 + p[1]**2)
    
    # 하단 포인트는 기존 방식으로 찾기
    # 도로 마스크에서 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or len(contours) == 0:
        return fallback_perspective_transform(img, mask, road_polygon)
    
    # 가장 큰 윤곽선 선택
    main_contour = max(contours, key=cv2.contourArea)
    all_points = main_contour.reshape(-1, 2)
    
    if len(all_points) == 0:
        return fallback_perspective_transform(img, mask, road_polygon)
    
    # 왼쪽 하단: x값 최소, y값 최대
    left_bottom_idx = np.lexsort((all_points[:, 0], -all_points[:, 1]))[0]
    bottom_left = all_points[left_bottom_idx]
    
    # 오른쪽 하단: x값 최대, y값 최대
    right_bottom_idx = np.lexsort((-all_points[:, 0], -all_points[:, 1]))[0]
    bottom_right = all_points[right_bottom_idx]
    
    # 포인트 추가 검증: y 좌표가 위아래 구분에 맞게 정렬
    if top_left[1] > bottom_left[1] or top_right[1] > bottom_right[1]:
        return fallback_perspective_transform(img, mask, road_polygon)
    
    # 포인트 추가 검증: x 좌표가 좌우 구분에 맞게 정렬
    if top_left[0] > top_right[0] or bottom_left[0] > bottom_right[0]:
        return fallback_perspective_transform(img, mask, road_polygon)
    
    # 원근 변환에 사용할 소스 포인트
    src_points = np.float32([
        bottom_left,    # 좌하단
        bottom_right,   # 우하단
        top_left,       # 좌상단
        top_right       # 우상단
    ])
    
    # 타겟 포인트 설정 (변환 후 이미지에서의 위치)
    dst_points = np.float32([
        [0, height-1],               # 왼쪽 하단
        [width-1, height-1],         # 오른쪽 하단
        [0, 0],                      # 왼쪽 상단
        [width-1, 0]                 # 오른쪽 상단
    ])
    
    # 원근 변환 행렬 계산 및 적용
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(img, M, (width, height))
    warped_mask = cv2.warpPerspective(mask, M, (width, height))
    
    return warped_img, warped_mask, M, src_points

def fallback_perspective_transform(img, mask, road_polygon=None):
    """
    코너 검출이 실패했을 때 사용하는 대체 원근 변환 함수
    기존의 방식을 사용
    """
    height, width = mask.shape
    
    # 도로 마스크에서 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and len(contours) > 0:
        # 가장 큰 윤곽선 선택
        main_contour = max(contours, key=cv2.contourArea)
        all_points = main_contour.reshape(-1, 2)
        
        if len(all_points) > 0:
            # 왼쪽 하단: x값 최소, y값 최대
            left_bottom_idx = np.lexsort((all_points[:, 0], -all_points[:, 1]))[0]
            left_bottom = all_points[left_bottom_idx]
            
            # 오른쪽 하단: x값 최대, y값 최대
            right_bottom_idx = np.lexsort((-all_points[:, 0], -all_points[:, 1]))[0]
            right_bottom = all_points[right_bottom_idx]
            
            # 상단 부분의 포인트 추출 (y값이 하위 10% 이내인 점들)
            y_threshold = np.percentile(all_points[:, 1], 10)
            top_points = all_points[all_points[:, 1] <= y_threshold]
            
            if len(top_points) > 0:
                # 왼쪽 상단: 상단 부분에서 x값이 가장 작은 점
                left_top_idx = np.argmin(top_points[:, 0])
                left_top = top_points[left_top_idx]
                
                # 오른쪽 상단: 상단 부분에서 x값이 가장 큰 점
                right_top_idx = np.argmax(top_points[:, 0])
                right_top = top_points[right_top_idx]
            else:
                # 상단 부분 포인트가 부족한 경우, y값을 우선 정렬하여 가장 작은 y값을 가진 포인트 중에서 선택
                sorted_by_y = all_points[np.argsort(all_points[:, 1])]
                top_n_points = sorted_by_y[:max(5, len(sorted_by_y) // 10)]  # 상위 10% 또는 최소 5개
                
                # 이 포인트들 중에서 왼쪽/오른쪽 구분
                left_top_idx = np.argmin(top_n_points[:, 0])
                left_top = top_n_points[left_top_idx]
                
                right_top_idx = np.argmax(top_n_points[:, 0])
                right_top = top_n_points[right_top_idx]
            
            src_points = np.float32([left_bottom, right_bottom, left_top, right_top])
        else:
            # 포인트가 없는 경우 기본값 사용
            src_points = get_default_perspective_points(road_polygon, height, width)
    else:
        # 윤곽선이 없는 경우 기본값 사용
        src_points = get_default_perspective_points(road_polygon, height, width)
    
    # 타겟 포인트 설정 (변환 후 이미지에서의 위치)
    dst_points = np.float32([
        [0, height-1],               # 왼쪽 하단
        [width-1, height-1],         # 오른쪽 하단
        [0, 0],                      # 왼쪽 상단
        [width-1, 0]                 # 오른쪽 상단
    ])
    
    # 원근 변환 행렬 계산 및 적용
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(img, M, (width, height))
    warped_mask = cv2.warpPerspective(mask, M, (width, height))
    
    return warped_img, warped_mask, M, src_points
