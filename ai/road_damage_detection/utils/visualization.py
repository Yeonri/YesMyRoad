import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.font_utils import put_text_on_image

def save_analysis_results(road_area, crack_area, pothole_area, warped_road_area, warped_crack_area, warped_pothole_area, img_num):
    """분석 결과를 텍스트 파일로 저장하는 함수"""
    # 결과 저장 경로
    result_dir = './result/'
    
    # 디렉토리가 없으면 생성
    os.makedirs(result_dir, exist_ok=True)
    
    # 파일 경로 설정
    file_path = os.path.join(result_dir, f'analysis_result_{img_num}.txt')
    
    # 결과 작성
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"도로 영역 분석 결과:\n")
        f.write(f"도로 면적: {road_area} 픽셀\n")
        f.write(f"균열 면적: {crack_area} 픽셀 ({crack_area/road_area*100:.2f}%)\n")
        f.write(f"포트홀 면적: {pothole_area} 픽셀 ({pothole_area/road_area*100:.2f}%)\n")
        f.write(f"총 손상 면적: {crack_area + pothole_area} 픽셀 ({(crack_area + pothole_area)/road_area*100:.2f}%)\n\n")
        
        f.write(f"원근 보정된 도로 영역 분석 결과:\n")
        f.write(f"도로 면적: {warped_road_area} 픽셀\n")
        f.write(f"균열 면적: {warped_crack_area} 픽셀 ({warped_crack_area/warped_road_area*100:.2f}%)\n")
        f.write(f"포트홀 면적: {warped_pothole_area} 픽셀 ({warped_pothole_area/warped_road_area*100:.2f}%)\n")
        f.write(f"총 손상 면적: {warped_crack_area + warped_pothole_area} 픽셀 ({(warped_crack_area + warped_pothole_area)/warped_road_area*100:.2f}%)\n")
    
    print(f"분석 결과가 {file_path}에 저장되었습니다.")

def save_key_images(original_img, overlay_original, img_with_points, overlay_warped, img_num):
    """핵심 이미지 4개를 하나의 이미지 파일로 저장하는 함수"""
    # 결과 저장 경로
    result_dir = './result/'
    
    # 디렉토리가 없으면 생성
    os.makedirs(result_dir, exist_ok=True)
    
    # 파일 경로 설정
    file_path = os.path.join(result_dir, f'key_images_{img_num}.jpg')
    
    # 2x2 그리드로 이미지 배치
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    plt.imshow(original_img)
    plt.title('원본 이미지')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(overlay_original)
    plt.title('손상 감지 (원본)')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_with_points)
    plt.title('원근 변환 포인트')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(overlay_warped)
    plt.title('손상 감지 (원근 보정)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"도로 손상 분석 결과 (이미지 {img_num})", fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # 이미지 저장
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"핵심 이미지가 {file_path}에 저장되었습니다.")

def visualize_corner_detection(img, mask):
    """코너 검출 과정과 결과를 시각화하는 함수 - 하이브리드 방식 시각화"""
    height, width = mask.shape
    
    # 코너 검출에 사용된 마스크 준비
    mask_smooth = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Harris 코너 검출
    corners = cv2.cornerHarris(mask_smooth, blockSize=5, ksize=3, k=0.04)
    
    # 결과 이진화 및 팽창
    corners_dilated = cv2.dilate(corners, None)
    _, corners_binary = cv2.threshold(corners_dilated, 0.01 * corners_dilated.max(), 255, 0)
    corners_binary = np.uint8(corners_binary)

    # 코너 좌표 추출
    corner_coords = np.where(corners_binary > 0)
    all_corners = []
    for i in range(len(corner_coords[0])):
        all_corners.append([corner_coords[1][i], corner_coords[0][i]])  # x, y 순서로 저장
    
    # 이미지 상단/하단 구분
    h_mid = height // 2
    w_mid = width // 2
    
    # 상단 영역의 코너 포인트 분류
    top_left_corners = [p for p in all_corners if p[0] < w_mid and p[1] < h_mid]
    top_right_corners = [p for p in all_corners if p[0] >= w_mid and p[1] < h_mid]
    
    # 최적의 상단 코너 포인트 계산
    selected_corners = []
    if top_left_corners:
        top_left = min(top_left_corners, key=lambda p: (p[0]-w_mid)**2 + p[1]**2)
        selected_corners.append(top_left)
    
    if top_right_corners:
        top_right = min(top_right_corners, key=lambda p: (p[0]-w_mid)**2 + p[1]**2)
        selected_corners.append(top_right)
    
    # 하단 포인트 계산 (기존 방식)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) > 0:
        main_contour = max(contours, key=cv2.contourArea)
        all_points = main_contour.reshape(-1, 2)
        
        if len(all_points) > 0:
            # 왼쪽 하단: x값 최소, y값 최대
            left_bottom_idx = np.lexsort((all_points[:, 0], -all_points[:, 1]))[0]
            bottom_left = all_points[left_bottom_idx]
            selected_corners.append(bottom_left)
            
            # 오른쪽 하단: x값 최대, y값 최대
            right_bottom_idx = np.lexsort((-all_points[:, 0], -all_points[:, 1]))[0]
            bottom_right = all_points[right_bottom_idx]
            selected_corners.append(bottom_right)
    
    # 시각화
    plt.figure(figsize=(15, 10))
    
    # 원본 마스크
    plt.subplot(2, 3, 1)
    plt.imshow(mask, cmap='gray')
    plt.title('원본 마스크')
    plt.axis('off')
    
    # 스무딩된 마스크
    plt.subplot(2, 3, 2)
    plt.imshow(mask_smooth, cmap='gray')
    plt.title('스무딩된 마스크')
    plt.axis('off')
    
    # Harris 코너 검출 결과
    plt.subplot(2, 3, 3)
    plt.imshow(corners, cmap='jet')
    plt.title('Harris 코너 검출 결과')
    plt.axis('off')
    
    # 이진화된 코너 결과
    plt.subplot(2, 3, 4)
    plt.imshow(corners_binary, cmap='gray')
    plt.title('이진화된 코너 결과')
    plt.axis('off')
    
    # 상단 코너와 윤곽선 구분
    plt.subplot(2, 3, 5)
    visualization = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 상단 영역 표시 (밝은 회색)
    visualization[:h_mid, :] = [64, 64, 64]
    
    # 윤곽선 표시
    if contours and len(contours) > 0:
        cv2.drawContours(visualization, [main_contour], 0, [0, 128, 0], 2)
    
    # 상단 코너 표시
    for p in top_left_corners:
        cv2.circle(visualization, tuple(p), 2, [255, 0, 0], -1)  # 빨간색
    for p in top_right_corners:
        cv2.circle(visualization, tuple(p), 2, [0, 255, 0], -1)  # 초록색
    
    plt.imshow(visualization)
    plt.title('상단 코너와 전체 윤곽선')
    plt.axis('off')
    
    # 선택된 코너 포인트
    plt.subplot(2, 3, 6)
    img_with_corners = img.copy()
    point_names = ['좌상단(코너)', '우상단(코너)', '좌하단(윤곽선)', '우하단(윤곽선)']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, p in enumerate(selected_corners):
        if i < len(point_names):
            cv2.circle(img_with_corners, tuple(p), 10, colors[i], -1)
            # PIL로 한글 텍스트 추가
            img_with_corners = put_text_on_image(
                img_with_corners, 
                point_names[i], 
                (p[0] + 15, p[1] + 15), 
                font_size=0.7, 
                color=colors[i]
            )
    
    plt.imshow(img_with_corners)
    plt.title('선택된 하이브리드 코너 포인트')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("코너 검출 및 윤곽선 기반 포인트 선택 과정", fontsize=20)
    plt.subplots_adjust(top=0.9)
    plt.show()

def visualize_results(original_img, road_mask, lane_mask, left_lane_mask, right_lane_mask,
                     road_lane_mask, road_polygon, mask_crack, mask_pothole, 
                     warped_img, warped_mask, warped_crack, warped_pothole, corner_points,
                     edges, masked_edges, img_num):
    """결과 시각화 함수"""
    plt.figure(figsize=(20, 15))
    
    # 첫 번째 행: 기본 이미지와 마스크
    plt.subplot(3, 4, 1)
    plt.imshow(original_img)
    plt.title('원본 이미지')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(road_mask, cmap='gray')
    plt.title('도로 마스크')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(lane_mask, cmap='gray')
    plt.title('차선 마스크')
    plt.axis('off')
    
    # 왼쪽/오른쪽 차선 구분
    plt.subplot(3, 4, 4)
    lane_combined = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
    lane_combined[left_lane_mask == 255] = [255, 0, 0]  # 왼쪽 차선: 빨간색
    lane_combined[right_lane_mask == 255] = [0, 0, 255]  # 오른쪽 차선: 파란색
    plt.imshow(lane_combined)
    plt.title('왼쪽/오른쪽 차선 구분')
    plt.axis('off')
    
    # 두 번째 행: 도로 분석
    plt.subplot(3, 4, 5)
    plt.imshow(road_lane_mask, cmap='gray')
    plt.title('차선 내 도로 마스크')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    polygon_viz = original_img.copy()
    if road_polygon is not None:
        cv2.polylines(polygon_viz, [road_polygon], True, (0, 255, 0), 3)
    plt.imshow(polygon_viz)
    plt.title('도로 폴리곤')
    plt.axis('off')
    
    # 손상 감지 시각화 (원본)
    plt.subplot(3, 4, 7)
    overlay_original = original_img.copy()
    alpha = 0.5
    road_overlay = overlay_original.copy()
    road_overlay[road_lane_mask == 255] = [0, 255, 0]  # 도로: 초록색
    overlay_original = cv2.addWeighted(road_overlay, alpha, overlay_original, 1-alpha, 0)
    overlay_original[mask_crack == 255] = [255, 0, 0]  # 균열: 빨간색
    overlay_original[mask_pothole == 255] = [0, 0, 255]  # 포트홀: 파란색
    plt.imshow(overlay_original)
    plt.title('손상 감지 (원본)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(warped_img)
    plt.title('원근 보정 이미지')
    plt.axis('off')
    
    # 세 번째 행: 추가 분석
    plt.subplot(3, 4, 9)
    plt.imshow(edges, cmap='gray')
    plt.title('캐니 엣지 (전체)')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(masked_edges, cmap='gray')
    plt.title('캐니 엣지 (상단 25% 제외)')
    plt.axis('off')
    
    # 원근 변환 포인트 시각화
    plt.subplot(3, 4, 11)
    img_with_points = original_img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    point_names = ['왼쪽 아래', '오른쪽 아래', '왼쪽 위', '오른쪽 위']
    
    for i, point in enumerate(corner_points):
        cv2.circle(img_with_points, (int(point[0]), int(point[1])), 10, colors[i], -1)
        # PIL로 한글 텍스트 추가
        img_with_points = put_text_on_image(
            img_with_points, 
            point_names[i], 
            (int(point[0]) + 15, int(point[1]) + 15), 
            font_size=0.7, 
            color=colors[i]
        )
    plt.imshow(img_with_points)
    plt.title('원근 변환 포인트')
    plt.axis('off')
    
    # 원근 보정된 손상 감지
    plt.subplot(3, 4, 12)
    overlay_warped = warped_img.copy()
    road_overlay_warped = overlay_warped.copy()
    road_overlay_warped[warped_mask == 255] = [0, 255, 0]
    overlay_warped = cv2.addWeighted(road_overlay_warped, alpha, overlay_warped, 1-alpha, 0)
    overlay_warped[warped_crack == 255] = [255, 0, 0]
    overlay_warped[warped_pothole == 255] = [0, 0, 255]
    plt.imshow(overlay_warped)
    plt.title('손상 감지 (원근 보정)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("도로 손상 감지 및 분석 과정", fontsize=20)
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    # 손상 면적 계산 및 출력
    road_area = np.count_nonzero(road_lane_mask)
    crack_area = np.count_nonzero(cv2.bitwise_and(mask_crack, road_lane_mask))
    pothole_area = np.count_nonzero(cv2.bitwise_and(mask_pothole, road_lane_mask))
    
    print(f"\n도로 영역 분석 결과:")
    print(f"도로 면적: {road_area} 픽셀")
    print(f"균열 면적: {crack_area} 픽셀 ({crack_area/road_area*100:.2f}%)")
    print(f"포트홀 면적: {pothole_area} 픽셀 ({pothole_area/road_area*100:.2f}%)")
    print(f"총 손상 면적: {crack_area + pothole_area} 픽셀 ({(crack_area + pothole_area)/road_area*100:.2f}%)")

    # 원근 보정된 이미지에서의 손상 면적 계산 및 출력
    warped_road_area = np.count_nonzero(warped_mask)
    warped_crack_area = np.count_nonzero(cv2.bitwise_and(warped_crack, warped_mask))
    warped_pothole_area = np.count_nonzero(cv2.bitwise_and(warped_pothole, warped_mask))
    
    print(f"\n원근 보정된 도로 영역 분석 결과:")
    print(f"도로 면적: {warped_road_area} 픽셀")
    print(f"균열 면적: {warped_crack_area} 픽셀 ({warped_crack_area/warped_road_area*100:.2f}%)")
    print(f"포트홀 면적: {warped_pothole_area} 픽셀 ({warped_pothole_area/warped_road_area*100:.2f}%)")
    print(f"총 손상 면적: {warped_crack_area + warped_pothole_area} 픽셀 ({(warped_crack_area + warped_pothole_area)/warped_road_area*100:.2f}%)")

    # 분석 결과 텍스트 파일로 저장
    save_analysis_results(road_area, crack_area, pothole_area, warped_road_area, warped_crack_area, warped_pothole_area, img_num)
    
    # 핵심 이미지 저장
    save_key_images(original_img, overlay_original, img_with_points, overlay_warped, img_num)
