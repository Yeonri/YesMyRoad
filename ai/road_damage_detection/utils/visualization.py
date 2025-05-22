"""
시각화 관련 모듈

이 모듈은 도로 손상 감지 및 분석 결과를 시각화하는 기능을 제공합니다.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from .font_utils import put_text_on_image
from .analysis import get_lrpci_grade, calculate_lrpci
from .detection import analyze_crack_severity, check_pothole_emergency


def visualize_results_masked(original_img, undistorted_img, road_mask, lane_mask, left_lane_mask, right_lane_mask,
                     road_lane_mask, road_polygon, mask_crack, mask_pothole, 
                     warped_img, warped_mask, warped_crack, warped_pothole, corner_points,
                     original_lane_mask, masked_edges, color_mask, final_mask,
                     img_num=None, pixel_to_cm_ratio=1.0):
    """
    도로 손상 감지 및 분석 결과를 시각화하는 함수

    Process:
    1. 4x3 그리드로 12개 이미지 구성 (원본, 왜곡 보정, 도로 마스크, 엣지 마스크, 색상 마스크 등)
    2. 도로 및 손상 영역 시각화
    3. 손상 면적 및 비율 계산
    4. 균열 심각도 및 포트홀 응급 상황 분석
    5. LrPCI 계산 및 보수 방법 결정
    6. 분석 결과 출력 및 저장

    Parameters:
        - original_img (numpy.ndarray) : 원본 이미지
        - undistorted_img (numpy.ndarray) : 왜곡 보정된 이미지
        - road_mask (numpy.ndarray) : 도로 마스크
        - lane_mask (numpy.ndarray) : 차선 마스크
        - left_lane_mask (numpy.ndarray) : 왼쪽 차선 마스크
        - right_lane_mask (numpy.ndarray) : 오른쪽 차선 마스크
        - road_lane_mask (numpy.ndarray) : 차선 내부 도로 마스크
        - road_polygon (numpy.ndarray) : 도로 폴리곤 좌표
        - mask_crack (numpy.ndarray) : 균열 마스크
        - mask_pothole (numpy.ndarray) : 포트홀 마스크
        - warped_img (numpy.ndarray) : 원근 변환된 이미지
        - warped_mask (numpy.ndarray) : 원근 변환된 도로 마스크
        - warped_crack (numpy.ndarray) : 원근 변환된 균열 마스크
        - warped_pothole (numpy.ndarray) : 원근 변환된 포트홀 마스크
        - corner_points (numpy.ndarray) : 원근 변환 코너 포인트
        - original_lane_mask (numpy.ndarray) : 원본 차선 마스크 (연장 전)
        - masked_edges (numpy.ndarray) : 엣지 마스크 (ROI 적용)
        - color_mask (numpy.ndarray) : 색상 기반 마스크
        - final_mask (numpy.ndarray) : 엣지와 색상 마스크 결합
        - img_num (str) : 이미지 번호 (저장 시 사용)
        - pixel_to_cm_ratio (float) : 픽셀 대 센티미터 비율

    Returns:
        - lrpci (float) : LrPCI 점수
        - warped_crack_ratio (float) : 원근 보정된 이미지의 균열 비율
        - warped_pothole_ratio (float) : 원근 보정된 이미지의 포트홀 비율
        - crack_severity (float) : 균열 심각도
        - crack_type (str) : 균열 유형
        - crack_info (dict) : 균열 상세 정보
        - pothole_info (dict) : 포트홀 상세 정보
    """
    plt.figure(figsize=(22, 16))
   
    # 1. 원본 이미지
    plt.subplot(4, 3, 1)
    plt.imshow(original_img)
    plt.title('1. 원본 이미지')
    plt.axis('off')

    # 2. 왜곡 보정 이미지
    plt.subplot(4, 3, 2)
    plt.imshow(undistorted_img)
    plt.title('2. 왜곡 보정 이미지')
    plt.axis('off')
    
    # 3. 도로 마스크
    plt.subplot(4, 3, 3)
    plt.imshow(road_mask, cmap='gray')
    plt.title('3. 도로 마스크')
    plt.axis('off')
    
    # 4. 엣지 마스크
    plt.subplot(4, 3, 4)
    plt.imshow(masked_edges, cmap='gray')
    plt.title('4. 엣지 마스크')
    plt.axis('off')
    
    # 5. 색상 마스크
    plt.subplot(4, 3, 5)
    plt.imshow(color_mask, cmap='gray')
    plt.title('5. 색상 마스크')
    plt.axis('off')
    
    # 6. 엣지+색상 마스크
    plt.subplot(4, 3, 6)
    plt.imshow(final_mask, cmap='gray')
    plt.title('6. 엣지+색상 마스크')
    plt.axis('off')
    
    # 7. 차선 마스크 (RANSAC 적용 전)
    plt.subplot(4, 3, 7)
    plt.imshow(original_lane_mask, cmap='gray')  # RANSAC 적용 전 차선 마스크
    plt.title('7. 차선 마스크 (원본)')
    plt.axis('off')
    
    # 8. 차선 마스크 (RANSAC 적용 후)
    plt.subplot(4, 3, 8)
    lane_combined = np.zeros((undistorted_img.shape[0], undistorted_img.shape[1], 3), dtype=np.uint8)
    lane_combined[left_lane_mask == 255] = [255, 0, 0]  # 왼쪽 차선: 빨간색
    lane_combined[right_lane_mask == 255] = [0, 0, 255]  # 오른쪽 차선: 파란색
    plt.imshow(lane_combined)
    plt.title('8. 차선 마스크 (연장 적용 후)')
    plt.axis('off')
    
    # 9. 최종 도로 및 차선 감지
    plt.subplot(4, 3, 9)
    road_overlay = undistorted_img.copy()
    alpha = 0.5
    
    # 도로 영역 반투명 초록색으로 표시
    road_highlight = road_overlay.copy()
    road_highlight[road_lane_mask == 255] = [0, 255, 0]  # 도로: 초록색
    road_overlay = cv2.addWeighted(road_highlight, alpha, road_overlay, 1-alpha, 0)
    
    # 차선 표시
    road_overlay[lane_mask == 255] = [255, 255, 0]  # 차선: 노란색
    
    # 도로 폴리곤 표시
    if road_polygon is not None:
        cv2.polylines(road_overlay, [road_polygon], True, (0, 255, 255), 2)
    
    plt.imshow(road_overlay)
    plt.title('9. 최종 도로 및 차선 감지')
    plt.axis('off')
    
    # 10. 손상 감지 (왜곡 보정)
    plt.subplot(4, 3, 10)
    overlay_undistorted = undistorted_img.copy()
    alpha = 0.5
    road_highlight = overlay_undistorted.copy()
    road_highlight[road_lane_mask == 255] = [0, 255, 0]  # 도로: 초록색
    overlay_undistorted = cv2.addWeighted(road_highlight, alpha, overlay_undistorted, 1-alpha, 0)
    overlay_undistorted[mask_crack == 255] = [255, 0, 0]  # 균열: 빨간색
    overlay_undistorted[mask_pothole == 255] = [0, 0, 255]  # 포트홀: 파란색
    
    plt.imshow(overlay_undistorted)
    plt.title('10. 손상 감지 (왜곡 보정)')
    plt.axis('off')

    # 11. 원근 변환 포인트
    plt.subplot(4, 3, 11)
    img_with_points = undistorted_img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    point_names = ['왼쪽 위', '오른쪽 위', '왼쪽 아래', '오른쪽 아래']
    
    # 도로 영역 윤곽선
    if road_polygon is not None:
        cv2.polylines(img_with_points, [road_polygon], True, (0, 255, 255), 2)
    
    # 변환 포인트 표시
    for i, point in enumerate(corner_points):
        cv2.circle(img_with_points, (int(point[0]), int(point[1])), 10, colors[i], -1)
        img_with_points = put_text_on_image(
            img_with_points, 
            point_names[i], 
            (int(point[0]) + 15, int(point[1]) + 15), 
            font_size=0.7, 
            color=colors[i]
        )
    
    plt.imshow(img_with_points)
    plt.title('11. 원근 변환 포인트')
    plt.axis('off')

    # 12. 손상 감지 (원근 보정) - 코너 포인트 외부 영역 제거
    plt.subplot(4, 3, 12)
    plt.imshow(warped_img)  # 이미 마스킹된 이미지 사용
    plt.title('12. 손상 감지 (원근 보정)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle("도로 손상 감지 및 분석 결과", fontsize=16)
    plt.subplots_adjust(top=0.94)
    
    # 원본 이미지에서의 손상 면적 계산
    road_area = np.count_nonzero(road_lane_mask)
    crack_area = np.count_nonzero(cv2.bitwise_and(mask_crack, road_lane_mask))
    pothole_area = np.count_nonzero(cv2.bitwise_and(mask_pothole, road_lane_mask))
    
    # 원본 이미지의 손상률 계산
    crack_ratio = crack_area / road_area if road_area > 0 else 0
    pothole_ratio = pothole_area / road_area if road_area > 0 else 0
    total_damage_ratio = crack_ratio + pothole_ratio
    
    # 원근 보정된 이미지에서의 손상 면적 계산
    warped_road_area = np.count_nonzero(warped_mask)
    warped_crack_area = np.count_nonzero(cv2.bitwise_and(warped_crack, warped_mask))
    warped_pothole_area = np.count_nonzero(cv2.bitwise_and(warped_pothole, warped_mask))
    
    # 원근 보정된 이미지의 손상률 계산
    warped_crack_ratio = warped_crack_area / warped_road_area if warped_road_area > 0 else 0
    warped_pothole_ratio = warped_pothole_area / warped_road_area if warped_road_area > 0 else 0
    warped_total_damage_ratio = warped_crack_ratio + warped_pothole_ratio
    
    # 균열 심각도 분석 - 원근 보정된 이미지 사용
    crack_severity, crack_type, crack_info = analyze_crack_severity(
        cv2.bitwise_and(warped_crack, warped_mask), pixel_to_cm_ratio
    )
    
    # 포트홀 응급 상황 확인 - 원근 보정된 이미지 사용
    is_emergency, pothole_info = check_pothole_emergency(
        cv2.bitwise_and(warped_pothole, warped_mask), 
        pixel_to_cm_ratio
    )
    
    # 원근 보정된 이미지 기반으로 LrPCI 계산
    lrpci = calculate_lrpci(warped_crack_ratio, warped_pothole_ratio, crack_severity, crack_type, pothole_info)
    
    # 원본 이미지 분석 결과 출력
    print(f"\n도로 영역 분석 결과 (왜곡 보정 이미지):")
    print(f"도로 면적: {road_area} 픽셀")
    print(f"균열 면적: {crack_area} 픽셀 ({crack_ratio*100:.2f}%)")
    print(f"포트홀 면적: {pothole_area} 픽셀 ({pothole_ratio*100:.2f}%)")
    print(f"총 손상 면적: {crack_area + pothole_area} 픽셀 ({total_damage_ratio*100:.2f}%)")
    
    # 원근 보정된 이미지 분석 결과 출력 (LrPCI 포함)
    print(f"\n도로 영역 분석 결과 (원근 보정된 이미지):")
    print(f"도로 면적: {warped_road_area} 픽셀")
    print(f"균열 면적: {warped_crack_area} 픽셀 ({warped_crack_ratio*100:.2f}%)")
    print(f"포트홀 면적: {warped_pothole_area} 픽셀 ({warped_pothole_ratio*100:.2f}%)")
    print(f"총 손상 면적: {warped_crack_area + warped_pothole_area} 픽셀 ({warped_total_damage_ratio*100:.2f}%)")
    
    # LrPCI 등급 계산
    grade, description = get_lrpci_grade(lrpci)
    print(f"LrPCI 평가 점수: {lrpci}/100.00 ({grade}: {description})")
    
    plt.show()
    
    return lrpci, warped_crack_ratio, warped_pothole_ratio, crack_severity, crack_type, crack_info, pothole_info
