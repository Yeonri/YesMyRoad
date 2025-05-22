"""
도로 손상 감지 시스템의 메인 스크립트

이 스크립트는 도로 이미지를 입력받아 손상을 감지하고 분석하는 전체 파이프라인을 실행합니다.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils.font_utils import setup_matplotlib_font
from utils.calibration import undistort_image
from utils.road_extraction import create_roi_mask, extract_road_mask, extract_road_between_lanes
from utils.lane_detection import detect_lane_lines
from utils.detection import apply_yolo_on_masked_area
from utils.transformation import extract_road_corners_direct, perspective_transform_direct_masked
from utils.visualization import visualize_results_masked


def main():
    """
    전체 도로 손상 감지 파이프라인을 실행하는 메인 함수
    """
    # 폰트 설정
    setup_matplotlib_font()
    
    # 모델 로드
    model_path = './model/yolo11-seg.pt'
    model = YOLO(model_path)
    print(f"YOLOv11-seg 모델 '{model_path}'을(를) 로드했습니다.")
    
    # 캘리브레이션 파일 경로
    calibration_file = './model/camera_calibration_result.npy'
    if not os.path.exists(calibration_file):
        print(f"캘리브레이션 파일이 존재하지 않습니다: {calibration_file}")
        print("원본 이미지를 사용합니다.")
    else:
        print(f"카메라 캘리브레이션 데이터 '{calibration_file}'을(를) 사용합니다.")
    
    # 이미지 로드
    img_num = input("테스트할 이미지 번호를 입력하세요: ")
    img_path = f'./image/test_image_{img_num}.jpg'
    print(f"이미지 '{img_path}'을(를) 로드했습니다.")
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"이미지를 로드할 수 없습니다: {img_path}")
        return
    
    # 이미지 왜곡 보정 적용
    undistorted_bgr, calibration_info = undistort_image(img_bgr, calibration_file)
    mtx, dist, roi = calibration_info
    
    if mtx is None:
        print("캘리브레이션 파일을 로드할 수 없거나 잘못된 형식입니다. 원본 이미지를 사용합니다.")
        undistorted_bgr = img_bgr
    else:
        print("카메라 캘리브레이션을 적용하여 이미지 왜곡을 보정했습니다.")
    
    # BGR -> RGB 변환
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
    img_height, img_width = undistorted_rgb.shape[:2]
    
    # ROI를 자동으로 trapezoid로 설정
    use_roi = True
    roi_type = "trapezoid"
    
    # ROI 마스크 생성
    roi_mask = create_roi_mask(undistorted_bgr, use_roi=True, roi_type=roi_type)
    print(f"ROI 타입 '{roi_type}'을(를) 자동으로 적용했습니다.")
    
    # ROI가 비어있는지 확인
    if np.count_nonzero(roi_mask) == 0:
        print("경고: ROI 마스크가 비어 있습니다. 기본 사다리꼴 ROI를 대신 사용합니다.")
        roi_mask = create_roi_mask(undistorted_bgr, use_roi=True, roi_type="trapezoid")
    
    # 도로 마스크 생성
    road_mask = extract_road_mask(undistorted_bgr, use_roi=use_roi, roi_mask=roi_mask)
    
    # 차선 마스크 생성 (추가된 반환값: color_mask, final_mask)
    lane_mask, left_lane_mask, right_lane_mask, original_lane_mask, masked_edges, color_mask, final_mask = detect_lane_lines(undistorted_bgr, use_roi=use_roi, roi_mask=roi_mask)
    
    # 차선 내부의 도로와 아래 부분까지 마스크
    road_lane_mask, road_polygon = extract_road_between_lanes(undistorted_bgr, road_mask, left_lane_mask, right_lane_mask)
    
    # 도로 마스크나 차선 마스크가 비어 있는지 확인
    if np.count_nonzero(road_lane_mask) == 0:
        print("경고: 도로 차선 마스크가 비어 있습니다. ROI 설정을 다시 확인하세요.")
        # 기본 마스크 사용
        road_lane_mask = np.ones((img_height, img_width), dtype=np.uint8) * 255
        if use_roi and roi_mask is not None:
            road_lane_mask = cv2.bitwise_and(road_lane_mask, roi_mask)
    
    # 마스크된 영역에서 YOLO 추론
    mask_crack, mask_pothole = apply_yolo_on_masked_area(model, undistorted_rgb, road_lane_mask)
    
    # 원근 변환을 위한 코드
    try:
        # 도로 마스크에서 직접 코너 포인트 추출
        corner_points = extract_road_corners_direct(road_lane_mask, roi_mask)
        
        # 마스킹이 적용된 원근 변환 적용
        warped_img, warped_mask, perspective_matrix, src_points = perspective_transform_direct_masked(
            undistorted_rgb, road_lane_mask, roi_mask)
        
        # 손상 마스크에도 원근 변환 적용
        warped_crack = cv2.warpPerspective(mask_crack, perspective_matrix, (img_width, img_height))
        warped_pothole = cv2.warpPerspective(mask_pothole, perspective_matrix, (img_width, img_height))
        
        # 변환된 손상 마스크도 동일한 영역으로 마스킹
        # 변환된 코너 포인트 영역 마스크 생성
        height, width = img_height, img_width
        road_width_pixels = width / 2
        road_length_pixels = height  # 간단하게 이미지 높이로 설정
        
        left_margin = (width - road_width_pixels) / 2
        right_margin = left_margin + road_width_pixels
        
        transformed_mask = np.zeros((height, width), dtype=np.uint8)
        transformed_rect = np.int32([
            [left_margin, 0],                   # 왼쪽 상단
            [right_margin, 0],                  # 오른쪽 상단
            [right_margin, road_length_pixels], # 오른쪽 하단
            [left_margin, road_length_pixels]   # 왼쪽 하단
        ])
        cv2.fillPoly(transformed_mask, [transformed_rect], 255)
        
        # 손상 마스크 마스킹
        warped_crack = cv2.bitwise_and(warped_crack, transformed_mask)
        warped_pothole = cv2.bitwise_and(warped_pothole, transformed_mask)
    except Exception as e:
        print(f"원근 변환 중 오류 발생: {e}")
        # 원근 변환 실패 시 원본 이미지 및 마스크 사용
        warped_img = undistorted_rgb.copy()
        warped_mask = road_lane_mask.copy()
        warped_crack = mask_crack.copy()
        warped_pothole = mask_pothole.copy()
        
        # 기본 코너 포인트 (이미지 전체)
        src_points = np.float32([
            [0, 0], [img_width-1, 0], [0, img_height-1], [img_width-1, img_height-1]
        ])
        corner_points = src_points
    
    # 결과 시각화
    lrpci = visualize_results_masked(
        img_rgb, undistorted_rgb, road_mask, lane_mask, left_lane_mask, right_lane_mask,
        road_lane_mask, road_polygon, mask_crack, mask_pothole, 
        warped_img, warped_mask, warped_crack, warped_pothole, 
        corner_points, original_lane_mask, masked_edges, color_mask, final_mask, img_num
    )
    
if __name__ == "__main__":
    main()
