import cv2
import numpy as np
from ultralytics import YOLO

# 자체 모듈 임포트
from utils import setup_matplotlib_font
from road_detection import (
    extract_road_mask, 
    detect_lane_lines, 
    extract_road_between_lanes, 
    perspective_transform
)
from damage_detection import apply_yolo_on_masked_area
from visualization import visualize_results, analyze_damage_area

def process_image(img_path, model):
    """이미지를 처리하고 결과를 반환하는 함수"""
    print(f"이미지 '{img_path}'을(를) 처리합니다.")
    
    # 이미지 로드
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"이미지 '{img_path}'를 로드할 수 없습니다.")
        return None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_rgb.shape[:2]
    
    # 도로 마스크 생성
    road_mask = extract_road_mask(img_bgr)
    
    # 차선 마스크 생성
    lane_mask, left_lane_mask, right_lane_mask, edges, masked_edges = detect_lane_lines(img_bgr)
    
    # 차선 내부의 도로와 아래 부분까지 마스크
    road_lane_mask, road_polygon = extract_road_between_lanes(img_bgr, road_mask, left_lane_mask, right_lane_mask)
    
    # 마스크된 영역에서 YOLO 추론
    mask_crack, mask_pothole = apply_yolo_on_masked_area(model, img_rgb, road_lane_mask)
    
    # 원근 변환 적용 - road_lane_mask를 사용
    warped_img, warped_mask, perspective_matrix, corner_points = perspective_transform(
        img_rgb, road_lane_mask, road_polygon)
    
    # 손상 마스크에도 원근 변환 적용
    warped_crack = cv2.warpPerspective(mask_crack, perspective_matrix, (img_width, img_height))
    warped_pothole = cv2.warpPerspective(mask_pothole, perspective_matrix, (img_width, img_height))
    
    # 결과 시각화
    visualize_results(
        img_rgb, road_mask, lane_mask, left_lane_mask, right_lane_mask,
        road_lane_mask, road_polygon, mask_crack, mask_pothole, 
        warped_img, warped_mask, warped_crack, warped_pothole, corner_points,
        edges, masked_edges
    )
    
    # 손상 면적 분석
    damage_stats = analyze_damage_area(
        road_lane_mask, mask_crack, mask_pothole, 
        warped_mask, warped_crack, warped_pothole
    )
    
    return {
        'images': {
            'original': img_rgb,
            'road_mask': road_mask,
            'lane_mask': lane_mask,
            'left_lane_mask': left_lane_mask,
            'right_lane_mask': right_lane_mask,
            'road_lane_mask': road_lane_mask,
            'mask_crack': mask_crack,
            'mask_pothole': mask_pothole,
            'warped_img': warped_img,
            'warped_mask': warped_mask,
            'warped_crack': warped_crack,
            'warped_pothole': warped_pothole
        },
        'geometry': {
            'road_polygon': road_polygon,
            'corner_points': corner_points,
            'perspective_matrix': perspective_matrix
        },
        'damage_stats': damage_stats
    }

def main():
    # 폰트 설정
    setup_matplotlib_font()
    
    # 모델 로드
    model_path = './model/yolo11l-seg.pt'
    try:
        model = YOLO(model_path)
        print(f"YOLOv11-seg 모델 '{model_path}'을(를) 로드했습니다.")
    except Exception as e:
        print(f"모델 로드 중 오류가 발생했습니다: {e}")
        return
    
    # 이미지 처리
    while True:
        img_num = input("테스트할 이미지 번호를 입력하세요 (종료하려면 'q' 입력): ")
        
        if img_num.lower() == 'q':
            break
            
        try:
            img_path = f'./image/test_image_{img_num}.jpg'
            result = process_image(img_path, model)
            
            if result is None:
                print("이미지 처리에 실패했습니다.")
                continue
                
            # 여기서 result를 사용하여 추가 분석이나 저장 작업 가능
            
        except Exception as e:
            print(f"이미지 처리 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()