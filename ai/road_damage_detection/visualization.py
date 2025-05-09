import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(original_img, road_mask, lane_mask, left_lane_mask, right_lane_mask,
                     road_lane_mask, road_polygon, mask_crack, mask_pothole, 
                     warped_img, warped_mask, warped_crack, warped_pothole, corner_points,
                     edges, masked_edges):
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
    cv2.polylines(polygon_viz, [road_polygon], True, (0, 255, 0), 3)
    plt.imshow(polygon_viz)
    plt.title('도로 폴리곤')
    plt.axis('off')
    
    # 손상 감지 시각화
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
        cv2.putText(img_with_points, f'{point_names[i]}', 
                   (int(point[0]) + 15, int(point[1]) + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
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

def analyze_damage_area(road_lane_mask, mask_crack, mask_pothole, warped_mask, warped_crack, warped_pothole):
    """손상 면적 분석 및 결과 출력"""
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
    
    return {
        'original': {
            'road_area': road_area,
            'crack_area': crack_area,
            'pothole_area': pothole_area,
            'crack_percent': crack_area/road_area*100,
            'pothole_percent': pothole_area/road_area*100,
            'total_damage_percent': (crack_area + pothole_area)/road_area*100
        },
        'warped': {
            'road_area': warped_road_area,
            'crack_area': warped_crack_area,
            'pothole_area': warped_pothole_area,
            'crack_percent': warped_crack_area/warped_road_area*100,
            'pothole_percent': warped_pothole_area/warped_road_area*100,
            'total_damage_percent': (warped_crack_area + warped_pothole_area)/warped_road_area*100
        }
    }