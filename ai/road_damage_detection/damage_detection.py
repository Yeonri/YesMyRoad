import cv2
import numpy as np

def apply_yolo_on_masked_area(model, img, mask):
    """마스크된 영역에서 YOLO 추론 수행"""
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