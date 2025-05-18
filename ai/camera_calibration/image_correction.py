import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def undistort_single_image(calibration_file, input_image_path, output_image_path=None, visualize=True):
    """
    캘리브레이션 파일을 활용하여 특정 이미지의 왜곡을 보정합니다.
    
    매개변수:
    - calibration_file: 캘리브레이션 결과가 저장된 .npy 파일 경로
    - input_image_path: 왜곡 보정할 이미지 파일 경로
    - output_image_path: 보정된 이미지를 저장할 경로 (None이면 저장하지 않음)
    - visualize: 원본 이미지와 보정된 이미지를 시각화할지 여부
    
    반환값:
    - 왜곡이 보정된 이미지 (numpy 배열)
    """
    # 캘리브레이션 결과 불러오기
    print(f"캘리브레이션 파일 '{calibration_file}' 불러오는 중...")
    try:
        calibration_data = np.load(calibration_file, allow_pickle=True).item()
        mtx = calibration_data['camera_matrix']  # 카메라 내부 매개변수 행렬
        dist = calibration_data['dist_coeffs']   # 왜곡 계수
        print("캘리브레이션 데이터 불러오기 성공!")
        print("\n카메라 행렬 (Camera Matrix):")
        print(mtx)
        print("\n왜곡 계수 (Distortion Coefficients):")
        print(dist)
    except Exception as e:
        print(f"캘리브레이션 파일을 불러오는 중 오류 발생: {e}")
        return None
    
    # 이미지 불러오기
    print(f"이미지 '{input_image_path}' 불러오는 중...")
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {input_image_path}")
        return None
    
    # 이미지 크기 가져오기
    h, w = img.shape[:2]
    print(f"이미지 크기: {w}x{h}")
    
    # 1. 최적의 카메라 행렬 구하기
    # alpha=1: 모든 픽셀을 유지, alpha=0: 불필요한 픽셀 제거
    print("왜곡 보정 중...")
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 2. 이미지 왜곡 보정 (undistort 메서드 사용)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    # 3. 결과 이미지에서 불필요한 픽셀 잘라내기
    x, y, w, h = roi
    if all(val > 0 for val in [x, y, w, h]):
        dst_cropped = dst[y:y+h, x:x+w]
        print(f"ROI 영역으로 잘라내기: ({x}, {y}, {w}, {h})")
    else:
        dst_cropped = dst
        print("잘라낼 ROI 영역이 유효하지 않습니다. 원본 크기 유지.")
    
    # 4. 보정된 이미지 저장 (지정된 경우)
    if output_image_path:
        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 잘라낸 이미지와 원본 크기 이미지 모두 저장
        cv2.imwrite(output_image_path, dst_cropped)
        
        # 원본 크기의 보정된 이미지 파일명 생성
        base_name, ext = os.path.splitext(output_image_path)
        uncropped_output = f"{base_name}_uncropped{ext}"
        cv2.imwrite(uncropped_output, dst)
        
        print(f"보정된 이미지 저장 완료: {output_image_path}")
        print(f"잘라내지 않은 보정 이미지 저장 완료: {uncropped_output}")
    
    # 5. 결과 시각화 (옵션)
    if visualize:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(221)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('원본 이미지', fontsize=14)
        plt.axis('off')
        
        plt.subplot(222)
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.title('보정된 이미지 (원본 크기)', fontsize=14)
        plt.axis('off')
        
        plt.subplot(223)
        plt.imshow(cv2.cvtColor(dst_cropped, cv2.COLOR_BGR2RGB))
        plt.title('보정된 이미지 (잘라냄)', fontsize=14)
        plt.axis('off')
        
        # 차이 시각화 (원본 - 보정됨)
        diff = cv2.absdiff(img, dst)
        # 차이를 더 잘 보이게 하기 위해 밝기 향상
        diff = cv2.convertScaleAbs(diff, alpha=5, beta=0)
        
        plt.subplot(224)
        plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
        plt.title('원본과 보정 이미지의 차이 (5배 증폭)', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("왜곡 보정 완료!")
    return dst_cropped

# 사용 예시
if __name__ == "__main__":
    # 매개변수 설정
    calibration_file = "./model/camera_calibration_result.npy"  # 캘리브레이션 결과 파일
    input_image = "./image/test_image_18.jpg"  # 보정할 이미지 경로
    output_image = "./result/checkerboard_01_undistorted.jpg"  # 결과 저장 경로
    
    # 이미지 왜곡 보정 실행
    undistorted_img = undistort_single_image(
        calibration_file=calibration_file,
        input_image_path=input_image,
        output_image_path=output_image,
        visualize=True
    )