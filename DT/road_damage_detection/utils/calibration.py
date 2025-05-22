"""
카메라 캘리브레이션 및 이미지 왜곡 보정 모듈

이 모듈은 카메라 캘리브레이션 결과를 사용하여 이미지의 왜곡을 보정하는 기능을 제공합니다.
"""

import cv2
import numpy as np


def undistort_image(img, calibration_file):
    """
    카메라 캘리브레이션 결과를 사용하여 이미지 왜곡을 보정하는 함수

    Process:
    1. 캘리브레이션 파일에서 카메라 매트릭스와 왜곡 계수 로드
    2. 왜곡 보정에 사용할 새로운 카메라 매트릭스 계산
    3. 이미지 왜곡 보정 적용
    4. ROI 영역 잘라내기
    5. 예외 처리를 통한 오류 관리

    Parameters:
        - img (numpy.ndarray) : 원본 이미지
        - calibration_file (str) : 캘리브레이션 결과가 저장된 파일 경로

    Returns:
        - undistorted (numpy.ndarray) : 왜곡 보정된 이미지
        - (mtx, dist, roi) (tuple) : 카메라 행렬, 왜곡 계수, ROI 정보
    """
    try:
        # 캘리브레이션 결과 로드
        calibration_data = np.load(calibration_file, allow_pickle=True).item()
        
        # 키 이름 확인 및 수정 부분
        mtx = calibration_data.get('camera_matrix')  # 'mtx' 대신 'camera_matrix'
        dist = calibration_data.get('dist_coeffs')   # 'dist' 대신 'dist_coeffs'
        
        if mtx is None or dist is None:
            raise ValueError("캘리브레이션 데이터에 카메라 행렬 또는 왜곡 계수가 없습니다.")
        
        # 원본 이미지 크기
        h, w = img.shape[:2]
        
        # 왜곡 보정 후 이미지 크기 계산을 위한 새로운 카메라 행렬 얻기
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # 왜곡 보정 적용
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # ROI 잘라내기
        if roi[2] > 0 and roi[3] > 0:  # 유효한 ROI 영역이 있는 경우
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted, (mtx, dist, roi)
    except Exception as e:
        print(f"이미지 왜곡 보정 중 오류 발생: {e}")
        return img, (None, None, None)
    