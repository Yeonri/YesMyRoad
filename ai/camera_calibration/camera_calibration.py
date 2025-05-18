import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# 체스보드의 내부 코너 개수 설정
CHECKERBOARD = (10, 7)  # 내부 코너 개수 (가로, 세로)

# 3D 공간에서의 체스보드 코너 좌표를 저장할 배열 준비
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# 수정된 부분: np.mgrid 사용 방식 변경
# 이 부분이 'T' member 오류를 일으킬 수 있음
objp[:, :2] = np.array([(j, i) for i in range(CHECKERBOARD[1]) for j in range(CHECKERBOARD[0])], dtype=np.float32)

# 각 체스보드 이미지에서 검출된 코너의 좌표를 저장할 배열
objpoints = []  # 3D 공간 좌표
imgpoints = []  # 2D 이미지 좌표

# 체스보드 이미지 경로를 './image/'로 설정
images = glob.glob('./image/*.jpg')  # jpg 파일 검색
if not images:
    # jpg 파일이 없으면 png 파일도 검색
    images = glob.glob('./image/*.png')
if not images:
    # 다른 이미지 형식도 검색
    images = glob.glob('./image/*.*')

print(f"발견된 이미지 파일 수: {len(images)}")

# 코너 검출 결과를 시각화할지 여부
visualize = True

# 이미지 크기를 저장할 변수
img_shape = None

# 체스보드 코너 검출
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    
    # 이미지를 읽을 수 없는 경우 건너뛰기
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {fname}")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이미지 크기 저장
    if img_shape is None:
        img_shape = gray.shape[::-1]
    
    # 이미지 전처리를 통한 코너 검출 향상
    # 대비 강화
    gray = cv2.equalizeHist(gray)
    
    # 체스보드 코너 검출 - 다양한 옵션 시도
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                            cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                            cv2.CALIB_CB_FAST_CHECK + 
                                            cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # 코너 검출 실패 시 다른 방법 시도
    if not ret:
        # 이미지 전처리를 더 강화하여 다시 시도
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, corners = cv2.findChessboardCorners(gray_blur, CHECKERBOARD, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_FAST_CHECK + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # 코너 검출 성공 시
    if ret:
        # 서브픽셀 정확도로 코너 위치 조정
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 검출된 코너 저장
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        # 검출된 코너 시각화
        if visualize:
            img_copy = img.copy()  # 원본 이미지 보존
            cv2.drawChessboardCorners(img_copy, CHECKERBOARD, corners2, ret)
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            plt.title(f'Detected Corners: Image {idx+1} - {fname}')
            plt.show()
    else:
        print(f"코너 검출 실패: {fname}")
        # 실패한 이미지 디버깅용 시각화
        if visualize:
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f'Original Image (Detection Failed): {fname}')
            plt.show()

# 캘리브레이션 수행
if len(objpoints) > 0:
    print(f"총 {len(objpoints)}개 이미지에서 체스보드 코너 검출 성공")
    
    try:
        # 캘리브레이션 실행
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None)
        
        # 결과 출력
        print("\n===== 카메라 캘리브레이션 결과 =====")
        print("카메라 행렬 (Camera Matrix):")
        print(mtx)
        print("\n왜곡 계수 (Distortion Coefficients):")
        print(dist)
        
        # 재투영 오차 계산
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        print(f"\n총 재투영 오차 (Total Reprojection Error): {mean_error/len(objpoints)}")
        
        # 결과 저장
        calibration_result = {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs
        }
        np.save('./model/camera_calibration_result.npy', calibration_result)
        print("\n캘리브레이션 결과가 'model/camera_calibration_result.npy'에 저장되었습니다.")
        
        # 왜곡 보정 예시 (첫 번째 이미지 사용)
        if len(images) > 0:
            img = cv2.imread(images[0])
            if img is not None:  # 이미지를 읽을 수 있는지 확인
                h, w = img.shape[:2]
                
                # 최적의 카메라 행렬 구하기
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                
                # 왜곡 보정
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
                
                # ROI 영역으로 이미지 자르기
                x, y, w, h = roi
                if all(val > 0 for val in [x, y, w, h]):
                    dst = dst[y:y+h, x:x+w]
                
                # 원본과 보정된 이미지 출력
                plt.figure(figsize=(16, 8))
                plt.subplot(121)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title('Original Image')
                plt.subplot(122)
                plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
                plt.title('Undistorted Image')
                plt.tight_layout()
                plt.show()
                
                # 보정된 이미지 저장
                cv2.imwrite('./result/calibration_result.jpg', dst)
                print("보정된 이미지가 'result/calibration_result.jpg'에 저장되었습니다.")
    except Exception as e:
        print(f"캘리브레이션 중 오류 발생: {e}")
else:
    print("체스보드 코너 검출에 실패하여 캘리브레이션을 수행할 수 없습니다.")
    print("체스보드 크기(CHECKERBOARD)가 올바른지 확인하세요. 현재 설정: ", CHECKERBOARD)