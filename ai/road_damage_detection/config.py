class Config:
    # 모델 설정
    MODEL_PATH = './model/yolo11l-seg.pt'
    
    # 이미지 경로 형식
    IMAGE_PATH_FORMAT = './image/test_image_{}.jpg'
    
    # 도로 탐지 설정
    ROAD_LOWER_HSV = [0, 0, 60]
    ROAD_UPPER_HSV = [180, 30, 160]
    ROAD_TOP_CUTOFF_RATIO = 0.25
    
    # 차선 탐지 설정
    LANE_CANNY_LOW = 50
    LANE_CANNY_HIGH = 150
    WHITE_LANE_LOWER_HSV = [0, 0, 200]
    WHITE_LANE_UPPER_HSV = [180, 30, 255]
    YELLOW_LANE_LOWER_HSV = [10, 60, 80]
    YELLOW_LANE_UPPER_HSV = [40, 255, 255]
    
    # 호프 변환 설정
    HOUGH_THRESHOLD = 40
    MIN_LINE_LENGTH = 30
    MAX_LINE_GAP = 5
    
    # 시각화 설정
    FIGURE_SIZE = (20, 15)
    ALPHA = 0.5
    ROAD_COLOR = [0, 255, 0]  # 초록색
    CRACK_COLOR = [255, 0, 0]  # 빨간색  
    POTHOLE_COLOR = [0, 0, 255]  # 파란색