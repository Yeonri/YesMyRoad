"""
도로 손상 감지 시스템 유틸리티 패키지

이 패키지는 도로 손상 감지에 필요한 다양한 유틸리티 함수와 모듈을 제공합니다.
"""

from .font_utils import setup_matplotlib_font, put_text_on_image
from .calibration import undistort_image
from .road_extraction import create_roi_mask, extract_road_mask, extract_road_between_lanes
from .lane_detection import detect_lane_lines
from .detection import apply_yolo_on_masked_area, analyze_crack_severity, check_pothole_emergency
from .transformation import extract_road_corners_direct, perspective_transform_direct_masked
from .analysis import calculate_lrpci, get_lrpci_grade
from .visualization import visualize_results_masked
