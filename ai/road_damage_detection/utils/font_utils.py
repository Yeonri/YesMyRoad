import cv2
import numpy as np
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

def setup_matplotlib_font():
    """matplotlib 한글 폰트 설정"""
    # 운영체제별 기본 폰트 설정
    system = platform.system()
    if system == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우의 경우 맑은 고딕
    elif system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 우분투 기준 나눔고딕

    # 폰트 패스가 유효한지 확인
    try:
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except:
        print("기본 폰트를 사용합니다. 한글이 깨질 수 있습니다.")
        plt.rcParams['font.family'] = 'NanumGothic'  # 나눔고딕이 설치되어 있으면 사용

    # 그래프에서 마이너스 폰트 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

def put_text_on_image(img, text, position, font_size=1, color=(255, 0, 0), thickness=2):
    """한글 텍스트를 이미지에 표시하는 함수"""
    # PIL 이미지로 변환
    pil_img = Image.fromarray(img)
    
    # 그리기 객체 생성
    draw = ImageDraw.Draw(pil_img)
    
    # 폰트 설정 (시스템에 맞게 조절 필요)
    system = platform.system()
    if system == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우의 경우 맑은 고딕
    elif system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    else:  # Linux
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 우분투 기준 나눔고딕
    
    try:
        font = ImageFont.truetype(font_path, int(font_size * 20))
    except:
        print("폰트를 로드할 수 없습니다. 기본 폰트를 사용합니다.")
        font = ImageFont.load_default()
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color[::-1])  # RGB -> BGR 변환
    
    # NumPy 배열로 변환하여 반환
    return np.array(pil_img)
