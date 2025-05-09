import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_matplotlib_font():
    """matplotlib에서 한글 폰트 설정"""
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