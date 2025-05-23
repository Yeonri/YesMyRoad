"""
도로 손상 분석 모듈

이 모듈은 도로 손상 정도 평가 및 분석 관련 기능을 제공합니다.
"""


def calculate_lrpci(crack_ratio, pothole_ratio, crack_severity, crack_type, pothole_info):
    """
    도로 포장 상태를 평가하는 LrPCI(Local road Pavement Condition Index) 계산 함수

    Process:
    1. 기본 점수 설정 (완벽한 상태 = 100)
    2. 포트홀 감점 계산 (직경, 개수, 면적에 따른 감점)
    3. 균열 감점 계산 (유형, 면적, 심각도에 따른 감점)
    4. 총 감점 계산 및 최종 점수 도출

    Parameters:
        - crack_ratio (float) : 0~1 사이의 균열율
        - pothole_ratio (float) : 0~1 사이의 포트홀 비율
        - crack_severity (float) : 0~1 사이의 균열 심각도
        - crack_type (str) : 균열 유형 (선형 균열, 블록형 균열, 악어 균열 등)
        - pothole_info (dict) : 포트홀 정보 (개수, 최대 직경 등)

    Returns:
        - lrpci (float) : 0~100 사이의 LrPCI 값
    """
    # 기본값 설정: 완벽한 상태의 도로는 100점에서 시작
    base_score = 100
    
    # 포트홀 감점 계산 (포트홀이 있으면 더 많은 감점)
    pothole_deduction = 0
    if pothole_ratio > 0:
        pothole_count = pothole_info.get('potholes_count', 0)
        max_diameter_cm = pothole_info.get('max_diameter_cm', 0)
        
        # 포트홀 직경에 따른 심각도 분류
        if max_diameter_cm >= 30:  # 심각: 30cm 이상은 심각한 포트홀
            pothole_deduction = 30 + (pothole_count * 2.5)  # 기본 30점 감점, 추가 포트홀당 2.5점 추가 감점
        elif max_diameter_cm >= 15:  # 중간: 15~30cm는 중간 수준
            pothole_deduction = 15 + (pothole_count * 2)    # 기본 15점 감점, 추가 포트홀당 2점 추가 감점
        else:  # 경미: 15cm 미만은 경미한 수준
            pothole_deduction = 5 + (pothole_count * 1.5)   # 기본 5점 감점, 추가 포트홀당 1.5점 추가 감점
            
        # 포트홀 면적 비율이 높으면 추가 감점
        pothole_deduction += pothole_ratio * 100 * 0.3
        
        # 최대 감점은 40점으로 제한
        pothole_deduction = min(pothole_deduction, 40)
    
    # 균열 감점 계산
    crack_deduction = 0
    if crack_ratio > 0:
        # 균열 유형에 따른 가중치 설정
        crack_type_weights = {
            "선형 균열": 1.0,       # 선형 균열은 기본 가중치
            "블록형 균열": 1.5,     # 블록형 균열은 더 심각
            "악어 균열": 2.0,       # 악어 균열은 가장 심각
            "없음": 0
        }
        
        crack_type_weight = crack_type_weights.get(crack_type, 1.0)
        
        # 균열 면적 비율과 심각도, 유형에 따른 감점
        if crack_ratio >= 0.5:  # 50% 이상은 심각
            crack_deduction = 25 * crack_type_weight
        elif crack_ratio >= 0.2:  # 20~50%는 중간
            crack_deduction = 15 * crack_type_weight
        else:  # 20% 미만은 경미
            crack_deduction = 5 * crack_type_weight
            
        # 균열 면적과 심각도에 따른 추가 감점
        crack_deduction += (crack_ratio * 100 * 0.2) + (crack_severity * 10)
        
        # 최대 감점은 25점으로 제한
        crack_deduction = min(crack_deduction, 25)
    
    # 총 감점 계산
    total_deduction = pothole_deduction + crack_deduction
    
    # 최종 LrPCI 계산 (최소값은 0)
    lrpci = max(0, base_score - total_deduction)
    
    return round(lrpci, 2)


def get_lrpci_grade(lrpci):
    """
    LrPCI 점수에 따른 등급과 설명을 반환하는 함수

    Parameters:
        - lrpci (float) : LrPCI 점수 (0-100)

    Returns:
        - grade (str) : 등급 (A, B, C, D, F)
        - description (str) : 등급 설명
    """
    if lrpci >= 90:
        return "A", "우수 (양호한 상태, 일상 유지보수만 필요)"
    elif lrpci >= 75:
        return "B", "양호 (경미한 손상, 예방적 유지보수 필요)"
    elif lrpci >= 60:
        return "C", "보통 (보통 수준의 손상, 일반 보수 필요)"
    elif lrpci >= 40:
        return "D", "불량 (심각한 손상, 긴급 보수 필요)"
    else:
        return "F", "매우 불량 (위험 수준의 손상, 전체 재포장 필요)"
    