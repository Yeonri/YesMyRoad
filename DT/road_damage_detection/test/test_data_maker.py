import pandas as pd
import json
from datetime import timezone

# 기존 CSV 파일 불러오기
capture_point_df = pd.read_csv("data/capture_point.csv", encoding="cp949")
capture_damage_df = pd.read_csv("data/capture_damage.csv", encoding="cp949")
damage_category_df = pd.read_csv("data/damage_category.csv", encoding="cp949")

# 변환 대상 컬럼: capture_timestamp -> datetime
capture_point_df["capture_timestamp"] = pd.to_datetime(capture_point_df["capture_timestamp"])

# 변환 수행
structured_records = []

for cp in capture_point_df.itertuples():
    detections = capture_damage_df[capture_damage_df.capture_point_id == cp.capture_point_id] \
        .merge(damage_category_df, on="category_id")["category_name"].tolist()
    
    lat, lon = map(float, cp.location.replace("POINT(", "").replace(")", "").split())
    
    structured_records.append({
        "capture_timestamp_utc": cp.capture_timestamp.tz_localize("Asia/Seoul").astimezone(timezone.utc).isoformat(),
        "location": {
            "latitude": lon,
            "longitude": lat,
            "accuracy_meters": cp.accuracy_meters
        },
        "image_info": {
            "image_url": cp.image_url,
            "risk": cp.risk
        },
        "detections": [{"category_name": d} for d in detections]
    })

# JSON 파일로 저장
with open("structured_capture_data.json", "w", encoding="utf-8") as f:
    json.dump(structured_records, f, ensure_ascii=False, indent=2)