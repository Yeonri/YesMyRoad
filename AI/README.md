# 🛣️ AI 기반 도로파손 감지 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11n-00FFFF?logo=yolo)](https://ultralytics.com/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Quantized-FF6F00?logo=tensorflow)](https://www.tensorflow.org/lite)
[![SAM](https://img.shields.io/badge/SAM-Meta-1877F2?logo=meta)](https://segment-anything.com/)


> AI Hub 지자체 도로 정비 데이터를 활용한 이중 검증 도로파손 감지 시스템. 모바일 온디바이스 Detection과 서버 기반 SAM Segmentation을 통한 정밀 검증 파이프라인

## ✨ 데모

- **사용 데이터셋:** [지자체 도로 정비 AI 학습용 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=557)

## 🔖 목차

1. [시스템 아키텍처](#시스템-아키텍처)
2. [주요 기능](#주요-기능)
3. [기술 스택](#기술-스택)
4. [시작하기](#시작하기)
5. [모바일용 모델 구축](#모바일용-모델-구축)
6. [서버용 모델 구축](#서버용-모델-구축)
7. [폴더 구조](#폴더-구조)
8. [라이선스](#라이선스)

## 🏗️ 시스템 아키텍처

```
📱 모바일 디바이스 (1차 검출)
├── YOLO11n Detection Model
├── TensorFlow Lite (양자화)
└── 실시간 도로파손 감지

         ⬇️ 의심 구간 전송

🖥️ GPU 서버 (2차 정밀 검증)
├── YOLO11 Segmentation
├── SAM (Segment Anything Model)
└── 고정밀도 마스크 생성
```

## 🎯 주요 기능

### 📱 모바일 온디바이스 (1차 검출)
- **경량 Detection**: YOLO11n 기반 실시간 도로파손 감지
- **TensorFlow Lite**: Float16/Int8 양자화로 모바일 최적화
- **빠른 추론**: 갤럭시 스마트폰에서 실시간 처리

### 🖥️ 서버 기반 (2차 정밀 검증)
- **고정밀 Segmentation**: YOLO11 + SAM 결합 모델
- **정밀 마스크 생성**: Segment Anything Model로 픽셀 단위 분할
- **GPU 가속**: CUDA 기반 고성능 처리

### 📊 데이터 전처리
- **이중 파이프라인**: 모바일용/서버용 별도 데이터 변환
- **스마트 샘플링**: 포트홀 비율 조정 및 데이터 균형 최적화
- **병렬 처리**: 멀티프로세싱을 통한 고속 변환

## 🛠️ 기술 스택

| 구분 | 모바일 온디바이스 | 서버 기반 |
|:-----|:-----------------|:----------|
| **AI 모델** | YOLO11n Detection | YOLO11 Segmentation + SAM |
| **최적화** | TensorFlow Lite | CUDA GPU 가속 |
| **타겟 환경** | Android/iOS | Linux GPU 서버 |
| **처리 방식** | 실시간 추론 | 배치 정밀 검증 |

## 🚀 시작하기

> Python 3.8 이상, CUDA 11.8+ (서버용)가 설치되어 있어야 합니다.

```bash
# 1. 저장소 클론
git clone [repository-url]
cd ai-road-damage-detection

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 기본 패키지 설치
pip install ultralytics opencv-python numpy pillow
pip install tensorflow  # TFLite 변환용
pip install wandb       # 실험 추적용 (선택사항)

# 4. SAM 관련 패키지 설치 (서버용)
pip install segment-anything
pip install torch torchvision  # CUDA 버전

# 5. SAM 모델 체크포인트 다운로드
mkdir sam_models
cd sam_models
# ViT-B (기본, 375MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# ViT-L (큰 모델, 1.25GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# ViT-H (최대 모델, 2.56GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..

# 6. AI Hub 데이터셋 다운로드
# https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=557
```

## 📱 모바일용 모델 구축

### 🔄 데이터 전처리 (Detection용)

```bash
# 기본 COCO → YOLO Detection 변환
python mobile_data_converter.py --source_images_dir ./images --source_labels_dir ./labels

# 샘플링 옵션 포함
python mobile_data_converter.py --source_images_dir ./images --source_labels_dir ./labels \
    --total_usage_ratio 0.5 --pothole_presence_ratio 0.3 \
    --split_ratios_train_val_test 0.7 0.2 0.1

# 고급 옵션 (병렬 처리)
python mobile_data_converter.py --source_images_dir ./images --source_labels_dir ./labels \
    --output_dir ./mobile_dataset --num_workers 8 --seed 42
```

### 🎓 모바일 Detection 모델 학습

```bash
# 기본 학습 (모바일 최적화)
python train_mobile_detection.py --data-yaml ./mobile_dataset/dataset.yaml --epochs 200

# 경량화 중심 설정
python train_mobile_detection.py --data-yaml ./config/mobile_data.yaml \
    --model yolo11n.pt --epochs 300 --batch-size 128 \
    --img-size 640 --optimizer AdamW --device 0 \
    --wandb-project mobile_road_detection

# 클래스 이름 설정
python train_mobile_detection.py --data-yaml ./dataset.yaml \
    --names '도로균열,도로홀' --epochs 150
```

### ⚡ TensorFlow Lite 양자화

```python
# quantize_yolo.py 설정 수정
PT_MODEL_PATH = './runs/detect/train/weights/best.pt'    # 모바일 모델 경로
OUTPUT_TFLITE_BASENAME = 'mobile_road_detection'         # 출력 파일명
IMAGE_SIZE = 640                                         # 입력 이미지 크기
QUANTIZATION_TYPE = 'int8'                              # 모바일 최적화

# 실행
python quantize_yolo.py
```

## 🖥️ 서버용 모델 구축

### 🎭 데이터 전처리 (Segmentation + SAM용)

```bash
# 기본 COCO → YOLO Segmentation 변환
python server_data_converter.py --input ./json_files --output ./server_dataset --image_dir ./images

# SAM 적용 정밀 변환
python server_data_converter.py --input ./annotations --output ./server_advanced \
    --image_dir ./raw_images --adaptive --workers 8 \
    --use_sam_for_potholes --sam_checkpoint_path ./sam_models/sam_vit_b_01ec64.pth \
    --sam_model_type vit_b --sam_device cuda

# 샘플링 포함 서버용 데이터셋
python server_data_converter.py --input ./annotations --output ./server_sampled \
    --image_dir ./raw_images --sample_ratio 0.3 --target_pothole_ratio_in_sample 0.4 \
    --random_seed 42 --buffer 15.0
```

### 🎓 서버 Segmentation 모델 학습

```bash
# 기본 서버 모델 학습
python train_server_segmentation.py --data-dir ./server_dataset --names 'crack,pothole'

# W&B 로깅 포함 고성능 학습
python train_server_segmentation.py --data-dir ./server_dataset --names 'crack,pothole' \
    --wandb --wandb-project server-road-segmentation --epochs 300

# GPU 다중 사용 고성능 설정
python train_server_segmentation.py --data-dir ./server_dataset --names 'crack,pothole' \
    --model yolo11m-seg.pt --epochs 400 --batch-size 16 \
    --img-size 1024 --device 0,1,2,3 --workers 32
```

## 📁 폴더 구조

```
.
├── 📱 모바일 온디바이스 파일들
│   ├── mobile_data_converter.py      # COCO → YOLO Detection 변환
│   ├── train_mobile_detection.py     # YOLO11n Detection 학습
│   └── quantize_yolo.py             # TensorFlow Lite 변환 및 양자화
│
├── 🖥️ 서버 기반 파일들
│   ├── server_data_converter.py      # COCO → YOLO Segmentation + SAM 변환
│   └── train_server_segmentation.py  # YOLO11 Segmentation 학습
│
├── 📊 데이터셋 구조
│   ├── mobile_dataset/              # 모바일용 Detection 데이터
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/                  # YOLO Detection 형식
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── dataset.yaml
│   │
│   └── server_dataset/              # 서버용 Segmentation 데이터
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/                  # YOLO Segmentation 형식 (폴리곤)
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── dataset.yaml
│
├── 🤖 모델 저장소
│   ├── mobile_models/               # 모바일 배포용 모델
│   │   ├── best.pt                  # PyTorch 원본
│   │   ├── mobile_road_detection.tflite    # TFLite 양자화 모델
│   │   └── model_info.txt
│   │
│   ├── server_models/               # 서버 검증용 모델
│   │   ├── best.pt                  # YOLO Segmentation 모델
│   │   ├── last.pt
│   │   └── validation_results.json
│   │
│   └── sam_models/                  # SAM 체크포인트
│       ├── sam_vit_b_01ec64.pth     # ViT-B (375MB)
│       ├── sam_vit_l_0b3195.pth     # ViT-L (1.25GB)
│       └── sam_vit_h_4b8939.pth     # ViT-H (2.56GB)
│
└── 📋 학습 결과
    ├── mobile_runs/                 # 모바일 모델 학습 로그
    │   └── detect/
    └── server_runs/                 # 서버 모델 학습 로그
        └── segment/
```

## 🔧 주요 파라미터 비교

### 📱 모바일용 vs 🖥️ 서버용

| 구분 | 모바일 Detection | 서버 Segmentation |
|:-----|:----------------|:------------------|
| **모델 크기** | ~6MB (INT8) | ~50MB+ |
| **입력 크기** | 640x640 | 896x896+ |
| **배치 크기** | 128 | 16 |
| **처리 시간** | ~50ms | ~200ms |
| **정확도** | mAP50: 0.85+ | Mask mAP50: 0.90+ |

### 🎯 SAM 모델 타입별 특성

| 모델 | 크기 | VRAM | 추론 시간 | 정확도 |
|:-----|:-----|:-----|:---------|:-------|
| **ViT-B** | 375MB | 4GB | ~1s | 높음 |
| **ViT-L** | 1.25GB | 8GB | ~2s | 매우 높음 |
| **ViT-H** | 2.56GB | 16GB | ~3s | 최고 |

## 🎯 감지 클래스 및 성능

### 감지 대상
- **Class 0**: 도로균열 (Crack)
- **Class 1**: 도로홀 (Pothole)

### 성능 지표

**모바일 Detection 모델**
- **mAP50**: 0.85+ (IoU 0.5에서의 평균 정밀도)
- **mAP50-95**: 0.72+ (IoU 0.5-0.95에서의 평균 정밀도)
- **추론 속도**: 20 FPS (Galaxy S23)

**서버 Segmentation 모델**
- **Mask mAP50**: 0.90+ (마스크 기준 평균 정밀도)
- **Box mAP50**: 0.92+ (바운딩 박스 기준 평균 정밀도)
- **SAM 정밀도**: IoU 0.95+ (픽셀 단위 분할)

## 🖥️ 서버 환경 요구사항

### 최소 사양
- **GPU**: RTX 3060 (12GB VRAM)
- **RAM**: 16GB
- **CUDA**: 11.8+
- **Python**: 3.8+

### 권장 사양
- **GPU**: RTX 4090 (24GB VRAM)
- **RAM**: 32GB
- **CUDA**: 12.0+
- **SSD**: 100GB+ 여유 공간

## 🚧 알려진 이슈 및 해결방안

### 모바일 모델
- **메모리 부족**: 배치 크기를 64 이하로 조정
- **정확도 저하**: INT8 대신 Float16 양자화 사용
- **속도 저하**: 입력 크기를 320x320으로 축소

### 서버 모델
- **CUDA OOM**: SAM 모델 타입을 ViT-B로 변경
- **SAM 속도**: 배치 처리 대신 개별 이미지 처리
- **디스크 공간**: 불필요한 중간 파일 정기 삭제

## 🔄 워크플로우

```bash
# 1단계: 데이터 준비
python mobile_data_converter.py --source_images_dir ./raw_images --source_labels_dir ./raw_labels
python server_data_converter.py --input ./raw_labels --output ./server_data --image_dir ./raw_images --use_sam_for_potholes

# 2단계: 모델 학습
python train_mobile_detection.py --data-yaml ./mobile_dataset/dataset.yaml --epochs 200
python train_server_segmentation.py --data-dir ./server_dataset --epochs 300

# 3단계: 모델 최적화
python quantize_yolo.py  # 모바일용 TFLite 변환

# 4단계: 배포
# 모바일: mobile_road_detection.tflite → Android/iOS 앱
# 서버: server_models/best.pt → GPU 서버 배포
```
---

**AI 기반 도로파손 감지 시스템**은 모바일 실시간 검출과 서버 정밀 검증을 결합한 하이브리드 AI 솔루션입니다.