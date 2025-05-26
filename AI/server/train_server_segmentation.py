import os
import sys
import json
import argparse
import logging
import random
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_yolo11_seg_direct.log')
    ]
)
logger = logging.getLogger('YOLO11-SEG-Trainer-Direct')

class YOLO11SegTrainer:
    def __init__(self, args):
        self.args = args
        self.data_path = args.input
        self.output_dir_base = args.output
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.img_size = args.img_size
        self.device = args.device
        self.workers = args.workers
        self.seed = args.random_seed
        self.model_type = args.model
        self.pretrained = args.pretrained
        self.cache = args.cache
        
        self.use_wandb = args.wandb
        self.wandb_project = args.wandb_project
        self.wandb_entity = args.wandb_entity
        self.wandb_name = args.wandb_name

        if self.args.names:
            self.class_names = [name.strip() for name in args.names.split(',')]
        else:
            self.class_names = self._try_load_names_from_yaml()
            if not self.class_names:
                logger.error("클래스 이름을 찾을 수 없습니다. --names 인자를 사용하거나 data 디렉토리에 yaml 파일을 포함하세요.")
                sys.exit(1)

        logger.info(f"사용될 클래스: {self.class_names}")

        random.seed(self.seed)
        np.random.seed(self.seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = os.path.splitext(os.path.basename(self.model_type))[0]
        self.run_name = args.run_name if args.run_name else f"{model_basename}_run_{timestamp}"

        if self.use_wandb and not self.wandb_name:
            self.wandb_name = self.run_name

        if not self.output_dir_base:
            self.ultralytics_project_dir = "runs/segment"
        else:
            self.ultralytics_project_dir = self.output_dir_base

        self.final_output_dir_local = os.path.join(self.ultralytics_project_dir, self.run_name)

        logger.info(f"실행 이름: {self.run_name}")
        logger.info(f"Ultralytics 로컬 프로젝트 디렉토리: {self.ultralytics_project_dir}")
        logger.info(f"예상 로컬 결과 디렉토리: {self.final_output_dir_local}")

        self.dataset_yaml_path = None

    def _try_load_names_from_yaml(self):
        names = []
        for yaml_name in ['data.yaml', 'dataset.yaml']:
            yaml_path_candidate = Path(self.data_path) / yaml_name
            if yaml_path_candidate.exists():
                try:
                    with open(yaml_path_candidate, 'r', encoding='utf-8') as f:
                        data_yaml = yaml.safe_load(f)
                    if 'names' in data_yaml:
                        if isinstance(data_yaml['names'], list):
                            names = data_yaml['names']
                        elif isinstance(data_yaml['names'], dict):
                            names = [data_yaml['names'][i] for i in sorted(data_yaml['names'].keys())]
                        logger.info(f"{yaml_path_candidate}에서 클래스 이름 로드 성공.")
                        return names
                except Exception as e:
                    logger.warning(f"{yaml_path_candidate} 로드 중 오류: {e}")
        logger.info(f"{self.data_path} 에서 클래스 이름 YAML 파일을 찾지 못했습니다.")
        return names

    def setup_directories(self):
        if self.ultralytics_project_dir:
             os.makedirs(self.ultralytics_project_dir, exist_ok=True)
        os.makedirs(self.final_output_dir_local, exist_ok=True)

    def create_yaml_config(self):
        logger.info("YOLO 학습 설정 YAML 파일 생성 중...")

        self.dataset_yaml_path = os.path.join(self.final_output_dir_local, "data_config_for_run.yaml")

        dataset_config = {
            'path': os.path.abspath(self.data_path),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(self.class_names)}
        }

        try:
            with open(self.dataset_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(dataset_config, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
            logger.info(f"데이터셋 설정 파일 생성 완료: {self.dataset_yaml_path}")
        except Exception as e:
            logger.error(f"YAML 파일 생성 실패: {e}")
            sys.exit(1)

    def train_model(self):
        if not self.dataset_yaml_path:
            logger.error("데이터셋 YAML 설정 파일 경로가 설정되지 않았습니다.")
            return False

        logger.info("YOLO 11-SEG 모델 학습 시작...")

        try:
            from ultralytics import YOLO
            import torch
            
            if self.use_wandb:
                try:
                    import wandb
                    logger.info("Weights & Biases 로깅 설정 중...")
                except ImportError:
                    logger.warning("wandb 패키지가 설치되지 않았습니다. pip install wandb로 설치해주세요.")
                    self.use_wandb = False
        except ImportError as e:
            logger.error(f"필수 패키지를 임포트할 수 없습니다: {e}")
            logger.error("pip install ultralytics torch torchvision torchaudio 명령어로 설치해주세요.")
            return False

        cuda_available = torch.cuda.is_available()
        device_to_use = self.device if self.device else ('0' if cuda_available else 'cpu')
        logger.info(f"학습에 사용할 장치: {device_to_use}")

        try:
            logger.info(f"모델 로드 중: {self.model_type}")
            model = YOLO(self.model_type)

            train_args = {
                'data': self.dataset_yaml_path,
                'epochs': self.epochs,
                'imgsz': self.img_size,
                'batch': self.batch_size,
                'device': device_to_use,
                'workers': self.workers,
                'name': self.run_name,
                'project': self.ultralytics_project_dir,
                'cache': self.cache,
                'task': 'segment',
                'verbose': True,
                'seed': self.seed,
                'lr0': 0.001,
                'lrf': 0.05,
                'weight_decay': 0.0003,
                'warmup_epochs': 5.0,
                'optimizer': 'AdamW',
                'patience': 20,
                'val': True,
                'save_hybrid': True,
                'save': True,
                'save_period': 5,
                'amp': True,
            }
            
            if self.use_wandb:
                train_args.update({
                    'project': self.wandb_project or 'YOLO11-SEG-Training',
                    'entity': self.wandb_entity,
                    'name': self.wandb_name,
                    'upload_dataset': False,
                    'bbox_interval': 1,
                    'mask_interval': 1,
                    'plots': True,
                })
                
                wandb_tags = [f"model-{os.path.basename(self.model_type)}", 
                             f"resolution-{self.img_size}",
                             f"batch-{self.batch_size}"]
                
                wandb.init(
                    project=train_args['project'],
                    entity=train_args['entity'],
                    name=train_args['name'],
                    tags=wandb_tags,
                    notes=f"YOLOv11-SEG 학습: {len(self.class_names)}개 클래스, 이미지 크기 {self.img_size}",
                    config={
                        "model_type": self.model_type,
                        "image_size": self.img_size,
                        "batch_size": self.batch_size,
                        "epochs": self.epochs,
                        "classes": self.class_names,
                        "lr0": train_args['lr0'],
                        "lrf": train_args['lrf'],
                        "weight_decay": train_args['weight_decay'],
                        "warmup_epochs": train_args['warmup_epochs'],
                    }
                )
                
                logger.info(f"Weights & Biases 로깅 설정 완료: {wandb.run.name} ({wandb.run.id})")

            logger.info(f"Ultralytics 학습 설정: {json.dumps(train_args, indent=2)}")
            model.train(**train_args)

            logger.info(f"학습 완료! 결과는 다음 폴더에 저장되었습니다: {self.final_output_dir_local}")

            best_model_path = os.path.join(self.final_output_dir_local, 'weights', 'best.pt')
            if not os.path.exists(best_model_path):
                logger.warning(f"최적 모델 파일(best.pt)을 찾을 수 없습니다: {best_model_path}")
                last_model_path = os.path.join(self.final_output_dir_local, 'weights', 'last.pt')
                if os.path.exists(last_model_path):
                    best_model_path = last_model_path
                    logger.info(f"last.pt 모델로 평가를 진행합니다: {best_model_path}")
                else:
                    logger.error("평가할 모델 파일을 찾을 수 없습니다.")
                    return True

            logger.info("검증 데이터에서 최종 모델 평가 중...")
            eval_model = YOLO(best_model_path)
            val_results = eval_model.val(data=self.dataset_yaml_path, split='val', device=device_to_use, batch=self.batch_size)

            logger.info("검증 결과:")
            if hasattr(val_results, 'box') and hasattr(val_results.box, 'map'):
                logger.info(f"  Box mAP50-95: {val_results.box.map:.4f}")
                logger.info(f"  Box mAP50: {val_results.box.map50:.4f}")
            if hasattr(val_results, 'seg') and hasattr(val_results.seg, 'map'):
                logger.info(f"  Seg mAP50-95: {val_results.seg.map:.4f}")
                logger.info(f"  Seg mAP50: {val_results.seg.map50:.4f}")
                
            if self.use_wandb:
                final_metrics = {}
                if hasattr(val_results, 'box') and hasattr(val_results.box, 'map'):
                    final_metrics["box_map"] = val_results.box.map
                    final_metrics["box_map50"] = val_results.box.map50
                if hasattr(val_results, 'seg') and hasattr(val_results.seg, 'map'):
                    final_metrics["seg_map"] = val_results.seg.map
                    final_metrics["seg_map50"] = val_results.seg.map50
                
                wandb.log(final_metrics)
                
                if os.path.exists(best_model_path):
                    artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
                    artifact.add_file(best_model_path, name="best.pt")
                    wandb.log_artifact(artifact)
                
                wandb.finish()
                logger.info("Weights & Biases 로깅 완료 및 종료")

            return True

        except Exception as e:
            logger.error(f"학습 또는 평가 중 오류 발생: {e}", exc_info=True)
            if self.use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                except:
                    pass
            return False

    def run(self):
        logger.info(f"YOLO 11-SEG 학습 파이프라인 시작 (Run: {self.run_name})")
        success = False
        try:
            self.setup_directories()
            self.create_yaml_config()
            success = self.train_model()
        except Exception as e:
            logger.error(f"파이프라인 실행 중 예외 발생: {e}", exc_info=True)
            success = False

        if success:
            logger.info("학습 파이프라인 성공적으로 완료!")
        else:
            logger.error("학습 파이프라인 중 오류 발생 또는 실패.")
        return success

def main():
    parser = argparse.ArgumentParser(
        description='YOLO 11-SEG 모델 학습 스크립트 (기존 YOLO 형식 데이터셋 사용)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
사용 예:
  python %(prog)s --input /path/to/yolo_dataset --names 'crack,pothole' --epochs 100
  python %(prog)s --input ./my_dataset --names 'crack,pothole' --model yolov8n-seg.pt \\
    --batch_size 16 --run_name exp001
  python %(prog)s --input ./my_dataset --names 'crack,pothole' --wandb --wandb_project "my-yolo-project"
"""
    )

    parser.add_argument('--input', required=True, help='YOLO 형식 데이터셋 루트 디렉토리 경로')
    parser.add_argument('--names', default=None, help="클래스 이름 목록 (쉼표 구분). data-dir 내 yaml에 없거나 재정의 시 사용.")
    parser.add_argument('--model', default='last.pt', help='모델 아키텍처 또는 가중치 경로 (기본값: last.pt)')
    parser.add_argument('--output', default='', help='로컬 출력물의 기본 디렉토리 (기본값: runs/segment)')
    parser.add_argument('--run_name', default=None, help='실행 이름 및 로컬 결과 폴더 이름 (기본값: 자동 생성)')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기 (기본값: 16)')
    parser.add_argument('--epochs', type=int, default=200, help='학습 에폭 수 (기본값: 200)')
    parser.add_argument('--img_size', type=int, default=896, help='입력 이미지 크기 (기본값: 896)')
    parser.add_argument('--device', default='', help='학습 장치 (예: 0,1 또는 cpu, 기본값: 자동 감지)')
    parser.add_argument('--workers', type=int, default=16, help='데이터 로딩 워커 수 (기본값: 16)')
    parser.add_argument('--random_seed', type=int, default=42, help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--pretrained', action='store_true', help='사전 학습된 가중치 사용')
    parser.add_argument('--cache', action='store_true', help='데이터셋 캐싱 활성화 (RAM 사용 증가)')
    parser.add_argument('--verbose', action='store_true', help='자세한 로깅 활성화')
    
    parser.add_argument('--wandb', action='store_true', help='Weights & Biases 로깅 활성화')
    parser.add_argument('--wandb_project', default=None, help='Weights & Biases 프로젝트 이름')
    parser.add_argument('--wandb_entity', default=None, help='Weights & Biases 엔티티/조직 이름')
    parser.add_argument('--wandb_name', default=None, help='Weights & Biases 실행 이름')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("상세 로깅 모드가 활성화되었습니다.")

    if not os.path.isdir(args.input):
        logger.error(f"데이터 디렉토리가 존재하지 않거나 디렉토리가 아닙니다: {args.input}")
        sys.exit(1)

    trainer = YOLO11SegTrainer(args)
    success = trainer.run()

    if success:
        logger.info("YOLO 11-SEG 모델 학습 최종 완료!")
        sys.exit(0)
    else:
        logger.error("YOLO 11-SEG 모델 학습 최종 실패.")
        sys.exit(1)

if __name__ == "__main__":
    main()