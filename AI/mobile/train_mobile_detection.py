import os
import sys
import json
import argparse
import logging
import random
import yaml
from datetime import datetime
import numpy as np
from pathlib import Path
import shutil

try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yolo11n_training.log')
    ]
)
logger = logging.getLogger('yolo11n_trainer')

class YOLO11nDetectTrainer:
    def __init__(self, args):
        self.args = args
        self.user_yaml_path = Path(args.input).resolve()
        self.output_dir_base = Path(args.output) if args.output else Path("runs/detect")
        
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.img_size = args.img_size
        self.device = args.device
        self.workers = args.workers
        self.seed = args.random_seed
        self.model_type = args.model
        self.pretrained = args.pretrained
        self.cache = args.cache
        self.use_wandb = args.use_wandb

        self.class_names = []
        self.wandb_run = None

        self.project_dir_ultralytics = None
        self.run_name_ultralytics = None
        self.run_specific_files_output_dir = None

        self.true_dataset_root_path = None
        self.yaml_for_training_path = None

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _extract_class_names_from_yaml_content(self, yaml_content):
        names_list = []
        if 'names' in yaml_content:
            if isinstance(yaml_content['names'], list):
                names_list = yaml_content['names']
            elif isinstance(yaml_content['names'], dict):
                try:
                    sorted_keys = sorted([k for k in yaml_content['names'] if isinstance(k, int)])
                    if len(sorted_keys) == len(yaml_content['names']):
                        names_list = [yaml_content['names'][k] for k in sorted_keys]
                    else:
                        logger.debug("YAML 'names' 딕셔너리 키가 정수가 아니거나 혼합됨. 값 순서대로 추출.")
                        names_list = list(yaml_content['names'].values())
                except TypeError:
                    logger.warning("YAML 'names' 딕셔너리 키 정렬 불가. 값 순서대로 추출.")
                    names_list = list(yaml_content['names'].values())
        return names_list

    def prepare_dataset_config(self):
        logger.info(f"사용자 제공 데이터셋 설정 파일: {self.user_yaml_path}")
        if not self.user_yaml_path.is_file():
            logger.error(f"지정된 데이터 YAML 파일 '{self.user_yaml_path}'을(를) 찾을 수 없습니다.")
            sys.exit(1)
        
        try:
            with open(self.user_yaml_path, 'r', encoding='utf-8') as f:
                user_yaml_content = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"'{self.user_yaml_path}' 파일 로드 중 오류: {e}")
            sys.exit(1)
        
        if 'path' not in user_yaml_content:
            logger.error(f"'{self.user_yaml_path}' 파일에 필수 키 'path'가 없습니다.")
            sys.exit(1)
        
        user_defined_path_str = user_yaml_content['path']
        user_defined_path = Path(user_defined_path_str)
        
        if user_defined_path.is_absolute():
            self.true_dataset_root_path = user_defined_path.resolve()
        else:
            self.true_dataset_root_path = (self.user_yaml_path.parent / user_defined_path).resolve()
        
        logger.info(f"데이터셋 실제 루트 경로 결정: {self.true_dataset_root_path}")
        
        if not self.true_dataset_root_path.is_dir():
            logger.error(f"결정된 데이터셋 루트 경로 '{self.true_dataset_root_path}'가 존재하지 않거나 디렉토리가 아닙니다.")
            sys.exit(1)

        if self.args.names:
            self.class_names = [name.strip() for name in self.args.names.split(',')]
            logger.info(f"--names 인자로 클래스 이름 설정/재정의: {self.class_names}")
        elif 'names' in user_yaml_content:
            self.class_names = self._extract_class_names_from_yaml_content(user_yaml_content)
            logger.info(f"'{self.user_yaml_path}' 파일에서 클래스 이름 로드: {self.class_names}")
        else:
            logger.error(f"클래스 이름을 결정할 수 없습니다. '{self.user_yaml_path}'에 'names'를 추가하거나 --names 인자를 사용하세요.")
            sys.exit(1)
        
        if not self.class_names:
            logger.error("클래스 이름 목록이 비어있습니다."); sys.exit(1)
        logger.info(f"최종 사용될 클래스 이름: {self.class_names} (개수: {len(self.class_names)})")

        final_yaml_content = user_yaml_content.copy()
        final_yaml_content['path'] = str(self.true_dataset_root_path)
        final_yaml_content['names'] = {i: name for i, name in enumerate(self.class_names)}

        for split in ['train', 'val', 'test']:
            if split in final_yaml_content:
                split_path_str = final_yaml_content[split]
                abs_split_dir_path = self.true_dataset_root_path / split_path_str
                if not abs_split_dir_path.is_dir():
                    msg = f"YAML 내 '{split}' 경로 '{split_path_str}' (해석된 경로: {abs_split_dir_path})가 디렉토리가 아닙니다."
                    if split == 'train': logger.error(msg + " 학습 불가."); sys.exit(1)
                    else: logger.warning(msg + f" 해당 분할은 사용되지 않을 수 있습니다.")
                else:
                    logger.info(f"'{split}' 이미지 경로 확인: {abs_split_dir_path}")
            elif split == 'train':
                logger.error(f"사용자 YAML에 필수 키 'train' (이미지 폴더 경로)가 없습니다."); sys.exit(1)

        self.yaml_for_training_path = self.run_specific_files_output_dir / "final_dataset_for_training.yaml"
        try:
            with open(self.yaml_for_training_path, 'w', encoding='utf-8') as f:
                yaml.dump(final_yaml_content, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
            logger.info(f"학습용 데이터셋 YAML 파일 생성 (사용자 YAML 기반): {self.yaml_for_training_path}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"생성된 학습용 YAML 내용:\n{yaml.dump(final_yaml_content, allow_unicode=True, sort_keys=False, default_flow_style=False)}")
        except Exception as e:
            logger.error(f"학습용 YAML 파일 '{self.yaml_for_training_path}' 생성 실패: {e}")
            sys.exit(1)

    def setup_directories(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = Path(self.model_type).stem 
        self.run_name_ultralytics = f"{model_basename}_run_{timestamp}"
        self.project_dir_ultralytics = self.output_dir_base.resolve()
        self.run_specific_files_output_dir = self.project_dir_ultralytics / self.run_name_ultralytics
        try:
            self.run_specific_files_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"결과 저장 디렉토리 '{self.run_specific_files_output_dir}' 생성 실패: {e}")
            sys.exit(1)
        logger.info(f"Ultralytics 학습 결과 저장 프로젝트 경로: {self.project_dir_ultralytics}")
        logger.info(f"이번 실행(run)의 이름: {self.run_name_ultralytics}")
        logger.info(f"스크립트 생성 파일 (최종 YAML 등) 저장 경로: {self.run_specific_files_output_dir}")

    def train_model(self):
        if not self.yaml_for_training_path or not self.yaml_for_training_path.is_file():
            logger.error("학습에 필요한 dataset YAML 파일이 준비되지 않았습니다.")
            return False
        if not self.true_dataset_root_path or not self.true_dataset_root_path.is_dir():
            logger.error("데이터셋 루트 경로가 yaml_for_training_path에 올바르게 설정되지 않았습니다.")
            return False
                    
        logger.info(f"YOLO11n Detection 모델 학습 시작 (데이터 YAML: {self.yaml_for_training_path})...")
        
        bbox_labels_dir = self.true_dataset_root_path / "bbox_labels"
        labels_dir_link_target = self.true_dataset_root_path / "labels"
        
        if bbox_labels_dir.exists() and bbox_labels_dir.is_dir():
            logger.info(f"'bbox_labels' 디렉토리 존재 확인: {bbox_labels_dir}")
            if labels_dir_link_target.exists():
                if labels_dir_link_target.is_symlink():
                    logger.info(f"기존 'labels' 심볼릭 링크 제거: {labels_dir_link_target}")
                    labels_dir_link_target.unlink()
                elif labels_dir_link_target.resolve() == bbox_labels_dir.resolve():
                     logger.info(f"'labels' 디렉토리가 이미 'bbox_labels' 콘텐츠를 가리키거나 동일함: {labels_dir_link_target}")
                else:
                    backup_time = datetime.now().strftime("%Y%m%d%H%M%S")
                    backup_dir = self.true_dataset_root_path / f"labels_backup_{backup_time}"
                    logger.warning(f"기존 'labels' 디렉토리/심볼릭 링크를 백업: {labels_dir_link_target} -> {backup_dir}")
                    try:
                        shutil.move(str(labels_dir_link_target), str(backup_dir))
                    except Exception as e:
                        logger.error(f"기존 'labels' 백업 실패: {e}. 심볼릭 링크/복사를 진행하지 못할 수 있습니다.")
            
            if not labels_dir_link_target.exists():
                try:
                    os.symlink(str(bbox_labels_dir.resolve()), str(labels_dir_link_target.resolve()), target_is_directory=True)
                    logger.info(f"심볼릭 링크 생성 완료: '{bbox_labels_dir.resolve()}' -> '{labels_dir_link_target.resolve()}'")
                except OSError:
                    logger.warning(f"심볼릭 링크 생성 실패 (OSError). '{bbox_labels_dir}'를 '{labels_dir_link_target}'로 복사 시도...")
                    try:
                        shutil.copytree(str(bbox_labels_dir), str(labels_dir_link_target))
                        logger.info(f"디렉토리 복사 완료: '{bbox_labels_dir}' -> '{labels_dir_link_target}'")
                    except Exception as copy_err:
                        logger.error(f"디렉토리 복사 실패: {copy_err}. YOLO 학습이 레이블을 찾지 못할 수 있습니다.")
                except Exception as e:
                     logger.error(f"심볼릭 링크 생성 중 예외 발생: {e}. YOLO 학습이 레이블을 찾지 못할 수 있습니다.")
        elif labels_dir_link_target.is_dir():
             logger.info(f"'labels' 디렉토리({labels_dir_link_target})가 이미 존재하고 'bbox_labels'는 없음. 기존 'labels' 디렉토리 사용.")
        else:
            logger.warning(f"'bbox_labels' 및 'labels' 디렉토리 모두 찾을 수 없음 ({self.true_dataset_root_path} 기준). YOLO가 레이블을 찾지 못할 수 있습니다.")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            logger.error(f"ImportError: {e}. 'pip install ultralytics torch' 필요."); return False
            
        if self.use_wandb and wandb:
            try:
                wandb_config_args = vars(self.args).copy()
                wandb_config_args['user_yaml_path'] = str(self.user_yaml_path)
                wandb_config_args['yaml_for_training_path'] = str(self.yaml_for_training_path)
                wandb_config_args['true_dataset_root_path'] = str(self.true_dataset_root_path)
                wandb_config_args['actual_class_names'] = self.class_names
                wandb_config_args['resolved_run_name_ultralytics'] = self.run_name_ultralytics
                self.wandb_run = wandb.init(project=self.args.wandb_project, entity=self.args.wandb_entity,
                                            name=self.run_name_ultralytics, config=wandb_config_args)
                logger.info(f"W&B 로깅 활성화됨. 실행 URL: {self.wandb_run.url if self.wandb_run else 'N/A'}")
            except Exception as e:
                logger.error(f"W&B 초기화 실패: {e}. 로깅 없이 진행."); self.use_wandb = False; self.wandb_run = None

        cuda_available = torch.cuda.is_available()
        if self.device and self.device.lower() == 'cpu': device_to_use = 'cpu'
        elif cuda_available: device_to_use = self.device if self.device else '0'
        else: device_to_use = 'cpu'
        logger.info(f"CUDA 사용 가능: {cuda_available}, 학습 장치 결정: {device_to_use}")

        try:
            model = YOLO(self.model_type)
            
            train_args = {
                'data': str(self.yaml_for_training_path),
                'epochs': self.epochs, 'imgsz': self.img_size, 'batch': self.batch_size,
                'device': device_to_use, 'workers': self.workers,
                'project': str(self.project_dir_ultralytics), 'name': self.run_name_ultralytics,
                'cache': self.cache, 'task': 'detect', 'verbose': self.args.verbose,
                'seed': self.seed, 'pretrained': self.pretrained,
                'lr0': 0.001, 'lrf': 0.01, 'optimizer': 'AdamW', 'weight_decay': 0.0005,
                'warmup_epochs': 3.0, 'warmup_bias_lr': 0.1, 'cos_lr': True,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0,
                'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0,
                'flipud': 0.0, 'fliplr': 0.5, 'mosaic': 0.1, 'mixup': 0.1,
                'val': True, 'conf': 0.001, 'iou': 0.7, 'save': True,
                'save_period': 10, 'patience': 15, 'amp': True,
            }

            cli_override_params = {
                'lr0': self.args.lr0, 'lrf': self.args.lrf, 'mosaic': self.args.mosaic,
                'mixup': self.args.mixup, 'patience': self.args.patience,
                'optimizer': self.args.optimizer, 'save_period': self.args.save_period
            }
            
            overridden_keys = []
            for key, value in cli_override_params.items():
                if value is not None:
                    if key in train_args and train_args[key] != value: logger.info(f"학습 인자 재정의: '{key}' = {train_args[key]} -> {value}")
                    elif key not in train_args: logger.info(f"학습 인자 추가: '{key}' = {value}")
                    train_args[key] = value
                    overridden_keys.append(key)
            if overridden_keys: logger.info(f"CLI로 다음 학습 파라미터 설정/재정의: {', '.join(overridden_keys)}")
            else: logger.info("CLI 재정의된 추가 학습 하이퍼파라미터 없음.")

            logger.info(f"Ultralytics 학습 시작 인자 (최종):\n{json.dumps(train_args, indent=2)}")
            
            results = model.train(**train_args)
            actual_ultralytics_output_dir = Path(model.trainer.save_dir) if hasattr(model, 'trainer') and hasattr(model.trainer, 'save_dir') else (self.project_dir_ultralytics / self.run_name_ultralytics)
            logger.info(f"학습 완료! 실제 결과 저장 경로: {actual_ultralytics_output_dir}")

            logger.info("검증 데이터에서 최종 모델(best.pt) 평가 중...")
            best_model_path = actual_ultralytics_output_dir / 'weights' / 'best.pt'
            if not best_model_path.is_file():
                last_model_path = actual_ultralytics_output_dir / 'weights' / 'last.pt'
                if last_model_path.is_file(): best_model_path = last_model_path
                else: logger.error(f"평가할 모델 best.pt 또는 last.pt 없음."); return True
            
            eval_model = YOLO(str(best_model_path))
            validation_possible = False
            try:
                with open(self.yaml_for_training_path, 'r') as f: loaded_yaml = yaml.safe_load(f)
                if loaded_yaml and loaded_yaml.get('val') and (self.true_dataset_root_path / loaded_yaml['val']).is_dir():
                    validation_possible = True
            except Exception as e: logger.warning(f"검증용 YAML 확인 중 오류: {e}.")

            if validation_possible:
                val_results = eval_model.val(data=str(self.yaml_for_training_path), split='val',
                                             project=str(self.project_dir_ultralytics), 
                                             name=self.run_name_ultralytics + "_final_val_results",
                                             device=device_to_use, batch=self.batch_size, imgsz=self.img_size)
                if hasattr(val_results, 'box') and val_results.box is not None and hasattr(val_results.box, 'map'):
                    metrics_to_log = {"validation/final_box_map": val_results.box.map,
                                      "validation/final_box_map50": val_results.box.map50,
                                      "validation/final_box_map75": val_results.box.map75}
                    if self.use_wandb and self.wandb_run: self.wandb_run.log(metrics_to_log)
            else: logger.warning("최종 모델 검증 생략 (데이터셋 정보 부족 또는 접근 불가).")

            if self.use_wandb and self.wandb_run and best_model_path.is_file():
                try:
                    model_artifact_name = f"model-{self.wandb_run.id if hasattr(self.wandb_run, 'id') else self.run_name_ultralytics}"
                    model_artifact = wandb.Artifact(name=model_artifact_name, type="model", metadata={})
                    model_artifact.add_file(str(best_model_path), name="best.pt")
                    if self.yaml_for_training_path.is_file():
                         model_artifact.add_file(str(self.yaml_for_training_path), name="dataset_config_used_for_training.yaml")
                    self.wandb_run.log_artifact(model_artifact)
                    logger.info("모델 아티팩트 W&B 로깅 완료.")
                except Exception as e: logger.error(f"W&B 모델 아티팩트 로깅 오류: {e}")
            return True

        except Exception as e:
            logger.error(f"학습/평가 중 심각한 오류 발생: {e}")
            import traceback; logger.error(traceback.format_exc())
            if self.use_wandb and self.wandb_run:
                self.wandb_run.summary["error_message"] = str(e)
            return False

    def run(self):
       logger.info("YOLO11n Detection 학습 파이프라인 시작 (v7 - Direct User YAML)")
       success = False; wandb_exit_code = 1
       try:
           self.setup_directories()
           self.prepare_dataset_config()
           success = self.train_model()
           if success: wandb_exit_code = 0
       except KeyboardInterrupt: logger.warning("사용자 중단."); success = False; wandb_exit_code = 2
       except SystemExit as e: logger.error(f"스크립트 강제 종료 (코드: {e.code})."); success = (e.code == 0); wandb_exit_code = e.code if e.code is not None else 1
       except Exception as e: logger.error(f"파이프라인 실행 중 예외: {e}"); import traceback; logger.error(traceback.format_exc()); success = False; wandb_exit_code = 1
       finally:
           if self.use_wandb and wandb and self.wandb_run:
               try:
                   logger.info(f"W&B 실행 종료 시도 (코드: {wandb_exit_code}).")
                   self.wandb_run.finish(exit_code=wandb_exit_code)
                   logger.info("W&B 실행 종료 완료.")
               except Exception as e_finish: logger.error(f"W&B 실행 종료 중 오류: {e_finish}")
       return success

def main():
   parser = argparse.ArgumentParser(
       description="YOLO11n Detection 모델 학습 스크립트 (v7 - Direct User YAML Usage)",
       formatter_class=argparse.RawDescriptionHelpFormatter,
       epilog="""
사용 예:
 python %(prog)s --input /path/to/your/dataset_config.yaml --epochs 100 --model yolo11n.pt
 python %(prog)s --input ./config/my_data.yaml --names 'class1,class2' --model custom_yolo11n_backbone.yaml
"""
   )
   parser.add_argument('--input', required=True, help="사용자 정의 데이터셋 YAML 파일 경로. 'path', 'train', 'val', 'names' 키 포함 권장.")
   parser.add_argument('--model', default='yolo11n.pt', help="모델 아키텍처 또는 .pt 가중치 경로 (기본값: yolo11n.pt)")
   parser.add_argument('--output', default=None, help="학습 결과 최상위 디렉토리 (기본값: runs/detect)")
   parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수')
   parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
   parser.add_argument('--img_size', type=int, default=640, help='입력 이미지 크기')
   parser.add_argument('--names', default=None, help="클래스 이름 목록 (쉼표 구분). 데이터셋 YAML보다 우선.")
   parser.add_argument('--device', default='', help="학습 장치 (예: '0', '0,1', 'cpu')")
   parser.add_argument('--workers', type=int, default=8, help='데이터 로딩 워커 수')
   parser.add_argument('--random_seed', type=int, default=42, help='랜덤 시드')
   parser.add_argument('--pretrained', type=lambda x: (str(x).lower() in ['true', 'yes', '1']), default=True, help="사전 학습된 가중치 사용 여부 (기본값 True).")
   parser.add_argument('--cache', action='store_true', help='데이터셋 캐싱 활성화 (RAM에 로드)')
   
   parser.add_argument('--wandb_project', default='yolo11n_detect_runs', help='W&B 프로젝트 이름')
   parser.add_argument('--wandb_entity', default=None, help='W&B 엔티티 (팀/사용자 이름)')
   parser.add_argument('--no_wandb', dest='use_wandb', action='store_false', help='W&B 로깅 비활성화')
   parser.set_defaults(use_wandb=True)
   
   parser.add_argument('--verbose', action='store_true', help='상세 로깅 활성화.')
   
   parser.add_argument('--lr0', type=float, default=None, help='초기 학습률')
   parser.add_argument('--lrf', type=float, default=None, help='최종 학습률 계수')
   parser.add_argument('--mosaic', type=float, default=None, help='모자이크 증강 확률')
   parser.add_argument('--mixup', type=float, default=None, help='MixUp 증강 확률')
   parser.add_argument('--patience', type=int, default=None, help='EarlyStopping patience 에폭 수')
   parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'auto'], default=None, help='옵티마이저 선택')
   parser.add_argument('--save_period', type=int, default=None, help="체크포인트 저장 주기 (에폭). -1이면 best/last만.")

   args = parser.parse_args()

   if args.verbose:
       logger.setLevel(logging.DEBUG)
       logging.getLogger().setLevel(logging.DEBUG)
       logger.info("상세 로깅(DEBUG) 활성화됨.")

   global torch
   try:
       import torch
   except ImportError:
       logger.error("'torch' 라이브러리 없음. 'pip install torch torchvision torchaudio'로 설치해주세요."); sys.exit(1)

   if args.use_wandb and wandb is None:
       logger.warning("W&B 사용 요청되었으나 'wandb' 라이브러리 없음. W&B 로깅 없이 진행."); args.use_wandb = False
   elif args.use_wandb:
       try:
           if not (os.getenv("WANDB_API_KEY") or (hasattr(wandb.api, 'api_key') and wandb.api.api_key)):
               if not wandb.login(anonymous="allow", timeout=10):
                   logger.warning("W&B 자동 로그인 실패. W&B 로깅 없이 진행."); args.use_wandb = False
       except Exception as e_login:
           logger.error(f"W&B 로그인 확인/시도 중 오류: {e_login}. W&B 로깅 없이 진행."); args.use_wandb = False

   trainer = YOLO11nDetectTrainer(args)
   success = trainer.run()
   sys.exit(0 if success else 1)

if __name__ == "__main__":
   main()