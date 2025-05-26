#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import yaml
import multiprocessing
import random
import shutil
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CATEGORY_MAP = {
   1: 0,
   2: 1
}
POTHOLE_CATEGORY_ID = 2
YOLO_POTHOLE_CLASS_ID = str(CATEGORY_MAP[POTHOLE_CATEGORY_ID])

def calculate_polygon_bbox(coords):
   if not coords or len(coords) < 6:
       logger.debug(f"폴리곤 bbox용 좌표 부족: {coords}")
       return None
   try:
       x_coords = np.array(coords[0::2])
       y_coords = np.array(coords[1::2])
       min_x, max_x = np.min(x_coords), np.max(x_coords)
       min_y, max_y = np.min(y_coords), np.max(y_coords)
       width, height = max_x - min_x, max_y - min_y
       if width <= 0 or height <= 0:
           logger.debug(f"폴리곤에서 음수 또는 0 너비/높이: w={width}, h={height}, 좌표 {coords}")
           return None
       return [float(min_x), float(min_y), float(width), float(height)]
   except Exception as e:
       logger.error(f"좌표 {coords}에 대한 폴리곤 bbox 계산 오류: {e}")
       return None

def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
   if not bbox or len(bbox) < 4 or img_width <= 0 or img_height <= 0:
       logger.debug(f"YOLO 변환용 잘못된 입력: bbox={bbox}, img_w={img_width}, img_h={img_height}")
       return None
   try:
       x_min, y_min, w, h = bbox
       
       x_min_clamped = max(0.0, float(x_min))
       y_min_clamped = max(0.0, float(y_min))
       x_max_clamped = min(float(img_width), float(x_min) + float(w))
       y_max_clamped = min(float(img_height), float(y_min) + float(h))

       w_clamped = x_max_clamped - x_min_clamped
       h_clamped = y_max_clamped - y_min_clamped

       if w_clamped <= 0 or h_clamped <= 0:
           logger.debug(f"클램핑된 bbox에 음수 또는 0 너비/높이: w_c={w_clamped}, h_c={h_clamped}. 원본 bbox: {bbox}, img_dims: ({img_width},{img_height})")
           return None
           
       center_x = (x_min_clamped + w_clamped / 2) / img_width
       center_y = (y_min_clamped + h_clamped / 2) / img_height
       norm_w = w_clamped / img_width
       norm_h = h_clamped / img_height
       
       yolo_coords = [
           round(max(0.0, min(1.0, center_x)), 6),
           round(max(0.0, min(1.0, center_y)), 6),
           round(max(0.0, min(1.0, norm_w)), 6),
           round(max(0.0, min(1.0, norm_h)), 6)
       ]
       return yolo_coords
   except Exception as e:
       logger.error(f"COCO bbox {bbox}를 YOLO로 변환하는 중 오류: {e}")
       return None

def coco_to_yolo_lines(coco_data, img_width, img_height):
   yolo_lines = []
   annotations = coco_data.get('annotations', [])
   if not isinstance(annotations, list):
       return yolo_lines

   for ann in annotations:
       category_id = ann.get('category_id')
       if category_id not in CATEGORY_MAP:
           continue
       
       yolo_class_id = CATEGORY_MAP[category_id]
       coco_bbox = None 
       ann_id_for_log = ann.get('id', 'N/A')

       if category_id == 1:
           segmentation = ann.get('segmentation')
           if segmentation and isinstance(segmentation, list) and \
              len(segmentation) > 0 and isinstance(segmentation[0], list) and segmentation[0]:
               coco_bbox = calculate_polygon_bbox(segmentation[0])
           if coco_bbox is None:
               logger.debug(f"어노테이션 ID {ann_id_for_log} (균열): segmentation에서 bbox 계산 실패: {segmentation}")

       elif category_id == 2:
           bbox_field_value = ann.get('bbox')
           if isinstance(bbox_field_value, list) and len(bbox_field_value) == 4:
               if bbox_field_value[2] > 0 and bbox_field_value[3] > 0:
                   coco_bbox = bbox_field_value
                   logger.debug(f"어노테이션 ID {ann_id_for_log} (포트홀): 'bbox' 필드에서 사용: {coco_bbox}")
               else:
                   logger.debug(f"어노테이션 ID {ann_id_for_log} (포트홀): 'bbox' 필드 {bbox_field_value}에 0 또는 음수 너비/높이. 'segmentation' 시도.")
           else:
               logger.debug(f"어노테이션 ID {ann_id_for_log} (포트홀): 'bbox' 필드가 유효한 4개 리스트가 아님 (값: {bbox_field_value}). 'segmentation' 시도.")
           
           if coco_bbox is None:
               segmentation = ann.get('segmentation')
               if segmentation and isinstance(segmentation, list) and \
                  len(segmentation) > 0 and isinstance(segmentation[0], list) and \
                  len(segmentation[0]) == 4:
                   
                   potential_bbox_from_seg = segmentation[0]
                   if all(isinstance(coord, (int, float)) for coord in potential_bbox_from_seg) and \
                      potential_bbox_from_seg[2] > 0 and potential_bbox_from_seg[3] > 0:
                       coco_bbox = potential_bbox_from_seg
                       logger.debug(f"어노테이션 ID {ann_id_for_log} (포트홀): 'segmentation' 필드에서 bbox 사용: {coco_bbox}")
                   else:
                       logger.warning(f"어노테이션 ID {ann_id_for_log} (포트홀): 'segmentation' 데이터 {potential_bbox_from_seg}가 유효한 bbox [x,y,w,h] 형식이 아님 (w>0, h>0). 어노테이션 건너뛰기.")
               else:
                   logger.warning(f"어노테이션 ID {ann_id_for_log} (포트홀): 'bbox' 필드에서 유효한 bbox를 얻지 못했고 'segmentation' 필드가 예상 형식이 아님. 어노테이션 건너뛰기.")

       if coco_bbox:
           yolo_coords = convert_coco_bbox_to_yolo(coco_bbox, img_width, img_height)
           if yolo_coords:
               yolo_lines.append(f"{yolo_class_id} {' '.join(map(str, yolo_coords))}")
           else:
               logger.warning(f"어노테이션 ID {ann_id_for_log} (카테고리 {category_id}): COCO bbox {coco_bbox}를 YOLO 형식으로 변환 실패. img_width={img_width}, img_height={img_height}. 어노테이션 건너뛰기.")
   return yolo_lines

def has_pothole(coco_data_dict):
   if not isinstance(coco_data_dict, dict):
       logger.debug("has_pothole용 잘못된 입력 타입. dict 예상됨.")
       return False
   for annotation in coco_data_dict.get('annotations', []):
       if annotation.get('category_id') == POTHOLE_CATEGORY_ID:
           return True
   return False

def scan_and_match_files(source_images_dir, source_labels_dir):
   source_images_path_str = str(source_images_dir)
   source_labels_path_str = str(source_labels_dir)
   matched_files = []

   logger.info(f"{source_labels_path_str}의 모든 레이블 파일 사전 스캔 중 (os.walk 사용)...")
   label_file_map = {}
   duplicate_stems_log = set()
   label_candidates_for_tqdm = []
   for dirpath, _, filenames in os.walk(source_labels_path_str):
       for filename in filenames:
           if filename.lower().endswith('.json'):
               label_candidates_for_tqdm.append(Path(dirpath) / filename)

   for label_path in tqdm(label_candidates_for_tqdm, desc="레이블 맵 구축 중", unit="파일"):
       stem = label_path.stem
       if stem not in label_file_map:
           label_file_map[stem] = label_path
       elif stem not in duplicate_stems_log:
           logger.warning(f"중복 레이블 stem '{stem}' 발견. 첫 번째 사용: {label_file_map[stem]}. 무시: {label_path}")
           duplicate_stems_log.add(stem)
   logger.info(f"{len(label_file_map)}개의 고유 레이블 stem으로 레이블 맵 구축 완료.")
   if duplicate_stems_log: logger.warning(f"총 {len(duplicate_stems_log)}개 stem에 중복이 있었음.")

   logger.info(f"{source_images_path_str}의 이미지 파일 스캔 중 (os.walk 사용)...")
   image_extensions_tuple = tuple(['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'])
   all_image_paths_for_tqdm = []
   for dirpath, _, filenames in os.walk(source_images_path_str):
       for filename in filenames:
           if filename.lower().endswith(image_extensions_tuple):
               all_image_paths_for_tqdm.append(Path(dirpath) / filename)
   
   logger.info(f"{len(all_image_paths_for_tqdm)}개의 잠재적 이미지 파일 발견. 레이블 맵과 매칭 중...")

   for img_path in tqdm(all_image_paths_for_tqdm, desc="이미지와 레이블 매칭 중", unit="파일"):
       img_stem = img_path.stem
       potential_label_path = label_file_map.get(img_stem)
       if potential_label_path:
           try:
               with open(potential_label_path, 'r', encoding='utf-8') as f:
                   coco_json_data = json.load(f)
               
               img_info_found = False
               img_width, img_height = None, None
               for img_entry in coco_json_data.get('images', []):
                   if img_entry.get('file_name') == img_path.name:
                       img_width = img_entry.get('width')
                       img_height = img_entry.get('height')
                       if img_width and img_height and int(img_width) > 0 and int(img_height) > 0:
                           img_info_found = True
                           break
               
               if img_info_found:
                   pothole_detected = has_pothole(coco_json_data)
                   matched_files.append({
                       'img_path': img_path, 'label_path': potential_label_path,
                       'has_pothole': pothole_detected, 'width': int(img_width),
                       'height': int(img_height), 'coco_data': coco_json_data
                   })
               else:
                   logger.debug(f"이미지 항목 '{img_path.name}' (stem: {img_stem})이 JSON {potential_label_path}에서 찾아지지 않거나 잘못됨 (너비/높이). 건너뛰기.")
           except json.JSONDecodeError:
               logger.warning(f"레이블 {potential_label_path} (img '{img_path.name}')에 JSONDecodeError. 건너뛰기.")
           except Exception as e:
               logger.warning(f"레이블 {potential_label_path} (img '{img_path.name}') 처리 중 오류: {e}. 건너뛰기.")
       else:
           logger.debug(f"이미지 {img_path} (stem: {img_stem})에 대한 레이블이 레이블 맵에 없음.")
           
   logger.info(f"{len(matched_files)}개의 유효하고 매칭된 이미지-레이블 쌍 발견.")
   return matched_files

def apply_sampling_and_ratio_logic(all_files, total_usage_ratio, pothole_presence_ratio):
   if not all_files: return []
   num_total_initial = len(all_files)
   num_total_to_select = int(num_total_initial * total_usage_ratio)
   if num_total_to_select == 0:
       logger.warning(f"total_usage_ratio ({total_usage_ratio})를 {num_total_initial}개 파일에 적용 -> 0개 이미지. 처리할 파일 없음.")
       return []
   num_total_to_select = min(num_total_to_select, num_total_initial)
   
   pothole_files = [f for f in all_files if f['has_pothole']]
   other_files = [f for f in all_files if not f['has_pothole']]
   random.shuffle(pothole_files)
   random.shuffle(other_files)

   num_pothole_target = int(num_total_to_select * pothole_presence_ratio)
   selected_final_files = []
   
   num_potholes_can_take = min(num_pothole_target, len(pothole_files))
   selected_final_files.extend(pothole_files[:num_potholes_can_take])
   
   num_remaining_needed = num_total_to_select - len(selected_final_files)
   if num_remaining_needed > 0:
       num_others_to_take = min(num_remaining_needed, len(other_files))
       selected_final_files.extend(other_files[:num_others_to_take])
   
   if len(selected_final_files) < num_total_to_select:
       remaining_pothole_pool = pothole_files[num_potholes_can_take:]
       num_still_needed = num_total_to_select - len(selected_final_files)
       num_additional_potholes = min(num_still_needed, len(remaining_pothole_pool))
       selected_final_files.extend(remaining_pothole_pool[:num_additional_potholes])

   logger.info(f"초기 매칭: {num_total_initial}. 사용 목표: {num_total_to_select} (total_usage_ratio: {total_usage_ratio}).")
   logger.info(f"포트홀 목표 (최소): {num_pothole_target} (pothole_presence_ratio: {pothole_presence_ratio}).")
   actual_potholes_sel = len([f for f in selected_final_files if f['has_pothole']])
   logger.info(f"실제 선택된 포트홀: {actual_potholes_sel}. 총 선택: {len(selected_final_files)}.")
   
   if len(selected_final_files) < num_total_to_select and num_total_initial >= num_total_to_select :
       logger.warning(f"목표 {num_total_to_select}보다 적은 {len(selected_final_files)}개 파일만 선택할 수 있었음. "
                      f"이는 데이터 분포와 pothole_presence_ratio 제약 때문일 수 있음 "
                      f"(사용 가능한 포트홀: {len(pothole_files)}, 사용 가능한 기타: {len(other_files)}).")
   random.shuffle(selected_final_files)
   return selected_final_files

def split_data(files, train_ratio, val_ratio, test_ratio):
   if not (abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9):
       raise ValueError("Train, val, test 비율의 합이 1.0이어야 함")
   random.shuffle(files)
   total_files = len(files)
   
   train_end_idx = int(round(total_files * train_ratio))
   val_end_idx = train_end_idx + int(round(total_files * val_ratio))
   
   train_files = files[:train_end_idx]
   val_files = files[train_end_idx:val_end_idx]
   test_files = files[val_end_idx:]

   current_sum = len(train_files) + len(val_files) + len(test_files)
   if current_sum < total_files:
       test_files.extend(files[current_sum:])
   elif current_sum > total_files:
       diff = current_sum - total_files
       test_files = test_files[:-diff] if len(test_files) >= diff else []

   logger.info(f"데이터 분할: Train ({len(train_files)}), Val ({len(val_files)}), Test ({len(test_files)})")
   return {'train': train_files, 'val': val_files, 'test': test_files}

def process_and_copy_file_worker(task_args):
   file_info, split_name, output_images_split_dir, output_labels_split_dir = task_args
   img_path = file_info['img_path']
   coco_data = file_info['coco_data']
   img_width, img_height = file_info['width'], file_info['height']
   
   processed_pothole_annotations_in_file = 0

   dest_img_path = output_images_split_dir / img_path.name
   image_copied_successfully = False

   try:
       shutil.copy2(img_path, dest_img_path)
       image_copied_successfully = True
   except Exception as e:
       logger.error(f"이미지 {img_path}를 {dest_img_path}로 복사 실패: {e}")
       return f"IMG_COPY_ERROR:{img_path.name} ({e})", 0

   yolo_txt_name = img_path.stem + ".txt"
   dest_label_path = output_labels_split_dir / yolo_txt_name
   
   try:
       yolo_lines = coco_to_yolo_lines(coco_data, img_width, img_height)
       
       if yolo_lines:
           with open(dest_label_path, 'w', encoding='utf-8') as f:
               f.write('\n'.join(yolo_lines) + '\n')
           
           for line in yolo_lines:
               if line.strip().startswith(YOLO_POTHOLE_CLASS_ID + ' '):
                   processed_pothole_annotations_in_file += 1
           
           return "SUCCESS", processed_pothole_annotations_in_file
       else:
           if image_copied_successfully:
               try:
                   os.remove(dest_img_path)
                   logger.info(f"정리됨: 유효한 YOLO 어노테이션이 없어 이미지 {dest_img_path} 제거.")
               except OSError as rm_e:
                   logger.error(f"정리 실패: 유효한 어노테이션이 없는 후 이미지 {dest_img_path}를 제거할 수 없음. {rm_e}")
           return "NO_ANNOTATIONS_SKIPPED", 0

   except Exception as e:
       logger.error(f"{img_path.name}에 대한 레이블 처리 오류: {e}", exc_info=True)
       if image_copied_successfully:
           try:
               os.remove(dest_img_path)
               logger.info(f"정리됨: 레이블 처리 오류로 이미지 {dest_img_path} 제거: {e}")
           except OSError as rm_e:
               logger.error(f"정리 실패: 레이블 처리 오류 후 이미지 {dest_img_path}를 제거할 수 없음. {rm_e}")
       return f"LABEL_PROCESSING_ERROR:{img_path.name} ({e})", 0

def create_yolo_dataset_yaml(output_dir, class_names_map):
   yaml_path = Path(output_dir) / 'dataset.yaml'
   sorted_class_names = [class_names_map[i] for i in sorted(class_names_map.keys())]
   
   yaml_data = {
       'path': Path(output_dir).resolve().as_posix(),
       'train': Path('images/train').as_posix(),
       'val': Path('images/val').as_posix(),
       'test': Path('images/test').as_posix(),
       'nc': len(sorted_class_names),
       'names': sorted_class_names
   }
   try:
       with open(yaml_path, 'w', encoding='utf-8') as f:
           yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True, default_flow_style=None)
       logger.info(f"YOLO dataset.yaml 생성됨: {yaml_path}")
   except Exception as e:
       logger.error(f"dataset.yaml 생성 실패: {e}")

def main():
   parser = argparse.ArgumentParser(
       description="고급 COCO에서 YOLO Bbox 변환기 및 데이터셋 구성기 (os.walk 최적화 스캔, 엄격한 에러 처리).",
       formatter_class=argparse.RawTextHelpFormatter
   )
   parser.add_argument('--input', required=True, type=Path, help="원본 COCO JSON 레이블 파일의 루트 디렉토리 (재귀 스캔)")
   parser.add_argument('--image_dir', required=True, type=Path, help="원본 소스 이미지 파일의 루트 디렉토리 (재귀 스캔)")
   parser.add_argument('--output', default='yolo_road_dataset', type=Path, help="새 YOLO 데이터셋을 저장할 디렉토리 (기본값: yolo_road_dataset)")
   parser.add_argument('--sample_ratio', type=float, default=1.0, help="매칭된 총 쌍 중 사용할 비율 (0.0-1.0, 기본값: 1.0)")
   parser.add_argument('--target_pothole_ratio_in_sample', type=float, default=0.0, help="포트홀이 있는 선택된 이미지의 최소 비율 (0.0-1.0, 기본값: 0.0, 최소값 강제 안함)")
   parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=('TRAIN', 'VAL', 'TEST'), help="train, val, test 분할 비율 (합이 1.0, 기본값: 0.8 0.1 0.1)")
   parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2 if os.cpu_count() else 1), help="워커 프로세스 수 (기본값: 자동, os.cpu_count() - 2)")
   parser.add_argument('--verbose', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="로깅 레벨 (기본값: INFO)")
   parser.add_argument('--random_seed', type=int, default=None, help="재현 가능성을 위한 랜덤 시드 (분할 및 샘플링 셔플링에 영향)")
   
   args = parser.parse_args()

   if args.random_seed is not None:
       random.seed(args.random_seed)
       np.random.seed(args.random_seed)
       logger.info(f"랜덤 시드 사용: {args.random_seed}")
       
   logger.setLevel(args.verbose.upper())

   if not (0.0 <= args.sample_ratio <= 1.0):
       logger.critical("sample_ratio는 0.0과 1.0 사이여야 함."); return 1
   if not (0.0 <= args.target_pothole_ratio_in_sample <= 1.0):
       logger.critical("target_pothole_ratio_in_sample은 0.0과 1.0 사이여야 함."); return 1
   if not (abs(sum(args.split_ratio) - 1.0) < 1e-9 and \
           all(0.0 <= r <= 1.0 for r in args.split_ratio) and \
           len(args.split_ratio) == 3):
       logger.critical("split_ratio는 합이 1.0인 3개의 음이 아닌 수여야 함."); return 1
   if not args.image_dir.is_dir():
       logger.critical(f"소스 이미지 디렉토리를 찾을 수 없음: {args.image_dir}"); return 1
   if not args.input.is_dir():
       logger.critical(f"소스 레이블 디렉토리를 찾을 수 없음: {args.input}"); return 1
   if args.workers < 1:
       logger.warning(f"workers ({args.workers})가 1보다 작음. 1로 설정.")
       args.workers = 1

   if args.output.exists():
       logger.warning(f"출력 디렉토리 {args.output}가 이미 존재함. 파일이 덮어쓰여질 수 있음.")
   else:
       logger.info(f"출력 디렉토리 생성: {args.output}")
   for split_type in ['train', 'val', 'test']:
       (args.output / 'images' / split_type).mkdir(parents=True, exist_ok=True)
       (args.output / 'labels' / split_type).mkdir(parents=True, exist_ok=True)

   logger.info(f"--- 데이터셋 준비 시작 (os.walk 최적화 스캔, 엄격한 에러 처리) ---")
   logger.info(f"  소스 이미지 디렉토리: {args.image_dir.resolve()}")
   logger.info(f"  소스 레이블 디렉토리: {args.input.resolve()}")
   logger.info(f"  출력 디렉토리: {args.output.resolve()}")
   logger.info(f"  총 사용 비율: {args.sample_ratio}")
   logger.info(f"  포트홀 존재 비율 (최소): {args.target_pothole_ratio_in_sample}")
   logger.info(f"  분할 비율 (Train/Val/Test): {args.split_ratio}")
   logger.info(f"  워커 수: {args.workers}")
   logger.info(f"  로그 레벨: {args.verbose.upper()}")
   logger.info(f"  랜덤 시드: {args.random_seed if args.random_seed is not None else '설정 안됨'}")

   all_matched_files = scan_and_match_files(args.image_dir, args.input)
   if not all_matched_files:
       logger.error("초기 스캔 및 매칭 후 유효한 이미지-레이블 쌍을 찾을 수 없음. 종료."); return 1

   files_for_processing = apply_sampling_and_ratio_logic(all_matched_files, args.sample_ratio, args.target_pothole_ratio_in_sample)
   if not files_for_processing:
       logger.error("sample_ratio 및 target_pothole_ratio_in_sample 로직 적용 후 선택된 파일이 없음. 종료."); return 1
   
   split_file_sets = split_data(files_for_processing, *args.split_ratio)
   
   tasks = []
   for split_name, file_list in split_file_sets.items():
       if not file_list:
           logger.info(f"'{split_name}' 분할에 할당된 파일이 없음.")
           continue
       output_img_split_dir = args.output / 'images' / split_name
       output_lbl_split_dir = args.output / 'labels' / split_name
       for file_info in file_list:
           tasks.append((file_info, split_name, output_img_split_dir, output_lbl_split_dir))

   if not tasks:
       logger.warning("처리할 작업이 없음 (모든 분할이 비어있거나 선택된 파일이 없음). 종료."); return 0

   logger.info(f"{args.workers}개 워커를 사용하여 {len(tasks)}개 파일 처리 시작...")
   status_counts = {"SUCCESS":0, "NO_ANNOTATIONS_SKIPPED":0, "IMG_COPY_ERROR":0, "LABEL_PROCESSING_ERROR":0, "OTHER_ERROR":0}
   total_successfully_converted_potholes = 0

   try:
       with multiprocessing.Pool(processes=args.workers) as pool:
           results_from_workers = list(tqdm(pool.map(process_and_copy_file_worker, tasks), total=len(tasks), desc="변환/복사 중", unit="파일"))
   except Exception as e_pool:
       logger.critical(f"멀티프로세싱 중 치명적 오류 발생: {e_pool}", exc_info=True)
       return 1

   for worker_result_tuple in results_from_workers:
       status_string, pothole_count_in_file = worker_result_tuple
       
       base_status = status_string.split(":")[0] if ":" in status_string else status_string
       if base_status not in status_counts:
           logger.warning(f"워커로부터 예상치 못한 상태 수신: {status_string}")
           base_status = "OTHER_ERROR"
       status_counts[base_status] = status_counts.get(base_status, 0) + 1
       
       if base_status == "SUCCESS":
           total_successfully_converted_potholes += pothole_count_in_file
           
       if base_status != status_string and logger.isEnabledFor(logging.DEBUG):
           logger.debug(f"워커가 상세 상태 반환: {status_string}")

   logger.info("\n--- 변환 요약 ---")
   total_results_received = sum(status_counts.values())
   logger.info(f"제출된 작업: {len(tasks)}. 받은 결과: {total_results_received}.")
   for k, v in status_counts.items():
       if v > 0:
           logger.info(f"  {k}: {v}")
   
   logger.info(f"  성공적으로 변환된 '도로(홀)' (포트홀, YOLO 클래스 ID {YOLO_POTHOLE_CLASS_ID}) 어노테이션: {total_successfully_converted_potholes}")

   class_names_map_for_yaml = {
       CATEGORY_MAP[1]: "도로균열",
       CATEGORY_MAP[2]: "도로(홀)"
   }
   create_yolo_dataset_yaml(args.output, class_names_map_for_yaml)
   
   logger.info("--- 데이터셋 준비 완료 ---")

   successful_outcomes = status_counts.get("SUCCESS", 0)
   if successful_outcomes == 0 and len(tasks) > 0:
       logger.error("치명적: 유효한 어노테이션으로 성공적으로 처리된 파일이 없음. 출력 데이터셋이 비어있거나 불완전할 가능성.")
       return 1

   total_critical_errors = status_counts.get("IMG_COPY_ERROR", 0) + status_counts.get("LABEL_PROCESSING_ERROR", 0) + status_counts.get("OTHER_ERROR",0)
   if total_critical_errors > 0:
       logger.warning(f"경고: {total_critical_errors}개 파일이 처리 중 치명적 오류를 만나 제외됨. 자세한 내용은 로그를 확인하세요.")
   
   files_not_included = len(tasks) - successful_outcomes
   if files_not_included > 0 and successful_outcomes > 0 :
        logger.info(f"{files_not_included}개 파일이 오류 또는 어노테이션 부족으로 최종 데이터셋에 포함되지 않음.")
   elif successful_outcomes > 0 and files_not_included == 0:
       logger.info("선택된 모든 파일이 성공적으로 처리됨!")

   return 0

if __name__ == "__main__":
   try:
       import yaml
       import numpy
   except ImportError as e:
       print(f"치명적 오류: 라이브러리 누락: {e.name}. PyYAML과 NumPy를 설치하세요 (예: pip install PyYAML numpy).", file=sys.stderr)
       sys.exit(1)
   
   if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
       current_start_method = multiprocessing.get_start_method(allow_none=True)
       if current_start_method != 'spawn':
           try:
               multiprocessing.set_start_method('spawn', force=True)
               if '--verbose' in sys.argv and 'DEBUG' in sys.argv[sys.argv.index('--verbose') + 1].upper():
                    print(f"디버그: 멀티프로세싱 시작 방법을 'spawn'으로 설정. 이전: {current_start_method}", file=sys.stderr)
           except RuntimeError as e_mp:
               print(f"경고: 멀티프로세싱 시작 방법을 'spawn'으로 설정할 수 없음: {e_mp}", file=sys.stderr)

   sys.exit(main())