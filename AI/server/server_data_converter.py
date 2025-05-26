import os
import json
import argparse
import logging
import random
import numpy as np
import cv2
import shutil
import tempfile
import time
import threading
import queue
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from shapely.geometry import LineString, Polygon, MultiPolygon

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable
from shapely.geometry import LineString, Polygon
from shapely.errors import TopologicalError
from scipy.signal import savgol_filter
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger('Road2YOLO-SEG')

class ThreadSafeCounter:
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.lock = Lock()
    
    def increment(self, amount=1):
        with self.lock:
            self.value += amount
            return self.value
    
    def get(self):
        with self.lock:
            return self.value

class ThreadSafeDict:
    def __init__(self):
        self.data = {}
        self.lock = Lock()
    
    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)
    
    def set(self, key, value):
        with self.lock:
            self.data[key] = value

class ThreadSafeSet:
    def __init__(self):
        self.data = set()
        self.lock = Lock()
    
    def add(self, item):
        with self.lock:
            self.data.add(item)
    
    def contains(self, item):
        with self.lock:
            return item in self.data
    
    def size(self):
        with self.lock:
            return len(self.data)

class ImageProcessor:
    def __init__(self, queue, converter):
        self.queue = queue
        self.converter = converter
        self.running = True

    def process(self):
        while self.running:
            try:
                task = self.queue.get(timeout=0.1)
                if task is None:
                    self.running = False
                    try:
                        self.queue.task_done()
                    except ValueError:
                        pass
                    break

                image_file, annotations, image_width, image_height, split = task
                self.converter._process_single_image(image_file, annotations, image_width, image_height, split)
                self.queue.task_done()

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"작업 처리 중 오류 발생 (task: {task if 'task' in locals() else 'N/A'}): {e}", exc_info=True)
                try:
                    self.queue.task_done()
                except ValueError:
                    pass

class Road2YOLOSeg:
    def __init__(self, input_path, output_path, image_dir=None, buffer_size=10.0,
                 adaptive=False, split_ratio=[0.7, 0.2, 0.1], num_workers=1,
                 use_sam_for_potholes=False,
                 sam_checkpoint_path=None, sam_model_type="vit_b", sam_device="cpu",
                 sam_polygon_epsilon_factor=0.005, sam_min_contour_area=10,
                 sample_ratio=1.0, target_pothole_ratio_in_sample=None, random_seed=None
                 ):

        self.use_sam_for_potholes = use_sam_for_potholes
        
        self._check_system()

        self.input_path = Path(input_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.image_dir = Path(image_dir).resolve() if image_dir else None
        self.buffer_size = buffer_size
        self.adaptive = adaptive
        self.split_ratio = split_ratio
        self.num_workers = max(1, min(num_workers, os.cpu_count() or 1))
        logger.info(f"사용할 워커 수: {self.num_workers}")

        self.sample_ratio = np.clip(sample_ratio, 0.0, 1.0)
        self.target_pothole_ratio_in_sample = target_pothole_ratio_in_sample
        if self.target_pothole_ratio_in_sample is not None:
            self.target_pothole_ratio_in_sample = np.clip(target_pothole_ratio_in_sample, 0.0, 1.0)
            logger.info(f"샘플 내 목표 포트홀 비율: {self.target_pothole_ratio_in_sample*100:.1f}%")
        else:
            logger.info("샘플 내 포트홀 비율 조정 사용 안 함.")
        self.random_seed = random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
            logger.info(f"샘플링 랜덤 시드: {self.random_seed}")

        self.file_locks = {}
        self.file_locks_mutex = Lock() 
        self.task_queue = queue.Queue(maxsize=self.num_workers*10)
        self.processed_images = ThreadSafeSet()
        self.scanned_image_files = set()
        self.image_path_cache = {}
        self.image_cache_lock = Lock()

        self.labels_path = self.output_path/'labels'
        self.images_path = self.output_path/'images'
        self.splits = ['train','val','test']
        self._create_folder_structure()
        self.temp_dir = tempfile.mkdtemp(prefix="road2yolo_")
        logger.debug(f"임시 폴더 생성: {self.temp_dir}")

        self._initialize_stats()

        self.sam_predictor = None
        self.sam_model_lock = Lock()
        self.sam_polygon_epsilon_factor = sam_polygon_epsilon_factor
        self.sam_min_contour_area = sam_min_contour_area
        self._load_sam_model(sam_checkpoint_path, sam_model_type, sam_device)

    def _initialize_stats(self):
        self.stats = {
            "scanned_json_files": ThreadSafeCounter(),
            "total_images_in_json": ThreadSafeCounter(),
            "total_pothole_images_in_json": ThreadSafeCounter(),
            "total_other_images_in_json": ThreadSafeCounter(),
            "target_sample_size": ThreadSafeCounter(),
            "final_sampled_images": ThreadSafeCounter(),
            "final_sampled_pothole": ThreadSafeCounter(),
            "final_sampled_other": ThreadSafeCounter(),
            "processed_images": ThreadSafeCounter(),
            "processed_annotations": {1: ThreadSafeCounter(), 2: ThreadSafeCounter()},
            "skipped_images_no_annotation": ThreadSafeCounter(),
            "skipped_images_not_found": ThreadSafeCounter(),
            "skipped_images_move_fail": ThreadSafeCounter(),
            "errors_json_scan": ThreadSafeCounter(),
            "errors_image_processing": ThreadSafeCounter(),
            "errors_annotation_processing": ThreadSafeCounter(),
            "errors_file_move": ThreadSafeCounter(),
            "errors_missing_imagedir": ThreadSafeCounter(),
            "split_counts": {split: ThreadSafeCounter() for split in self.splits},
            "sam_processed_potholes": ThreadSafeCounter(),
            "sam_errors_pothole": ThreadSafeCounter()
        }

    def _load_sam_model(self, checkpoint_path, model_type, device):
        if not self.use_sam_for_potholes:
            return
        if not SAM_AVAILABLE:
            logger.error("SAM 사용 불가: 'segment_anything' 라이브러리 없음.")
            self.use_sam_for_potholes = False
            return
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.error(f"SAM 사용 불가: 체크포인트 파일 없음 ({checkpoint_path}).")
            self.use_sam_for_potholes = False
            return
        try:
            logger.info(f"SAM 모델 로딩 중... ({model_type}/{device})")
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
            logger.info("SAM 모델 로딩 완료.")
            logger.info("도로홀(category_id=2) 처리에 SAM을 사용합니다.")
        except Exception as e:
            logger.error(f"SAM 모델 로딩 실패: {e}", exc_info=True)
            self.use_sam_for_potholes = False

    def _check_system(self):
        required = ['numpy', 'cv2', 'shapely', 'scipy']
        missing = [lib for lib in required if not self._try_import(lib)]
        if missing:
            logger.warning(f"필요 라이브러리 누락: {', '.join(missing)}")
        if self.use_sam_for_potholes and not SAM_AVAILABLE:
            logger.warning("'segment_anything' 라이브러리가 설치되지 않았습니다.")

    def _try_import(self, library_name):
        try:
            __import__(library_name)
            return True
        except ImportError:
            return False

    def _create_folder_structure(self):
        self.labels_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
        for split in self.splits:
            (self.labels_path/split).mkdir(exist_ok=True)
            (self.images_path/split).mkdir(exist_ok=True)

    def _get_file_lock(self, filepath):
        filepath_str = str(filepath)
        with self.file_locks_mutex:
            if filepath_str not in self.file_locks:
                self.file_locks[filepath_str] = Lock()
            return self.file_locks[filepath_str]

    def process_directory(self):
        logger.info(f"처리 시작: 입력={self.input_path}, 출력={self.output_path}")
        logger.info(f"전체 이미지 샘플링 비율: {self.sample_ratio*100:.1f}%")

        json_files = list(self.input_path.rglob('*.json'))
        if not json_files:
            logger.warning(f"JSON 파일 없음: {self.input_path}")
            return

        pothole_items, other_items = self._scan_and_classify_images(json_files)
        num_pothole_available = len(pothole_items)
        num_other_available = len(other_items)
        num_total_unique = num_pothole_available + num_other_available
        logger.info(f"스캔 완료: 총 고유 이미지 {num_total_unique}개 (포트홀 포함: {num_pothole_available}개, 미포함: {num_other_available}개)")

        if num_total_unique == 0:
            logger.warning("스캔 결과 처리할 이미지 없음.")
            self._cleanup()
            self.print_stats()
            return

        sampled_image_items = self._perform_sampling(pothole_items, other_items)
        final_sample_size = len(sampled_image_items)
        if final_sample_size == 0:
            logger.info("샘플링된 이미지 없음. 종료.")
            self._cleanup()
            self.print_stats()
            return

        self._process_sampled_items(sampled_image_items)

    def _scan_and_classify_images(self, json_files):
        pothole_image_items = []
        other_image_items = []
        self.scanned_image_files.clear()
        
        available_images = {}
        if self.image_dir:
            logger.info(f"이미지 디렉토리 스캔 중: {self.image_dir}")
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                for img_path in self.image_dir.glob(f"**/*{ext}"):
                    available_images[img_path.stem] = img_path
                    if 'wetsunny' in img_path.stem:
                        alt_name = img_path.stem.replace('wetsunny', 'wet')
                        available_images[alt_name] = img_path
                    elif 'rainny' in img_path.stem:
                        alt_name = img_path.stem.replace('rainny', 'wet')
                        available_images[alt_name] = img_path
            
            logger.info(f"이미지 디렉토리에서 {len(available_images)}개 파일 인덱싱 완료")
        
        matched_count = 0
        total_image_entries = 0
        
        for json_file_path in tqdm(json_files, desc="JSON 파일 스캔 및 이미지 매칭 중"):
            self.stats["scanned_json_files"].increment()
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                annotations = data.get('annotations', [])
                ann_by_imgid = {}
                pothole_imgids = set()
                
                for ann in annotations:
                    img_id = ann.get('image_id')
                    if img_id is not None:
                        ann_by_imgid.setdefault(img_id, []).append(ann)
                        if ann.get('category_id') == 2:
                            pothole_imgids.add(img_id)
                
                for img_info in data.get('images', []):
                    total_image_entries += 1
                    img_id = img_info.get('id')
                    img_file = img_info.get('file_name')
                    
                    if not img_id or not img_file:
                        continue
                    
                    if img_file in self.scanned_image_files:
                        continue
                    
                    img_basename = Path(img_file).stem
                    img_exists = img_basename in available_images
                    
                    if not self.image_dir:
                        img_exists = True
                    
                    if img_exists or not self.image_dir:
                        self.scanned_image_files.add(img_file)
                        self.stats["total_images_in_json"].increment()
                        matched_count += 1
                        
                        if img_exists and self.image_dir:
                            actual_path = available_images[img_basename]
                            with self.image_cache_lock:
                                self.image_path_cache[img_file] = str(actual_path)
                        
                        item_data = {
                            "image_file": img_file, 
                            "annotations": ann_by_imgid.get(img_id, []),
                            "image_width": img_info.get('width', 1920), 
                            "image_height": img_info.get('height', 1080)
                        }
                        
                        if img_id in pothole_imgids:
                            pothole_image_items.append(item_data)
                            self.stats["total_pothole_images_in_json"].increment()
                        else:
                            other_image_items.append(item_data)
                            self.stats["total_other_images_in_json"].increment()
                    
            except Exception as e:
                logger.error(f"JSON 스캔/분류 오류 {json_file_path}: {e}", exc_info=True)
                self.stats["errors_json_scan"].increment()
        
        if self.image_dir:
            logger.info(f"JSON 항목 {total_image_entries}개 중 {matched_count}개 이미지 파일 매칭됨 ({matched_count/total_image_entries*100:.1f}%)")
        else:
            logger.info(f"이미지 디렉토리 미지정 모드: JSON에서 {matched_count}개 이미지 정보 처리")
        
        return pothole_image_items, other_image_items

    def _perform_sampling(self, pothole_items, other_items):
        num_pothole_available = len(pothole_items)
        num_other_available = len(other_items)
        num_total_unique = num_pothole_available + num_other_available
        sampled_items = []
        N_initial_target = 0

        if self.sample_ratio >= 1.0 and self.target_pothole_ratio_in_sample is None:
            sampled_items = pothole_items + other_items
            N_initial_target = num_total_unique
        elif self.sample_ratio <= 0.0:
            pass
        else:
            N_initial_target = int(round(num_total_unique * self.sample_ratio))
            if N_initial_target == 0 and num_total_unique > 0:
                N_initial_target = 1
            N_initial_target = min(N_initial_target, num_total_unique)

            S_p, S_o = 0, 0
            if self.target_pothole_ratio_in_sample is None:
                all_items = pothole_items + other_items
                sampled_items = random.sample(all_items, N_initial_target)
            else:
                N_p_target = int(round(N_initial_target * self.target_pothole_ratio_in_sample))
                N_o_target = N_initial_target - N_p_target
                
                S_p = min(N_p_target, num_pothole_available)
                if S_p < N_p_target:
                    logger.warning(f"포트홀 이미지 부족({num_pothole_available}<{N_p_target}). {S_p}개만 포함.")
                
                needed_other = N_initial_target - S_p
                S_o = min(needed_other, num_other_available)
                if S_o < needed_other:
                    logger.warning(f"비-포트홀 이미지 부족({num_other_available}<{needed_other}). {S_o}개만 포함.")

                sampled_p = random.sample(pothole_items, S_p) if S_p > 0 else []
                sampled_o = random.sample(other_items, S_o) if S_o > 0 else []
                sampled_items = sampled_p + sampled_o
                random.shuffle(sampled_items)

        final_size = len(sampled_items)
        final_pothole = sum(1 for item in sampled_items if any(a.get('category_id')==2 for a in item['annotations']))
        final_other = final_size - final_pothole
        self.stats["target_sample_size"].increment(N_initial_target)
        self.stats["final_sampled_images"].increment(final_size)
        self.stats["final_sampled_pothole"].increment(final_pothole)
        self.stats["final_sampled_other"].increment(final_other)
        ratio = final_pothole/final_size*100 if final_size>0 else 0
        logger.info(f"최종 샘플링 결과: 총 {final_size}개 이미지 (포트홀 포함: {final_pothole}개 [{ratio:.1f}%])")

        return sampled_items

    def _process_sampled_items(self, sampled_items):
        workers = []
        progress_bar = tqdm(total=len(sampled_items), desc="샘플링된 이미지 처리 중")
        try:
            if self.num_workers > 1:
                logger.info(f"{self.num_workers}개의 워커 스레드 시작.")
                for i in range(self.num_workers):
                    processor = ImageProcessor(self.task_queue, self)
                    worker = threading.Thread(target=processor.process, name=f"Worker-{i+1}", daemon=True)
                    worker.start()
                    workers.append((worker, processor))
            
            for item in sampled_items:
                if not item["annotations"]:
                    logger.debug(f"어노테이션 없음 건너뛰기: {item['image_file']}")
                    self.stats["skipped_images_no_annotation"].increment()
                    progress_bar.update(1)
                    continue
                split = self._get_optimal_split()
                task_data = (item["image_file"], item["annotations"], item["image_width"], item["image_height"], split)
                if self.num_workers > 1:
                    self.task_queue.put(task_data)
                else:
                    self._process_single_image(*task_data)
                    progress_bar.update(1)

            if self.num_workers > 1:
                logger.info("모든 작업 큐 추가 완료. 완료 대기...")
                self.task_queue.join()
                logger.info("모든 작업 완료.")
                final_n = self.stats['processed_images'].get() + self.stats['skipped_images_not_found'].get() + self.stats['skipped_images_move_fail'].get()
                if progress_bar.n < final_n:
                    progress_bar.update(final_n - progress_bar.n)
                if progress_bar.n < len(sampled_items):
                    progress_bar.update(len(sampled_items) - progress_bar.n)

        finally:
            if progress_bar:
                progress_bar.close()
            if self.num_workers > 1:
                for _ in range(self.num_workers):
                    self.task_queue.put(None)
                for worker, processor in workers:
                    worker.join(timeout=5.0)

    def _cleanup(self):
        try:
             if hasattr(self, 'temp_dir') and self.temp_dir and Path(self.temp_dir).exists():
                 shutil.rmtree(self.temp_dir)
                 logger.debug(f"임시 폴더 삭제: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"임시 폴더 정리 오류: {e}")

    def _process_single_image(self, image_file, annotations, image_width, image_height, split):
        if self.processed_images.contains(image_file):
            return

        original_image_path_abs = None
        try:
            if self.image_dir:
                original_image_path_abs = self.find_image_file(image_file)
                if not original_image_path_abs:
                    logger.warning(f"이미지 파일 찾기 실패: {image_file}. 건너뛰니다.")
                    self.stats["skipped_images_not_found"].increment()
                    return

                moved_successfully = self._move_image_file(original_image_path_abs, image_file, split)
                if not moved_successfully:
                    logger.error(f"이미지 이동 실패: {image_file}. 해당 이미지 처리를 중단합니다.")
                    self.stats["skipped_images_move_fail"].increment()
                    return
            else:
                if self.adaptive or self.use_sam_for_potholes:
                    logger.error("--image_dir 필요 (adaptive/SAM 사용 시).")
                    self.stats["errors_missing_imagedir"].increment()
                    return
                original_image_path_abs = None

            annotations_processed_count = self._convert_annotations(
                image_file, annotations, image_width, image_height, original_image_path_abs, split
            )

            if annotations_processed_count > 0:
                self.stats["processed_images"].increment()
                self.stats["split_counts"][split].increment()
                self.processed_images.add(image_file)
            else:
                 logger.warning(f"이미지 '{image_file}'에 대해 처리된 유효 어노테이션 없음 (또는 처리 실패).")

        except Exception as e:
            logger.error(f"이미지 처리 중 예상치 못한 오류 {image_file}: {e}", exc_info=True)
            self.stats["errors_image_processing"].increment()

    def _move_image_file(self, src_path_obj, image_file_name, split):
            if not isinstance(src_path_obj, Path) or not src_path_obj.is_file():
                return False
            src_lock = self._get_file_lock(src_path_obj)
            basename = Path(image_file_name).stem
            ext = src_path_obj.suffix
            target_filename = f"{basename}{ext}"
            dst_path = self.images_path/split/target_filename
            dst_lock = self._get_file_lock(dst_path)
            temp_path = Path(self.temp_dir)/f"{uuid.uuid4()}{ext}"
            try:
                with src_lock:
                    if not src_path_obj.exists():
                        return False
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path_obj, temp_path)
                with dst_lock:
                    if temp_path.exists():
                        shutil.move(str(temp_path), str(dst_path))
                    else:
                        logger.warning(f"임시 파일 이동 실패: {temp_path}")
                        return False
                with src_lock:
                    if src_path_obj.exists() and dst_path.exists():
                        try:
                            src_path_obj.unlink()
                        except Exception as e:
                            logger.warning(f"원본 삭제 실패: {src_path_obj}, {e}")
                    elif not dst_path.exists():
                        logger.error(f"이동 최종 확인 실패: {dst_path}")
                        return False
                return True
            except Exception as e:
                logger.error(f"이미지 이동 오류 {src_path_obj}->{dst_path}: {e}", exc_info=True)
                self.stats["errors_file_move"].increment()
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except OSError:
                    pass
                return False

    def _get_optimal_split(self):
        counts = {s: self.stats["split_counts"][s].get() for s in self.splits}
        total = sum(counts.values())
        if total == 0:
            rand = random.random()
            cum_ratio = 0.0
            for i, ratio in enumerate(self.split_ratio):
                cum_ratio += ratio
                if rand < cum_ratio:
                    return self.splits[i]
            return self.splits[-1]
        target = {self.splits[i]: self.split_ratio[i] for i in range(len(self.splits))}
        current = {s: c/total if total>0 else 0 for s, c in counts.items()}
        diff = {s: target[s]-current[s] for s in self.splits}
        return max(diff, key=diff.get)

    def find_image_file(self, image_file_name_str):
        if not self.image_dir:
            return None
        
        img_path = Path(image_file_name_str)
        basename = img_path.stem
        
        with self.image_cache_lock:
            cache_str = self.image_path_cache.get(image_file_name_str)
            if cache_str:
                cache_path = Path(cache_str)
                if cache_path.is_file():
                    return cache_path
        
        direct_path = self.image_dir/image_file_name_str
        if direct_path.is_file():
            with self.image_cache_lock:
                self.image_path_cache[image_file_name_str] = str(direct_path)
                return direct_path
        
        for split in ['train', 'val', 'test']:
            split_path = self.image_dir/split/image_file_name_str
            if split_path.is_file():
                with self.image_cache_lock:
                    self.image_path_cache[image_file_name_str] = str(split_path)
                    logger.debug(f"하위 폴더 {split}에서 이미지 찾음: {split_path}")
                    return split_path
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            alt_path = self.image_dir/f"{basename}{ext}"
            if alt_path.is_file():
                with self.image_cache_lock:
                    self.image_path_cache[image_file_name_str] = str(alt_path)
                    return alt_path
            
            for split in ['train', 'val', 'test']:
                split_alt_path = self.image_dir/split/f"{basename}{ext}"
                if split_alt_path.is_file():
                    with self.image_cache_lock:
                        self.image_path_cache[image_file_name_str] = str(split_alt_path)
                        logger.debug(f"하위 폴더 {split}에서 대체 확장자로 이미지 찾음: {split_alt_path}")
                        return split_alt_path
        
        try:
            for found in self.image_dir.rglob(f"{basename}.*"):
                if found.is_file():
                    with self.image_cache_lock:
                        self.image_path_cache[image_file_name_str] = str(found)
                        logger.debug(f"재귀 검색으로 이미지 찾음: {found}")
                        return found
        except Exception as e:
            logger.warning(f"이미지 검색 오류(rglob): {e}")
        
        return None

    def _mask_to_yolo_polygons_sam(self, binary_mask, image_width, image_height):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.sam_min_contour_area:
            return []
        epsilon = self.sam_polygon_epsilon_factor * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) >= 3:
            norm_flat = []
            for pt in approx:
                x, y = pt[0]
                nx = np.clip(x/image_width, 0., 1.)
                ny = np.clip(y/image_height, 0., 1.)
                norm_flat.extend([round(nx, 6), round(ny, 6)])
            if len(set(map(tuple, np.round(np.array(norm_flat).reshape(-1, 2) * 1000)))) < 3:
                logger.debug("Degenerate polygon detected after approxPolyDP, skipping.")
                return []
            return [norm_flat]
        return []

    def _sam_bbox_to_polygon(self, image_rgb_obj, coco_bbox, image_width, image_height):
        if not self.sam_predictor or image_rgb_obj is None:
            return None
        try:
            x_min, y_min, w, h = coco_bbox
            input_box = np.array([x_min, y_min, x_min+w, y_min+h])
            with self.sam_model_lock:
                self.sam_predictor.set_image(image_rgb_obj)
                masks, scores, _ = self.sam_predictor.predict(box=input_box[None, :], multimask_output=False)
            binary_mask = (masks[0].astype(np.uint8)*255)
            poly_list = self._mask_to_yolo_polygons_sam(binary_mask, image_width, image_height)
            if poly_list:
                self.stats["sam_processed_potholes"].increment()
                return poly_list[0]
            else:
                self.stats["sam_errors_pothole"].increment()
                return None
        except Exception as e:
            logger.error(f"SAM 오류(bbox:{coco_bbox}): {e}", exc_info=True)
            self.stats["sam_errors_pothole"].increment()
            return None

    def _convert_annotations(self, image_file, annotations, image_width, image_height, original_image_path_abs=None, split='train'):
        basename = Path(image_file).stem
        output_file = self.labels_path/split/f"{basename}.txt"
        output_lock = self._get_file_lock(output_file)
        processed_count = 0
        yolo_lines = []
        loaded_bgr = None
        loaded_rgb = None
        binary_adaptive = None
        needs_adaptive = self.adaptive
        needs_sam = self.use_sam_for_potholes and self.sam_predictor
        
        if (needs_adaptive or needs_sam):
            ext = Path(original_image_path_abs).suffix if original_image_path_abs else '.jpg'
            moved_image_path = self.images_path/split/f"{basename}{ext}"
            
            if moved_image_path.is_file():
                try:
                    loaded_bgr = cv2.imread(str(moved_image_path))
                    if loaded_bgr is not None:
                        logger.debug(f"이동된 이미지 로드 성공: {moved_image_path}")
                        if needs_sam:
                            loaded_rgb = cv2.cvtColor(loaded_bgr, cv2.COLOR_BGR2RGB)
                        if needs_adaptive:
                            _, binary_adaptive_cand = self.preprocess_crack_image(loaded_bgr)
                            if binary_adaptive_cand is not None:
                                binary_adaptive = binary_adaptive_cand
                            else:
                                logger.warning(f"Adaptive용 이진 이미지 생성 실패: {image_file}")
                    else:
                        logger.warning(f"이미지 로드 실패(imread): {moved_image_path}")
                except Exception as e:
                    logger.error(f"이미지 로드/전처리 오류({moved_image_path}): {e}")
            
            elif original_image_path_abs and original_image_path_abs.is_file():
                try:
                    loaded_bgr = cv2.imread(str(original_image_path_abs))
                    if loaded_bgr is not None:
                        if needs_sam:
                            loaded_rgb = cv2.cvtColor(loaded_bgr, cv2.COLOR_BGR2RGB)
                        if needs_adaptive:
                            _, binary_adaptive_cand = self.preprocess_crack_image(loaded_bgr)
                            if binary_adaptive_cand is not None:
                                binary_adaptive = binary_adaptive_cand
                            else:
                                logger.warning(f"Adaptive용 이진 이미지 생성 실패: {image_file}")
                    else:
                        logger.warning(f"이미지 로드 실패(imread): {original_image_path_abs}")
                except Exception as e:
                    logger.error(f"이미지 로드/전처리 오류({image_file}): {e}")
    
        for ann in annotations:
            cat_id = ann.get('category_id')
            if cat_id is None:
                continue
            cls_id = cat_id - 1
            if cls_id < 0:
                continue
            yolo_coords = None
            try:
                if cat_id == 1:
                    seg = ann.get('segmentation')
                    if not seg or not isinstance(seg, list) or not seg[0]:
                        continue
                    for poly_flat in seg:
                        if len(poly_flat) < 4:
                            continue
                        shp_poly = None
                        if self.adaptive and binary_adaptive is not None:
                            wids, _ = self.measure_crack_width(binary_adaptive, poly_flat)
                            shp_poly = self.create_adaptive_polygon(poly_flat, wids)
                        else:
                            shp_poly = self.polyline_to_polygon(poly_flat)
                        if shp_poly and not shp_poly.is_empty:
                            coords = self.polygon_to_yolo_format(shp_poly, image_width, image_height)
                            if coords:
                                yolo_lines.append(f"{cls_id} {' '.join(map(str, coords))}")
                                self.stats["processed_annotations"][cat_id].increment()
                                processed_count += 1
                elif cat_id == 2:
                    bbox = ann.get('segmentation')
                    if not bbox or not isinstance(bbox, list) or len(bbox[0]) < 4:
                        continue
                    bbox = bbox[0]
                    
                    if needs_sam and loaded_rgb is not None:
                        yolo_coords = self._sam_bbox_to_polygon(loaded_rgb, bbox, image_width, image_height)
                    else:
                        if needs_sam and loaded_rgb is None:
                            logger.debug(f"SAM용 이미지 로드 실패({image_file}), bbox 대체.")
                        shp_poly = self.bbox_to_polygon(bbox)
                        if shp_poly and not shp_poly.is_empty:
                            yolo_coords = self.polygon_to_yolo_format(shp_poly, image_width, image_height)
                            
                    if yolo_coords:
                        yolo_lines.append(f"{cls_id} {' '.join(map(str, yolo_coords))}")
                        self.stats["processed_annotations"][cat_id].increment()
                        processed_count += 1
            except Exception as e:
                logger.error(f"어노테이션 오류(ann:{ann.get('id','N/A')},img:{image_file}): {e}", exc_info=False)
                self.stats["errors_annotation_processing"].increment()
    
        if yolo_lines:
            with output_lock:
                try:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(yolo_lines) + '\n')
                except IOError as e:
                    logger.error(f"라벨 쓰기 오류 {output_file}: {e}")
                    return 0
        return processed_count

    def polyline_to_polygon(self, polyline_coords):
        pts = []
        L = len(polyline_coords)
        i = 0
        while i + 1 < L:
            pts.append((polyline_coords[i], polyline_coords[i+1]))
            i += 2
        if len(pts) < 2:
            raise ValueError("폴리라인 포인트 부족")
        line = LineString(pts)
        buf = max(0.1, self.buffer_size)
        dil = line.buffer(buf, cap_style=2, join_style=2)
        if not dil.is_valid:
            dil = dil.buffer(0)
            if not dil.is_valid:
                return None
        return dil

    def bbox_to_polygon(self, bbox_coords):
        if len(bbox_coords) < 4:
            raise ValueError("Bbox 좌표 부족")
        x, y, w, h = bbox_coords[:4]
        if w <= 0 or h <= 0:
            raise ValueError("Bbox 크기 오류")
        pgc = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
        poly = Polygon(pgc)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                return None
        return poly

    def polygon_to_yolo_format(self, polygon, image_width, image_height):
        if image_width <= 0 or image_height <= 0:
            return []
        try:
            if isinstance(polygon, MultiPolygon):
                polygon = max(polygon.geoms, key=lambda p: p.area)
            if not hasattr(polygon, 'exterior') or polygon.exterior is None:
                return []
            coords = list(polygon.exterior.coords)
            if len(coords) < 3:
                return []
            if coords[0] == coords[-1] and len(coords) > 1:
                coords = coords[:-1]
            norm = []
            for x, y in coords:
                nx = round(np.clip(x/image_width, 0., 1.), 6)
                ny = round(np.clip(y/image_height, 0., 1.), 6)
                norm.extend([nx, ny])
            if len(norm) < 6:
                return []
            return norm
        except Exception as e:
            logger.error(f"YOLO 형식 변환 오류: {e}", exc_info=True)
            return []

    def preprocess_crack_image(self, bgr_image):
        try:
            if bgr_image is None:
                raise ValueError("입력 이미지 없음")
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            eq = clahe.apply(gray)
            blur = cv2.GaussianBlur(eq, (5, 5), 0)
            kern = np.ones((7, 7), np.uint8)
            top = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kern)
            _, binary = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            return gray, binary
        except Exception as e:
            logger.error(f"이미지 전처리 오류: {e}", exc_info=True)
            return None, None

    def measure_crack_width(self, binary_image, polyline_coords, sampling_interval=10):
        if binary_image is None or len(polyline_coords) < 4:
            return [], self.buffer_size
        pts = []
        h, w = binary_image.shape[:2]
        L = len(polyline_coords)
        i = 0
        while i + 1 < L:
            x = int(np.clip(polyline_coords[i], 0, w-1))
            y = int(np.clip(polyline_coords[i+1], 0, h-1))
            pts.append((x, y))
            i += 2
        if len(pts) < 2:
            return [], self.buffer_size
        widths = []
        for i in range(1, len(pts)):
            p1, p2 = np.array(pts[i-1]), np.array(pts[i])
            vec = p2 - p1
            leng = np.linalg.norm(vec)
            if leng < 1e-3:
                continue
            n_samp = max(2, int(leng/sampling_interval))
            norm = np.array([-vec[1], vec[0]])/leng
            for j in range(n_samp):
                t = j/(n_samp-1) if n_samp > 1 else 0.5
                pt = p1 + t*vec
                cx, cy = int(pt[0]), int(pt[1])
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                wid = self.measure_width_at_point(binary_image, cx, cy, norm[0], norm[1], 30)
                if wid > 0:
                    widths.append(wid)
        avg_w = np.mean(widths) if widths else self.buffer_size
        avg_w = np.clip(avg_w, 1., 40.)
        return widths, avg_w

    def measure_width_at_point(self, binary_image, x, y, normal_x, normal_y, max_dist):
        h, w = binary_image.shape[:2]
        if not(0 <= x < w and 0 <= y < h):
            return 0
        dp, dn = 0, 0
        for i in range(max_dist):
            px, py = int(round(x+i*normal_x)), int(round(y+i*normal_y))
            if not(0 <= px < w and 0 <= py < h and binary_image[py, px] == 255):
                dp = i
                break
        else:
            dp = max_dist
        for i in range(1, max_dist+1):
            px, py = int(round(x-i*normal_x)), int(round(y-i*normal_y))
            if not(0 <= px < w and 0 <= py < h and binary_image[py, px] == 255):
                dn = i
                break
        else:
            dn = max_dist
        return dp + dn

    def create_adaptive_polygon(self, polyline_coords, measured_widths, min_width=2.0, max_width=30.0):
        pts = []
        L = len(polyline_coords)
        i = 0
        while i + 1 < L:
            pts.append((polyline_coords[i], polyline_coords[i+1]))
            i += 2
        if len(pts) < 2:
            return LineString(pts).buffer(self.buffer_size/2., cap_style=2) if len(pts) > 0 else None
        line = LineString(pts)
        if not measured_widths:
            buf = max(0.1, self.buffer_size/2., min_width/2.)
            return line.buffer(buf, cap_style=2, join_style=2)
        filt = np.clip(measured_widths, min_width, max_width)
        smth = np.copy(filt)
        if len(filt) >= 5:
            win = min(len(filt), 11)
            if win % 2 == 0:
                win -= 1
            if win >= 3:
                order = min(2, win-1)
                try:
                    s_cand = savgol_filter(filt, win, order)
                    smth = np.maximum(s_cand, min_width/2.)
                except Exception as e:
                    logger.warning(f"폭 스무딩 오류: {e}")
        avg_buf = np.mean(smth)/2. if len(smth) > 0 else min_width/2.
        avg_buf = max(0.1, avg_buf)
        dil = line.buffer(avg_buf, cap_style=2, join_style=2)
        if not dil.is_valid:
            dil = dil.buffer(0)
            if not dil.is_valid:
                return line.buffer(self.buffer_size/2., cap_style=2)
        return dil

    def print_stats(self):
        print("\n=== 변환 통계 ===")
        print(f"스캔된 JSON 파일: {self.stats['scanned_json_files'].get()}")
        print(f"JSON 내 총 고유 이미지 수: {self.stats['total_images_in_json'].get()}")
        print(f"  - 도로홀 포함 이미지: {self.stats['total_pothole_images_in_json'].get()}")
        print(f"  - 도로홀 미포함 이미지: {self.stats['total_other_images_in_json'].get()}")
        print(f"샘플링 설정:")
        print(f"  - 전체 샘플링 비율 목표: {self.sample_ratio*100:.1f}%")
        if self.target_pothole_ratio_in_sample is not None:
            print(f"  - 샘플 내 목표 포트홀 비율: {self.target_pothole_ratio_in_sample*100:.1f}%")
        else:
            print("  - 샘플 내 포트홀 비율 조정: 사용 안 함")
        print(f"샘플링 결과:")
        final_sampled = self.stats['final_sampled_images'].get()
        final_pothole = self.stats['final_sampled_pothole'].get()
        final_other = self.stats['final_sampled_other'].get()
        actual_ratio = final_pothole / final_sampled * 100 if final_sampled > 0 else 0
        print(f"  - 최종 샘플링된 이미지 수: {final_sampled} (목표: {self.stats['target_sample_size'].get()})")
        print(f"  - 최종 샘플 내 포트홀 포함: {final_pothole}개 ({actual_ratio:.1f}%)")
        print(f"  - 최종 샘플 내 포트홀 미포함: {final_other}개")
        print(f"최종 처리 성공 이미지 (유효 어노테이션 포함): {self.stats['processed_images'].get()}")
        print(f"처리된 어노테이션:")
        print(f"  - 도로균열 (cat_id=1): {self.stats['processed_annotations'][1].get()}")
        print(f"  - 도로홀 (cat_id=2): {self.stats['processed_annotations'][2].get()}")
        if self.use_sam_for_potholes:
            print(f"    - SAM으로 처리된 도로홀: {self.stats.get('sam_processed_potholes', ThreadSafeCounter(0)).get()}")
            print(f"    - SAM 처리 실패 도로홀: {self.stats.get('sam_errors_pothole', ThreadSafeCounter(0)).get()}")
        print(f"건너뛴 이미지:")
        print(f"  - 어노테이션 없음: {self.stats['skipped_images_no_annotation'].get()}")
        print(f"  - 이미지 파일 못 찾음: {self.stats['skipped_images_not_found'].get()}")
        print(f"  - 이미지 이동 실패: {self.stats['skipped_images_move_fail'].get()}")
        print(f"발생한 오류:")
        print(f"  - JSON 스캔 오류: {self.stats['errors_json_scan'].get()}")
        if 'errors_missing_imagedir' in self.stats:
             print(f"  - 이미지 디렉토리 누락 오류: {self.stats['errors_missing_imagedir'].get()}")
        print(f"  - 이미지 처리 오류: {self.stats['errors_image_processing'].get()}")
        print(f"  - 어노테이션 처리 오류: {self.stats['errors_annotation_processing'].get()}")
        print(f"  - 파일 이동 오류 (로깅됨): {self.stats['errors_file_move'].get()}")

        total_processed_final = self.stats['processed_images'].get()
        print(f"데이터셋 분할 (총 {total_processed_final}개 최종 처리 이미지 기준):")
        for split in self.splits:
            count = self.stats['split_counts'][split].get()
            ratio = count/total_processed_final*100 if total_processed_final > 0 else 0
            print(f"  - {split.capitalize()}: {count}개 ({ratio:.1f}%)")
        print(f"출력 폴더: {self.output_path}")

    def create_yolo_dataset_yaml(self, class_names=['도로균열', '도로홀']):
        yaml_content = f"""# YOLOv8 Segmentation Dataset (v6.1)
path: {self.output_path.resolve()}
train: images/train
val: images/val
test: images/test
names:\n"""
        for i, name in enumerate(class_names):
            yaml_content += f"  {i}: {name}\n"
        yaml_path = self.output_path / 'dataset.yaml'
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            logger.info(f"dataset.yaml 생성 완료: {yaml_path}")
        except IOError as e:
            logger.error(f"dataset.yaml 생성 실패: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='도로균열 및 도로홀 데이터를 YOLO-SEG 형식으로 변환 (v6.1 - 안정성 개선)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""사용 예시:
    python %(prog)s --input ./json --output ./yolo_output --image_dir ./images
    python %(prog)s --input ./json --output ./yolo_sampled --image_dir ./img \\
        --sample_ratio 0.2 --target_pothole_ratio_in_sample 0.4 --random_seed 42 \\
        --adaptive --workers 8 \\
        --use_sam_for_potholes --sam_checkpoint_path ./sam.pth --sam_device cuda
        """
    )
    parser.add_argument('--input', required=True, help='JSON 폴더')
    parser.add_argument('--output', required=True, help='출력 폴더')
    parser.add_argument('--image_dir', help='원본 이미지 폴더')
    parser.add_argument('--buffer', type=float, default=10., help='고정 균열 버퍼')
    parser.add_argument('--adaptive', action='store_true', help='적응형 균열 폭')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 로깅')
    parser.add_argument('--split_ratio', type=str, default='0.8,0.1,0.1', help='Train/Val/Test 분할 비율')
    parser.add_argument('--workers', type=int, default=1, help='병렬 워커 수')
    sampling_group = parser.add_argument_group('Target Ratio Sampling')
    sampling_group.add_argument('--sample_ratio', type=float, default=1.0, help='사용할 전체 이미지 비율 (0.0 ~ 1.0)')
    sampling_group.add_argument('--target_pothole_ratio_in_sample', type=float, default=None, help='샘플 내 목표 포트홀 비율 (0.0 ~ 1.0)')
    sampling_group.add_argument('--random_seed', type=int, default=None, help='샘플링 랜덤 시드')
    sam_group = parser.add_argument_group('SAM Parameters')
    sam_group.add_argument('--use_sam_for_potholes', action='store_true', help='도로홀에 SAM 사용')
    sam_group.add_argument('--sam_checkpoint_path', type=str, default=None, help='SAM 체크포인트 경로')
    sam_group.add_argument('--sam_model_type', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'], help='SAM 모델 타입')
    sam_group.add_argument('--sam_device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='SAM 실행 장치')
    sam_group.add_argument('--sam_polygon_epsilon_factor', type=float, default=0.005, help='SAM 폴리곤 근사 계수')
    sam_group.add_argument('--sam_min_contour_area', type=int, default=10, help='SAM 폴리곤 최소 면적')
    
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
       logging.getLogger().setLevel(logging.INFO)
    if not Path(args.input).is_dir():
       logger.error(f"입력 폴더 없음: {args.input}")
       return 1
    img_dir_needed = args.adaptive or args.use_sam_for_potholes or (args.image_dir is not None)
    if img_dir_needed and not args.image_dir:
       logger.error("--image_dir 필요")
       return 1
    if args.image_dir and not Path(args.image_dir).is_dir():
       logger.error(f"이미지 폴더 없음: {args.image_dir}")
       return 1
    if args.use_sam_for_potholes:
       if not SAM_AVAILABLE:
           logger.error("'segment_anything' 없음")
           return 1
       if not args.sam_checkpoint_path or not Path(args.sam_checkpoint_path).exists():
           logger.error(f"SAM 체크포인트 없음: {args.sam_checkpoint_path}")
           return 1
    if not (0.0 <= args.sample_ratio <= 1.0):
       logger.error(f"--sample_ratio 범위 오류")
       return 1
    if args.target_pothole_ratio_in_sample is not None and not (0.0 <= args.target_pothole_ratio_in_sample <= 1.0):
       logger.error(f"--target_pothole_ratio_in_sample 범위 오류")
       return 1
    if args.sample_ratio >= 1.0 and args.target_pothole_ratio_in_sample is not None:
       logger.warning("--target_pothole_ratio_in_sample은 --sample_ratio < 1.0 일 때만 유효.")

    try:
       splits = [float(x.strip()) for x in args.split_ratio.split(',')]
       assert len(splits) == 3 and np.isclose(sum(splits), 1.)
    except:
       logger.error(f"분할 비율 오류: {args.split_ratio}")
       splits = [0.7, 0.2, 0.1]

    workers = args.workers if args.workers > 0 else 1

    start_time = time.time()
    converter = None
    try:
        converter = Road2YOLOSeg(
            input_path=args.input, output_path=args.output, image_dir=args.image_dir,
            buffer_size=args.buffer, adaptive=args.adaptive, split_ratio=splits, num_workers=workers,
            use_sam_for_potholes=args.use_sam_for_potholes, sam_checkpoint_path=args.sam_checkpoint_path,
            sam_model_type=args.sam_model_type, sam_device=args.sam_device,
            sam_polygon_epsilon_factor=args.sam_polygon_epsilon_factor, sam_min_contour_area=args.sam_min_contour_area,
            sample_ratio=args.sample_ratio, target_pothole_ratio_in_sample=args.target_pothole_ratio_in_sample,
            random_seed=args.random_seed
        )
        converter.process_directory()
        logger.info(f"변환 완료!")
        return 0
    except KeyboardInterrupt:
       logger.warning("사용자에 의해 프로세스가 중단되었습니다.")
       return 130
    except Exception as e:
       logger.error(f"처리 중 예기치 않은 오류 발생: {e}", exc_info=True)
       return 1
    finally:
        if converter is not None and hasattr(converter, '_cleanup'):
           logger.info("임시 파일 정리 시도...")
           converter._cleanup()
        end_time = time.time()
        logger.info(f"총 실행 시간: {end_time - start_time:.2f} 초")
        if 'converter' in locals() and converter is not None:
            logger.info("최종 통계 요약:")
            converter.print_stats()
            converter.create_yolo_dataset_yaml()

if __name__ == '__main__':
   import sys
   exit_code = main()
   sys.exit(exit_code)