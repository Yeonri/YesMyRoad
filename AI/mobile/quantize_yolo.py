import sys
import argparse
from ultralytics import YOLO
import tensorflow as tf
from pathlib import Path

def quantize_yolo_to_tflite(pt_model_path, output_basename, img_size, quant_type='float16', data_yaml=None):
    try:
        print(f"원본 PyTorch 모델 로딩 중: {pt_model_path}")
        model = YOLO(pt_model_path)
        print("모델 로딩 완료.")

        output_path = None

        if quant_type == 'float16':
            print(f"\n--- Float16 TFLite 변환 시작 (imgsz={img_size}) ---")
            output_path = model.export(format='tflite', imgsz=img_size, half=True, simplify=True)
            print(f"--- Float16 TFLite 변환 완료 ---")
            print(f"저장된 파일: {output_path}")

        elif quant_type == 'int8':
            print(f"\n--- Int8 TFLite 변환 시작 (imgsz={img_size}) ---")
            if not data_yaml:
                print("오류: Int8 양자화를 위해서는 data.yaml 파일 경로가 필요합니다.")
                return None
            print(f"보정 데이터셋 사용: {data_yaml}")
            output_path = model.export(format='tflite', imgsz=img_size, int8=True, data=data_yaml, simplify=True)
            print(f"--- Int8 TFLite 변환 완료 ---")
            print(f"저장된 파일: {output_path}")

        else:
            print(f"오류: 지원하지 않는 양자화 유형입니다: {quant_type}. ('float16' 또는 'int8' 사용)")
            return None

        if output_path and tf.__version__:
            try:
                interpreter = tf.lite.Interpreter(model_path=str(output_path))
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print("\n--- 생성된 TFLite 모델 정보 ---")
                print("입력 텐서:")
                for detail in input_details:
                    print(f"  이름: {detail['name']}, 형태: {detail['shape']}, 타입: {detail['dtype']}")
                print("출력 텐서:")
                for detail in output_details:
                    print(f"  이름: {detail['name']}, 형태: {detail['shape']}, 타입: {detail['dtype']}")
                print("-----------------------------\n")
            except Exception as e:
                print(f"경고: 생성된 TFLite 모델 정보 확인 중 오류 발생: {e}")
                print("      (TensorFlow Lite 런타임과 모델 호환성 문제일 수 있습니다.)")

        return str(output_path)

    except FileNotFoundError:
        print(f"오류: 모델 파일 또는 데이터 파일을 찾을 수 없습니다.")
        print(f"  모델 경로 확인: {pt_model_path}")
        if quant_type == 'int8':
            print(f"  데이터 YAML 경로 확인: {data_yaml}")
        return None
    except Exception as e:
        print(f"TFLite 변환 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(
        description='YOLO 모델을 TensorFlow Lite로 변환하고 양자화',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
사용 예:
  python %(prog)s --input ./best.pt --output my_model --quantization float16
  python %(prog)s --input ./model.pt --output tflite_model --quantization int8 --data ./dataset.yaml
  python %(prog)s --input ./yolo11n.pt --img_size 320 --quantization int8 --data ./data.yaml
"""
    )

    parser.add_argument('--input', required=True, help='원본 PyTorch 모델 파일 경로 (.pt)')
    parser.add_argument('--output', required=True, help='출력 TFLite 모델 기본 이름 (확장자 제외)')
    parser.add_argument('--img_size', type=int, default=640, help='모델 입력 이미지 크기 (기본값: 640)')
    parser.add_argument('--quantization', choices=['float16', 'int8'], default='float16', 
                       help='양자화 유형 (기본값: float16)')
    parser.add_argument('--data', default=None, help='Int8 양자화용 데이터셋 YAML 파일 경로 (Int8 시 필수)')
    parser.add_argument('--verbose', action='store_true', help='상세 정보 출력')

    args = parser.parse_args()

    if args.verbose:
        print(f"=== 변환 설정 ===")
        print(f"입력 모델: {args.input}")
        print(f"출력 이름: {args.output}")
        print(f"이미지 크기: {args.img_size}x{args.img_size}")
        print(f"양자화 방식: {args.quantization}")
        if args.quantization == 'int8':
            print(f"데이터 YAML: {args.data}")
        print(f"==================\n")

    if not Path(args.input).exists():
        print(f"오류: 입력 모델 파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    if args.quantization == 'int8':
        if not args.data:
            print("오류: Int8 양자화를 위해서는 --data 인자로 데이터셋 YAML 파일을 지정해야 합니다.")
            sys.exit(1)
        if not Path(args.data).exists():
            print(f"오류: 데이터셋 YAML 파일을 찾을 수 없습니다: {args.data}")
            sys.exit(1)

    print(f"선택된 양자화 유형: {args.quantization}")
    print(f"모델 입력 크기: {args.img_size}x{args.img_size}")

    tflite_model_path = quantize_yolo_to_tflite(
        args.input,
        args.output,
        args.img_size,
        args.quantization,
        args.data if args.quantization == 'int8' else None
    )

    if tflite_model_path:
        print(f"\n성공적으로 TFLite 모델을 생성했습니다: {tflite_model_path}")
        
        if args.verbose:
            try:
                file_size = Path(tflite_model_path).stat().st_size
                print(f"변환된 모델 크기: {file_size / (1024*1024):.2f} MB")
            except:
                pass
                
        sys.exit(0)
    else:
        print("\nTFLite 모델 생성에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()