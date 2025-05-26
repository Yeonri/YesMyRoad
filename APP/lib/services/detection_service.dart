import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:collection';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../utils/image_converter.dart';
import '../utils/nms.dart';

enum _Command { init, busy, ready, detect, result }

class DetectionService {
  static const String _modelPath = 'assets/models/bbox_model_float32.tflite';
  static const String _labelPath = 'assets/models/labels.txt';

  static Future<DetectionService> initialize() async {
    final ReceivePort receivePort = ReceivePort();
    final Isolate isolate = await Isolate.spawn(_DetectorIsolate.run, receivePort.sendPort);

    final DetectionService service = DetectionService._(
      isolate,
      await _loadModel(),
      await _loadLabels(),
    );

    receivePort.listen((message) {
      final Map<String, dynamic> command = message as Map<String, dynamic>;
      service._handleCommand(command);
    });

    return service;
  }

  static Future<Interpreter> _loadModel() async {
    try {
      final interpreterOptions = InterpreterOptions();

      if (defaultTargetPlatform == TargetPlatform.android) {
        try {
          interpreterOptions.useNnApiForAndroid = true;
          debugPrint('NNAPI Delegate 활성화 시도 (useNnApiForAndroid = true 사용)');
        } catch (e) {
          debugPrint('useNnApiForAndroid 설정 중 예외 발생: $e');
          debugPrint('XNNPack Delegate로 fallback 시도');
          interpreterOptions.addDelegate(XNNPackDelegate());
        }
      } else {
        debugPrint('Android 플랫폼이 아니므로 XNNPack Delegate 사용');
        interpreterOptions.addDelegate(XNNPackDelegate());
      }

      interpreterOptions.threads = 4;

      debugPrint('모델 로드 시도: $_modelPath');
      final interpreter = await Interpreter.fromAsset(
        _modelPath,
        options: interpreterOptions,
      );

      debugPrint('모델 로드 성공: 입력 텐서 크기 ${interpreter.getInputTensors().first.shape}');
      return interpreter;

    } catch (e) {
      debugPrint('모델 로드 중 최종 오류 발생 (useNnApiForAndroid 사용 시도 포함): $e');
      debugPrint('Delegate 없이 CPU로 모델 재로드 시도...');
      try {
        final cpuOptions = InterpreterOptions()..threads = 2;

        final interpreter = await Interpreter.fromAsset(
          _modelPath,
          options: cpuOptions,
        );
        debugPrint('Delegate 없이 CPU로 모델 로드 성공');
        return interpreter;
      } catch (e2) {
        debugPrint('Delegate 없이 CPU로 모델 로드도 실패: $e2');
        rethrow;
      }
    }
  }

  static Future<List<String>> _loadLabels() async {
    return (await rootBundle.loadString(_labelPath)).split('\n');
  }

  DetectionService._(this._isolate, this._interpreter, this._labels) {
    _frameQueue = FrameQueue(this);
  }

  final Isolate _isolate;
  final Interpreter _interpreter;
  final List<String> _labels;
  late final FrameQueue _frameQueue;

  late final SendPort _sendPort;
  bool _isReady = false;

  final StreamController<Map<String, dynamic>> resultsStream =
  StreamController<Map<String, dynamic>>();

  Map<String, dynamic> getQueueStatus() {
    return {
      'queueSize': _frameQueue.queueLength,
      'maxQueueSize': _frameQueue.maxQueueSize,
      'isProcessing': _frameQueue.isProcessing,
      'lastProcessingTime': _frameQueue.lastProcessingTime,
      'averageProcessingTime': _frameQueue.averageProcessingTime,
      'maxProcessingTime': _frameQueue.maxProcessingTime,
    };
  }

  void _handleCommand(Map<String, dynamic> command) {
    final _Command code = command['code'];
    final args = command['args'];

    switch (code) {
      case _Command.init:
        _sendPort = args[0] as SendPort;
        final rootIsolateToken = RootIsolateToken.instance!;
        _sendPort.send({
          'code': _Command.init,
          'args': [rootIsolateToken, _interpreter.address, _labels],
        });
        break;
      case _Command.ready:
        _isReady = true;
        _frameQueue.setReady();
        break;
      case _Command.busy:
        _isReady = false;
        break;
      case _Command.result:
        _isReady = true;
        resultsStream.add(args[0] as Map<String, dynamic>);
        _frameQueue.setReady();
        break;
      default:
        debugPrint('Unknown command: $code');
    }
  }

  void processFrame(CameraImage cameraImage) {
    _frameQueue.addFrame(cameraImage);
  }

  void stop() {
    _isolate.kill();
    resultsStream.close();
  }
}

class FrameQueue {
  final int maxQueueSize = 12;
  final Queue<_FrameTask> _queue = Queue<_FrameTask>();
  bool _isProcessing = false;
  final DetectionService _service;

  int _lastProcessingTime = 0;
  final List<int> _processingTimes = [];
  DateTime? _processingStartTime;

  FrameQueue(this._service);

  void addFrame(CameraImage frame) {
    if (_queue.length >= maxQueueSize) {
      _queue.removeFirst();
    }

    _queue.add(_FrameTask(
      frame: frame,
      timestamp: DateTime.now(),
    ));

    if (_queue.length % 5 == 0) {
      debugPrint('프레임 큐 상태: ${_queue.length}/$maxQueueSize');
    }

    _processNextFrameIfReady();
  }

  void _processNextFrameIfReady() {
    if (_isProcessing || _queue.isEmpty || !_service._isReady) return;

    _isProcessing = true;
    final task = _queue.removeFirst();

    _processingStartTime = DateTime.now();

    _service._sendPort.send({
      'code': _Command.detect,
      'args': [task.frame],
    });
  }

  void setReady() {
    if (_isProcessing && _processingStartTime != null) {
      final now = DateTime.now();
      _lastProcessingTime = now.difference(_processingStartTime!).inMilliseconds;

      _processingTimes.add(_lastProcessingTime);
      if (_processingTimes.length > 10) {
        _processingTimes.removeAt(0);
      }
    }

    _isProcessing = false;
    _processNextFrameIfReady();
  }

  int get queueLength => _queue.length;
  bool get isProcessing => _isProcessing;
  int get lastProcessingTime => _lastProcessingTime;
  double get averageProcessingTime {
    if (_processingTimes.isEmpty) return 0;
    return _processingTimes.reduce((a, b) => a + b) / _processingTimes.length;
  }
  double get maxProcessingTime {
    if (_processingTimes.isEmpty) return 0;
    return _processingTimes.reduce(math.max).toDouble();
  }
}

class _FrameTask {
  final CameraImage frame;
  final DateTime timestamp;

  _FrameTask({required this.frame, required this.timestamp});
}

class _DetectorIsolate {
  final SendPort _sendPort;
  int _modelInputSize = 224;
  Interpreter? _interpreter;
  List<String>? _labels;

  _DetectorIsolate(this._sendPort);

  static void run(SendPort sendPort) {
    final ReceivePort receivePort = ReceivePort();
    final _DetectorIsolate isolate = _DetectorIsolate(sendPort);

    sendPort.send({
      'code': _Command.init,
      'args': [receivePort.sendPort],
    });

    receivePort.listen((message) async {
      if (message is Map<String, dynamic>) {
        final _Command code = message['code'];
        final args = message['args'];

        if (code == _Command.init) {
          final rootIsolateToken = args[0] as RootIsolateToken;
          BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);

          isolate._interpreter = Interpreter.fromAddress(args[1] as int);
          isolate._modelInputSize = isolate._interpreter!.getInputTensors().first.shape[1];
          isolate._labels = args[2] as List<String>;

          sendPort.send({'code': _Command.ready});
        } else {
          await isolate._handleCommand(message);
        }
      }
    });
  }

  Future<void> _handleCommand(Map<String, dynamic> command) async {
    final _Command code = command['code'];
    final args = command['args'];

    switch (code) {
      case _Command.detect:
        _sendPort.send({'code': _Command.busy});
        _processImage(args[0] as CameraImage);
        break;
      default:
        debugPrint('Unknown command in isolate: $code');
    }
  }

  void _processImage(CameraImage cameraImage) async {
    final startTime = DateTime.now().millisecondsSinceEpoch;

    try {
      final image = await ImageConverter.convertCameraImageToImage(cameraImage);

      if (image != null) {
        final results = _analyzeImage(image, startTime);

        if (results['conf'].any((double c) => c > 0.25)) {
          bool isLandscape = image.width > image.height;
          results['isLandscape'] = isLandscape;

          Uint8List jpegImage = ImageConverter.encodeImageToJpeg(image);
          results['jpegImage'] = jpegImage;
        }

        _sendPort.send({
          'code': _Command.result,
          'args': [results],
        });
      } else {
        _sendPort.send({'code': _Command.ready});
      }
    } catch (e, stackTrace) {
      debugPrint('이미지 처리 오류: $e');
      debugPrint('스택 트레이스: $stackTrace');
      _sendPort.send({'code': _Command.ready});
    }
  }

  Map<String, dynamic> _analyzeImage(dynamic image, int startTime) {
    try {
      final conversionTime = DateTime.now().millisecondsSinceEpoch - startTime;
      final preprocessStart = DateTime.now().millisecondsSinceEpoch;

      debugPrint('원본 이미지 정보: ${image.width} x ${image.height}, 포맷: ${image.format}');

      final preprocessResult = ImageConverter.preprocessImageWithPadding(image, _modelInputSize);
      final processedImage = preprocessResult.tensor;

      debugPrint('전처리된 이미지 차원: ${processedImage.length} x ${processedImage[0].length} x ${processedImage[0][0].length}');
      debugPrint('이미지 샘플 픽셀 값 (0,0): ${processedImage[0][0].toString()}');
      debugPrint('패딩 정보: scale=${preprocessResult.scale}, padX=${preprocessResult.padX}, padY=${preprocessResult.padY}');

      final lastY = processedImage.length - 1;
      final lastX = processedImage[0].length - 1;
      debugPrint('이미지 샘플 픽셀 값 ($lastX,$lastY): ${processedImage[lastY][lastX].toString()}');

      final preprocessTime = DateTime.now().millisecondsSinceEpoch - preprocessStart;
      final inferenceStart = DateTime.now().millisecondsSinceEpoch;

      final output = _runInference(processedImage);
      final rawOutput = (output.first as List).first as List<List<double>>;

      debugPrint('모델 출력 차원: ${rawOutput.length} x ${rawOutput[0].length}');
      debugPrint('예상 출력 차원: 6 x 8400');

      if (rawOutput.isNotEmpty && rawOutput[0].isNotEmpty) {
        debugPrint('출력 샘플 [0,0]: ${rawOutput[0][0]}');
        debugPrint('출력 샘플 [4,0]: ${rawOutput[4][0]}');
      }

      final numLabels = _labels?.length ?? 0;
      final (idx, boxes, confidences) = NMS.apply(
          rawOutput,
          numLabels + 4,
          confidenceThreshold: 0.6,
          iouThreshold: 0.2
      );

      debugPrint('감지된 객체 수: ${idx.length}');

      final convertedBoxes = _convertBoxesToOriginalCoordinates(
          boxes,
          preprocessResult.scale,
          preprocessResult.padX,
          preprocessResult.padY,
          image.width,
          image.height
      );

      if (convertedBoxes.isNotEmpty) {
        for (int i = 0; i < convertedBoxes.length; i++) {
          debugPrint('감지 #$i - 클래스: ${_labels?[idx[i]]}, 신뢰도: ${confidences[i]}, 박스: ${convertedBoxes[i]}');
        }
      } else {
        debugPrint('감지된 객체 없음');
      }

      List<String> classes = [];
      if (idx.isNotEmpty && _labels != null) {
        classes = idx.map((i) => _labels![i]).toList();
      }

      final inferenceTime = DateTime.now().millisecondsSinceEpoch - inferenceStart;
      final totalTime = DateTime.now().millisecondsSinceEpoch - startTime;

      debugPrint('추론 시간: $inferenceTime ms');
      debugPrint('전처리 시간: $preprocessTime ms');
      debugPrint('전체 처리 시간: $totalTime ms');

      return {
        'cls': classes,
        'box': convertedBoxes,
        'conf': confidences,
        'stats': {
          'Conversion time:': conversionTime.toString(),
          'Pre-processing time:': preprocessTime.toString(),
          'Inference time:': inferenceTime.toString(),
          'Total prediction time:': totalTime.toString(),
          'Frame': '${image.width} X ${image.height}',
        }
      };
    } catch (e, stackTrace) {
      debugPrint('분석 오류: $e');
      debugPrint('스택 트레이스: $stackTrace');
      return {
        'cls': <String>[],
        'box': <List<double>>[],
        'conf': <double>[],
        'error': e.toString(),
        'stats': {
          'Error': '이미지 분석 중 오류가 발생했습니다',
        }
      };
    }
  }

  List<List<double>> _convertBoxesToOriginalCoordinates(
      List<List<double>> boxes,
      double scale,
      double padX,
      double padY,
      int originalWidth,
      int originalHeight
      ) {
    final convertedBoxes = <List<double>>[];

    for (var box in boxes) {
      final cx = box[0];
      final cy = box[1];
      final w = box[2];
      final h = box[3];

      final cxPixel = cx * _modelInputSize;
      final cyPixel = cy * _modelInputSize;
      final wPixel = w * _modelInputSize;
      final hPixel = h * _modelInputSize;

      final originalCx = (cxPixel - padX) / scale;
      final originalCy = (cyPixel - padY) / scale;
      final originalW = wPixel / scale;
      final originalH = hPixel / scale;

      final constrainedCx = originalCx.clamp(0.0, originalWidth.toDouble());
      final constrainedCy = originalCy.clamp(0.0, originalHeight.toDouble());
      final constrainedW = originalW.clamp(0.0, originalWidth.toDouble());
      final constrainedH = originalH.clamp(0.0, originalHeight.toDouble());

      convertedBoxes.add([constrainedCx, constrainedCy, constrainedW, constrainedH]);
    }

    return convertedBoxes;
  }

  List<Object> _runInference(List<List<List<double>>> imageMatrix) {
    debugPrint('모델 입력 텐서 차원: [1, ${imageMatrix.length}, ${imageMatrix[0].length}, ${imageMatrix[0][0].length}]');

    final inputTensor = _interpreter!.getInputTensors().first;
    final outputTensor = _interpreter!.getOutputTensors().first;
    debugPrint('모델 입력 텐서 정보: 이름=${inputTensor.name}, 형태=${inputTensor.shape}, 타입=${inputTensor.type}');
    debugPrint('모델 출력 텐서 정보: 이름=${outputTensor.name}, 형태=${outputTensor.shape}, 타입=${outputTensor.type}');

    if (imageMatrix.length != inputTensor.shape[1] ||
        imageMatrix[0].length != inputTensor.shape[2] ||
        imageMatrix[0][0].length != inputTensor.shape[3]) {
      debugPrint('⚠️ 경고: 입력 이미지 차원이 모델 입력 차원과 일치하지 않습니다!');
      debugPrint('입력 이미지: [${imageMatrix.length}, ${imageMatrix[0].length}, ${imageMatrix[0][0].length}]');
      debugPrint('모델 기대 입력: [${inputTensor.shape[1]}, ${inputTensor.shape[2]}, ${inputTensor.shape[3]}]');
    }

    final input = [imageMatrix];

    final outputShape = outputTensor.shape;

    debugPrint('모델 출력 형태: ${outputShape.toString()}');

    final expectedOutputShape = [1, 6, 8400];
    final isOutputShapeMatching = outputShape.length == expectedOutputShape.length &&
        outputShape.asMap().entries.every((entry) =>
        entry.value == expectedOutputShape[entry.key] ||
            entry.value == -1);

    debugPrint('출력 차원 일치 여부: $isOutputShapeMatching');
    if (!isOutputShapeMatching) {
      debugPrint('⚠️ 경고: 모델 출력 차원이 예상과 다릅니다!');
      debugPrint('예상 출력 차원: $expectedOutputShape');
      debugPrint('실제 출력 차원: $outputShape');
    }

    final outputs = List<double>.filled(
        outputShape.reduce((a, b) => a * b),
        0
    ).reshape(outputShape);

    final outputMap = <int, Object>{};
    outputMap[0] = outputs;

    try {
      debugPrint('모델 추론 시작...');

      _interpreter!.runForMultipleInputs([input], outputMap);

      debugPrint('모델 추론 완료');

      final outList = outputMap[0] as List;
      if (outList.isNotEmpty) {
        final firstOut = outList[0] as List;
        if (firstOut.isNotEmpty) {
          debugPrint('출력 첫 번째 요소: ${firstOut[0]}');

          debugPrint('첫 번째 바운딩 박스 좌표: [${firstOut[0][0]}, ${firstOut[1][0]}, ${firstOut[2][0]}, ${firstOut[3][0]}]');

          final classScores = <double>[];
          for (int i = 4; i < firstOut.length && i < 10; i++) {
            classScores.add(firstOut[i][0]);
          }
          debugPrint('첫 번째 클래스 점수 (일부): $classScores');
        }
      }

      return outputMap.values.toList();
    } catch (e, stackTrace) {
      debugPrint('⚠️ 모델 추론 중 오류 발생: $e');
      debugPrint('스택 트레이스: $stackTrace');
      rethrow;
    }
  }
}

extension ListReshape on List<double> {
  List<dynamic> reshape(List<int> shape) {
    if (shape.isEmpty) return this;

    int total = shape.reduce((a, b) => a * b);
    if (total != this.length) {
      throw Exception('Cannot reshape a list of length ${this.length} into shape $shape with total elements $total');
    }

    if (shape.length == 1) return this;

    List<dynamic> result = [];
    int subListSize = this.length ~/ shape[0];

    for (int i = 0; i < shape[0]; i++) {
      final subList = this.sublist(i * subListSize, (i + 1) * subListSize);
      if (shape.length > 2) {
        final newShape = shape.sublist(1);
        result.add(subList.reshape(newShape));
      } else {
        result.add(subList);
      }
    }

    return result;
  }
}