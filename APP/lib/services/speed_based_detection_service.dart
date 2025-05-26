import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:geolocator/geolocator.dart';
import '../utils/image_converter.dart';
import '../utils/nms.dart';

enum _Command { init, busy, ready, detect, result }

class SpeedBasedDetectionService {
  static const String _modelPath = 'assets/models/bbox_model_float32.tflite';
  static const String _labelPath = 'assets/models/labels.txt';

  static Future<SpeedBasedDetectionService> initialize() async {
    final ReceivePort receivePort = ReceivePort();
    final Isolate isolate = await Isolate.spawn(_DetectorIsolate.run, receivePort.sendPort);

    final SpeedBasedDetectionService service = SpeedBasedDetectionService._(
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
      final interpreterOptions = InterpreterOptions()..threads = 2;
      interpreterOptions.addDelegate(XNNPackDelegate());

      debugPrint('모델 로드 시도: $_modelPath');
      final interpreter = await Interpreter.fromAsset(
        _modelPath,
        options: interpreterOptions,
      );
      debugPrint('모델 로드 성공: 입력 텐서 크기 ${interpreter.getInputTensors().first.shape}');
      return interpreter;
    } catch (e) {
      debugPrint('모델 로드 실패: $e');
      rethrow;
    }
  }

  static Future<List<String>> _loadLabels() async {
    return (await rootBundle.loadString(_labelPath)).split('\n');
  }

  SpeedBasedDetectionService._(this._isolate, this._interpreter, this._labels) {
    _startSpeedTracking();
  }

  final Isolate _isolate;
  final Interpreter _interpreter;
  final List<String> _labels;

  late final SendPort _sendPort;
  bool _isReady = false;

  Position? _previousPosition;
  double _estimatedSpeed = 0.0;
  int _dynamicFrameSkip = 30;
  StreamSubscription? _positionSubscription;
  StreamSubscription? _accelerometerSubscription;

  double _velocityEstimateAccel = 0.0;
  DateTime _lastAccelUpdate = DateTime.now();

  final StreamController<Map<String, dynamic>> resultsStream =
  StreamController<Map<String, dynamic>>();

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
        break;
      case _Command.busy:
        _isReady = false;
        break;
      case _Command.result:
        _isReady = true;
        resultsStream.add(args[0] as Map<String, dynamic>);
        break;
      default:
        debugPrint('Unknown command: $code');
    }
  }

  void _startSpeedTracking() {
    _positionSubscription = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 1,
      ),
    ).listen((Position position) {
      _updateSpeedFromGPS(position);
    });

    _accelerometerSubscription = accelerometerEvents.listen(
          (AccelerometerEvent event) {
        _updateSpeedFromAccelerometer(event);
      },
    );
  }

  void _updateSpeedFromGPS(Position currentPosition) {
    if (_previousPosition != null) {
      double distance = Geolocator.distanceBetween(
        _previousPosition!.latitude,
        _previousPosition!.longitude,
        currentPosition.latitude,
        currentPosition.longitude,
      );

      double timeDiff = currentPosition.timestamp!.difference(_previousPosition!.timestamp!).inMilliseconds / 1000.0;

      if (timeDiff > 0) {
        _estimatedSpeed = (distance / timeDiff) * 3.6;

        _estimatedSpeed = (_estimatedSpeed * 0.7) + (_velocityEstimateAccel * 0.3);

        _updateFrameSkipBasedOnSpeed();
      }
    }
    _previousPosition = currentPosition;
  }

  void _updateSpeedFromAccelerometer(AccelerometerEvent event) {
    DateTime now = DateTime.now();
    double dt = now.difference(_lastAccelUpdate).inMilliseconds / 1000.0;

    double magnitude = math.sqrt(event.x * event.x + event.y * event.y + event.z * event.z);
    double linearAccel = (magnitude - 9.81).abs();

    _velocityEstimateAccel += linearAccel * dt * 3.6;

    _velocityEstimateAccel *= 0.99;

    _lastAccelUpdate = now;
  }

  void _updateFrameSkipBasedOnSpeed() {
    if (_estimatedSpeed < 10) {
      _dynamicFrameSkip = 30;
    } else if (_estimatedSpeed < 30) {
      _dynamicFrameSkip = 15;
    } else if (_estimatedSpeed < 60) {
      _dynamicFrameSkip = 5;
    } else {
      _dynamicFrameSkip = 1;
    }

    debugPrint('속도: ${_estimatedSpeed.toStringAsFixed(1)} km/h, 프레임 스킵: ${_dynamicFrameSkip + 1}');
  }

  void processFrame(CameraImage cameraImage) {
    if (_isReady) {
      _sendPort.send({
        'code': _Command.detect,
        'args': [cameraImage, _estimatedSpeed, _dynamicFrameSkip],
      });
    }
  }

  void stop() {
    _positionSubscription?.cancel();
    _accelerometerSubscription?.cancel();
    _isolate.kill();
    resultsStream.close();
  }
}

class _DetectorIsolate {
  final SendPort _sendPort;
  int _modelInputSize = 224;
  Interpreter? _interpreter;
  List<String>? _labels;
  int _frameCounter = 0;

  _DetectorIsolate(this._sendPort);

  static void run(SendPort sendPort) {
    final ReceivePort receivePort = ReceivePort();
    final _DetectorIsolate isolate = _DetectorIsolate(sendPort);

    receivePort.listen((message) async {
      final Map<String, dynamic> command = message as Map<String, dynamic>;
      await isolate._handleCommand(command);
    });

    sendPort.send({
      'code': _Command.init,
      'args': [receivePort.sendPort],
    });
  }

  Future<void> _handleCommand(Map<String, dynamic> command) async {
    final _Command code = command['code'];
    final args = command['args'];

    switch (code) {
      case _Command.init:
        final rootIsolateToken = args[0] as RootIsolateToken;
        BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);

        _interpreter = Interpreter.fromAddress(args[1] as int);
        _modelInputSize = _interpreter!.getInputTensors().first.shape[1];
        _labels = args[2] as List<String>;

        _sendPort.send({'code': _Command.ready});
        break;
      case _Command.detect:
        final cameraImage = args[0] as CameraImage;
        final estimatedSpeed = args[1] as double;
        final frameSkip = args[2] as int;

        _sendPort.send({'code': _Command.busy});
        _processImage(cameraImage, estimatedSpeed, frameSkip);
        break;
      default:
        debugPrint('Unknown command in isolate: $code');
    }
  }

  void _processImage(CameraImage cameraImage, double speed, int frameSkip) async {
    final startTime = DateTime.now().millisecondsSinceEpoch;

    try {
      if (_frameCounter % (frameSkip + 1) != 0) {
        _frameCounter++;
        _sendPort.send({'code': _Command.ready});
        return;
      }

      final image = await ImageConverter.convertCameraImageToImage(cameraImage);

      if (image != null) {
        final results = _analyzeImage(image, startTime);

        results['speed'] = speed;
        results['frameSkip'] = frameSkip;

        Uint8List jpegImage = ImageConverter.encodeImageToJpeg(image);
        results['jpegImage'] = jpegImage;

        _sendPort.send({
          'code': _Command.result,
          'args': [results],
        });
      } else {
        _sendPort.send({'code': _Command.ready});
      }

      _frameCounter++;
      if (_frameCounter > 1000) _frameCounter = 0;
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

      final preprocessResult = ImageConverter.preprocessImageWithPadding(image, _modelInputSize);
      final processedImage = preprocessResult.tensor;

      final preprocessTime = DateTime.now().millisecondsSinceEpoch - preprocessStart;
      final inferenceStart = DateTime.now().millisecondsSinceEpoch;

      final output = _runInference(processedImage);
      final rawOutput = (output.first as List).first as List<List<double>>;

      final numLabels = _labels?.length ?? 0;
      final (idx, boxes, confidences) = NMS.apply(
          rawOutput,
          numLabels + 4,
          confidenceThreshold: 0.6,
          iouThreshold: 0.2
      );

      final convertedBoxes = _convertBoxesToOriginalCoordinates(
          boxes,
          preprocessResult.scale,
          preprocessResult.padX,
          preprocessResult.padY,
          image.width,
          image.height
      );

      List<String> classes = [];
      if (idx.isNotEmpty && _labels != null) {
        classes = idx.map((i) => _labels![i]).toList();
      }

      final inferenceTime = DateTime.now().millisecondsSinceEpoch - inferenceStart;
      final totalTime = DateTime.now().millisecondsSinceEpoch - startTime;

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
    final input = [imageMatrix];
    final outputShape = _interpreter!.getOutputTensors().first.shape;
    final outputs = List<double>.filled(
        outputShape.reduce((a, b) => a * b),
        0
    ).reshape(outputShape);

    final outputMap = <int, Object>{};
    outputMap[0] = outputs;

    _interpreter!.runForMultipleInputs([input], outputMap);
    return outputMap.values.toList();
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