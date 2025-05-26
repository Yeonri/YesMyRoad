import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:geolocator/geolocator.dart';
import 'package:path_provider/path_provider.dart';
import 'package:intl/intl.dart';
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';
import '../main.dart';
import '../services/api_service.dart';
import '../services/detection_service.dart';
import '../utils/screen_utils.dart';
import '../utils/frame_rate_tester.dart';
import '../utils/image_converter.dart';
import '../models/detection_result.dart';

class DetectionScreen extends StatefulWidget {
  const DetectionScreen({Key? key}) : super(key: key);

  @override
  _DetectionScreenState createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  DetectionService? _detectionService;
  StreamSubscription? _subscription;
  StreamSubscription? _positionSubscription;
  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;
  final ApiService _apiService = ApiService();
  final FrameRateTester _frameRateTester = FrameRateTester();

  int _detectionCount = 0;
  int _frameSkip = 30;
  int _frameCounter = 0;
  double _currentSpeed = 0.0;

  List<String> _detectedClasses = [];
  List<List<double>> _boundingBoxes = [];
  List<double> _confidenceScores = [];

  final StreamController<DetectionResult> _uploadQueue = StreamController<DetectionResult>();

  Timer? _monitorTimer;
  Map<String, dynamic> _queueStatus = {
    'queueSize': 0,
    'maxQueueSize': 12,
    'isProcessing': false,
    'lastProcessingTime': 0,
    'averageProcessingTime': 0.0,
    'maxProcessingTime': 0.0,
  };
  List<int> _queueSizeHistory = [];
  List<int> _processingTimeHistory = [];

  bool _showQueueMonitor = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    // 가속도계 리스너 추가
    _accelerometerSubscription = accelerometerEvents.listen((AccelerometerEvent event) {
      _updateOrientationFromSensor(event);
    });

    _initializeSystem();
    _uploadQueue.stream.listen(_uploadDetection);

    _monitorTimer = Timer.periodic(const Duration(milliseconds: 200), (timer) {
      if (_detectionService != null) {
        final status = _detectionService!.getQueueStatus();
        setState(() {
          _queueStatus = status;

          _queueSizeHistory.add(status['queueSize'] as int);
          if (_queueSizeHistory.length > 50) {
            _queueSizeHistory.removeAt(0);
          }

          if (status['lastProcessingTime'] > 0) {
            _processingTimeHistory.add(status['lastProcessingTime'] as int);
            if (_processingTimeHistory.length > 50) {
              _processingTimeHistory.removeAt(0);
            }
          }
        });
      }
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // 여기서 방향 업데이트 호출
    _updateOrientation();
  }

  // 센서 이벤트로부터 방향 감지 및 업데이트
  void _updateOrientationFromSensor(AccelerometerEvent event) {
    // 가속도계 기반 방향 감지 (가로/세로 및 좌/우)
    final double x = event.x;
    final double y = event.y;

    // 값이 너무 작으면 무시 (노이즈 제거)
    if (x.abs() < 1.0 && y.abs() < 1.0) return;

    DeviceOrientation newOrientation;

    if (x.abs() > y.abs()) {
      newOrientation = x > 0
          ? DeviceOrientation.landscapeRight
          : DeviceOrientation.landscapeLeft;
    } else {
      newOrientation = y > 0
          ? DeviceOrientation.portraitUp
          : DeviceOrientation.portraitDown;
    }

    // 이전 방향과 다른 경우에만 업데이트
    if (newOrientation != ImageConverter.currentOrientation) {
      setState(() {
        ImageConverter.setOrientation(newOrientation);
        debugPrint('센서 기반 방향 업데이트: ${ImageConverter.currentOrientation}');
      });
    }
  }

  void _updateOrientation() {
    if (mounted) {
      try {
        final orientation = MediaQuery.of(context).orientation;

        // 기본 방향 설정
        DeviceOrientation defaultOrientation;
        if (orientation == Orientation.portrait) {
          defaultOrientation = DeviceOrientation.portraitUp;
        } else {
          // 가로 모드에서는 가속도계 데이터로 구분 필요
          // 기본값으로 landscapeLeft 설정
          defaultOrientation = DeviceOrientation.landscapeLeft;
        }

        // 가속도계 데이터로 더 정확한 방향 감지
        ImageConverter.getAccurateDeviceOrientation().then((detectedOrientation) {
          setState(() {
            ImageConverter.setOrientation(detectedOrientation);
            debugPrint('디바이스 방향 업데이트: ${ImageConverter.currentOrientation}');
          });
        }).catchError((_) {
          // 오류 시 MediaQuery 기반 방향 사용
          ImageConverter.setOrientation(defaultOrientation);
        });
      } catch (e) {
        // MediaQuery가 아직 준비되지 않은 경우
        debugPrint('방향 설정 오류: $e - 기본 방향 사용');
      }
    }
  }

  @override
  void didChangeMetrics() {
    super.didChangeMetrics();
    // 화면 크기나 방향이 변경될 때 호출됨
    _updateOrientation();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _stopDetection();
    } else if (state == AppLifecycleState.resumed) {
      _initializeSystem();
      // 앱이 다시 시작될 때 방향 업데이트
      _updateOrientation();
    }
  }

  void _initializeSystem() async {
    await _initializeCamera();
    _startSpeedTracking();

    DetectionService.initialize().then((detector) {
      setState(() {
        _detectionService = detector;
        _subscription = detector.resultsStream.stream.listen(_handleDetectionResults);
      });
    });
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      debugPrint('사용 가능한 카메라가 없습니다');
      return;
    }

    // 카메라 선택 및 기본 방향 설정
    final CameraDescription camera = cameras[0];

    // 카메라 센서 방향 설정
    ImageConverter.setSensorOrientation(camera.sensorOrientation);
    debugPrint('카메라 센서 방향: ${camera.sensorOrientation}');

    try {
      _cameraController = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888,
      );

      await _cameraController!.initialize();
      await _cameraController!.setFlashMode(FlashMode.off);
      await _cameraController!.setFocusMode(FocusMode.auto);
      await _cameraController!.setExposureMode(ExposureMode.auto);

      if (_cameraController!.value.isInitialized) {
        ScreenUtils.setPreviewSize(_cameraController!.value.previewSize!);
      }

      await _cameraController!.startImageStream(_processFrame);

      setState(() {});
    } catch (e) {
      debugPrint('카메라 초기화 오류: $e');
    }
  }

  void _startSpeedTracking() {
    _positionSubscription = Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 1,
      ),
    ).listen((Position position) {
      if (position.speed > 0) {
        setState(() {
          _currentSpeed = position.speed * 3.6;
        });
        _updateFrameSkipBasedOnSpeed();
      }
    });
  }

  void _updateFrameSkipBasedOnSpeed() {
    int newFrameSkip;

    if (_currentSpeed < 20) {
      newFrameSkip = 15;
    } else if (_currentSpeed < 50) {
      newFrameSkip = 7;
    } else {
      newFrameSkip = 3;
    }

    if (_frameSkip != newFrameSkip) {
      setState(() {
        _frameSkip = newFrameSkip;
      });
      debugPrint('속도: ${_currentSpeed.toStringAsFixed(1)} km/h, 프레임 스킵: ${_frameSkip + 1}');
    }
  }

  void _processFrame(CameraImage cameraImage) {
    _frameRateTester.countFrame();

    if (_frameCounter % (_frameSkip + 1) == 0) {
      _detectionService?.processFrame(cameraImage);
    }

    _frameCounter++;
    if (_frameCounter > 1000) _frameCounter = 0;
  }

  void _handleDetectionResults(Map<String, dynamic> results) {
    setState(() {
      if (results.containsKey('jpegImage')) {
        Uint8List imageData = results['jpegImage'];
        // 이미지 방향 정보 가져오기 (있는 경우)
        bool isLandscape = results['isLandscape'] ?? false;
        _saveAndUploadDetection(imageData, isLandscape);
      }

      _detectedClasses = results['cls'] ?? [];
      _boundingBoxes = results['box'] ?? [];
      _confidenceScores = results['conf'] ?? [];
    });
  }

  Future<void> _saveAndUploadDetection(Uint8List imageData, bool isLandscape) async {
    try {
      // 방향 정보가 포함된 파일명 생성
      String timestamp = DateFormat('yyyyMMdd_HHmmss').format(DateTime.now());
      String directory = (await getApplicationDocumentsDirectory()).path;
      String directionInfo = isLandscape ? "landscape" : "portrait";
      String fileName = '$directory/road_defect_${directionInfo}_$timestamp.jpg';

      // 향상된 저장 메서드 호출
      File imageFile = await ImageConverter.saveImageWithCorrectOrientation(
          imageData,
          customFileName: fileName
      );

      // 위치정보 및 업로드 처리는 기존과 동일
      Position position;
      try {
        position = await Geolocator.getCurrentPosition(
          desiredAccuracy: LocationAccuracy.medium,
          timeLimit: const Duration(seconds: 5),
        );
      } catch (e) {
        debugPrint('위치 정보 획득 실패: $e');
        position = Position(
            longitude: 0,
            latitude: 0,
            timestamp: DateTime.now(),
            accuracy: -1,
            altitude: 0,
            altitudeAccuracy: 0,
            heading: 0,
            headingAccuracy: 0,
            speed: 0,
            speedAccuracy: 0,
            floor: 0,
            isMocked: false
        );
      }

      String defectType = _determineDefectType(_detectedClasses);
      _uploadQueue.add(DetectionResult(
        imageFile: imageFile,
        position: position,
        timestamp: DateTime.now(),
        defectType: defectType,
      ));

      setState(() {
        _detectionCount++;
      });
    } catch (e) {
      debugPrint('이미지 저장 및 처리 오류: $e');
    }
  }

  String _determineDefectType(List<String> classes) {
    if (classes.contains('pothole')) {
      return 'POTHOLE';
    } else if (classes.contains('crack')) {
      return 'CRACK';
    }
    return 'UNKNOWN';
  }

  Future<void> _uploadDetection(DetectionResult detection) async {
    try {
      bool success = await _apiService.reportRoadDefect(
        detection.position.longitude,
        detection.position.latitude,
        detection.position.accuracy,
        detection.timestamp,
        detection.imageFile,
      );

      if (success) {
        debugPrint('도로 손상 데이터 업로드 성공: ${detection.defectType}');
      } else {
        debugPrint('도로 손상 데이터 업로드 실패');
      }
    } catch (e) {
      debugPrint('업로드 오류: $e');
    }
  }

  void _stopDetection() {
    _cameraController?.stopImageStream();
    _detectionService?.stop();
    _subscription?.cancel();
    _positionSubscription?.cancel();
  }

  Widget _buildQueueVisualization() {
    Color queueColor;
    if (_queueStatus['queueSize'] > _queueStatus['maxQueueSize'] * 0.8) {
      queueColor = Colors.red;
    } else if (_queueStatus['queueSize'] > _queueStatus['maxQueueSize'] * 0.5) {
      queueColor = Colors.orange;
    } else {
      queueColor = Colors.green;
    }

    final processingColor = _queueStatus['isProcessing'] ? Colors.blue : Colors.grey;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '큐 모니터링',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 14),
              ),
              GestureDetector(
                onTap: () {
                  setState(() {
                    _showQueueMonitor = !_showQueueMonitor;
                  });
                },
                child: Icon(
                  _showQueueMonitor ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down,
                  color: Colors.white,
                  size: 20,
                ),
              ),
            ],
          ),

          if (_showQueueMonitor) ...[
            const SizedBox(height: 8),

            Text(
              '큐 상태: ${_queueStatus['queueSize']}/${_queueStatus['maxQueueSize']} ${_queueStatus['isProcessing'] ? '(처리 중)' : '(대기 중)'}',
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
            const SizedBox(height: 8),

            Container(
              width: double.infinity,
              height: 24,
              decoration: BoxDecoration(
                color: Colors.grey.shade800,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Row(
                children: [
                  Container(
                    width: MediaQuery.of(context).size.width * 0.7 *
                        (_queueStatus['queueSize'] / _queueStatus['maxQueueSize']),
                    decoration: BoxDecoration(
                      color: queueColor,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  if (_queueStatus['isProcessing'])
                    Container(
                      width: 4,
                      color: processingColor,
                    ),
                ],
              ),
            ),

            const SizedBox(height: 8),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '마지막: ${_queueStatus['lastProcessingTime']}ms',
                  style: const TextStyle(color: Colors.white, fontSize: 12),
                ),
                Text(
                  '평균: ${_queueStatus['averageProcessingTime'].toStringAsFixed(1)}ms',
                  style: const TextStyle(color: Colors.white, fontSize: 12),
                ),
                Text(
                  '최대: ${_queueStatus['maxProcessingTime'].toStringAsFixed(1)}ms',
                  style: const TextStyle(color: Colors.white, fontSize: 12),
                ),
              ],
            ),

            const SizedBox(height: 8),

            SizedBox(
              height: 50,
              child: _buildHistoryGraph(_queueSizeHistory, Colors.cyan, _queueStatus['maxQueueSize'].toDouble()),
            ),

            const SizedBox(height: 4),
            const Text(
              '큐 크기 변화',
              style: TextStyle(color: Colors.white70, fontSize: 10),
            ),

            const SizedBox(height: 8),

            SizedBox(
              height: 50,
              child: _buildHistoryGraph(_processingTimeHistory, Colors.amber, 1000),
            ),

            const SizedBox(height: 4),
            const Text(
              '처리 시간 변화 (ms)',
              style: TextStyle(color: Colors.white70, fontSize: 10),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildHistoryGraph(List<int> history, Color color, double maxValue) {
    if (history.isEmpty) {
      return Container();
    }

    return CustomPaint(
      size: Size.infinite,
      painter: _HistoryGraphPainter(
        history: history,
        color: color,
        maxValue: maxValue,
      ),
    );
  }

  Widget _buildStatusItem(String label, String value) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 12,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _monitorTimer?.cancel();
    _stopDetection();
    _accelerometerSubscription?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    _frameRateTester.stopTesting();
    _uploadQueue.close();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      extendBodyBehindAppBar: true,
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController!),

          Positioned(
            top: MediaQuery.of(context).padding.top + 10,
            left: 10,
            child: IconButton(
              icon: const Icon(
                Icons.arrow_back,
                color: Colors.white,
                size: 30,
              ),
              onPressed: () {
                Navigator.pop(context);
              },
            ),
          ),

          Positioned(
            top: MediaQuery.of(context).padding.top + 50,
            left: 20,
            right: 20,
            child: _buildQueueVisualization(),
          ),

          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _buildStatusItem('속도', '${_currentSpeed.toStringAsFixed(1)} km/h'),
                  _buildStatusItem('감지', '$_detectionCount'),
                  _buildStatusItem('프레임 스킵', '${_frameSkip + 1}'),
                  _buildStatusItem('FPS', '${_frameRateTester.currentFps}'),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _HistoryGraphPainter extends CustomPainter {
  final List<int> history;
  final Color color;
  final double maxValue;

  _HistoryGraphPainter({
    required this.history,
    required this.color,
    required this.maxValue,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    final fillPaint = Paint()
      ..color = color.withOpacity(0.2)
      ..style = PaintingStyle.fill;

    final path = Path();
    final fillPath = Path();

    if (history.isEmpty) return;

    final xStep = size.width / (history.length - 1);

    double startX = 0;
    double startY = size.height - (history.first / maxValue * size.height);
    path.moveTo(startX, startY);
    fillPath.moveTo(startX, size.height);
    fillPath.lineTo(startX, startY);

    for (int i = 1; i < history.length; i++) {
      final x = i * xStep;
      final y = size.height - (history[i] / maxValue * size.height);
      path.lineTo(x, y);
      fillPath.lineTo(x, y);
    }

    fillPath.lineTo(size.width, size.height);
    fillPath.close();

    canvas.drawPath(fillPath, fillPaint);
    canvas.drawPath(path, paint);

    final gridPaint = Paint()
      ..color = Colors.white30
      ..strokeWidth = 0.5;

    for (int i = 1; i <= 3; i++) {
      final y = size.height * (1 - i / 4);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }
  }

  @override
  bool shouldRepaint(_HistoryGraphPainter oldDelegate) {
    return oldDelegate.history != history;
  }
}