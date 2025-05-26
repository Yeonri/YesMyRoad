import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:geolocator/geolocator.dart';
import 'package:path_provider/path_provider.dart';
import 'package:intl/intl.dart';
import 'package:flutter/services.dart';
import '../main.dart';
import '../services/speed_based_detection_service.dart';
import '../services/local_storage_service.dart';
import '../utils/screen_utils.dart';
import '../utils/image_converter.dart';
import '../models/local_detection_record.dart';

class RecordingDetectionScreen extends StatefulWidget {
  const RecordingDetectionScreen({Key? key}) : super(key: key);

  @override
  _RecordingDetectionScreenState createState() => _RecordingDetectionScreenState();
}

class _RecordingDetectionScreenState extends State<RecordingDetectionScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  SpeedBasedDetectionService? _detectionService;
  StreamSubscription? _detectionSubscription;
  final LocalStorageService _storageService = LocalStorageService();

  bool _isRecording = false;
  double _currentSpeed = 0.0;
  int _frameSkip = 30;
  int _savedCount = 0;
  int _totalFramesProcessed = 0;
  int _framesPerSecond = 0;
  int _lastSecondFrames = 0;
  Timer? _fpsTimer;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    _initializeCamera();

    _fpsTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        _framesPerSecond = _lastSecondFrames;
        _lastSecondFrames = 0;
      });
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _updateOrientation();
  }

  void _updateOrientation() {
    if (mounted) {
      try {
        final orientation = MediaQuery.of(context).orientation;
        if (orientation == Orientation.portrait) {
          ImageConverter.setOrientation(DeviceOrientation.portraitUp);
        } else {
          ImageConverter.setOrientation(DeviceOrientation.landscapeLeft);
        }
        debugPrint('디바이스 방향 업데이트: ${ImageConverter.currentOrientation}');
      } catch (e) {
        debugPrint('방향 설정 오류: $e - 기본 방향 사용');
      }
    }
  }

  @override
  void didChangeMetrics() {
    _updateOrientation();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _stopRecording();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
      _updateOrientation();
    }
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      debugPrint('사용 가능한 카메라가 없습니다');
      return;
    }

    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.high,
      enableAudio: false,
    );

    try {
      await _cameraController!.initialize();
      await _cameraController!.setFlashMode(FlashMode.off);
      await _cameraController!.setFocusMode(FocusMode.auto);
      await _cameraController!.setExposureMode(ExposureMode.auto);

      if (_cameraController!.value.isInitialized) {
        ScreenUtils.setPreviewSize(_cameraController!.value.previewSize!);
      }

      setState(() {});
    } catch (e) {
      debugPrint('카메라 초기화 오류: $e');
    }
  }

  Future<void> _startRecording() async {
    if (_isRecording) return;

    setState(() {
      _isRecording = true;
      _savedCount = 0;
      _totalFramesProcessed = 0;
    });

    _detectionService = await SpeedBasedDetectionService.initialize();

    _detectionSubscription = _detectionService!.resultsStream.stream.listen(_handleDetectionResults);

    if (_cameraController != null && _cameraController!.value.isInitialized) {
      await _cameraController!.startImageStream(_processFrame);
    }
  }

  void _processFrame(CameraImage cameraImage) {
    _lastSecondFrames++;
    _totalFramesProcessed++;

    if (_detectionService != null && _isRecording) {
      _detectionService!.processFrame(cameraImage);
    }
  }

  void _handleDetectionResults(Map<String, dynamic> results) async {
    setState(() {
      _currentSpeed = results['speed'] ?? 0.0;
      _frameSkip = results['frameSkip'] ?? 30;
    });

    if (results.containsKey('jpegImage')) {
      Uint8List imageData = results['jpegImage'];
      await _saveRecordToLocal(imageData, results);
    }
  }

  Future<void> _saveRecordToLocal(Uint8List imageData, Map<String, dynamic> results) async {
    try {
      bool isLandscape = results['isLandscape'] ?? false;
      String directionInfo = isLandscape ? "landscape" : "portrait";

      String timestamp = DateFormat('yyyyMMdd_HHmmss_SSS').format(DateTime.now());
      String directory = (await getApplicationDocumentsDirectory()).path;
      String fileName = '$directory/road_record_${directionInfo}_$timestamp.jpg';

      File imageFile = await ImageConverter.saveImageWithCorrectOrientation(
          imageData,
          customFileName: fileName
      );

      Position position;
      try {
        position = await Geolocator.getCurrentPosition(
          desiredAccuracy: LocationAccuracy.high,
          timeLimit: const Duration(seconds: 2),
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

      List<String> detectedClasses = results['cls'] ?? [];
      String defectType = _determineDefectType(detectedClasses);

      LocalDetectionRecord record = LocalDetectionRecord(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        imageFile: imageFile,
        position: position,
        timestamp: DateTime.now(),
        defectType: defectType,
        speed: _currentSpeed,
        detectedObjects: detectedClasses,
        boundingBoxes: results['box'] ?? [],
        confidenceScores: results['conf'] ?? [],
      );

      bool saved = await _storageService.saveRecord(record);

      if (saved) {
        setState(() {
          _savedCount++;
        });
        debugPrint('기록 저장 성공: ${record.defectType} at ${record.speed.toStringAsFixed(1)} km/h, 파일: ${imageFile.path}');
        debugPrint('이미지 크기: ${await imageFile.length()} 바이트, 방향: $directionInfo');
      } else {
        debugPrint('기록 저장 실패');
      }
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

  void _stopRecording() {
    if (!_isRecording) return;

    _cameraController?.stopImageStream();
    _detectionService?.stop();
    _detectionSubscription?.cancel();

    setState(() {
      _isRecording = false;
      _currentSpeed = 0.0;
    });
  }

  void _toggleRecording() {
    if (_isRecording) {
      _stopRecording();
    } else {
      _startRecording();
    }
  }

  @override
  void dispose() {
    _stopRecording();
    _fpsTimer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
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
                _stopRecording();
                Navigator.pop(context);
              },
            ),
          ),

          if (_isRecording)
            Positioned(
              top: MediaQuery.of(context).padding.top + 10,
              right: 20,
              child: Container(
                width: 12,
                height: 12,
                decoration: const BoxDecoration(
                  color: Colors.red,
                  shape: BoxShape.circle,
                ),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 500),
                  curve: Curves.easeInOut,
                ),
              ),
            ),

          Positioned(
            bottom: 100,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _buildStatusItem('속도', '${_currentSpeed.toStringAsFixed(1)} km/h'),
                      _buildStatusItem('저장 수', '$_savedCount'),
                      _buildStatusItem('FPS', '$_framesPerSecond'),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      _buildStatusItem('프레임 스킵', '${_frameSkip + 1}'),
                      _buildStatusItem('총 프레임', '$_totalFramesProcessed'),
                    ],
                  ),
                ],
              ),
            ),
          ),

          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _toggleRecording,
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    color: _isRecording ? Colors.red : Colors.white,
                    shape: BoxShape.circle,
                    border: Border.all(
                      color: Colors.white,
                      width: 3,
                    ),
                  ),
                  child: Icon(
                    _isRecording ? Icons.stop : Icons.fiber_manual_record,
                    color: _isRecording ? Colors.white : Colors.red,
                    size: 40,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusItem(String label, String value) {
    return Column(
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
}