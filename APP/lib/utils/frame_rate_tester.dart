import 'dart:async';
import 'package:flutter/foundation.dart';

class FrameRateTester {
  int _frameCount = 0;
  int _fps = 0;
  bool _isTestingFrameRate = false;
  Timer? _timer;

  FrameRateTester() {
    startTesting();
  }

  void startTesting() {
    if (_isTestingFrameRate) return;

    _isTestingFrameRate = true;
    _frameCount = 0;

    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      _fps = _frameCount;
      _frameCount = 0;
      debugPrint('현재 FPS: $_fps');
    });
  }

  void stopTesting() {
    _isTestingFrameRate = false;
    _timer?.cancel();
    _timer = null;
  }

  void countFrame() {
    if (_isTestingFrameRate) {
      _frameCount++;
    }
  }

  int get currentFps => _fps;
}