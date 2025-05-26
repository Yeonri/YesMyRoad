import 'dart:io';
import 'package:geolocator/geolocator.dart';

class DetectionResult {
  final File imageFile;
  final Position position;
  final DateTime timestamp;
  final String defectType;

  DetectionResult({
    required this.imageFile,
    required this.position,
    required this.timestamp,
    required this.defectType,
  });

  double get accuracy => position.accuracy;

}