import 'dart:io';
import 'dart:convert';
import 'package:geolocator/geolocator.dart';

class LocalDetectionRecord {
  final String id;
  final File imageFile;
  final Position position;
  final DateTime timestamp;
  final String defectType;
  final double speed;
  final List<String> detectedObjects;
  final List<List<double>> boundingBoxes;
  final List<double> confidenceScores;
  bool isUploaded;
  bool isSelected;

  LocalDetectionRecord({
    required this.id,
    required this.imageFile,
    required this.position,
    required this.timestamp,
    required this.defectType,
    required this.speed,
    required this.detectedObjects,
    required this.boundingBoxes,
    required this.confidenceScores,
    this.isUploaded = false,
    this.isSelected = false,
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'imageFilePath': imageFile.path,
      'longitude': position.longitude,
      'latitude': position.latitude,
      'accuracy': position.accuracy,
      'timestamp': timestamp.toIso8601String(),
      'defectType': defectType,
      'speed': speed,
      'detectedObjects': detectedObjects,
      'boundingBoxes': boundingBoxes,
      'confidenceScores': confidenceScores,
      'isUploaded': isUploaded,
    };
  }

  factory LocalDetectionRecord.fromJson(Map<String, dynamic> json) {
    return LocalDetectionRecord(
      id: json['id'],
      imageFile: File(json['imageFilePath']),
      position: Position(
        longitude: json['longitude'],
        latitude: json['latitude'],
        timestamp: DateTime.parse(json['timestamp']),
        accuracy: json['accuracy'],
        altitude: 0,
        altitudeAccuracy: 0,
        heading: 0,
        headingAccuracy: 0,
        speed: 0,
        speedAccuracy: 0,
        floor: 0,
        isMocked: false,
      ),
      timestamp: DateTime.parse(json['timestamp']),
      defectType: json['defectType'],
      speed: json['speed'],
      detectedObjects: List<String>.from(json['detectedObjects']),
      boundingBoxes: (json['boundingBoxes'] as List).map((e) => List<double>.from(e)).toList(),
      confidenceScores: List<double>.from(json['confidenceScores']),
      isUploaded: json['isUploaded'] ?? false,
    );
  }

  double get accuracy => position.accuracy;
}