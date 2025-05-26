import 'dart:io';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

class ApiService {
  final String baseUrl = 'http://43.200.186.24/api';

  Future<bool> reportRoadDefect(
      double longitude,
      double latitude,
      double accuracy,
      DateTime timestamp,
      File imageFile,
      ) async {
    try {
      final uri = Uri.parse('$baseUrl/upload');

      var request = http.MultipartRequest('POST', uri);

      request.fields['longitude'] = longitude.toString();
      request.fields['latitude'] = latitude.toString();
      // request.fields['accuracy'] = accuracy.toString();
      // request.fields['timestamp'] = timestamp.toIso8601String();

      var imageStream = http.ByteStream(imageFile.openRead());
      var imageLength = await imageFile.length();

      var multipartFile = http.MultipartFile(
        'image',
        imageStream,
        imageLength,
        filename: imageFile.path.split('/').last,
        contentType: MediaType('image', 'jpeg'),
      );

      request.files.add(multipartFile);

      var response = await request.send();

      return response.statusCode >= 200 && response.statusCode < 300;
    } catch (e) {
      debugPrint('API 요청 오류: $e');
      return false;
    }
  }

  Future<bool> reportRoadDefectWithDetails(
      String defectType,
      double longitude,
      double latitude,
      double accuracy,
      DateTime timestamp,
      File imageFile,
      double speed,
      List<String> detectedObjects,
      ) async {
    try {
      final uri = Uri.parse('$baseUrl/upload');

      var request = http.MultipartRequest('POST', uri);

      // request.fields['defectType'] = defectType;
      request.fields['longitude'] = longitude.toString();
      request.fields['latitude'] = latitude.toString();
      // request.fields['accuracy'] = accuracy.toString();
      // request.fields['timestamp'] = timestamp.toIso8601String();

      // request.fields['speed'] = speed.toString();
      // request.fields['detectedObjects'] = jsonEncode(detectedObjects);
      // request.fields['mode'] = 'recording'; // 녹화 모드임을 명시

      var imageStream = http.ByteStream(imageFile.openRead());
      var imageLength = await imageFile.length();

      var multipartFile = http.MultipartFile(
        'image',
        imageStream,
        imageLength,
        filename: imageFile.path.split('/').last,
        contentType: MediaType('image', 'jpeg'),
      );

      request.files.add(multipartFile);

      request.headers['Content-Type'] = 'multipart/form-data';
      request.headers['Accept'] = 'application/json';

      var response = await request.send();

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final responseData = await response.stream.bytesToString();
        debugPrint('API 응답: $responseData');
        return true;
      } else {
        debugPrint('API 에러: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      debugPrint('API 요청 오류: $e');
      return false;
    }
  }
}