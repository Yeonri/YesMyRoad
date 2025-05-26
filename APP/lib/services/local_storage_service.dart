import 'dart:io';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import '../models/local_detection_record.dart';

class LocalStorageService {
  static const String _recordsFileName = 'detection_records.json';

  Future<List<LocalDetectionRecord>> loadRecords() async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final file = File('${directory.path}/$_recordsFileName');

      if (!await file.exists()) {
        return [];
      }

      final jsonString = await file.readAsString();
      final List<dynamic> jsonList = jsonDecode(jsonString);

      return jsonList.map((json) => LocalDetectionRecord.fromJson(json)).toList();
    } catch (e) {
      debugPrint('기록 불러오기 오류: $e');
      return [];
    }
  }

  Future<bool> saveRecord(LocalDetectionRecord record) async {
    try {
      final records = await loadRecords();
      records.add(record);

      final directory = await getApplicationDocumentsDirectory();
      final file = File('${directory.path}/$_recordsFileName');

      final jsonString = jsonEncode(records.map((r) => r.toJson()).toList());
      await file.writeAsString(jsonString);

      return true;
    } catch (e) {
      debugPrint('기록 저장 오류: $e');
      return false;
    }
  }

  Future<bool> updateRecords(List<LocalDetectionRecord> records) async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final file = File('${directory.path}/$_recordsFileName');

      final jsonString = jsonEncode(records.map((r) => r.toJson()).toList());
      await file.writeAsString(jsonString);

      return true;
    } catch (e) {
      debugPrint('기록 업데이트 오류: $e');
      return false;
    }
  }

  Future<bool> deleteUploadedRecords() async {
    try {
      final records = await loadRecords();
      final unuploadedRecords = records.where((r) => !r.isUploaded).toList();

      for (var record in records.where((r) => r.isUploaded)) {
        if (await record.imageFile.exists()) {
          await record.imageFile.delete();
        }
      }

      return await updateRecords(unuploadedRecords);
    } catch (e) {
      debugPrint('업로드된 기록 삭제 오류: $e');
      return false;
    }
  }

  Future<bool> deleteRecord(String recordId) async {
    try {
      final records = await loadRecords();
      final record = records.firstWhere((r) => r.id == recordId);

      if (await record.imageFile.exists()) {
        await record.imageFile.delete();
      }

      records.removeWhere((r) => r.id == recordId);

      return await updateRecords(records);
    } catch (e) {
      debugPrint('기록 삭제 오류: $e');
      return false;
    }
  }
}