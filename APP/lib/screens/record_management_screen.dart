import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../services/local_storage_service.dart';
import '../services/api_service.dart';
import '../models/local_detection_record.dart';

class RecordManagementScreen extends StatefulWidget {
  const RecordManagementScreen({Key? key}) : super(key: key);

  @override
  _RecordManagementScreenState createState() => _RecordManagementScreenState();
}

class _RecordManagementScreenState extends State<RecordManagementScreen> {
  final LocalStorageService _storageService = LocalStorageService();
  final ApiService _apiService = ApiService();

  List<LocalDetectionRecord> _records = [];
  bool _isLoading = true;
  bool _isUploading = false;
  bool _isSelectMode = false;
  bool _selectAll = false;

  @override
  void initState() {
    super.initState();
    _loadRecords();
  }

  Future<void> _loadRecords() async {
    setState(() => _isLoading = true);
    try {
      final records = await _storageService.loadRecords();
      setState(() {
        _records = records;
        _isLoading = false;
      });
    } catch (e) {
      debugPrint('기록 불러오기 오류: $e');
      setState(() => _isLoading = false);
    }
  }

  void _toggleSelectMode() {
    setState(() {
      _isSelectMode = !_isSelectMode;
      if (!_isSelectMode) {
        for (var record in _records) {
          record.isSelected = false;
        }
        _selectAll = false;
      }
    });
  }

  void _toggleSelectAll() {
    setState(() {
      _selectAll = !_selectAll;
      for (var record in _records) {
        if (!record.isUploaded) {
          record.isSelected = _selectAll;
        }
      }
    });
  }

  void _toggleRecordSelection(LocalDetectionRecord record) {
    if (record.isUploaded) return;
    setState(() {
      record.isSelected = !record.isSelected;
      final unuploadedRecords = _records.where((r) => !r.isUploaded).toList();
      final selectedUnuploaded = unuploadedRecords.where((r) => r.isSelected).toList();
      _selectAll = unuploadedRecords.isNotEmpty && selectedUnuploaded.length == unuploadedRecords.length;
    });
  }

  Future<void> _uploadSelectedRecords() async {
    final selectedRecords = _records.where((r) => r.isSelected && !r.isUploaded).toList();
    if (selectedRecords.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('업로드할 기록을 선택해주세요.')),
      );
      return;
    }

    setState(() => _isUploading = true);
    int successCount = 0;
    int failCount = 0;

    for (var record in selectedRecords) {
      try {
        bool success = await _apiService.reportRoadDefectWithDetails(
          record.defectType,
          record.position.longitude,
          record.position.latitude,
          record.position.accuracy,
          record.timestamp,
          record.imageFile,
          record.speed,
          record.detectedObjects,
        );
        if (success) {
          record.isUploaded = true;
          record.isSelected = false;
          successCount++;
        } else {
          failCount++;
        }
      } catch (e) {
        failCount++;
        debugPrint('업로드 오류: $e');
      }
    }

    await _storageService.updateRecords(_records);

    setState(() {
      _isUploading = false;
      _selectAll = false;
      _isSelectMode = false;
    });

    String message = '업로드 완료: 성공 $successCount개';
    if (failCount > 0) {
      message += ', 실패 $failCount개';
    }
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  Future<void> _deleteUploadedRecords() async {
    final result = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('확인'),
        content: const Text('업로드된 모든 기록을 삭제하시겠습니까?\n(이미지 파일도 함께 삭제됩니다)'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('취소'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('삭제'),
          ),
        ],
      ),
    );

    if (result == true) {
      await _storageService.deleteUploadedRecords();
      await _loadRecords();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('업로드된 기록이 삭제되었습니다.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final unuploadedCount = _records.where((r) => !r.isUploaded).length;
    final uploadedCount = _records.where((r) => r.isUploaded).length;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text(
          _isSelectMode ? '${_records.where((r) => r.isSelected).length}개 선택됨' : '기록 관리',
          style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        iconTheme: const IconThemeData(color: Colors.white),
        actions: [
          if (_isSelectMode)
            Center(
              child: Row(
                children: [
                  const Text('전체 선택', style: TextStyle(color: Colors.white)),
                  Checkbox(
                    value: _selectAll,
                    onChanged: (_) => _toggleSelectAll(),
                    activeColor: Colors.white,
                    checkColor: Colors.blueAccent,
                  ),
                  const SizedBox(width: 16),
                ],
              ),
            ),
          IconButton(
            icon: Icon(_isSelectMode ? Icons.close : Icons.select_all),
            onPressed: _toggleSelectMode,
          ),
          IconButton(
            icon: const Icon(Icons.delete_sweep),
            onPressed: _deleteUploadedRecords,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: Stack(
        children: [
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  Color(0xFF1a1a2e),
                  Color(0xFF16213e),
                  Color(0xFF0f3460),
                ],
              ),
            ),
          ),

          Positioned(
            top: -100,
            right: -100,
            child: Container(
              width: 300,
              height: 300,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.blueAccent.withOpacity(0.1),
              ),
            ),
          ),

          Positioned(
            bottom: -150,
            left: -150,
            child: Container(
              width: 400,
              height: 400,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.redAccent.withOpacity(0.08),
              ),
            ),
          ),

          SafeArea(
            child: _records.isEmpty
                ? const Center(
              child: Text(
                '저장된 기록이 없습니다.',
                style: TextStyle(fontSize: 18, color: Colors.white70),
              ),
            )
                : GridView.builder(
              padding: const EdgeInsets.all(16),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
                childAspectRatio: 0.9,
              ),
              itemCount: _records.length,
              itemBuilder: (context, index) {
                final record = _records[index];
                return _buildRecordCard(record);
              },
            ),
          ),

          if (_isSelectMode)
            Positioned(
              bottom: 24,
              left: 24,
              right: 24,
              child: ElevatedButton(
                onPressed: _isUploading ? null : _uploadSelectedRecords,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blueAccent,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
                child: _isUploading
                    ? const SizedBox(
                  height: 24,
                  width: 24,
                  child: CircularProgressIndicator(color: Colors.white),
                )
                    : Text(
                  '선택한 ${selectedRecordCount()}개 기록 업로드',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  int selectedRecordCount() {
    return _records.where((r) => r.isSelected && !r.isUploaded).length;
  }

  Widget _buildRecordCard(LocalDetectionRecord record) {
    final formattedDate = DateFormat('yyyy-MM-dd HH:mm').format(record.timestamp);

    return GestureDetector(
      onTap: () {
        if (_isSelectMode && !record.isUploaded) {
          _toggleRecordSelection(record);
        }
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          color: Colors.white.withOpacity(0.1),
          border: Border.all(
            color: record.isSelected ? Colors.blueAccent.withOpacity(0.7) : Colors.transparent,
            width: 2,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 8,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Stack(
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(
                  child: ClipRRect(
                    borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
                    child: Image.file(
                      record.imageFile,
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stackTrace) {
                        return Container(
                          color: Colors.grey[800],
                          child: const Icon(Icons.error_outline, color: Colors.white),
                        );
                      },
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        formattedDate,
                        style: const TextStyle(fontSize: 10, color: Colors.white70),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            record.defectType,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                              color: Colors.white,
                            ),

                          ),
                          Text(
                            '${record.speed.toStringAsFixed(1)} km/h',
                            style: const TextStyle(fontSize: 10, color: Colors.white70),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),

            if (record.isUploaded)
              Positioned(
                top: 8,
                right: 8,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.green.withOpacity(0.8),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.cloud_done, color: Colors.white, size: 14),
                      SizedBox(width: 4),
                      Text('완료', style: TextStyle(color: Colors.white, fontSize: 12)),
                    ],
                  ),
                ),
              ),

            if (_isSelectMode && !record.isUploaded)
              Positioned(
                top: 8,
                left: 8,
                child: GestureDetector(
                  onTap: () => _toggleRecordSelection(record),
                  child: Container(
                    width: 24,
                    height: 24,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: Colors.white.withOpacity(0.8),
                    ),
                    child: Center(
                      child: Icon(
                        record.isSelected ? Icons.check_circle : Icons.radio_button_unchecked,
                        color: Colors.blueAccent,
                        size: 20,
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}