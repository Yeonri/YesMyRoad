import 'package:flutter/material.dart';

class BoundingBox extends StatelessWidget {
  const BoundingBox({
    Key? key,
    required this.box,
    required this.name,
    required this.score,
  }) : super(key: key);

  final List<double> box;
  final String name;
  final double score;

  @override
  Widget build(BuildContext context) {
    final double screenWidth = MediaQuery
        .of(context)
        .size
        .width;
    final double screenHeight = MediaQuery
        .of(context)
        .size
        .height;

    // 화면 크기와 카메라 미리보기 크기 간의 비율 계산
    // 이미지 크기 정적 값 적용 -> 실시간 바운딩 박스 형식으로 변경을 진행해야 함
    final double originalWidth = 480.0; // 로그에서 확인한 원본 이미지 너비
    final double originalHeight = 720.0; // 로그에서 확인한 원본 이미지 높이

    final double scaleX = screenWidth / originalWidth;
    final double scaleY = screenHeight / originalHeight;

    // box -> [cx, cy, w, h] 형식
    final double scaledWidth = box[2] * scaleX;
    final double scaledHeight = box[3] * scaleY;
    final double scaledLeft = (box[0] * scaleX) -
        (scaledWidth / 2); // 중심점 -> 좌상단 변환
    final double scaledTop = (box[1] * scaleY) -
        (scaledHeight / 2); // 중심점 -> 좌상단 변환

    debugPrint('BoundingBox - 원본 좌표: ${box.toString()}');
    debugPrint('BoundingBox - 화면 비율: $scaleX x $scaleY');
    debugPrint(
        'BoundingBox - 변환된 좌표: left=$scaledLeft, top=$scaledTop, width=$scaledWidth, height=$scaledHeight');

    final Color borderColor;
    if (name == 'pothole') {
      borderColor = Colors.red;
    } else if (name == 'crack') {
      borderColor = Colors.orange;
    } else {
      borderColor = Colors.green;
    }

    return Positioned(
      left: scaledLeft,
      top: scaledTop,
      width: scaledWidth,
      height: scaledHeight,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: borderColor, width: 2),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: borderColor.withOpacity(0.7),
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(2),
                  bottomRight: Radius.circular(2),
                ),
              ),
              child: Text(
                "${name.toUpperCase()} ${(score * 100).toInt()}%",
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}