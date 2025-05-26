import 'package:flutter/material.dart';
import 'dart:math';

class ScreenUtils {
  static late Size _previewSize;

  static void setPreviewSize(Size size) {
    _previewSize = size;
  }

  static double getPreviewRatio() {
    if (_previewSize.width == 0 || _previewSize.height == 0) {
      return 1.0;
    }
    return max(_previewSize.height, _previewSize.width) /
        min(_previewSize.height, _previewSize.width);
  }

  static Size screenPreviewSize(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    return Size(width, width * getPreviewRatio());
  }

  static double deviceHeight(BuildContext context) {
    return MediaQuery.of(context).size.height;
  }

  static double deviceWidth(BuildContext context) {
    return MediaQuery.of(context).size.width;
  }
}