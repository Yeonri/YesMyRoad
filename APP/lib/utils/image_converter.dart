import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img_lib;
import 'dart:io' show Platform, File;
import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'dart:math' as math;
import 'package:path_provider/path_provider.dart';
import 'package:intl/intl.dart';

class PreprocessResult {
  final List<List<List<double>>> tensor;
  final double scale;
  final double padX;
  final double padY;

  PreprocessResult(this.tensor, this.scale, this.padX, this.padY);
}

class ImageConverter {
  static DeviceOrientation currentOrientation = DeviceOrientation.portraitUp;

  static int _sensorOrientation = 90;

  static void setOrientation(DeviceOrientation orientation) {
    currentOrientation = orientation;
    debugPrint('방향 설정됨: $orientation');
  }

  static void setSensorOrientation(int orientation) {
    _sensorOrientation = orientation;
    debugPrint('센서 방향 설정됨: $orientation');
  }

  static Future<DeviceOrientation> getAccurateDeviceOrientation() async {
    try {
      final AccelerometerEvent event = await accelerometerEvents.first;

      final double x = event.x;
      final double y = event.y;

      if (x.abs() > y.abs()) {
        return x > 0
            ? DeviceOrientation.landscapeRight
            : DeviceOrientation.landscapeLeft;
      } else {
        return y > 0
            ? DeviceOrientation.portraitUp
            : DeviceOrientation.portraitDown;
      }
    } catch (e) {
      return DeviceOrientation.portraitUp;
    }
  }

  static Future<img_lib.Image?> convertCameraImageToImage(CameraImage cameraImage) async {
    img_lib.Image? image;

    try {
      if (cameraImage.format.group == ImageFormatGroup.yuv420) {
        image = _convertYUV420ToImage(cameraImage);
      } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
        image = img_lib.Image.fromBytes(
          width: cameraImage.width,
          height: cameraImage.height,
          bytes: cameraImage.planes[0].bytes.buffer,
          order: img_lib.ChannelOrder.rgba,
        );
      } else if (cameraImage.format.group == ImageFormatGroup.jpeg) {
        image = _convertJPEGToImage(cameraImage);
      } else if (cameraImage.format.group == ImageFormatGroup.nv21) {
        image = _convertNV21ToImage(cameraImage);
      }

      if (image != null) {
        image = _rotateImageBasedOnOrientation(image);
      }

      return image;
    } catch (e) {
      print('이미지 변환 오류: $e');
      return null;
    }
  }

  static img_lib.Image _rotateImageBasedOnOrientation(img_lib.Image image) {
    if (Platform.isAndroid) {
      switch (currentOrientation) {
        case DeviceOrientation.portraitUp:
          return img_lib.copyRotate(image, angle: 90);

        case DeviceOrientation.landscapeLeft:
          if (image.height > image.width) {
            return img_lib.copyRotate(image, angle: 0);
          }
          return image;

        case DeviceOrientation.landscapeRight:
          if (image.height > image.width) {
            return img_lib.copyRotate(image, angle: 180);
          }
          return image;

        case DeviceOrientation.portraitDown:
          return img_lib.copyRotate(image, angle: 270);

        default:
          return image;
      }
    } else if (Platform.isIOS) {
      switch (currentOrientation) {
        case DeviceOrientation.portraitUp:
          return image;

        case DeviceOrientation.landscapeLeft:
          if (image.height > image.width) {
            return img_lib.copyRotate(image, angle: -90);
          }
          return image;

        case DeviceOrientation.landscapeRight:
          if (image.height > image.width) {
            return img_lib.copyRotate(image, angle: 90);
          }
          return image;

        case DeviceOrientation.portraitDown:
          return img_lib.copyRotate(image, angle: 180);

        default:
          return image;
      }
    }

    return image;
  }

  static img_lib.Image _convertYUV420ToImage(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;

    final uvRowStride = cameraImage.planes[1].bytesPerRow;
    final uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

    final yPlane = cameraImage.planes[0].bytes;
    final uPlane = cameraImage.planes[1].bytes;
    final vPlane = cameraImage.planes[2].bytes;

    final image = img_lib.Image(width: width, height: height);

    var uvIndex = 0;

    for (var y = 0; y < height; y++) {
      var pY = y * width;
      var pUV = uvIndex;

      for (var x = 0; x < width; x++) {
        final yValue = yPlane[pY];
        final uValue = uPlane[pUV];
        final vValue = vPlane[pUV];

        final r = (yValue + 1.402 * (vValue - 128)).clamp(0, 255).toInt();
        final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)).clamp(0, 255).toInt();
        final b = (yValue + 1.772 * (uValue - 128)).clamp(0, 255).toInt();

        image.setPixelRgba(x, y, r, g, b, 255);

        pY++;
        if (x % 2 == 1 && uvPixelStride == 2) {
          pUV += uvPixelStride;
        } else if (x % 2 == 1 && uvPixelStride == 1) {
          pUV++;
        }
      }

      if (y % 2 == 1) {
        uvIndex += uvRowStride;
      }
    }
    return image;
  }

  static img_lib.Image _convertJPEGToImage(CameraImage cameraImage) {
    final bytes = cameraImage.planes[0].bytes;
    return img_lib.decodeJpg(bytes)!;
  }

  static img_lib.Image _convertNV21ToImage(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;
    final yPlane = cameraImage.planes[0].bytes;
    final uvPlane = cameraImage.planes[1].bytes;

    final image = img_lib.Image(width: width, height: height);

    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final yIndex = y * width + x;
        final uvIndex = (y ~/ 2) * (width ~/ 2) * 2 + (x ~/ 2) * 2;

        final yValue = yPlane[yIndex];
        final uValue = uvPlane[uvIndex];
        final vValue = uvPlane[uvIndex + 1];

        final r = (yValue + 1.402 * (vValue - 128)).clamp(0, 255).toInt();
        final g = (yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128)).clamp(0, 255).toInt();
        final b = (yValue + 1.772 * (uValue - 128)).clamp(0, 255).toInt();

        image.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return image;
  }

  static Uint8List encodeImageToJpeg(img_lib.Image image, {int quality = 85}) {
    return Uint8List.fromList(img_lib.encodeJpg(image, quality: quality));
  }

  static Future<File> saveImageWithCorrectOrientation(Uint8List imageData, {String? customFileName}) async {
    try {
      final img_lib.Image? decodedImage = img_lib.decodeJpg(imageData);
      if (decodedImage == null) throw Exception('이미지 디코딩 실패');

      debugPrint('원본 이미지 크기: ${decodedImage.width}x${decodedImage.height}');

      bool isImageLandscape = decodedImage.width > decodedImage.height;

      bool shouldBeLandscape =
          currentOrientation == DeviceOrientation.landscapeLeft ||
              currentOrientation == DeviceOrientation.landscapeRight;

      debugPrint('현재 이미지 방향: ${isImageLandscape ? "가로" : "세로"}, 앱 방향: ${shouldBeLandscape ? "가로" : "세로"}');

      img_lib.Image processedImage;

      if (shouldBeLandscape) {
        if (!isImageLandscape) {
          processedImage = img_lib.copyRotate(decodedImage, angle: -90);
          debugPrint('가로 모드: 이미지를 가로로 회전');
        } else {
          processedImage = decodedImage;
          debugPrint('가로 모드: 이미 가로 이미지');
        }
      } else {
        if (isImageLandscape) {
          processedImage = img_lib.copyRotate(decodedImage, angle: 90);
          debugPrint('세로 모드: 이미지를 세로로 회전');
        } else {
          processedImage = decodedImage;
          debugPrint('세로 모드: 이미 세로 이미지');
        }
      }

      if (currentOrientation == DeviceOrientation.portraitDown) {
        processedImage = img_lib.copyRotate(processedImage, angle: 180);
        debugPrint('거꾸로 세로 모드: 추가 180도 회전');
      } else if (currentOrientation == DeviceOrientation.landscapeRight) {
        if (isImageLandscape || processedImage.width > processedImage.height) {
          processedImage = img_lib.copyRotate(processedImage, angle: 180);
          debugPrint('오른쪽 가로 모드: 가로 이미지 추가 회전');
        }
      }

      String timestamp = DateFormat('yyyyMMdd_HHmmss').format(DateTime.now());
      String directory = (await getApplicationDocumentsDirectory()).path;
      String fileName = customFileName ?? '$directory/image_$timestamp.jpg';

      final file = File(fileName);
      await file.writeAsBytes(Uint8List.fromList(
          img_lib.encodeJpg(processedImage, quality: 90)
      ));

      debugPrint('이미지 저장 완료: $fileName (${processedImage.width}x${processedImage.height})');

      return file;
    } catch (e) {
      debugPrint('이미지 저장 오류: $e');

      String timestamp = DateFormat('yyyyMMdd_HHmmss').format(DateTime.now());
      String directory = (await getApplicationDocumentsDirectory()).path;
      String fileName = customFileName ?? '$directory/fallback_$timestamp.jpg';

      final file = File(fileName);
      await file.writeAsBytes(imageData);

      debugPrint('오류로 인한 원본 저장: $fileName');

      return file;
    }
  }

  static PreprocessResult preprocessImageWithPadding(img_lib.Image image, int targetSize) {
    final originalWidth = image.width;
    final originalHeight = image.height;

    final scaleW = targetSize / originalWidth;
    final scaleH = targetSize / originalHeight;
    final scale = scaleW < scaleH ? scaleW : scaleH;

    final scaledWidth = (originalWidth * scale).toInt();
    final scaledHeight = (originalHeight * scale).toInt();

    final scaledImage = img_lib.copyResize(
      image,
      width: scaledWidth,
      height: scaledHeight,
      interpolation: img_lib.Interpolation.linear,
    );

    final paddedImage = img_lib.Image(width: targetSize, height: targetSize);
    img_lib.fill(paddedImage, color: img_lib.ColorRgb8(0, 0, 0));

    final padX = (targetSize - scaledWidth) / 2;
    final padY = (targetSize - scaledHeight) / 2;

    img_lib.compositeImage(
      paddedImage,
      scaledImage,
      dstX: padX.toInt(),
      dstY: padY.toInt(),
    );

    final tensor = List.generate(
      targetSize,
          (y) => List.generate(
        targetSize,
            (x) {
          final pixel = paddedImage.getPixel(x, y);
          return [
            pixel.r / 255.0,
            pixel.g / 255.0,
            pixel.b / 255.0
          ];
        },
      ),
    );

    return PreprocessResult(tensor, scale, padX, padY);
  }
}