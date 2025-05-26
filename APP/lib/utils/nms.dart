import 'dart:math';

class NMS {
  static (List<int>, List<List<double>>, List<double>) apply(
      List<List<double>> rawOutput,
      int count, {
        double confidenceThreshold = 0.25,
        double iouThreshold = 0.45,
      }) {
    List<int> bestClasses = [];
    List<double> bestScores = [];
    List<int> boxesToSave = [];

    for (int i = 0; i < rawOutput.shape[1]; i++) {
      double bestScore = 0;
      int bestCls = -1;

      for (int j = 4; j < count; j++) {
        double clsScore = rawOutput[j][i];
        if (clsScore > bestScore) {
          bestScore = clsScore;
          bestCls = j - 4;
        }
      }

      if (bestScore > confidenceThreshold) {
        bestClasses.add(bestCls);
        bestScores.add(bestScore);
        boxesToSave.add(i);
      }
    }

    List<List<double>> candidateBoxes = [];
    for (var index in boxesToSave) {
      List<double> savedBox = [];
      for (int i = 0; i < 4; i++) {
        savedBox.add(rawOutput[i][index]);
      }
      candidateBoxes.add(savedBox);
    }

    var sortedIndices = List<int>.generate(bestScores.length, (i) => i)
      ..sort((a, b) => bestScores[b].compareTo(bestScores[a]));

    List<int> sortedBestClasses = [];
    List<List<double>> sortedCandidateBoxes = [];
    List<double> sortedBestScores = [];

    for (var i in sortedIndices) {
      sortedBestClasses.add(bestClasses[i]);
      sortedCandidateBoxes.add(candidateBoxes[i]);
      sortedBestScores.add(bestScores[i]);
    }

    List<List<double>> finalBboxes = [];
    List<double> finalScores = [];
    List<int> finalClasses = [];

    while (sortedCandidateBoxes.isNotEmpty) {
      var bbox1xywh = sortedCandidateBoxes.removeAt(0);
      finalBboxes.add(bbox1xywh);
      var bbox1xyxy = _xywh2xyxy(bbox1xywh);
      finalScores.add(sortedBestScores.removeAt(0));
      var class1 = sortedBestClasses.removeAt(0);
      finalClasses.add(class1);

      List<int> indexesToRemove = [];
      for (int i = 0; i < sortedCandidateBoxes.length; i++) {
        if (class1 == sortedBestClasses[i]) {
          if (_computeIou(bbox1xyxy, _xywh2xyxy(sortedCandidateBoxes[i])) >
              iouThreshold) {
            indexesToRemove.add(i);
          }
        }
      }

      for (var index in indexesToRemove.reversed) {
        sortedCandidateBoxes.removeAt(index);
        sortedBestClasses.removeAt(index);
        sortedBestScores.removeAt(index);
      }
    }

    return (finalClasses, finalBboxes, finalScores);
  }

  static List<double> _xywh2xyxy(List<double> bbox) {
    double halfWidth = bbox[2] / 2;
    double halfHeight = bbox[3] / 2;
    return [
      bbox[0] - halfWidth,
      bbox[1] - halfHeight,
      bbox[0] + halfWidth,
      bbox[1] + halfHeight,
    ];
  }

  static double _computeIou(List<double> bbox1, List<double> bbox2) {
    double xLeft = max(bbox1[0], bbox2[0]);
    double yTop = max(bbox1[1], bbox2[1]);
    double xRight = min(bbox1[2], bbox2[2]);
    double yBottom = min(bbox1[3], bbox2[3]);

    if (xRight < xLeft || yBottom < yTop) {
      return 0;
    }

    double intersectionArea = (xRight - xLeft) * (yBottom - yTop);
    double bbox1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    double bbox2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    double unionArea = bbox1Area + bbox2Area - intersectionArea;

    return intersectionArea / unionArea;
  }
}

extension ListShape on List {
  List<int> get shape {
    List<int> dimensions = [];
    Object? current = this;

    while (current is List) {
      dimensions.add((current as List).length);
      if ((current as List).isEmpty) break;
      current = (current as List).first;
    }

    return dimensions;
  }
}