package com.b201.image_api.service;

import com.b201.image_api.config.AwsConfig;
import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.NoSuchKeyException;

@Service
@RequiredArgsConstructor
@Slf4j
public class S3Service {

    private final S3Client s3Client;
    private final AwsConfig awsConfig;

    public ResponseEntity<?> downloadFromS3(String objectKey, HttpServletRequest request) {

        try {
            log.info("S3Service.downloadFromS3() - 요청된 파일명: {}", objectKey);
            GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                    .bucket(awsConfig.getBucketName())
                    .key(objectKey)
                    .build();

            ResponseInputStream<GetObjectResponse> s3ObjectStream = s3Client.getObject(getObjectRequest);

            MediaType contentType = getContentType(objectKey);

            return ResponseEntity.ok()
                    .contentType(contentType)
                    .body(new InputStreamResource(s3ObjectStream));

        } catch (NoSuchKeyException e) {
            log.warn("이미지를 찾을 수 없습니다: {}", objectKey);
            return ResponseEntity.status(404).build(); // 그냥 404 응답
        } catch (Exception e) {
            log.error("이미지 다운로드 오류", e);
            return ResponseEntity.status(500).build(); // 서버 내부 오류
        }
    }

    //<img src="https://yourdomain.com/images/some.jpg"
    //     onError="this.onerror=null; this.src='/default-image.jpg'" />
    // 이렇게 받도록 하자.

    private MediaType getContentType(String objectKey) {
        String lower = objectKey.toLowerCase();
        if (lower.endsWith(".png")) {
            return MediaType.IMAGE_PNG;
        } else if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            return MediaType.IMAGE_JPEG;
        } else if (lower.endsWith(".gif")) {
            return MediaType.IMAGE_GIF;
        } else {
            return MediaType.APPLICATION_OCTET_STREAM;
        }
    }
}
