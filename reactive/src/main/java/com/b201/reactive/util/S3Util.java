package com.b201.reactive.util;

import com.b201.reactive.config.AwsConfig;
import lombok.RequiredArgsConstructor;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.PresignedPutObjectRequest;
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest;
import org.slf4j.Logger;

import java.time.Duration;


@Service
@RequiredArgsConstructor
public class S3Util {

    private final AwsConfig awsConfig;
    private final S3Presigner s3Presigner;
    private static final Logger logger = LoggerFactory.getLogger(S3Util.class);

    // 외부 사용 X
    private String createPresignedUrl(String objectName) {

        PutObjectRequest objectRequest = PutObjectRequest.builder()
                .bucket(awsConfig.getBucketName())
                .key(objectName)
                .build();

        PutObjectPresignRequest presignRequest = PutObjectPresignRequest.builder()
                .signatureDuration(Duration.ofMinutes(10))  // The URL expires in 10 minutes.
                .putObjectRequest(objectRequest)
                .build();

        PresignedPutObjectRequest presignedRequest = s3Presigner.presignPutObject(presignRequest);

        String url = presignedRequest.url().toString();
        logger.info("Generated presignedURL: {}", url);
        return url;
    }

    // 실제 외부 사용 (비동기 처리)
    public Mono<String> createPresignedUrlAsync(String objectKey) {
        return Mono.fromCallable(() -> createPresignedUrl(objectKey))
                .subscribeOn(Schedulers.boundedElastic());
    }
}
