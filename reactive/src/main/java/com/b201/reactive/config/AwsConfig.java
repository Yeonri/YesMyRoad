package com.b201.reactive.config;

import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;

@Getter
@Configuration
public class AwsConfig {

    @Getter
    @Value("${aws.s3.bucket}")
    private String bucketName;

    @Value("${aws.region}")
    private String region;

    @Value("${aws.s3.credential.access-key}")
    private String accessKey;

    @Value("${aws.s3.credential.secret-key}")
    private String secretKey;

    @Bean
    public AwsBasicCredentials awsBasicCredentials() {
        return AwsBasicCredentials.create(accessKey, secretKey);
    }

    @Bean
    public StaticCredentialsProvider staticCredentialsProvider(AwsBasicCredentials awsBasicCredentials) {
        return StaticCredentialsProvider.create(awsBasicCredentials);
    }

    @Bean
    public S3Presigner s3Presigner(StaticCredentialsProvider staticCredentialsProvider) {
        return S3Presigner.builder()
                .region(Region.of(region))
                .credentialsProvider(staticCredentialsProvider)
                .build();
    }

    @Bean
    public S3Client s3Client(StaticCredentialsProvider staticCredentialsProvider) {
        return S3Client.builder()
                .region(Region.of(region))
                .credentialsProvider(staticCredentialsProvider)
                .build();
    }
}
