# Image-server

[![Java](https://img.shields.io/badge/Java-21-red?logo=openjdk)](https://openjdk.org/projects/jdk/21/)  [![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.4.5-brightgreen?logo=springboot)](https://spring.io/projects/spring-boot)  [![AWS SDK for Java](https://img.shields.io/badge/AWS%20SDK-2.30.15-yellow?logo=amazonaws)](https://sdk.amazonaws.com/java/api/latest/software/amazon/awssdk/services/s3/package-summary.html)  [![Lombok](https://img.shields.io/badge/Lombok-1.18.30-orange?logo=lombok)](https://projectlombok.org/)


## 📝 목차

1. [담당자](#담당자)
2. [특징](#특징)
3. [기술 스택](#기술-스택)
4. [시작하기](#시작하기) [스크립트](#스크립트)
5. [환경 변수](#환경-변수)
6. [이유](#이유)

## 🧑‍🦲 담당자

- BE : 이세중

## 📌 특징

- ⚡️ **S3에서 이미지 GET 요청**

## 🔧 기술 스택

| 범주                   | 사용 기술                                              | 비고                                              |
| :--------------------- | :----------------------------------------------------- | :------------------------------------------------ |
| **언어**               | Java 21                                                | Gradle toolchain 설정                             |
| **빌드**               | Gradle 8.x                                             | `java`, `org.springframework.boot`, `io.spring.dependency-management` 플러그인 |
| **프레임워크**         | Spring Boot 3.4.5                                       |                                   ||
| **클라우드 SDK**       | AWS SDK for Java v2.30.15                              | S3 조회                                 
| **애노테이션 프로세싱** | Lombok 1.18.x                                          | `compileOnly` 및 `annotationProcessor` 설정        |

## 🏁 시작하기

> JDK 21 이상 및 Gradle 8.x 이상을 권장합니다.  

```bash
# 1. 저장소 클론
git clone https://lab.ssafy.com/s12-final/S12P31B201.git

# 2. 프로젝트 루트로 이동
cd reactive

# 3. 환경 변수 설정
# .env 파일을 프로젝트 루트에 생성하고 AWS 자격증명 등 필요한 값을 입력하세요.
cp /path/to/.env .env

# 4. 의존성 설치 및 빌드
./gradlew clean build
```

| 이름                                                | 설명                                            |
| :-------------------------------------------------- | :---------------------------------------------- |
| `SPRING_CONFIG_IMPORT`                              | `optional:file:.env[.properties]` 설정 불러오기  |
| `AWS_REGION`                                        | AWS 리전                                        |
| `AWS_S3_BUCKET`                                     | S3 버킷 이름                                    |
| `AWS_S3_CREDENTIAL_ACCESS_KEY`                      | S3 접근용 Access Key                            |
| `AWS_S3_CREDENTIAL_SECRET_KEY`                      | S3 접근용 Secret Key                            |
|


## 🔥 이유

### [ 만든 이유 ]
Nginx에 캐싱 되어있지 않은 이미지를 S3에서 당겨올 목적으로 만들었습니다.