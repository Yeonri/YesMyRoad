# Upload-server

 [![Java](https://img.shields.io/badge/Java-21-red?logo=java)](https://www.oracle.com/java/)  [![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.4.5-brightgreen?logo=springboot)](https://spring.io/projects/spring-boot)  [![WebFlux](https://img.shields.io/badge/Spring%20WebFlux-3.4.5-blue?logo=spring)](https://spring.io/projects/spring-framework)  [![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.x-purple?logo=apachekafka)](https://kafka.apache.org/)  [![AWS SDK](https://img.shields.io/badge/AWS%20SDK%20for%20Java-2.30.15-yellow?logo=amazonaws)](https://sdk.amazonaws.com/java/api/latest/) 


## 📝 목차

1. [담당자](#담당자)
2. [특징](#특징)
3. [기술 스택](#기술-스택)
4. [시작하기](#시작하기) [스크립트](#스크립트)
5. [환경 변수](#환경-변수)
6. [트러블 슈팅](#트러블-슈팅)

## 🧑‍🦲 담당자

- BE : 이세중

## 📌 특징

- ⚡️ **이벤트 기반의 아키텍처 설계** - kafka, SpringWebflux 
- 🛠️ **S3 presinedURL을 이용**
- 💾 **리눅스 tmpfs를 활용하여 이미지 처리**

## 🔧 기술 스택

| 범주                   | 사용 기술                                              | 비고                                              |
| :--------------------- | :----------------------------------------------------- | :------------------------------------------------ |
| **언어**               | Java 21                                                | Gradle toolchain 설정                             |
| **빌드**               | Gradle 8.x                                             | `java`, `org.springframework.boot`, `io.spring.dependency-management` 플러그인 |
| **프레임워크**         | Spring Boot 3.4.5                                       | WebFlux 모듈 사용                                  |
| **메시징**             | Spring Kafka                                           | Apache Kafka 연동                                 |
| **클라우드 SDK**       | AWS SDK for Java v2.30.15                              | S3 객체 저장/조회                                 
| **애노테이션 프로세싱** | Lombok 1.18.x                                          | `compileOnly` 및 `annotationProcessor` 설정        |
| **개발 편의**          | Spring Boot DevTools                                   | Hot reload 지원                                   |
| **테스트**             |`spring-boot-starter-test`, `reactor-test`, `spring-kafka-test` | 리액티브 및 메시징 테스트 환경 구성               |

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

# 5. 실행
java -jar build/libs/S12P31B201-0.0.1-SNAPSHOT.jar
```
## 🌎 환경변수


| 이름                                                | 설명                                            |
| :-------------------------------------------------- | :---------------------------------------------- |
| `SPRING_CONFIG_IMPORT`                              | `optional:file:.env[.properties]` 설정 불러오기 |
| `SPRING_WEBFLUX_MULTIPART_MAX_IN_MEMORY_SIZE`       | WebFlux 멀티파트 메모리 최대 크기 (예: `2MB`)   |
| `KAFKA_BOOTSTRAP_SERVERS`                           | Kafka 브로커 주소                               |
| `AWS_REGION`                                        | AWS 리전                                        |
| `AWS_S3_BUCKET`                                     | S3 버킷 이름                                    |
| `AWS_S3_CREDENTIAL_ACCESS_KEY`                      | S3 접근용 Access Key                            |
| `AWS_S3_CREDENTIAL_SECRET_KEY`                      | S3 접근용 Secret Key                            |
| `NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN`                   | Mapbox GL JS API 키     

## 🔥 트러블 슈팅

### 문제 1
저희 서비스는 디바이스 1대당 분당 10건이 넘는 요청을 처리해야 했고, 이 요청에는 단순 JSON 데이터뿐 아니라, 이미지 데이터도 함께 포함되어 있었습니다. 특히 이미지 처리 과정은 CPU 자원을 많이 사용하는 작업이었기 때문에, 기존의 블로킹 방식인 Spring MVC로는 병목이 발생했고, 디바이스의 응답 대기 시간이 길어지면서 모델 추론 중단 등 다양한 문제가 발생했습니다.

### 해결 과정
### [이벤트 기반 아키텍처 선정 이유]
upload서버는 gpu서버,api서버,GPU서버 사이에서 응답과 요청을 수없이 하기에 병렬 처리만으로는 부족하였습니다. 그리하여서 쓰레드를 효율적으로 활용할 수 있는 이벤트 처리 방식을 채택하였고 kafka와 SpringWebflux를 사용하였습니다.

### [tmpfs를 이용한 이유] 
2차 필터링을 위하여 이미지를 임시로 가지고 있어야 했습니다.
디바이스에서 S3로 바로 전송하는 것은 저희의 서비스에서는 불필요한 S3 PUT DELETE 요청이 많을 수 있다는 판단하에 다른 방법이 필요했습니다.

처음에는 ***Local disk***를 활용하려 했지만, diskI/O 병목으로 인하여 안정적으로 요청을 처리할 수 없었습니다. 그리하여서 redis를 활용하려 했지만, 싱글 쓰레드만을 지원하는 ***Redis***가 끊임 없이 이미지I/O가 
일어나는 저희의 서비스에서는 사용할 수 없었습니다.
그래서 램을 활용하면서 SpringWebflux의 동적으로 쓰레드를 할당하는 기능과 리눅스의 ***tmpfs***를 이용하여 해결하였습니다. 순각적으로 업로드 요청을 1000개 이상을 보내어도 안정적으로 처리할 수 있었습니다.

### [kafka 선정 이유]
RabbitMQ 대신 카프카를 채택한 이유는 빠른 이미지 처리를 위해서 메모리 버퍼를 이용할 생각이였고, 램만을 활용하는 rabbitMQ보다 디스크를 함께 활용할 수 있는 kafka가 더 좋을 것이라 판단했습니다.

### [S3 presignedURL응 사용한 이유]
외부 GPU 대여 서비스를 사용하였고, AWS 크리덴셜을 해당 서버에 두는 것은 위험하다고 생각했습니다. 그렇다고, 2차 필터링 된 이미지를 다시 업로드 서버로 받아오는 것도 부담스러웠습니다.
그래서 처음 이미지 처리 요청을 보낼 때, presignedURL을 담아서 GPU서버로 보내서 직접 이미지 PUT요청을 보내기로 하였습니다.

# 
### 문제 2
비동기 로직을 짰음에도 불구하고, 요청이 많아지면 성능이 급격하게 느려지는 현상이 있었습니다.

### 해결 과정
로그를 하나하나 남기는 과정에서 블로킹 발생하여서 느려졌었습니다. 현재 코드에는 시연을 위해서 로그를 남기는 과정을 남겼지만, 정상적인 서비스 출시시에는 로그를 지워야 합니다.


## 🙏 회고

비동기 방식에서는 로그를 어떻게 남겨야하는지 공부해보고 싶습니다. 로그를 다 지웠더니 트러블이 발생했을 때, 문제를 해결하려면 다른 서버 로그를 다 열어봐야 겨우 문제를 발견할 수 있었습니다.

<끝>
