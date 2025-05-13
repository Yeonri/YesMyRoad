# B201 도로 파손 관리 서비스

Spring Boot 기반의 도로 파손 탐지·관리·통계 API 프로젝트입니다.

## 📖 프로젝트 개요

* **목표**: AI로 검출된 도로 파손 이미지를 저장하고, 위치 기반 주소 매핑, 파손 상태 관리 및 다양한 통계 데이터를 제공
* **주요 기능**:

    * AI 탐지 결과 저장 및 주소 변환 (Vworld API)
    * 파손 상태 변경(보고됨 → 처리중 → 완료)
    * 전체·일별·주별·월별 파손 통계 API
    * 유지보수 현황 통계 API
    * 사용자 인증(JWT) · 회원가입 · 로그인 · 리프레시·로그아웃
    * CORS 설정 및 전역 예외 처리

## 🚀 기술 스택

* **Framework**: Spring Boot 3.4.4
* **언어**: Java 21
* **빌드 툴**: Gradle (org.springframework.boot plugin v3.4.4, io.spring.dependency-management v1.1.7)
* **DB**: MySQL 8.x (Spring Data JPA + Hibernate Spatial 6.6.11)
* **토큰 저장소**: Redis (Spring Data Redis)
* **인증**: JWT (io.jsonwebtoken jjwt 0.11.5)
* **유효성 검증**: Spring Boot Starter Validation (Jakarta Validation)
* **HTTP Client**: Spring WebClient (Spring Boot Starter Web)
* **API 문서화**: springdoc-openapi-starter-webmvc-ui 2.1.0
* **로깅**: SLF4J + Logback (Spring Boot 기본)
* **도구**: Lombok (annotation processing), Spring Boot DevTools

## ⚙️ 환경 설정

### 1. `.env.properties` 또는 OS 환경변수 설정

```properties
MYSQL_DRIVER=com.mysql.cj.jdbc.Driver
MYSQL_URL=localhost:3306
MYSQL_DATABASE=b201db
MYSQL_USER=dbuser
MYSQL_PASSWORD=dbpass
REDIS_HOST=localhost
REDIS_PORT=6379
ADDRESS_API_KEY=Vworld에서발급받은키
JWT_SECRET=충분히긴서명키
```

### 2. `application.yml`

```yaml
spring:
  config:
    import: optional:file:.env[.properties]

  datasource:
    driver-class-name: ${MYSQL_DRIVER}
    url: jdbc:mysql://${MYSQL_URL}/${MYSQL_DATABASE}?serverTimezone=Asia/Seoul
    username: ${MYSQL_USER}
    password: ${MYSQL_PASSWORD}
    hikari:
      maximum-pool-size: 10

  jpa:
    hibernate:
      ddl-auto: none
    properties:
      hibernate:
        dialect: org.hibernate.dialect.MySQLDialect
    defer-datasource-initialization: true

  data:
    redis:
      host: ${REDIS_HOST}
      port: ${REDIS_PORT}

address:
  key: ${ADDRESS_API_KEY}
jwt:
  secret: ${JWT_SECRET}
```

## 🏗️ 빌드 & 실행

```bash
git clone <repository-url>
cd api               # 프로젝트 루트에서 api 디렉터리로 이동
./gradlew bootRun    # 개발용 실행 (로컬 핫 리로드 포함)
# 또는
./gradlew build
java -jar build/libs/*.jar
```

## 🔑 API 문서

자세한 API 명세는 [Swagger UI](http://localhost:8080/swagger-ui/index.html) 또는
실행된 애플리케이션에서 `/v3/api-docs` 엔드포인트를 통해 확인하세요.
로그인, 토큰 발급, 파손 조회 등 실제 호출 시에는 JWT 토큰 및 환경 설정이 필요합니다.

## 📘 CORS 설정

* `https://k12b201.p.ssafy.io`, `http://localhost:3000`, `http://45.12.114.100:12956` 허용
* 메서드: GET, POST, PATCH, DELETE
* 헤더: Content-Type, Authorization
* 설정 위치: `SecurityConfig#corsConfigurationSource()`

## 🛡️ 예외 처리

* `@RestControllerAdvice` 활용
* Validation 오류: HTTP 400 + 필드별 메시지
* 인증/인가 오류: HTTP 401
* 중복/잘못된 파라미터: HTTP 400
* 그 외 예기치 못한 예외: HTTP 500 + 메시지

---

