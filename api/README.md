## YES, MY ROAD 백엔드 서비스

본 문서는 AI 기반 도로 파손 탐지 및 관리, 통계 대시보드를 제공하는 B201 프로젝트의 백엔드 서비스에 대한 설명입니다.

---

### 🔍 개요

* **목적**: 엣지 디바이스에서 전송된 AI 탐지 결과를 수집·저장하고, 관리용 API 제공 및 지역별·시간별 통계 대시보드 생성
* **주요 기능**:

    * AI 탐지 결과 수집 (`POST /api/detect`)
    * 캡처 포인트 목록 조회 및 상세 정보 제공
    * 파손 상태 업데이트
    * 지역별·일별·주간·월간 통계 대시보드
    * JWT 기반 사용자 인증 및 권한 관리

---

### 🛠️ 기술 스택

* **언어 & 프레임워크**: Java 21, Spring Boot
* **보안**: Spring Security, JWT
* **데이터베이스**: MySQL (Spring Data JPA), Redis (캐싱)
* **빌드 도구**: Gradle
* **검증 & 직렬화**: Jakarta Validation, Jackson (JavaTimeModule)
* **라이브러리**: Lombok

---

### ⚙️ 환경 설정

프로젝트 최상위 `application.yml` 또는 `.env` 파일에 다음 변수를 설정하세요:

```properties
# MySQL
MYSQL_DRIVER=com.mysql.cj.jdbc.Driver
MYSQL_URL=localhost:3306
MYSQL_DATABASE=b201_db
MYSQL_USER=your_db_user
MYSQL_PASSWORD=your_db_pass
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
# 주소 조회 API
ADDRESS_API_KEY=your_address_api_key
# JWT
JWT_SECRET=your_jwt_secret_key
```

로깅 수준은 `logging.level.com.b201.api` 설정 값을 통해 조정 가능합니다.

---

### 🚀 시작하기

1. **레포지토리 클론**

   ```bash
   git clone -b BE/dev https://lab.ssafy.com/s12-final/S12P31B201.git
   cd S12P31B201/api
   ```

2. **데이터베이스 설정**

    * MySQL에 `MYSQL_DATABASE` 이름으로 데이터베이스 생성
    * (선택) `src/main/resources/data.sql` 실행하여 초기 데이터 로드

3. **의존성 설치 및 실행**

   ```bash
   ./gradlew clean build
   ./gradlew bootRun
   ```

   기본 포트 `http://localhost:8080`으로 서비스가 실행됩니다.

---

### 📦 주요 API

#### 사용자 관리 (`/api/users`)

| 메서드  | 경로         | 설명                   |
|------|------------|----------------------|
| POST | `/signup`  | 회원 가입                |
| POST | `/login`   | 로그인 및 JWT 발급         |
| POST | `/refresh` | 리프레시 토큰으로 액세스 토큰 재발급 |
| POST | `/logout`  | 리프레시 토큰 무효화          |

#### AI 탐지 (`/api/detect`)

| 메서드  | 경로            | 설명                          |
|------|---------------|-----------------------------|
| POST | `/api/detect` | AI 탐지 결과 제출 (`AiResultDto`) |

#### 캡처 포인트 (`/api/capture-points`)

| 메서드 | 경로                               | 설명                   |
|-----|----------------------------------|----------------------|
| GET | `/api/capture-points`            | 사용자 권한 지역 내 전체 목록 조회 |
| GET | `/api/capture-points/{publicId}` | 특정 포인트 상세 정보 조회      |

#### 파손 관리 (`/api/damages`)

| 메서드   | 경로                        | 설명                                    |
|-------|---------------------------|---------------------------------------|
| PATCH | `/api/damages/{damageId}` | 특정 파손 상태 업데이트 (`StatusUpdateRequest`) |

#### 대시보드 (`/api/dashboard`)

*모든 대시보드 엔드포인트는 JWT 인증 후 사용자 지역 기준으로 데이터 제공*

| 메서드 | 경로                               | 설명                |
|-----|----------------------------------|-------------------|
| GET | `/api/dashboard/type`            | 파손 종류별 분포         |
| GET | `/api/dashboard/daily`           | 금일 vs 전일 비교 및 증감율 |
| GET | `/api/dashboard/weekly`          | 최근 7일 상태          |
| GET | `/api/dashboard/monthly`         | 월별 현황             |
| GET | `/api/dashboard/monthly-summary` | 월별 누적 요약          |
| GET | `/api/dashboard/districts`       | 구 단위 분포           |
| GET | `/api/dashboard/top3`            | 파손 건수 상위 3개 지역    |
| GET | `/api/dashboard/risk`            | 위험도별 분류           |
| GET | `/api/dashboard/region-count`    | 활성 구 개수           |

#### 유지보수 (`/api/maintenance`)

| 메서드 | 경로                                  | 설명         |
|-----|-------------------------------------|------------|
| GET | `/api/maintenance/overview`         | 전체 유지보수 개요 |
| GET | `/api/maintenance/completion-stats` | 상태별 완료 통계  |
| GET | `/api/maintenance/monthly-status`   | 월별 유지보수 현황 |
| GET | `/api/maintenance/districts`        | 구별 유지보수 분포 |

---

### 🧪 테스트

* Postman/CURL 사용
* 보호된 엔드포인트는 `Authorization: Bearer <access_token>` 헤더 필요
* `data.sql` 샘플 데이터를 참고하세요

