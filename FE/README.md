# 📌 AI 기반 실시간 도로파손 모니터링 서비스

[![License](https://img.shields.io/github/license/ssafy/S12P31B201)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)](https://www.typescriptlang.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)

> AI와 지도 기반의 대용량 도로파손 데이터 실시간 시각화·분석 서비스

## ✨ 데모

- **배포 URL:** https://k12b201.p.ssafy.io/
- **동영상:** `public/최종 시연 영상 2.5배속.mp4`

## 🔖 목차

1. [담당자](#담당자)
2. [특징](#특징)
3. [기술 스택](#기술-스택)
4. [시작하기](#시작하기)
5. [스크립트](#스크립트)
6. [폴더 구조](#폴더-구조)
7. [환경 변수](#환경-변수)
8. [라이선스](#라이선스)

## 담당자

- FE: 조현준, 김동환

## 특징

- ⚡️ **대용량 좌표 데이터 처리 최적화** – Mapbox GL JS, ECharts
- 🧩 **UI/UX 컴포넌트** – Radix UI, shadcn/ui, Embla Carousel
- 📊 **통계 시각화** – Apache ECharts
- 🔍 **상태/데이터 관리** – React Query, Zustand
- 🛠️ **코드 컨벤션** – EditorConfig, Prettier, ESLint, Tailwind CSS, TypeScript strict, Conventional Commits

## 기술 스택

| 범주                 | 사용 기술               | 비고                                             |
| :------------------- | :---------------------- | :----------------------------------------------- |
| **언어**       | TypeScript 5.x          | Strict 설정                                      |
| **프레임워크** | React 19, Next.js 15    | App Router                                       |
| **지도**       | Mapbox GL JS            | 대규모 좌표 렌더링 최적화                        |
| **통계**       | Apache ECharts          | 반응형 통계 컴포넌트, 고급 시각화                |
| **스타일**     | Tailwind CSS, shadcn/ui | CSS-in-JS 불필요                                 |
| **UI**         | Radix UI, Embla Carousel| 접근성·모던 UI                                   |
| **상태 관리**  | Zustand, React Query    | 서버/클라이언트 상태 분리                        |
| **툴링**       | ESLint, Prettier        | Conventional Commits, import 정렬, Tailwind 플러그인 |

## 시작하기

> 최소 Node.js 18 이상 / pnpm 8 이상을 권장합니다

```bash
# 1. 저장소 클론
git clone https://lab.ssafy.com/s12-final/S12P31B201.git

# 2. 폴더 이동
cd fe

# 3. 패키지 설치
pnpm install

# 4. 개발 서버 실행
pnpm dev
```

## 스크립트

| 명령어               | 설명               |
| :---------------- | :--------------- |
| `pnpm dev`        | 로컬 개발 서버(HMR) 실행 |
| `pnpm build`      | 프로덕션 번들링         |
| `pnpm start`      | 빌드된 앱 실행         |
| `pnpm lint`       | ESLint 검사        |
| `pnpm format`     | Prettier 포맷팅     |

## 폴더 구조

```
.
├─ public/                # 정적 파일 및 데이터 (이미지, 폰트 등)
├─ src/
│  ├─ app/                # Next.js App Router 구성 파일
│  │  ├─ auth/            # 인증 관련 페이지
│  │  ├─ analytics/       # 분석 대시보드 페이지
│  │  ├─ layout.tsx       # 루트 레이아웃 컴포넌트
│  │  └─ page.tsx         # 메인 페이지
│  ├─ components/         # 재사용 컴포넌트
│  │  ├─ ui/              # 기본 UI 컴포넌트 (버튼, 입력 등)
│  │  ├─ auth/            # 인증 관련 컴포넌트
│  │  ├─ dashboard/       # 대시보드 관련 컴포넌트
│  │  ├─ statistics/      # 통계 시각화 컴포넌트
│  │  └─ header/          # 헤더 컴포넌트
│  ├─ api/                # API 통신 래퍼 함수
│  ├─ assets/             # 이미지, 아이콘 등 정적 리소스
│  ├─ data/               # 샘플 데이터, geojson 파일
│  ├─ lib/                # 유틸리티 함수 및 헬퍼
│  ├─ store/              # Zustand 상태 관리 스토어
│  ├─ services/           # 서비스 로직 (데이터 처리, 비즈니스 로직)
│  ├─ providers/          # Context Provider 컴포넌트
│  ├─ types/              # 전역 TypeScript 타입 정의
│  └─ utils/              # 일반 유틸리티 함수
├─ .editorconfig          # 에디터 설정
├─ .prettierrc            # 코드 포맷팅 설정
├─ tailwind.config.ts     # Tailwind CSS 설정
├─ next.config.ts         # Next.js 설정
├─ tsconfig.json          # TypeScript 설정
└─ package.json           # 프로젝트 의존성 및 스크립트
```

## 환경 변수

| 이름                              |  설명                       |
| :------------------------------- | :------------------------- |
| `NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN` | Mapbox GL JS API 키 |

## 라이선스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

