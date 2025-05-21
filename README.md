# Yes My Road – Frontend

스마트폰에서 구동되는 온디바이스 AI로 도로 균열·포트홀 실시간 탐지 및 시각화 웹 대시보드


## 주요 기능

- 도로 파손이 발생한 위치를 지도 및 히트맵에 마커로 표시
- 렌더링 최적화: `useMemo`, `useCallback`, `React.memo` 활용
- 고성능 렌더링: mapbox-gl 기반 지도, Apache Echarts 기반 차트
- 지자체별 필터링: 공공 GeoJSON API를 이용한 행정구역별 데이터 필터링
- 결함 목록 검색, 정렬, 필터링
- 작업 현황 변경 및 상세 이미지 조회
- 도로 파손 및 도로 보수 관련 각종 지표 표시


## 사용 기술
- React 19  
- Next.js 15.3.1  
- Tailwind CSS  
- TypeScript  
- mapbox-gl  
- Apache Echarts  
- shadcn
- zustand

