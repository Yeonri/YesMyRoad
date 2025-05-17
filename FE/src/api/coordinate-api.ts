// src/api/coordinate-api.ts
import apiCaller from '@/api/api-utils'

export const coordinateAPI = {
  // 지도에 찍을 손상(캡처) 좌표 데이터를 GeoJSON 형식으로 조회합니다.
  getDefectLocations: () =>
    apiCaller('get', '/api/capture-points', '손상 데이터 조회 실패'),

  // 좌표 클릭 시 상세 정보 조회
  getDefectDetail: (publicId: string) =>
    apiCaller(
      'get',
      `/api/capture-points/${publicId}`,
      '손상 데이터 상세 조회 실패',
    ),
}
