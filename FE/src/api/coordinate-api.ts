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


// import apiClient from '@/api/api-client'

// export const coodAPI = {
//   // 지도에 찍을 손상(캡처) 좌표 데이터를 GeoJSON 형식으로 조회합니다.
//   getDefects: async () => {
//     try {
//       const response = await apiClient.get('/api/capture-points')
//       return { data: response.data, status: response.status, error: null}
//     } catch (error: any) {
//       console.error(`손상 데이터 조회 실패: ${error}`)
//       return { data: null, status: error.response.status, error }
//     }
//   },

//   // 좌표 클릭 시 상세 정보 조회
//   getDetailedDefects: async (publicId: string) => {
//     try {
//       const response = await apiClient.get(`/api/capture-points/${publicId}`)
//       return { data: response.data, status: response.status, error: null}
//     } catch (error: any) {
//       console.error(`손상 데이터 상세 조회 실패: ${error}`)
//       return { data: null, status: error.response.status, error }
//     }
//   },
// }
