// src/api/defect-api.ts
import axios from 'axios'
import apiClient from './api-client'

export const defectAPI = {
  // 도로 파손 건의 상태 변경
  updateRoadDamageStatus: async (damageId: number, status: string) => {
    try {
      const response = await apiClient.patch(`/api/damages/${damageId}`, { status })
      return { data: response.data, status: response.status, error: null }
    } catch (error: unknown) {
      // 1) unknown 타입 좁히기
      if (axios.isAxiosError(error) && error.response) {
        console.error(`지역별 도로파손 분포 지도 조회 실패: ${error.message}`)
        return {
          data: null,
          status: error.response.status,
          error, // 여기선 AxiosError 타입입니다
        }
      }

      // 2) AxiosError가 아니면 fallback
      console.error('알 수 없는 오류:', error)
      return { data: null, status: 500, error: new Error('Unknown error') }
    }
  }
}
