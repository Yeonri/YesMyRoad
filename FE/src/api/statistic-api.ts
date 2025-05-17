// src/api/statistic-api.ts
import apiClient from '@/api/api-client'
import apiCaller from '@/api/api-utils'
import axios from 'axios'

export const statisticAPI = {
  // 유형별 도로파손 분포
  getDamageReportByType: () =>
    apiCaller(
      'get',
      '/api/dashboard/type',
      '유형별 도로파손 분포 현황 조회 실패',
    ),

  // 일간 도로파손 보고 현황 조회
  getDamageDailyReport: () =>
    apiCaller(
      'get',
      '/api/dashboard/daily',
      '일간 도로파손 보고 현황 조회 실패',
    ),

  // 주간 도로파손 보고 현황 조회
  getDamageWeeklyReport: () =>
    apiCaller(
      'get',
      '/api/dashboard/weekly',
      '주간 도로파손 보고 현황 조회 실패',
    ),

  // 월간 도로파손 보고 현황 조회
  getDamageMonthlyReport: () =>
    apiCaller(
      'get',
      '/api/dashboard/monthly',
      '월간 도로파손 보고 현황 조회 실패',
    ),

  // 월별 도로파손 누적 탐지 현황
  getSummarizedDamageMonthlyReport: () =>
    apiCaller(
      'get',
      '/api/dashboard/monthly-summary',
      '월별 도로파손 누적 탐지 현황 조회 실패',
    ),

  // 지역별 도로파손 분포 지도
  getLocationallyDamageMap: async (city: string) => {
    try {
      // 제네릭으로 response.data 타입 지정
      const response = await apiClient.get('/api/dashboard/districts', {
        params: { city },
      })
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
  },

  // getLocationallyDamageMap: (city: string) =>
  //   apiCaller(
  //     'get',
  //     '/api/dashboard/districts',
  //     '지역별 도로파손 분포 지도 조회 실패',
  //     undefined,
  //     { params: { city } },
  //   ),

  // 도로파손 분포 Top3 지역
  getTop3Damagedlocations: () =>
    apiCaller(
      'get',
      '/api/dashboard/top3',
      '도로파손 분포 Top3 지역 조회 실패',
    ),
}
