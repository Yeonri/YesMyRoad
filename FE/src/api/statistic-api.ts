import apiClient from '@/api/api-client'

export const statisticAPI = {
  // 일간 도로파손 보고 현황 조회
  getDamageDailyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/daily')
      return { data: response.data, status: response.status, error: null}
    } catch (error: any) {
      console.error(`일간 도로파손 보고 현황 조회 실패: ${error}`)
      return { data: null, status: error.response.status, error }
    }
  },

  // 주간 도로파손 보고 현황 조회
  getDamageWeeklyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/weekly')
      return { data: response.data, status: response.status, error: null}
    } catch (error: any) {
      console.error(`주간 도로파손 보고 현황 조회 실패: ${error}`)
      return { data: null, status: error.response.status, error }
    }
  },

  // 월간 도로파손 보고 현황 조회
  getDamageMonthlyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/monthly')
      return { data: response.data, status: response.status, error: null}
    } catch (error: any) {
      console.error(`월간 도로파손 보고 현황 조회 실패: ${error}`)
      return { data: null, status: error.response.status, error }
    }
  },
}
