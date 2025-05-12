import apiClient from '@/api/api-client'

export const statisticAPI = {
  // 일간 도로파손 보고 현황 조회
  getDamageDailyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/daily')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`일간 도로파손 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },

  // 주간 도로파손 보고 현황 조회
  getDamageWeeklyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/weekly')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`주간 도로파손 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },

  // 월간 도로파손 보고 현황 조회
  getDamageMonthlyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/monthly')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`월간 도로파손 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },
}
