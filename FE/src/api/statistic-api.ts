import apiClient from '@/api/api-client'

export const statisticAPI = {
  // 일간 도로파손 보고 현황 조회
  getDamageDailyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/damage/daily')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`일간 도로파손 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },

  // 주간 도로파손 보고 현황 조회
  getDamageWeeklyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/damage/weekly')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`주간 도로파손 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },

  // 월간 도로파손 보고 현황 조회
  getDamageMonthlyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/damage/monthly')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`월간 도로파손 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },

  // 일간 도로보수 보고 현황 조회
  getRepairDailyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/repair/daily')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`일간 도로보수 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },

  // 주간 도로보수 보고 현황 조회
  getRepairWeeklyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/repair/weekly')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`주간 도로보수 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },

  // 월간 도로보수 보고 현황 조회
  getRepairMonthlyReport: async () => {
    try {
      const response = await apiClient.get('/api/dashboard/repair/monthly')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`월간 도로보수 보고 현황 조회 실패: ${error}`)
      throw error
    }
  },
}
