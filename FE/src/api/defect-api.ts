import apiClient from '@/api/api-client'

export const defectAPI = {
  checkDefects: async () => {
    try {
      const response = await apiClient.get('/api/capture-points')
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`손상 데이터 조회 실패: ${error}`)
      throw error
    }
  },

  checkDetailedDefects: async (publicId: string) => {
    try {
      const response = await apiClient.get(`/api/capture-points/${publicId}`)
      return { data: response.data, status: response.status }
    } catch (error) {
      console.error(`손상 데이터 상세 조회 실패: ${error}`)
      throw error
    }
  },
}
