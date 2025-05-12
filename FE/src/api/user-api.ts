import apiClient from '@/api/api-client'
import { LoginFormData, RegisterFormData } from '@/types/user'

export const userAPI = {
  // 로그인
  login: async (Form: LoginFormData) => {
    try {
      const response = await apiClient.post('/api/users/login', {
        id: Form.id,
        password: Form.password,
      })
      return { data: response.data, status: response.status, error: null }
    } catch (error: any) {
      console.error(`로그인에 실패하였습니다`)
      return { data: null, status: error.response.status, error }
    }
  },

  // 로그아웃
  logout: async () => {
    try {
      const response = await apiClient.post('/api/users/logout')
      return { data: response.data, status: response.status, error: null}
    } catch (error: any) {
      console.error(`로그아웃에 실패하였습니다`)
      return { data: null, status: error.response.status, error }
    }
  },

  // 회원가입
  register: async (Form: RegisterFormData) => {
    try {
      const response = await apiClient.post('/api/users/signup', {
        id: Form.id,
        password: Form.password,
        name: Form.name,
        regionId: Form.region,
      })
      return { data: response.data, status: response.status, error: null}
    } catch (error: any) {
      console.error(`회원가입에 실패하였습니다`)
      return { data: null, status: error.response.status, error }
    }
  },

  // 아이디 중복체크
  checkIdDuplication: async (userId: string) => {
    try {
      const response = await apiClient.get(`/api/users/id`, {
        params: { userId },
      })
      return { data: response.data, status: response.status, error: null}
    } catch (error: any) {
      console.error(`아이디 중복 확인에 실패하였습니다`)
      return { data: null, status: error.response.status, error }
    }
  },

  // 토큰 리프레시
  refresh: async () => {
    try {
      const response = await apiClient.post('/api/users/refresh')
      return { data: response.data, status: response.status, error: null}
    } catch (error: any) {
      console.error(`리프레시 토큰 발급에 실패하였습니다`)
      return { data: null, status: error.response.status, error }
    }
  },
}
