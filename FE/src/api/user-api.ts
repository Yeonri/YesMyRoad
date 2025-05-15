// src/api/user-api.ts
import apiCaller from '@/api/api-utils'
import { useUserStore } from '@/store/user-store'
// 추가된 import
import { LoginFormData, RegisterFormData } from '@/types/user'

// Zustand 스토어 직접 접근 함수
const getStoreState = () => {
  return useUserStore.getState()
}

export const userAPI = {
  // 로그인
  login: (Form: LoginFormData) =>
    apiCaller('post', '/api/users/login', {
      id: Form.id,
      password: Form.password,
    }),

  // 로그아웃 - 개선된 버전
  logout: async () => {
    const result = await apiCaller('post', '/api/users/logout')

    // 서버 응답 성공 여부와 관계없이 클라이언트 측 로그아웃 처리
    localStorage.removeItem('accessToken')
    localStorage.removeItem('refreshToken')
    localStorage.removeItem('userId')

    // Zustand 스토어 업데이트
    const { clearSession } = getStoreState()
    clearSession()

    return result
  },

  // 회원가입
  register: (Form: RegisterFormData) =>
    apiCaller('post', '/api/users/signup', {
      id: Form.id,
      password: Form.password,
      name: Form.name,
      regionId: Form.region,
    }),

  // 아이디 중복체크
  checkIdDuplication: (userId: string) =>
    apiCaller('get', '/api/users/id', { params: { userId } }),

  // 토큰 리프레시 - 개선된 버전
  refresh: async () => {
    try {
      const result = await apiCaller('post', '/api/users/refresh')

      if (result.status === 200 && result.data) {
        localStorage.setItem('accessToken', result.data.accessToken)
        localStorage.setItem('refreshToken', result.data.refreshToken)
        return result
      }

      // 실패 시 로그아웃 처리 (선택적)
      if (result.status !== 200) {
        const { clearSession } = getStoreState()
        clearSession()
        localStorage.removeItem('accessToken')
        localStorage.removeItem('refreshToken')
        localStorage.removeItem('userId')
      }

      return result
    } catch (error) {
      // 오류 발생 시 로그아웃 처리
      const { clearSession } = getStoreState()
      clearSession()
      localStorage.removeItem('accessToken')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('userId')
      throw error
    }
  },
}
