// src/api/user-api.ts
import apiCaller from '@/api/api-utils'
import TokenService from '@/services/token-service'
import { useUserStore } from '@/store/user-store'
import { LoginFormData, RegisterFormData } from '@/types/user'

// 스토어 접근 함수
const getStoreState = () => {
  return useUserStore.getState()
}

export const userAPI = {
  // 로그인
  login: async (Form: LoginFormData) => {
    const response = await apiCaller('post', '/api/users/login', {
      id: Form.id,
      password: Form.password,
    })

    // 로그인 성공 시 토큰 저장 및 사용자 정보 설정
    if (response.status === 200 && response.data) {
      TokenService.setTokens(
        response.data.accessToken,
        response.data.refreshToken,
      )
      const { setUser } = getStoreState()
      setUser(response.data.user)
    }

    return response
  },

  // 로그아웃
  logout: async () => {
    const result = await apiCaller('post', '/api/users/logout')

    // 서버 응답과 관계없이 클라이언트 측 로그아웃 처리
    TokenService.clearTokens()
    const { clearUser } = getStoreState()
    clearUser()

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
    apiCaller('get', '/api/users/id', undefined, { params: { userId } }),

  // 토큰 유효성 검증 및 갱신
  validateSession: async () => {
    return await TokenService.validateToken()
  },
}
