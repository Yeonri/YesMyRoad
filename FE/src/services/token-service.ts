// src/services/token-service.ts
import { useUserStore } from '@/store/user-store'
import axios from 'axios'

const baseURL = 'https://k12b201.p.ssafy.io'

// 토큰 관리를 위한 중앙화된 서비스
const TokenService = {
  // 토큰 저장
  setTokens: (accessToken: string, refreshToken: string): void => {
    localStorage.setItem('accessToken', accessToken)
    localStorage.setItem('refreshToken', refreshToken)
  },

  // 토큰 가져오기
  getAccessToken: (): string | null => {
    return localStorage.getItem('accessToken')
  },

  getRefreshToken: (): string | null => {
    return localStorage.getItem('refreshToken')
  },

  // 토큰 존재 여부 확인
  hasTokens: (): boolean => {
    return (
      !!localStorage.getItem('accessToken') &&
      !!localStorage.getItem('refreshToken')
    )
  },

  // 토큰 제거 (로그아웃)
  clearTokens: (): void => {
    localStorage.removeItem('accessToken')
    localStorage.removeItem('refreshToken')
    localStorage.removeItem('userId')
  },

  // 토큰 갱신
  refreshTokens: async (): Promise<boolean> => {
    try {
      const refreshToken = TokenService.getRefreshToken()
      if (!refreshToken) {
        return false
      }

      const response = await axios.post(
        `${baseURL}/api/users/refresh`,
        {},
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${refreshToken}`,
          },
        },
      )

      if (response.data.accessToken) {
        TokenService.setTokens(
          response.data.accessToken,
          response.data.refreshToken || refreshToken,
        )
        return true
      }
      return false
    } catch (error) {
      // 토큰 갱신 실패 시 모든 토큰 제거 및 사용자 정보 초기화
      TokenService.clearTokens()
      const { clearUser } = useUserStore.getState()
      clearUser()
      console.error(error)
      return false
    }
  },

  // 토큰 유효성 검증 (페이지 로드 시 호출)
  validateToken: async (): Promise<boolean> => {
    if (!TokenService.hasTokens()) {
      return false
    }

    try {
      // 토큰 유효성을 검증하는 간단한 API 호출
      // 여기서는 refresh API를 호출해 토큰이 유효한지 확인
      return await TokenService.refreshTokens()
    } catch (error) {
      console.error(error)
      return false
    }
  },
}

export default TokenService
