// src/api/api-client.ts
import TokenService from '@/services/token-service'
import { useUserStore } from '@/store/user-store'
import axios, {
  AxiosError,
  AxiosInstance,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from 'axios'

const baseURL = 'https://k12b201.p.ssafy.io'

const apiClient: AxiosInstance = axios.create({
  baseURL,
  headers: { 'Content-Type': 'application/json' },
})

// 토큰 리프레시 진행 여부 관련 변수들
let isRefreshing = false
interface QueueItem {
  resolve: (value: string | PromiseLike<string>) => void
  reject: (reason: unknown) => void
}
let refreshQueue: QueueItem[] = []

// 스토어 접근 함수
const getStoreState = () => {
  return useUserStore.getState()
}

// 로그아웃 처리 함수
const handleLogout = () => {
  // 토큰 제거
  TokenService.clearTokens()

  // 사용자 정보 초기화
  const { clearUser } = getStoreState()
  clearUser()

  // 로그인 페이지로 리다이렉션
  window.location.href = '/auth'
}

// 대기 중인 요청 처리 함수
const processQueue = (
  error: Error | null,
  token: string | null = null,
): void => {
  refreshQueue.forEach((promise) => {
    if (error) {
      promise.reject(error)
    } else {
      promise.resolve(token as string)
    }
  })

  refreshQueue = []
}

// request interceptor
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    if (config.url?.endsWith('/api/users/refresh')) {
      const refreshToken = TokenService.getRefreshToken()
      if (refreshToken) {
        config.headers['Authorization'] = `Bearer ${refreshToken}`
      }
    } else {
      const accessToken = TokenService.getAccessToken()
      if (accessToken) {
        config.headers['Authorization'] = `Bearer ${accessToken}`
      }
    }
    return config
  },
  (error: AxiosError): Promise<AxiosError> => {
    return Promise.reject(error)
  },
)

// response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean
    }

    // 로그인·회원가입 요청은 토큰 리프레시 로직 스킵
    if (
      originalRequest.url?.endsWith('/api/users/login') ||
      originalRequest.url?.endsWith('/api/users/signup')
    ) {
      return Promise.reject(error)
    }

    // 401 처리 로직
    if (error.response?.status === 401 && !originalRequest._retry) {
      // 리프레시 요청 자체가 401인 경우 - 리프레시 토큰도 만료됨
      if (originalRequest.url?.endsWith('/api/users/refresh')) {
        handleLogout()
        alert('세션이 만료되었습니다. 다시 로그인해주세요.')
        return Promise.reject(error)
      }

      // 재시도 플래그 설정
      originalRequest._retry = true

      if (!isRefreshing) {
        isRefreshing = true

        return new Promise<AxiosResponse | void>(async (resolve, reject) => {
          try {
            // 토큰 갱신 시도
            const refreshSuccess = await TokenService.refreshTokens()
            isRefreshing = false

            if (refreshSuccess) {
              // 대기 중인 요청 처리
              processQueue(null, TokenService.getAccessToken())

              // 원래 요청 재시도
              originalRequest.headers['Authorization'] =
                `Bearer ${TokenService.getAccessToken()}`
              resolve(apiClient(originalRequest))
            } else {
              processQueue(new Error('토큰 갱신 실패'), null)
              reject(error)
              // alert('세션이 만료되었습니다. 다시 로그인해주세요.')
              handleLogout()
            }
          } catch (refreshError) {
            isRefreshing = false
            processQueue(
              refreshError instanceof Error
                ? refreshError
                : new Error('토큰 갱신 실패'),
              null,
            )
            // alert('세션이 만료되었습니다. 다시 로그인해주세요.')
            handleLogout()
            reject(refreshError)
          }
        })
      } else {
        // 이미 리프레시 진행 중이면 대기열에 추가
        return new Promise<string>((resolve, reject) => {
          refreshQueue.push({ resolve, reject })
        })
          .then((token) => {
            originalRequest.headers['Authorization'] = `Bearer ${token}`
            return apiClient(originalRequest)
          })
          .catch((err) => {
            return Promise.reject(err)
          })
      }
    }

    // 기타 에러 처리
    if (error.response) {
      switch (error.response.status) {
        case 403: // 비인가
          alert('접근 권한이 없습니다')
          break
        case 404: // 리소스 없음
          alert('요청한 리소스를 찾을 수 없습니다')
          break
        case 500: // 서버 오류
          alert('서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요')
          break
        default: // 그 외 오류
          if (error.response.status !== 401) {
            // 401은 위에서 처리함
            alert(`${error.response.status}: 알 수 없는 오류가 발생했습니다`)
          }
      }
    } else if (error.request) {
      alert('서버에 연결할 수 없습니다')
    } else {
      alert('요청 설정 중 오류가 발생했습니다')
    }

    return Promise.reject(error)
  },
)

export default apiClient
