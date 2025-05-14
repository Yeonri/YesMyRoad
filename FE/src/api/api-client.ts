// src/api/api-client.ts
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

// 직접 토큰 리프레시 함수 구현
const refreshTokens = async (): Promise<string> => {
  try {
    const refreshToken = localStorage.getItem('refreshToken')
    if (!refreshToken) {
      throw new Error('리프레시 토큰이 없습니다')
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
      localStorage.setItem('accessToken', response.data.accessToken)
    }
    if (response.data.refreshToken) {
      localStorage.setItem('refreshToken', response.data.refreshToken)
    }

    return response.data.accessToken
  } catch (error) {
    localStorage.removeItem('accessToken')
    localStorage.removeItem('refreshToken')
    throw error
  }
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
    if (config.url === '/api/users/refresh') {
      const refreshToken = localStorage.getItem('refreshToken')
      if (refreshToken) {
        config.headers['Authorization'] = `Bearer ${refreshToken}`
      }
    } else {
      const accessToken = localStorage.getItem('accessToken')
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

//  response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & {
      _retry?: boolean
    }

    // (1) 로그인·회원가입 요청은 토큰 리프레시 로직 스킵
    if (
      originalRequest.url?.endsWith('/api/users/login') ||
      originalRequest.url?.endsWith('/api/users/signup')
    ) {
      return Promise.reject(error)
    }

    // (2) 기존 401 처리 로직
    if (error.response?.status === 401 && !originalRequest._retry) {
      // 리프레시 요청 자체가 401인 경우 - 리프레시 토큰도 만료됨
      if (originalRequest.url === '/api/users/refresh') {
        localStorage.removeItem('accessToken')
        localStorage.removeItem('refreshToken')
        alert('세션이 만료되었습니다. 다시 로그인해주세요.')
        // 로그인 페이지로 리다이렉트 (선택 사항)
        // window.location.href = '/login'
        return Promise.reject(error)
      }

      // 재시도 플래그 설정
      originalRequest._retry = true

      if (!isRefreshing) {
        isRefreshing = true

        return new Promise<AxiosResponse | void>(async (resolve, reject) => {
          try {
            // 직접 구현한 리프레시 함수 호출
            const newToken = await refreshTokens()
            isRefreshing = false

            // 대기 중인 요청 처리
            processQueue(null, newToken)

            // 원래 요청 재시도
            originalRequest.headers['Authorization'] = `Bearer ${newToken}`
            resolve(apiClient(originalRequest))
          } catch (refreshError) {
            isRefreshing = false
            processQueue(
              refreshError instanceof Error
                ? refreshError
                : new Error('토큰 갱신 실패'),
              null,
            )
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
