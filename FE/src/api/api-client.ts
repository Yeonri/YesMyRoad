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

// request interceptor
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  (error: AxiosError): Promise<AxiosError> => {
    return Promise.reject(error)
  },
)

// response interceptor
apiClient.interceptors.response.use(
  (response: AxiosResponse): AxiosResponse => response,
  (error: AxiosError): Promise<AxiosError> => {
    if (error.response) {
      switch (error.response.status) {
        case 401: // 인증 만료
          localStorage.removeItem('token')
          alert('인증이 만료되었습니다. 다시 로그인해주세요.')
          break
        case 403: // 비인가
          alert('접근 권한이 없습니다')
          break
        case 404: // 리소스 없음
          localStorage.removeItem('token')
          alert('요청한 리소스를 찾을 수 없습니다')
          break
        case 500: // 서버 오류
          localStorage.removeItem('token')
          alert('서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요')
          break
        default: // 그 외 오류
          alert(`${error.response.status}: 알 수 없는 오류가 발생했습니다`)
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
