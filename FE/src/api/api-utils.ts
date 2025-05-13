// src/utils/api-utils.ts
import apiClient from '@/api/api-client'
import axios, { AxiosRequestConfig } from 'axios'

type HttpMethod = 'get' | 'post' | 'put' | 'delete' | 'patch'

const apiCaller = async (
  method: HttpMethod,
  url: string,
  errorMessage: string,
  data?: unknown,
  config?: AxiosRequestConfig,
) => {
  try {
    let response

    switch (method) {
      case 'get':
        response = await apiClient.get(url, config)
        break
      case 'post':
        response = await apiClient.post(url, data, config)
        break
      case 'put':
        response = await apiClient.put(url, data, config)
        break
      case 'delete':
        response = await apiClient.delete(url, config)
        break
      case 'patch':
        response = await apiClient.patch(url, data, config)
        break
    }

    return { data: response.data, status: response.status, error: null }
  } catch (error) {
    if (axios.isAxiosError(error)) {
      // console.error(`${errorMessage}: ${error.message}`)
      // 로그인 endpoint는 401일 때 throw
      if (url.endsWith('/api/users/login')) {
        throw error
      }
      return { data: null, status: error.response?.status || null, error }
    }
    console.error(`알 수 없는 오류: ${error}`)
    throw error
  }
}

export default apiCaller
