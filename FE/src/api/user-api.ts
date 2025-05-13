// src/api/user-api.ts
import apiCaller from '@/api/api-utils'
import { LoginFormData, RegisterFormData } from '@/types/user'

export const userAPI = {
  // 로그인
  login: (Form: LoginFormData) =>
    apiCaller('post', '/api/users/login', '로그인에 실패하였습니다', {
      id: Form.id,
      password: Form.password,
    }),

  // 로그아웃
  logout: () =>
    apiCaller('post', '/api/users/logout', '로그아웃에 실패하였습니다'),

  // 회원가입
  register: (Form: RegisterFormData) =>
    apiCaller('post', '/api/users/signup', '회원가입에 실패하였습니다', {
      id: Form.id,
      password: Form.password,
      name: Form.name,
      regionId: Form.region,
    }),

  // 아이디 중복체크
  checkIdDuplication: (userId: string) =>
    apiCaller(
      'get',
      '/api/users/id',
      '아이디 중복 확인에 실패하였습니다',
      undefined,
      { params: { userId } },
    ),

  // 토큰 리프레시
  refresh: async () => {
    const result = await apiCaller(
      'post',
      '/api/users/refresh',
      '리프레시 토큰 발급에 실패하였습니다',
    )

    if (result.status === 200 && result.data) {
      localStorage.setItem('accessToken', result.data.accessToken)
      localStorage.setItem('refreshToken', result.data.refreshToken)
    }

    return result
  },
}
