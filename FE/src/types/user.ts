// 유저 타입
export interface User {
    id: string,
    name: string,
    region: number // 지역에 맞춘 id가 변환되어 들어감
}

// 로그인 폼 데이터
export interface LoginFormData {
    id: string,
    password: string
}

// 회원가입 폼 데이터
export interface RegisterFormData {
    id: string,
    password: string,
    name: string,
    region: number // 지역에 맞춘 id가 변환되어 들어감
}

// 서버로부터 받는 로그인 응답 데이터
export interface LoginResonse {
    user: User
    token: string
    refreshToken: string
}

// 인증상태
export interface AuthState {
    user: User | null
    isAuthenticated: boolean
    isLoading: boolean | null
    error: string | null
}