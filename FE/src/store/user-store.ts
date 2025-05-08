// src/store/user-store.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { AuthState, LoginFormData, RegisterFormData } from '@/types/user'
import { userAPI } from '@/api/user-api'

// 초기 상태
const initialState: AuthState = {
    user: null,
    isAuthenticated: false,
    isLoading: false,
    error: null
}

// 토큰 관리 유틸리티
const tokenUtils = {
    set: (token: string, refreshToken: string) => {
        localStorage.setItem('token', token)
        localStorage.setItem('refreshToken', refreshToken)
    },
    remove: () => {
        localStorage.removeItem('token')
        localStorage.removeItem('refreshToken')
    }
}

interface UserStore extends AuthState {
    login: (form: LoginFormData) => Promise<void>
    register: (form: RegisterFormData) => Promise<void>
    logout: () => Promise<void>
    refreshToken: () => Promise<void>
    clearError: () => void
}

export const useUserStore = create<UserStore>()(
    persist(
        (set) => {
            // 비동기 액션 처리를 위한 헬퍼 함수
            const handleAsync = async (
                apiCall: () => Promise<any>,
                onSuccess: (data: any) => Partial<AuthState>,
                onError?: (error: any) => Partial<AuthState>
            ) => {
                try {
                    set({ ...initialState, isLoading: true })
                    const { data } = await apiCall()
                    set(onSuccess(data))
                    return data
                } catch (error: any) {
                    set({
                        ...initialState,
                        error: error.message || '오류가 발생했습니다',
                        isLoading: false
                    })
                    if (onError) set(onError(error))
                    throw error
                }
            }

            return {
                ...initialState,

                // 로그인
                login: async (form) => {
                    await handleAsync(
                        () => userAPI.login(form),
                        (data) => {
                            tokenUtils.set(data.token, data.refreshToken)
                            return {
                                user: data.user,
                                isAuthenticated: true,
                                isLoading: false
                            }
                        }
                    )
                },

                // 회원가입
                register: async (form) => {
                    await handleAsync(
                        () => userAPI.register(form),
                        () => ({ isLoading: false })
                    )
                },

                // 로그아웃
                logout: async () => {
                    await handleAsync(
                        () => userAPI.logout(),
                        () => {
                            tokenUtils.remove()
                            return {
                                ...initialState,
                                isLoading: false
                            }
                        },
                        () => {
                            // 로그아웃 실패 시에도 클라이언트에서는 로그아웃 처리
                            tokenUtils.remove()
                            return {
                                ...initialState,
                                isLoading: false
                            }
                        }
                    )
                },

                // 토큰 갱신
                refreshToken: async () => {
                    await handleAsync(
                        () => userAPI.refresh(),
                        (data) => {
                            if (data.token) {
                                tokenUtils.set(data.token, data.refreshToken || localStorage.getItem('refreshToken') || '')
                            }
                            return {
                                user: data.user,
                                isAuthenticated: true,
                                isLoading: false
                            }
                        },
                        () => {
                            tokenUtils.remove()
                            return {
                                ...initialState,
                                error: '인증이 만료되었습니다',
                                isLoading: false
                            }
                        }
                    )
                },

                // 에러 상태 초기화
                clearError: () => set({ error: null })
            }
        },
        {
            name: 'user-storage',
            partialize: (state) => ({
                user: state.user,
                isAuthenticated: state.isAuthenticated
            })
        }
    )
)