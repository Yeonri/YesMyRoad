import { userAPI } from '@/api/user-api'
import {
  AuthState,
  LoginFormData,
  LoginResponse,
  RegisterFormData,
} from '@/types/user'
import { create } from 'zustand'
import { PersistOptions, createJSONStorage, persist } from 'zustand/middleware'

// --- API 응답/에러 타입 ---
type ApiResponse<T> = { data: T }
interface ApiError extends Error {
  message: string
}

// --- 스토어 상태 & 액션 타입 ---
export type UserStoreType = AuthState & {
  login: (form: LoginFormData) => Promise<void>
  register: (form: RegisterFormData) => Promise<void>
  logout: () => Promise<void>
  refreshToken: () => Promise<void>
  clearError: () => void
}

// --- 초기 상태 ---
const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
}

// --- 토큰 관리 유틸 ---
const tokenUtils = {
  set: (token: string, refreshToken: string) => {
    localStorage.setItem('token', token)
    localStorage.setItem('refreshToken', refreshToken)
  },
  remove: () => {
    localStorage.removeItem('token')
    localStorage.removeItem('refreshToken')
  },
}

// --- Persist 미들웨어 옵션 ---
const persistOptions: PersistOptions<
  UserStoreType,
  Pick<AuthState, 'user' | 'isAuthenticated'>
> = {
  name: 'user-storage', // 저장 key (필수) :contentReference[oaicite:4]{index=4}
  storage: createJSONStorage(() => localStorage), // storage 구현 :contentReference[oaicite:5]{index=5}
  partialize: (state) => ({
    // 저장할 상태 부분 선택 (선택)
    user: state.user,
    isAuthenticated: state.isAuthenticated,
  }),
}

// --- 스토어 생성: create<>()(persist(…)) 패턴 ---
export const useUserStore = create<UserStoreType>()(
  persist((set) => {
    // 공통 비동기 헬퍼
    const handleAsync = async <T>(
      apiCall: () => Promise<ApiResponse<T>>,
      onSuccess: (data: T) => Partial<AuthState>,
      onError?: (err: ApiError) => Partial<AuthState>,
    ): Promise<T> => {
      set({ ...initialState, isLoading: true })
      try {
        const { data } = await apiCall()
        set(onSuccess(data))
        return data
      } catch (e) {
        const err = e as ApiError
        set({
          ...initialState,
          isLoading: false,
          error: err.message ?? '오류가 발생했습니다',
        })
        if (onError) set(onError(err))
        throw err
      }
    }

    return {
      ...initialState,

      login: async (form) => {
        await handleAsync<LoginResponse>(
          () => userAPI.login(form),
          (data) => {
            tokenUtils.set(data.token, data.refreshToken)
            return { user: data.user, isAuthenticated: true, isLoading: false }
          },
        )
      },

      register: async (form) => {
        await handleAsync<void>(
          () => userAPI.register(form),
          () => ({ isLoading: false }),
        )
      },

      logout: async () => {
        await handleAsync<void>(
          () => userAPI.logout(),
          () => {
            tokenUtils.remove()
            return { ...initialState, isLoading: false }
          },
          () => {
            tokenUtils.remove()
            return { ...initialState, isLoading: false }
          },
        )
      },

      refreshToken: async () => {
        await handleAsync<Partial<LoginResponse>>(
          () => userAPI.refresh(),
          (data) => {
            if (data.token) {
              tokenUtils.set(
                data.token,
                data.refreshToken ?? localStorage.getItem('refreshToken') ?? '',
              )
            }
            return {
              user: data.user ?? null,
              isAuthenticated: !!data.user,
              isLoading: false,
            }
          },
          () => {
            tokenUtils.remove()
            return {
              ...initialState,
              error: '인증이 만료되었습니다',
              isLoading: false,
            }
          },
        )
      },

      clearError: () => set({ error: null }),
    }
  }, persistOptions),
)
