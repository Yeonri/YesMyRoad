// src/store/user-store.ts
import { AuthState, User } from '@/types/user'
import { create } from 'zustand'
import { PersistOptions, createJSONStorage, persist } from 'zustand/middleware'

// 스토어 상태 & 액션 타입
export type UserStoreType = AuthState & {
  setAuthenticated: (isAuthenticated: boolean) => void
  setUser: (user: User | null) => void
  clearSession: () => void
  clearError: () => void
}

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
}

// Persist 미들웨어 옵션
const persistOptions: PersistOptions<
  UserStoreType,
  Pick<AuthState, 'user' | 'isAuthenticated'>
> = {
  name: 'user-storage', // 저장 key
  storage: createJSONStorage(() => localStorage), // storage 구현
  partialize: (state) => ({
    // 저장할 상태 부분 선택
    user: state.user,
    isAuthenticated: state.isAuthenticated,
  }),
}

// 스토어 생성
export const useUserStore = create<UserStoreType>()(
  persist(
    (set) => ({
      ...initialState,

      // 인증 상태 설정
      setAuthenticated: (isAuthenticated: boolean) => set({ isAuthenticated }),

      // 사용자 정보 설정
      setUser: (user: User | null) => set({ user }),

      // 세션 초기화 (로그아웃 시 사용)
      clearSession: () => set({ user: null, isAuthenticated: false }),

      // 에러 초기화
      clearError: () => set({ error: null }),
    }),
    persistOptions,
  ),
)
