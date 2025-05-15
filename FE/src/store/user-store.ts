// src/store/user-store.ts
import { User } from '@/types/user'
import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

// 스토어 상태 & 액션 타입
export type UserStoreType = {
  user: User | null
  error: string | null
  setUser: (user: User | null) => void
  clearUser: () => void
  clearError: () => void
}

const initialState = { user: null, error: null }

// 스토어 생성
export const useUserStore = create<UserStoreType>()(
  persist(
    (set) => ({
      ...initialState,

      // 사용자 정보 설정
      setUser: (user: User | null) => set({ user }),

      // 사용자 정보 초기화 (로그아웃 시 사용)
      clearUser: () => set({ user: null }),

      // 에러 초기화
      clearError: () => set({ error: null }),
    }),
    {
      name: 'user-storage', // 저장 key
      storage: createJSONStorage(() => localStorage), // storage 구현
      partialize: (state) => ({
        // 저장할 상태 부분 선택
        user: state.user,
      }),
    },
  ),
)
