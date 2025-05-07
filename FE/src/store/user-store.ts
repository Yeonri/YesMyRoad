import { create } from "zustand"
import { User, AuthState } from "@/types/user"

export const useAuthStore = create<AuthState>((set) => ({
    isAuthenticated: false,
    user: null,
    token: null,
    refreshToken: null,
    isLoading: false,
    error: null,

    register: async (id: string, password: string, name: string) => {
        set({ isLoading: true, error: null })
        try {

        } catch (error) {

        }
    },
    login: async (id: string, password: string) => {
        set({ isLoading: true, error: null })
        try {

        } catch (error) {

        }
    },
    logout: () => {
        set({
            isAuthenticated: false,
            user: null,
            token: null,
            refreshToken: null,
            error: null
        })
    },
    refreshAuth: async () => {

    }
}))