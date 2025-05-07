export interface User {
    id: string,
    name: string
}

export interface AuthState {
    isAuthenticated: boolean
    user: User | null
    token: string | null
    refreshToken: string | null
    isLoading: boolean | null
    error: string | null

    register: (id: string, password: string, name: string) => void
    login: (id: string, password: string) => Promise<void>
    logout: () => void
    refreshAuth: () => Promise<void>
}