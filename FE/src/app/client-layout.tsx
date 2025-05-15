// src/app/client-layout.tsx
'use client'

import { userAPI } from '@/api/user-api'
import { useUserStore } from '@/store/user-store'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'

// src/app/client-layout.tsx

// src/app/client-layout.tsx

// 보호할 경로 목록
const protectedRoutes = ['/']
// 인증된 사용자가 접근하면 리다이렉션할 경로
const authRoutes = ['/auth']

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { isAuthenticated, clearSession } = useUserStore()
  const pathname = usePathname()
  const router = useRouter()
  const [isChecking, setIsChecking] = useState(true)

  // 앱 초기화 시 토큰 유효성 검증
  useEffect(() => {
    const validateTokens = async () => {
      setIsChecking(true)
      console.log('[ClientLayout] 토큰 유효성 검증 시작')

      const accessToken = localStorage.getItem('accessToken')
      const refreshToken = localStorage.getItem('refreshToken')

      // 토큰이 없거나 isAuthenticated가 false인 경우 인증 상태 초기화
      if ((!accessToken || !refreshToken) && isAuthenticated) {
        console.log(
          '[ClientLayout] 토큰이 없지만 인증상태가 true: 인증상태 초기화',
        )
        clearSession()
        setIsChecking(false)
        return
      }

      // 토큰이 있고 isAuthenticated가 true인 경우 토큰 유효성 검증
      if (accessToken && refreshToken && isAuthenticated) {
        try {
          // 가벼운 API 호출로 토큰 유효성 확인
          const response = await userAPI.refresh()

          if (response.status !== 200) {
            console.log('[ClientLayout] 토큰 유효성 검증 실패: 인증상태 초기화')
            // 유효하지 않은 토큰이면 세션 초기화
            clearSession()
            localStorage.removeItem('accessToken')
            localStorage.removeItem('refreshToken')
            localStorage.removeItem('userId')

            // 보호된 경로에 있으면 로그인 페이지로 리다이렉션
            if (protectedRoutes.includes(pathname)) {
              router.push('/auth')
            }
          } else {
            console.log('[ClientLayout] 토큰 유효성 검증 성공')
          }
        } catch (error) {
          console.error('[ClientLayout] 토큰 검증 중 오류:', error)
          clearSession()
          localStorage.removeItem('accessToken')
          localStorage.removeItem('refreshToken')
          localStorage.removeItem('userId')

          // 보호된 경로에 있으면 로그인 페이지로 리다이렉션
          if (protectedRoutes.includes(pathname)) {
            router.push('/auth')
          }
        }
      }

      setIsChecking(false)
    }

    validateTokens()
  }, []) // 앱 초기화 시 한 번만 실행

  // 기존 인증 상태 변경 시 리다이렉션 로직
  useEffect(() => {
    if (!isChecking) {
      if (protectedRoutes.includes(pathname) && !isAuthenticated) {
        console.log(
          '[인증 상태 변경] 인증되지 않은 사용자가 보호된 경로에 접근: 로그인 페이지로 리다이렉션',
        )
        router.push('/auth')
      } else if (authRoutes.includes(pathname) && isAuthenticated) {
        console.log(
          '[인증 상태 변경] 인증된 사용자가 로그인 페이지에 접근: 메인 페이지로 리다이렉션',
        )
        router.push('/')
      }
    }
  }, [isAuthenticated, pathname, router, isChecking])

  // 로딩 상태 추가
  if (isChecking) {
    return <div>인증 상태 확인 중...</div>
  }

  return <>{children}</>
}
