'use client'

import { userAPI } from '@/api/user-api'
import TokenService from '@/services/token-service'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'

// 보호할 경로 목록
const protectedRoutes = ['/']
// 인증된 사용자가 접근하면 리다이렉션할 경로
const authRoutes = ['/auth']

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  const router = useRouter()
  const [isChecking, setIsChecking] = useState(true)

  // QueryClient 인스턴스 생성 - useState를 사용하여 컴포넌트 리렌더링시에도 인스턴스가 유지되도록 함
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5 * 60 * 1000, // 5분
            gcTime: 10 * 60 * 1000, // 10분
            retry: 1, // 실패 시 1번 재시도
          },
        },
      }),
  )

  // 앱 초기화 시 세션 유효성 검증
  useEffect(() => {
    const validateSession = async () => {
      console.log('[ClientLayout] 마운트됨')
      console.log('[ClientLayout] 현재 경로:', pathname)
      setIsChecking(true)

      // 토큰이 있는지 확인
      const hasTokens = TokenService.hasTokens()
      console.log('[ClientLayout] 토큰 존재 여부:', hasTokens)

      if (hasTokens) {
        // 토큰 유효성 검증 및 필요시 갱신
        const isValid = await userAPI.validateSession()
        console.log('[ClientLayout] 토큰 유효성:', isValid)

        // 토큰이 유효하지 않으면 로그인 페이지로
        if (!isValid && protectedRoutes.includes(pathname)) {
          console.log(
            '[인증 보호] 세션이 유효하지 않음: 로그인 페이지로 리다이렉션',
          )
          router.push('/auth')
        }
        // 토큰이 유효하고 로그인 페이지에 접근하면 메인으로
        else if (isValid && authRoutes.includes(pathname)) {
          console.log(
            '[인증 보호] 인증된 사용자가 로그인 페이지에 접근: 메인 페이지로 리다이렉션',
          )
          router.push('/')
        }
      }
      // 토큰이 없고 보호된 경로에 접근하면 로그인 페이지로
      else if (protectedRoutes.includes(pathname)) {
        console.log(
          '[인증 보호] 인증되지 않은 사용자가 보호된 경로에 접근: 로그인 페이지로 리다이렉션',
        )
        router.push('/auth')
      }

      setIsChecking(false)
    }

    validateSession()
  }, [pathname, router])

  // 로딩 상태
  if (isChecking) {
    return <div>인증 상태 확인 중...</div>
  }

  // QueryClientProvider로 children 감싸기
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <ReactQueryDevtools initialIsOpen={false} /> {/* 개발 도구 추가 */}
    </QueryClientProvider>
  )
}
