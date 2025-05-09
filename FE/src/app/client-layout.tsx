'use client'

import { useUserStore } from '@/store/user-store'
import { usePathname, useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'

// 보호할 경로 목록
const protectedRoutes = ['/']
// 인증된 사용자가 접근하면 리다이렉션할 경로
const authRoutes = ['/login']

export default function ClientLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const { isAuthenticated } = useUserStore()
  const pathname = usePathname()
  const router = useRouter()
  const [isChecking, setIsChecking] = useState(true)

  // 디버깅용 로그 추가
  useEffect(() => {
    console.log('[ClientLayout] 마운트됨')
    console.log('[ClientLayout] 현재 경로:', pathname)
    console.log('[ClientLayout] 인증 상태:', isAuthenticated)

    // 초기 인증 상태 확인 후 라우팅 결정
    if (protectedRoutes.includes(pathname) && !isAuthenticated) {
      console.log(
        '[인증 보호] 인증되지 않은 사용자가 보호된 경로에 접근: 로그인 페이지로 리다이렉션',
      )
      router.push('/login')
    } else if (authRoutes.includes(pathname) && isAuthenticated) {
      console.log(
        '[인증 보호] 인증된 사용자가 로그인 페이지에 접근: 메인 페이지로 리다이렉션',
      )
      router.push('/')
    }

    setIsChecking(false)
  }, [pathname, isAuthenticated, router])

  // 인증 상태 변경 시 리다이렉션
  useEffect(() => {
    if (!isChecking) {
      if (protectedRoutes.includes(pathname) && !isAuthenticated) {
        console.log(
          '[인증 상태 변경] 인증되지 않은 사용자가 보호된 경로에 접근: 로그인 페이지로 리다이렉션',
        )
        router.push('/login')
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
    // return <></>
    return <div>인증 상태 확인 중...</div>
  }

  return <>{children}</>
}
