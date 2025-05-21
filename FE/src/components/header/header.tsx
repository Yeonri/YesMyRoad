'use client'

import { userAPI } from '@/api/user-api'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { User } from 'lucide-react'
import { useRouter } from 'next/navigation'

export default function Header() {
  const router = useRouter() // 라우터 훅 사용
  const handleLogout = async () => {
    const response = await userAPI.logout()
    // console.log(response)
    if (response.status === 200) {
      // 로컬 스토리지에서 인증 관련 데이터 삭제
      localStorage.removeItem('accessToken')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('userId')

      // clearSession()
      // alert('로그아웃하였습니다')

      // 로그인 페이지로 리디렉션
      router.push('/auth')
    }
  }

  return (
    <header className="bg-background sticky top-0 z-50 flex h-16 items-center gap-4 border-b px-4 md:px-6">
      <div className="flex items-center gap-2">
        <div className="hidden font-bold md:flex select-none">YES, MY ROAD</div>
      </div>

      <div className="flex flex-1 items-center gap-4 md:gap-2 lg:gap-4">
        <div className="ml-auto flex-1 md:flex-initial"></div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon">
              <User className="h-4 w-4" />
              <span className="sr-only">유저</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>계정 관리</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout}>로그아웃</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
