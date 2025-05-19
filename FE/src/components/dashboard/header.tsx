'use client'

// import { coodAPI } from '@/api/coordinate-api'
import { defectAPI } from '@/api/defect-api'
// import { maintenanceAPI } from '@/api/maintenance-api'
// import { statisticAPI } from '@/api/statistic-api'
// API 테스트
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
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet'
import { Bell, Menu, Search, Settings, User } from 'lucide-react'
import { useRouter } from 'next/navigation'

import { MobileNav } from '../mobile-nav'

// const PUBLIC_ID = '23569766-0f5a-4f54-ba9b-dba4dbf0b922'

export default function Header() {
  const router = useRouter() // 라우터 훅 사용
  // const { clearSession } = useUserStore()

  const handleLogout = async () => {
    const response = await userAPI.logout()
    console.log(response)
    if (response.status === 200) {
      // 로컬 스토리지에서 인증 관련 데이터 삭제
      localStorage.removeItem('accessToken')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('userId')

      // clearSession()
      alert('로그아웃하였습니다')

      // 로그인 페이지로 리디렉션
      router.push('/auth')
    }
  }

  const defectAPITest = async () => {
    const response = await defectAPI.updateRoadDamageStatus(10, 'COMPLETED')
    if (response.status === 200) {
      console.log(response.data)
    } else {
      alert('응 안돼~')
    }
  }

  return (
    <header className="bg-background sticky top-0 z-50 flex h-16 items-center gap-4 border-b px-4 md:px-6">
      {/* 얘네 뭐하는 코드지? 주석처리해도 고장이 안나는데??? 렌더링 되는것도 전혀 없고 */}
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline" size="icon" className="md:hidden">
            <Menu className="h-5 w-5" />
            <span className="sr-only">메뉴 토글</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="pr-0">
          <MobileNav /> {/* 특히 요녀석 */}
        </SheetContent>
      </Sheet>
      <div className="flex items-center gap-2">
        <div className="hidden font-bold md:flex">YES, MY ROAD</div>
      </div>

      <div className="flex flex-1 items-center gap-4 md:gap-2 lg:gap-4">
        <div className="ml-auto flex-1 md:flex-initial">
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon" className="relative">
              <Bell className="h-4 w-4" />
              <span className="absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-600 text-[10px] font-medium text-white">
                3
              </span>
              <span className="sr-only">알림</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>알림</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem>
              심각 수준의 포트홀이 한밭대로에 발생
            </DropdownMenuItem>
            <DropdownMenuItem>
              위험 수준의 도로 균열이 동서대로에 발생
            </DropdownMenuItem>
            <DropdownMenuItem>유지관리 팀이 현충원로 작업 중</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <Button variant="outline" size="icon">
          <Settings className="h-4 w-4" />
          <span className="sr-only">설정</span>
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon">
              <User className="h-4 w-4" />
              <span className="sr-only">유저</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>내 계정</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem>프로필</DropdownMenuItem>
            <DropdownMenuItem onClick={defectAPITest}>설정</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout}>로그아웃</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
