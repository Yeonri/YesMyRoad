'use client'

import { defectAPI } from '@/api/defect-api'
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
import { Input } from '@/components/ui/input'
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet'
import { Bell, Menu, Search, Settings, User } from 'lucide-react'
import { useRouter } from 'next/navigation'

import { MobileNav } from './mobile-nav'

const PUBLIC_ID = '23569766-0f5a-4f54-ba9b-dba4dbf0b922'

export default function Header() {
  const router = useRouter() // 라우터 훅 사용
  const handleLogout = async () => {
    const response = await userAPI.logout()
    if (response.status === 200) {
      // 로컬 스토리지에서 인증 관련 데이터 삭제
      localStorage.removeItem('token')
      localStorage.removeItem('refreshToken')
      localStorage.removeItem('userId')

      // console.log('로그아웃 성공')

      // 로그인 페이지로 리디렉션
      router.push('/login')
    }
  }

  return (
    <header className="bg-background sticky top-0 z-50 flex h-16 items-center gap-4 border-b px-4 md:px-6">
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline" size="icon" className="md:hidden">
            <Menu className="h-5 w-5" />
            <span className="sr-only">메뉴 토글</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="pr-0">
          <MobileNav />
        </SheetContent>
      </Sheet>
      <div className="flex items-center gap-2">
        <div className="hidden font-bold md:flex">YES, MY ROAD</div>
      </div>

      {/* API 연동 테스트용 버튼 */}
      <Button
        onClick={async () => {
          const response = await defectAPI.checkDefects()
          console.log(response.data)
        }}
      >
        손상 데이터 조회 테스트
      </Button>
      <Button
        onClick={async () => {
          const response = await defectAPI.checkDetailedDefects(PUBLIC_ID)
          console.log(response.data)
        }}
      >
        손상 데이터 상세 조회 테스트
      </Button>
      {/* API 연동 테스트용 버튼 */}

      <div className="flex flex-1 items-center gap-4 md:gap-2 lg:gap-4">
        <form className="ml-auto flex-1 md:flex-initial">
          <div className="relative">
            <Search className="text-muted-foreground absolute top-2.5 left-2.5 h-4 w-4" />
            <Input
              type="search"
              placeholder="결함 검색"
              className="bg-background w-full rounded-lg pl-8 md:w-[200px] lg:w-[300px]"
            />
          </div>
        </form>
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
            <DropdownMenuItem>설정</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout}>로그아웃</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  )
}
