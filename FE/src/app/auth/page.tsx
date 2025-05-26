'use client'

import background from '@/assets/images/background.png'
import AuthForm from '@/components/auth/auth-form'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import Image from 'next/image'
import { useState } from 'react'

export default function AuthPage() {
  // 로그인/회원가입 탭 상태 관리
  const [isLoginTab, setIsLoginTab] = useState<boolean>(true)
  const [backgroundError, setBackgroundError] = useState(false)

  return (
    <>
      <header className="bg-background sticky top-0 z-50 flex h-16 items-center gap-4 border-b px-4 md:px-6">
        <div className="hidden font-bold select-none md:flex">YES, MY ROAD</div>
      </header>
      <div className="relative h-[calc(100vh-4rem)] w-full select-none">
        <Image
          src={
            backgroundError ? '/assets/images/image-not-found.png' : background
          }
          alt="로그인 배경"
          fill
          style={{ objectFit: 'cover' }}
          priority
          className="z-0"
          onError={() => setBackgroundError(true)}
        />
        <div className="bg-white-500 absolute inset-0 z-10 flex items-center justify-center">
          <Tabs
            value={isLoginTab ? 'login' : 'register'}
            onValueChange={(value: string) => {
              setIsLoginTab(value === 'login')
              // 'login'이면 로그인, 'register'면 회원가입
            }}
            className="w-[400px] select-none"
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="login">로그인</TabsTrigger>
              <TabsTrigger value="register">회원가입</TabsTrigger>
            </TabsList>

            <Card className="mt-4">
              <CardHeader>
                <CardTitle>{isLoginTab ? '로그인' : '회원가입'}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* AuthForm 컴포넌트에 isLoginTab과 탭 변경 핸들러 전달 */}
                <AuthForm isLoginTab={isLoginTab} onTabChange={setIsLoginTab} />
              </CardContent>
            </Card>
          </Tabs>
        </div>
      </div>
    </>
  )
}
