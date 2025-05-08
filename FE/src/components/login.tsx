'use client'

import background from '@/assets/background.png'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import * as AccordionPrimitive from '@radix-ui/react-accordion'
import Image from 'next/image'
import React, { useState } from 'react'

// Accordion 컴포넌트의 Props 타입 정의
interface AccordionContentProps extends React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content> {
  className?: string
  children: React.ReactNode
}

const AccordionContent: React.FC<AccordionContentProps> = ({ className, children, ...props }) => {
  return (
    <AccordionPrimitive.Content
      data-slot="accordion-content"
      className="data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down overflow-hidden text-sm transition-all duration-300"
      {...props}
    >
      <div className={cn('pt-0 pb-4', className)}>{children}</div>
    </AccordionPrimitive.Content>
  )
}

const Login: React.FC = () => {
  // boolean 타입으로 변경된 탭 상태 (true: 로그인(account), false: 회원가입(password))
  const [isLoginTab, setIsLoginTab] = useState<boolean>(true)

  // 초기 탭 상태에 따라 아코디언 상태 설정 (항상 배열로 초기화)
  // isLoginTab이 true면 빈 배열(접힘), false면 ['name-item', 'region-item'](펼쳐짐)
  const [accordionValues, setAccordionValues] = useState<string[]>([])

  // 탭 변경 처리
  const handleTabChange = (value: string): void => {
    // 문자열 값을 boolean으로 변환 (account면 true, password면 false)
    const isLogin = value === 'account'
    setIsLoginTab(isLogin)

    // 탭에 따라 아코디언 상태 업데이트
    if (isLogin) {
      setAccordionValues([]) // 로그인 탭일 경우 아코디언 접기
    } else {
      setAccordionValues(['name-item', 'region-item']) // 회원가입 탭일 경우 모든 아코디언 확장
    }
  }

  // 아코디언 상태 변경 핸들러
  const handleAccordionChange = (value: string[]) => {
    // 로그인 탭에서는 항상 빈 배열 유지
    if (isLoginTab) {
      setAccordionValues([])
    } else {
      // 회원가입 탭에서는 사용자가 선택한 값으로 업데이트
      setAccordionValues(value)
    }
  }

  return (
    <>
      <header className="sticky top-0 z-50 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6">
        <div className="hidden font-bold md:flex">YES, MY ROAD</div>
      </header>
      <div className="relative w-full h-[calc(100vh-4rem)]">
        <Image src={background} alt="로그인 배경" fill style={{ objectFit: 'cover' }} priority className="z-0" />
        <div className="bg-white-500 absolute inset-0 flex items-center justify-center z-10">
          <Tabs
            value={isLoginTab ? 'account' : 'password'}
            onValueChange={(value: string) => handleTabChange(value)}
            className="w-[400px]"
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="account">로그인</TabsTrigger>
              <TabsTrigger value="password">회원가입</TabsTrigger>
            </TabsList>
            <Card className="mt-4">
              <CardHeader>
                {/* boolean 값으로 조건부 렌더링 */}
                <CardTitle>{isLoginTab ? '로그인' : '회원가입'}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-1">
                  <Label htmlFor="id">ID</Label>
                  <Input id="id" placeholder="아이디를 입력해주세요" />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="password">비밀번호</Label>
                  <Input id="password" type="password" placeholder="비밀번호를 입력해주세요" />
                </div>

                {/* 이름 및 지역 필드를 위한 아코디언 */}
                <AccordionPrimitive.Root
                  type="multiple"
                  value={accordionValues}
                  onValueChange={handleAccordionChange}
                  className="w-full"
                >
                  <AccordionPrimitive.Item value="name-item">
                    <AccordionContent>
                      <div className="space-y-1">
                        <Label>이름</Label>
                        <Input id="name" type="text" placeholder="이름을 입력해주세요" />
                      </div>
                    </AccordionContent>
                  </AccordionPrimitive.Item>
                  <AccordionPrimitive.Item value="region-item">
                    <AccordionContent>
                      <div className="space-y-1">
                        <Label>지역</Label>
                        <Input id="region" type="text" placeholder="지역을 선택해주세요" />
                      </div>
                    </AccordionContent>
                  </AccordionPrimitive.Item>
                </AccordionPrimitive.Root>
              </CardContent>
              <CardFooter>
                {/* boolean 값으로 버튼 텍스트 변경 */}
                <Button className="w-full">{isLoginTab ? '로그인' : '회원가입'}</Button>
              </CardFooter>
            </Card>
          </Tabs>
        </div>
      </div>
    </>
  )
}

export default Login
