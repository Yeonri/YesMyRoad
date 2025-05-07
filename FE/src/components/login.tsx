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

// 탭 값의 타입 정의
type TabValue = 'account' | 'password'

// 단순화된 아코디언 컴포넌트 (첫 번째 코드 스니펫에서 추출)
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
  // 현재 활성화된 탭을 추적하는 상태
  const [activeTab, setActiveTab] = useState<TabValue>('account')
  // 아코디언 상태 (회원가입에서는 열림, 로그인에서는 닫힘)
  const [accordionValue, setAccordionValue] = useState<string | undefined>(undefined)

  // 탭 변경 처리
  const handleTabChange = (value: TabValue): void => {
    setActiveTab(value)

    // 탭에 따라 아코디언 상태 업데이트
    if (value === 'password') {
      // 회원가입의 경우 아코디언 확장
      setAccordionValue('name-item')
    } else {
      // 로그인의 경우 아코디언 접기
      setAccordionValue(undefined)
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
            value={activeTab}
            onValueChange={(value: string) => handleTabChange(value as TabValue)}
            className="w-[400px]"
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="account">로그인</TabsTrigger>
              <TabsTrigger value="password">회원가입</TabsTrigger>
            </TabsList>
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>{activeTab === 'account' ? '로그인' : '회원가입'}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-1">
                  <Label htmlFor="id">ID</Label>
                  <Input id="id" placeholder="아이디를 입력해주세요" />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="password">비밀번호</Label>
                  <Input id="password" placeholder="비밀번호를 입력해주세요" type="password" />
                </div>

                {/* 이름 필드를 위한 아코디언 */}
                <AccordionPrimitive.Root type="single" value={accordionValue} collapsible className="w-full">
                  <AccordionPrimitive.Item value="name-item">
                    <AccordionContent>
                      <div className="space-y-1">
                        <Label>이름</Label>
                        <Input id="name" type="text" placeholder="이름을 입력해주세요" />
                      </div>
                    </AccordionContent>
                  </AccordionPrimitive.Item>
                </AccordionPrimitive.Root>
              </CardContent>
              <CardFooter>
                <Button className="w-full">{activeTab === 'account' ? '로그인' : '회원가입'}</Button>
              </CardFooter>
            </Card>
          </Tabs>
        </div>
      </div>
    </>
  )
}

export default Login
