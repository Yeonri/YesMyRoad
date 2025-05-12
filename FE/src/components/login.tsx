'use client'

import { userAPI } from '@/api/user-api'
import background from '@/assets/background.png'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { useUserStore } from '@/store/user-store'
import { LoginFormData, RegisterFormData } from '@/types/user'
import * as AccordionPrimitive from '@radix-ui/react-accordion'
import Image from 'next/image'
import { useRouter } from 'next/navigation'
import React, { useEffect, useRef, useState } from 'react'

// Accordion 컴포넌트의 Props 타입 정의
interface AccordionContentProps
  extends React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content> {
  className?: string
  children: React.ReactNode
}

const AccordionContent: React.FC<AccordionContentProps> = ({
  className,
  children,
  ...props
}) => {
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
  const router = useRouter()
  // Zustand 스토어에서 필요한 상태와 액션 가져오기
  const { login, register, isLoading, error, clearError, isAuthenticated } =
    useUserStore()

  console.log('[컴포넌트 마운트] 스토어 상태:', {
    isLoading,
    error,
    isAuthenticated,
  })

  // 로그인 상태가 변경될 때 리디렉션
  useEffect(() => {
    console.log('[인증 상태 변경]:', isAuthenticated)
    if (isAuthenticated) {
      console.log('[리디렉션] 인증 성공, 메인 페이지로 이동')
      router.push('/') // 메인 페이지로 리디렉션
    }
  }, [isAuthenticated, router])

  // boolean 타입으로 변경된 탭 상태 (true: 로그인(account), false: 회원가입(password))
  const [isLoginTab, setIsLoginTab] = useState<boolean>(true)

  // 초기 탭 상태에 따라 아코디언 상태 설정 (항상 배열로 초기화)
  const [accordionValues, setAccordionValues] = useState<string[]>([])

  // 폼 데이터 상태 관리
  const [formData, setFormData] = useState({
    id: '',
    password: '',
    name: '',
    region: 0, // 숫자 타입으로 설정
  })

  // 아이디 중복 확인용 플래그 변수
  const [idAvailability, setIdAvailability] = useState<number>(-1)
  // 초기 렌더링 감지용
  const isInitialRender = useRef(true)

  const handleCheckIdDuplication = async (
    e: React.MouseEvent<HTMLButtonElement>,
  ) => {
    e.preventDefault()
    if (formData.id.length !== 0) {
      const response = await userAPI.checkIdDuplication(formData.id)
      setIdAvailability(response.data.available)
      console.log(`함수 실행: ${idAvailability}`)

      // alert는 useEffect에서 처리
    } else {
      alert('아이디를 입력해주세요')
    }
  }

  useEffect(() => {
    // 초기 렌더링 시에는 알림 표시하지 않음
    if (isInitialRender.current) {
      isInitialRender.current = false
      return
    }

    // idAvailability 값에 따라 알림 표시
    if (idAvailability) {
      alert('사용할 수 있는 아이디입니다')
      console.log(`idAvailability: ${idAvailability}`)
    } else {
      alert('이미 존재하는 아이디입니다')
      console.log(`idAvailability: ${idAvailability}`)
    }
  }, [idAvailability]) // idAvailability가 변경될 때마다 실행

  // 디버깅용 - 폼 데이터 변화 감지
  useEffect(() => {
    console.log('[폼 데이터 상태]:', formData)
  }, [formData])

  // 입력 필드 변경 핸들러
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target
    console.log(`[입력 필드 변경] 필드: ${id}, 값: ${value}`)

    const newValue = id === 'region' ? parseInt(value) || 0 : value // region은 숫자로 변환
    console.log(`[변환된 값] ${id}: ${newValue}, 타입: ${typeof newValue}`)
    setFormData({ ...formData, [id]: newValue })
  }

  // 아이디 입력 필드 변경 핸들러
  const handleIdInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target
    console.log(`[입력 필드 변경] 필드: ${id}, 값: ${value}`)

    const newValue = id === 'region' ? parseInt(value) || 0 : value // region은 숫자로 변환
    console.log(`[변환된 값] ${id}: ${newValue}, 타입: ${typeof newValue}`)
    setFormData({ ...formData, [id]: newValue })
  }

  // 폼 제출 핸들러
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log('[폼 제출] 시작')
    console.log('[현재 탭]:', isLoginTab ? '로그인' : '회원가입')
    console.log('[제출 데이터]:', formData)

    clearError() // 이전 오류 초기화
    console.log('[에러 초기화 완료]')

    if (isLoginTab) {
      // 로그인 처리
      const loginData: LoginFormData = {
        id: formData.id,
        password: formData.password,
      }

      console.log('[로그인 요청 데이터]:', loginData)
      console.log('[로그인 API 호출 직전]')

      try {
        await login(loginData)
        console.log('[로그인 API 호출 완료]')
      } catch (err) {
        console.error('[로그인 실패]:', err)
      }
    } else {
      // 회원가입 처리
      if (!formData.name.trim()) {
        console.warn('[유효성 검사 실패] 이름이 비어있음')
        alert('이름을 입력해주세요')
        return
      }

      if (!formData.region) {
        console.warn('[유효성 검사 실패] 지역이 선택되지 않음')
        alert('지역을 선택해주세요')
        return
      }

      const registerData: RegisterFormData = {
        id: formData.id,
        password: formData.password,
        name: formData.name,
        region: formData.region,
      }

      console.log('[회원가입 요청 데이터]:', registerData)
      console.log('[회원가입 API 호출 직전]')

      try {
        await register(registerData)
        console.log('[회원가입 API 호출 완료]')
        alert('회원가입이 완료되었습니다. 로그인해주세요.')
        setIsLoginTab(true) // 로그인 탭으로 전환
        console.log('[로그인 탭으로 전환]')
      } catch (err) {
        console.error('[회원가입 실패]:', err)
      }
    }
  }

  // 탭 변경 처리
  const handleTabChange = (value: string): void => {
    // 문자열 값을 boolean으로 변환 (account면 true, password면 false)
    const isLogin = value === 'account'
    console.log(`[탭 변경] ${isLogin ? '로그인' : '회원가입'} 탭으로 전환`)
    setIsLoginTab(isLogin)
    setIdAvailability(-1)

    // 입력 필드 초기화
    setFormData({ id: '', password: '', name: '', region: 0 })
    console.log('[폼 데이터 초기화 완료]')

    clearError() // 탭 전환 시 오류 메시지 초기화
    console.log('[에러 상태 초기화 완료]')

    // 탭에 따라 아코디언 상태 업데이트
    if (isLogin) {
      setAccordionValues([]) // 로그인 탭일 경우 아코디언 접기
      console.log('[아코디언 상태] 모두 접기')
    } else {
      setAccordionValues(['name-item', 'region-item']) // 회원가입 탭일 경우 모든 아코디언 확장
      console.log('[아코디언 상태] 모두 펼치기')
    }
  }

  // 아코디언 상태 변경 핸들러
  const handleAccordionChange = (value: string[]) => {
    console.log('[아코디언 상태 변경]:', value)
    // 로그인 탭에서는 항상 빈 배열 유지
    if (isLoginTab) {
      setAccordionValues([])
      console.log('[아코디언 상태 강제 변경] 로그인 탭이므로 모두 접기')
    } else {
      // 회원가입 탭에서는 사용자가 선택한 값으로 업데이트
      setAccordionValues(value)
    }
  }

  // 스토어 상태 변화 모니터링
  useEffect(() => {
    console.log('[스토어 상태 변화]', { isLoading, error, isAuthenticated })
  }, [isLoading, error, isAuthenticated])

  return (
    <>
      <header className="bg-background sticky top-0 z-50 flex h-16 items-center gap-4 border-b px-4 md:px-6">
        <div className="hidden font-bold md:flex">YES, MY ROAD</div>
      </header>
      <div className="relative h-[calc(100vh-4rem)] w-full">
        <Image
          src={background}
          alt="로그인 배경"
          fill
          style={{ objectFit: 'cover' }}
          priority
          className="z-0"
        />
        <div className="bg-white-500 absolute inset-0 z-10 flex items-center justify-center">
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
                <CardTitle>{isLoginTab ? '로그인' : '회원가입'}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* 오류 메시지 표시 */}
                {error && (
                  <div className="rounded border border-red-400 bg-red-100 px-4 py-3 text-red-700">
                    {error}
                  </div>
                )}

                <form onSubmit={handleSubmit}>
                  <div className="space-y-1 pt-0 pb-4">
                    <Label htmlFor="id">ID</Label>
                    <div className="flex">
                      <Input
                        id="id"
                        placeholder="아이디를 입력해주세요"
                        value={formData.id}
                        onChange={handleIdInputChange}
                        required
                        className={`${!isLoginTab && idAvailability === 1 ? 'border-green-500' : ''} ${!isLoginTab && idAvailability === 0 ? 'border-red-500' : ''}`}
                      />
                      {!isLoginTab && (
                        <Button
                          className="ml-5"
                          onClick={(e) => {
                            handleCheckIdDuplication(e)
                            console.log(idAvailability)
                          }}
                        >
                          중복 확인
                        </Button>
                      )}
                    </div>
                  </div>
                  <div className="space-y-1 pt-0 pb-4">
                    <Label htmlFor="password">비밀번호</Label>
                    <Input
                      id="password"
                      type="password"
                      placeholder="비밀번호를 입력해주세요"
                      value={formData.password}
                      onChange={handleInputChange}
                      required
                    />
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
                          <Label htmlFor="name">이름</Label>
                          <Input
                            id="name"
                            type="text"
                            placeholder="이름을 입력해주세요"
                            value={formData.name}
                            onChange={handleInputChange}
                          />
                        </div>
                      </AccordionContent>
                    </AccordionPrimitive.Item>

                    <AccordionPrimitive.Item value="region-item">
                      <AccordionContent>
                        <div className="space-y-1">
                          <Label htmlFor="region">지역</Label>
                          <Input
                            id="region"
                            type="number"
                            placeholder="지역을 선택해주세요"
                            value={formData.region || ''}
                            onChange={handleInputChange}
                          />
                        </div>
                      </AccordionContent>
                    </AccordionPrimitive.Item>
                  </AccordionPrimitive.Root>
                  <CardFooter className="px-0 pt-6">
                    <Button
                      type="submit"
                      className="w-full"
                      disabled={isLoading}
                      onClick={async (e) => {
                        e.preventDefault()
                        clearError() // 에러 초기화

                        try {
                          // isLoginTab일 때만 로그인 로직 실행
                          if (isLoginTab) {
                            const response = await userAPI.login({
                              id: formData.id,
                              password: formData.password,
                            })

                            if (response.status === 200) {
                              // 토큰 저장
                              localStorage.setItem(
                                'token',
                                response.data.accessToken,
                              )
                              localStorage.setItem(
                                'refreshToken',
                                response.data.refreshToken,
                              )
                              localStorage.setItem(
                                'userId',
                                response.data.userId,
                              )



                              console.log('[인증 상태 업데이트] 로그인 성공')
                              router.push('/') // 로그인 성공 시 리디렉션
                            }
                          } else {
                            // 회원가입 로직
                            if (!formData.name.trim()) {
                              alert('이름을 입력해주세요')
                              return
                            }

                            if (!formData.region) {
                              alert('지역을 선택해주세요')
                              return
                            }

                            const registerResponse = await userAPI.register({
                              id: formData.id,
                              password: formData.password,
                              name: formData.name,
                              region: formData.region,
                            })

                            if (registerResponse.status === 200) {
                              alert(
                                '회원가입이 완료되었습니다. 로그인해주세요.',
                              )
                              setIsLoginTab(true)
                            }
                          }
                        } catch (err: any) {
                          // 에러 처리
                          console.error('API 호출 실패:', err)
                          // 에러 메시지 설정 (useState로 관리한다고 가정)
                          // setError(err.message || '로그인 중 오류가 발생했습니다');
                        } 
                      }}
                    >
                      {isLoading
                        ? '처리 중...'
                        : isLoginTab
                          ? '로그인'
                          : '회원가입'}
                    </Button>
                  </CardFooter>
                </form>
              </CardContent>
            </Card>
          </Tabs>
        </div>
      </div>
    </>
  )
}

export default Login
