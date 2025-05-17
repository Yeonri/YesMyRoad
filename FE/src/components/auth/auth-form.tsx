'use client'

import { userAPI } from '@/api/user-api'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
// 지역 ID 데이터 임포트
import regionIdMap from '@/data/region-id.json'
import { cn } from '@/lib/utils'
import { useUserStore } from '@/store/user-store'
import { LoginFormData, RegisterFormData } from '@/types/user'
import * as AccordionPrimitive from '@radix-ui/react-accordion'
import { isAxiosError } from 'axios'
import { Check, ChevronsUpDown } from 'lucide-react'
import { useRouter } from 'next/navigation'
import React, { useEffect, useState } from 'react'

// Accordion 컴포넌트의 Props 타입 정의
interface AccordionContentProps
  extends React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Content> {
  className?: string
  children: React.ReactNode
}

// 에러 응답 타입 정의
interface ErrorField {
  field: string | null
  defaultMessage: string
}

interface ErrorResponse {
  errors: ErrorField[]
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

// props 타입 정의
interface AuthFormProps {
  isLoginTab: boolean
  onTabChange: (isLogin: boolean) => void
}

const AuthForm: React.FC<AuthFormProps> = ({ isLoginTab, onTabChange }) => {
  const router = useRouter()

  // Zustand 스토어에서 필요한 상태와 액션 가져오기
  const { error, clearError } = useUserStore()

  const [isLoading, setIsLoading] = useState(false)  // 로딩 상태
  const [isSuccess, setIsSuccess] = useState(false)// 성공 상태
  const [formErrors, setFormErrors] = useState<{ [key: string]: string }>({})  // 폼 에러 메시지 관리
  const [generalError, setGeneralError] = useState<string>('')  // 일반 에러 메시지 (필드에 매핑되지 않는 에러)
  const [openRegion, setOpenRegion] = useState(false)  // 지역 선택기 상태

  // 초기 탭 상태에 따라 아코디언 상태 설정 (항상 배열로 초기화)
  const [accordionValues, setAccordionValues] = useState<string[]>(
    isLoginTab ? [] : ['name-item', 'region-item'],
  )

  // 폼 데이터 상태 관리 - 타입에 맞게 초기화
  const [formData, setFormData] = useState<RegisterFormData>({
    id: '',
    password: '',
    name: '',
    region: 0,
  })

  // 아이디 중복 확인용 플래그 변수 (-1: 미확인, 0: 중복, 1: 사용가능)
  const [idAvailability, setIdAvailability] = useState<number>(-1)

  // 탭 변경 시 상태 초기화
  useEffect(() => {
    setFormData({ id: '', password: '', name: '', region: 0 })
    setIdAvailability(-1)
    clearError()
    setIsLoading(false)
    setIsSuccess(false)
    setFormErrors({})
    setGeneralError('')

    // 탭에 따라 아코디언 상태 업데이트
    if (isLoginTab) {
      setAccordionValues([]) // 로그인 탭일 경우 아코디언 접기
    } else {
      setAccordionValues(['name-item', 'region-item']) // 회원가입 탭일 경우 모든 아코디언 확장
    }
  }, [isLoginTab, clearError])

  // 아이디 중복 확인 핸들러 - 직접 여기서 alert 표시
  const handleCheckIdDuplication = async (
    e: React.MouseEvent<HTMLButtonElement>,
  ) => {
    e.preventDefault()
    if (formData.id.length === 0) {
      alert('아이디를 입력해주세요')
      return
    }

    try {
      setIsLoading(true) // 중복 확인 중 로딩 상태 활성화
      const response = await userAPI.checkIdDuplication(formData.id)
      const available = response.data.available
      setIdAvailability(available)

      // 중복 확인 결과에 따른 alert 표시 (useEffect에서 이동)
      if (available === 1) {
        alert('사용할 수 있는 아이디입니다')
      } else if (available === 0) {
        alert('이미 존재하는 아이디입니다')
      }
    } catch (error) {
      console.error('아이디 중복 확인 실패:', error)
      alert('아이디 중복 확인에 실패했습니다.')
      setIdAvailability(-1)
    } finally {
      setIsLoading(false) // 중복 확인 완료 후 로딩 상태 비활성화
    }
  }

  // 입력 필드 변경 핸들러
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { id, value } = e.target

    // id 필드가 변경될 때 idAvailability 초기화
    if (id === 'id') {
      setIdAvailability(-1) // ID가 변경되면 중복확인 상태 초기화
    }

    // 에러 메시지 초기화
    if (formErrors[id]) {
      setFormErrors((prev) => ({ ...prev, [id]: '' }))
    }
    if (generalError) {
      setGeneralError('')
    }

    const newValue = id === 'region' ? parseInt(value) || 0 : value // region은 숫자로 변환
    setFormData({ ...formData, [id]: newValue })
  }

  // 서버 에러 처리 함수
  const handleServerErrors = (errorResponse: ErrorResponse) => {
    const newFormErrors: { [key: string]: string } = {}
    let newGeneralError = ''

    errorResponse.errors.forEach((err) => {
      if (err.field) {
        newFormErrors[err.field] = err.defaultMessage
      } else {
        newGeneralError = err.defaultMessage
      }
    })
    setFormErrors(newFormErrors)
    setGeneralError(newGeneralError)
  }

  // 폼 제출 핸들러
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    clearError() // 이전 오류 초기화
    setFormErrors({})
    setGeneralError('')
    setIsLoading(true) // 로딩 상태 활성화

    try {
      // isLoginTab일 때 로그인 로직 실행
      if (isLoginTab) {
        // 로그인 데이터 타입 적용
        const loginData: LoginFormData = {
          id: formData.id,
          password: formData.password,
        }

        // 로그인 요청 (토큰 저장 및 사용자 정보 설정은 API 함수 내부에서 처리)
        const response = await userAPI.login(loginData)

        if (response.status === 200) {
          // 성공 상태 설정
          setIsSuccess(true)

          // 성공 메시지 및 리디렉션
          alert('로그인 성공! 대시보드로 이동합니다.')
          router.push('/')
        }
      } else {
        // 회원가입 로직 - 이미 RegisterFormData 타입 사용 중
        if (!formData.name.trim()) {
          setFormErrors((prev) => ({ ...prev, name: '이름을 입력해주세요' }))
          setIsLoading(false)
          return
        }

        if (!formData.region) {
          setFormErrors((prev) => ({ ...prev, region: '지역을 선택해주세요' }))
          setIsLoading(false)
          return
        }

        // 아이디 중복 확인 여부 검사
        if (idAvailability !== 1) {
          setGeneralError('아이디 중복 확인을 먼저 진행해주세요')
          setIsLoading(false)
          return
        }

        const response = await userAPI.register(formData)

        if (response.status === 200 || response.status === 201) {
          // 성공 상태 설정
          setIsSuccess(true)

          // 회원가입 성공 메시지
          alert('회원가입이 완료되었습니다. 로그인 페이지로 이동합니다.')

          // 회원가입 성공 시 로그인 탭으로 전환
          onTabChange(true)

          // 폼과 상태 초기화
          setFormData({ id: '', password: '', name: '', region: 0 })
          setIsLoading(false)
          setIsSuccess(false)
        }
      }
    } catch (error) {
      // console.error('API 호출 실패:', error)

      // 에러 메시지 표시 (개선된 에러 처리)
      if (isAxiosError(error) && error.response?.data) {
        try {
          // 서버 응답에서 에러 형식 파싱
          const errorData = error.response.data as ErrorResponse
          if (errorData.errors) {
            handleServerErrors(errorData)
          } else {
            setGeneralError('알 수 없는 오류가 발생했습니다')
          }
        } catch (e) {
          setGeneralError(`요청 처리 중 오류가 발생했습니다: ${e}`)
        }
      } else {
        setGeneralError(
          isLoginTab
            ? '로그인 중 오류가 발생했습니다'
            : '회원가입 중 오류가 발생했습니다',
        )
      }

      setIsLoading(false) // 에러 발생 시 로딩 상태 비활성화
    }
  }

  // 버튼 텍스트 결정 함수
  const getButtonText = () => {
    const LABELS = {
      login: ['로그인', '대시보드 화면으로 이동중입니다'],
      signup: ['회원가입', '회원가입 완료'],
    }

    if (isSuccess) return LABELS[isLoginTab ? 'login' : 'signup'][1]
    if (isLoading) return '처리 중...'
    return LABELS[isLoginTab ? 'login' : 'signup'][0]
  }

  return (
    <>
      {/* 일반 에러 메시지 표시 */}
      {(generalError || error) && (
        <div className="mb-4 rounded border border-red-400 bg-red-100 px-4 py-3 text-red-700">
          {generalError || error}
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
              onChange={handleInputChange}
              required
              disabled={isLoading || isSuccess}
              className={`${!isLoginTab && idAvailability === 1 ? 'border-green-500' : ''} ${!isLoginTab && idAvailability === 0 ? 'border-red-500' : ''} ${formErrors.id ? 'border-red-500' : ''}`}
            />
            {!isLoginTab && (
              <Button
                className="ml-5"
                onClick={handleCheckIdDuplication}
                disabled={isLoading || isSuccess || !formData.id}
              >
                중복 확인
              </Button>
            )}
          </div>
          {/* ID 필드 에러 메시지 */}
          {formErrors.id && (
            <div className="mt-1 text-sm text-red-600">{formErrors.id}</div>
          )}
          {/* ID 중복 확인 상태 표시 */}
          {!isLoginTab && (
            <div className="mt-1 text-sm">
              {idAvailability === -1 && formData.id && (
                <span className="text-yellow-600">
                  아이디 중복 확인이 필요합니다
                </span>
              )}
              {idAvailability === 0 && (
                <span className="text-red-600">
                  이미 사용 중인 아이디입니다
                </span>
              )}
              {idAvailability === 1 && (
                <span className="text-green-600">사용 가능한 아이디입니다</span>
              )}
            </div>
          )}
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
            disabled={isLoading || isSuccess}
            className={formErrors.password ? 'border-red-500' : ''}
          />
          {/* 비밀번호 필드 에러 메시지 */}
          {formErrors.password && (
            <div className="mt-1 text-sm text-red-600">
              {formErrors.password}
            </div>
          )}
        </div>

        {/* 이름 및 지역 필드를 위한 아코디언 */}
        <AccordionPrimitive.Root
          type="multiple"
          value={accordionValues}
          onValueChange={value => setAccordionValues(isLoginTab ? [] : value)}
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
                  disabled={isLoading || isSuccess}
                  className={formErrors.name ? 'border-red-500' : ''}
                />
                {/* 이름 필드 에러 메시지 */}
                {formErrors.name && (
                  <div className="mt-1 text-sm text-red-600">
                    {formErrors.name}
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionPrimitive.Item>

          <AccordionPrimitive.Item value="region-item">
            <AccordionContent>
              <div className="space-y-1">
                <Label htmlFor="region">지역</Label>
                <Popover open={openRegion} onOpenChange={setOpenRegion}>
                  <PopoverTrigger asChild>
                    <Button
                      id="region"
                      variant="outline"
                      role="combobox"
                      aria-expanded={openRegion}
                      className={`w-full justify-between ${formErrors.region ? 'border-red-500' : ''
                        }`}
                      disabled={isLoading || isSuccess}
                    >
                      {formData.region
                        ? Object.entries(regionIdMap).find(
                          (entry) => entry[1] === formData.region,
                        )?.[0] || '지역을 선택해주세요'
                        : '지역을 선택해주세요'}
                      <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-full p-0">
                    <Command>
                      <CommandInput placeholder="지역 검색..." />
                      <CommandList>
                        <CommandEmpty>검색 결과가 없습니다.</CommandEmpty>
                        <CommandGroup>
                          {Object.entries(regionIdMap).map(
                            ([regionName, regionId]) => (
                              <CommandItem
                                key={regionId}
                                value={regionName}
                                onSelect={() => {
                                  // 지역 선택 시 region ID 값 설정
                                  setFormData({ ...formData, region: regionId })
                                  // 지역 관련 에러 메시지 초기화
                                  if (formErrors.region) {
                                    setFormErrors((prev) => ({
                                      ...prev,
                                      region: '',
                                    }))
                                  }
                                  setOpenRegion(false)
                                }}
                              >
                                <Check
                                  className={cn(
                                    'mr-2 h-4 w-4',
                                    formData.region === regionId
                                      ? 'opacity-100'
                                      : 'opacity-0',
                                  )}
                                />
                                {regionName}
                              </CommandItem>
                            ),
                          )}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
                {/* 지역 필드 에러 메시지 */}
                {formErrors.region && (
                  <div className="mt-1 text-sm text-red-600">
                    {formErrors.region}
                  </div>
                )}
              </div>
            </AccordionContent>
          </AccordionPrimitive.Item>
        </AccordionPrimitive.Root>

        <div className="pt-6">
          <Button
            type="submit"
            className={`w-full ${isSuccess ? 'bg-green-600 hover:bg-green-700' : ''}`}
            disabled={isLoading || isSuccess || (isLoginTab && (!formData.id.trim() || !formData.password.trim()))}
          >
            {getButtonText()}
          </Button>
        </div>
      </form>
    </>
  )
}

export default AuthForm
