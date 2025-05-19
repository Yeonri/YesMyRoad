'use client'

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Button } from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import address from '@/data/address.json'
import { cn } from '@/libs/utils'
import useAddressStore from '@/store/address-store'
import { AddressData, LocationInfo } from '@/types/address'
import { Check, ChevronDown } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

const AddressSelector: React.FC = () => {
  const addressData = useMemo(() => address as unknown as AddressData, [])
  const { setId, setLongitude, setLatitude, setLevel1, setLevel2, setLevel3 } =
    useAddressStore()

  // 팝오버 상태 관리
  const [provinceOpen, setProvinceOpen] = useState(false)
  const [cityOpen, setCityOpen] = useState(false)
  const [districtOpen, setDistrictOpen] = useState(false)

  // 선택된 행정구역 상태
  const [districtLevel1, setDistrictLevel1] = useState<string | null>(
    '대전광역시',
  )
  const [districtLevel2, setDistrictLevel2] = useState<string | null>('유성구')
  const [districtLevel3, setDistrictLevel3] = useState<string | null>(null)

  // 선행 행정구역의 하위 옵션
  const [districtLevel2Array, setDistrictLevel2Array] = useState<string[]>([])
  const [districtLevel3Array, setDistrictLevel3Array] = useState<string[]>([])

  // 시/도(level1) 변경시 시/군/구(level2) 목록 업데이트 및 초기 시/군/구(level2) 선택
  useEffect(() => {
    if (districtLevel1) {
      // level1 주소를 스토어에 저장
      setLevel1(districtLevel1)

      // level1에 해당하는 하위 지역(level2) 목록 가져오기
      const level1Data = addressData[districtLevel1]

      // 세종특별자치시처럼 직접 데이터를 가진 경우 id 값을 저장하고 level2를 비움
      if (level1Data && 'id' in level1Data && level1Data.district === null) {
        const locationInfo = level1Data as LocationInfo
        if (locationInfo.id) setId(locationInfo.id)
        if (locationInfo.longitude) setLongitude(locationInfo.longitude)
        if (locationInfo.latitude) setLatitude(locationInfo.latitude)

        setDistrictLevel2Array([])
        setDistrictLevel2(null)
        setLevel2(null)
        setDistrictLevel3Array([])
        setDistrictLevel3(null)
        setLevel3(null)
      }
      // 일반적인 지역의 경우 하위 district 목록 가져오기
      else if (level1Data && 'district' in level1Data && level1Data.district) {
        const districts = Object.keys(level1Data.district)
        setDistrictLevel2Array(districts)

        // 이미 설정된 districtLevel2가 없거나, 새로운 districts 목록에 현재 districtLevel2가 없는 경우에만 변경
        if (!districtLevel2 || !districts.includes(districtLevel2)) {
          // 첫 번째 district를 기본값으로 설정
          if (districts.length > 0) {
            setDistrictLevel2(districts[0])
          } else {
            setDistrictLevel2(null)
          }
        }
      } else {
        setDistrictLevel2Array([])
        setDistrictLevel2(null)
      }
    }
  }, [
    districtLevel1,
    addressData,
    setLevel1,
    setLevel2,
    setLevel3,
    setId,
    setLongitude,
    setLatitude,
    districtLevel2,
  ])

  // 시/군/구(level2) 변경시 하위 구(level3) 목록 업데이트 및 초기 하위 구(level3) 선택
  useEffect(() => {
    if (districtLevel1 && districtLevel2) {
      // level2 주소를 스토어에 저장
      setLevel2(districtLevel2)

      const level1Data = addressData[districtLevel1]

      if (level1Data && 'district' in level1Data && level1Data.district) {
        const level2Data = level1Data.district[districtLevel2]

        // 지역 ID, 위도, 경도 정보 스토어에 저장
        if (level2Data) {
          if (level2Data.id) setId(level2Data.id)
          if (level2Data.longitude) setLongitude(level2Data.longitude)
          if (level2Data.latitude) setLatitude(level2Data.latitude)

          // level3 존재 여부에 따라 처리
          if (level2Data.district) {
            const districts = Object.keys(level2Data.district)
            setDistrictLevel3Array(districts)

            // 첫번째 하위 구를 기본값으로 설정
            if (districts.length > 0) {
              setDistrictLevel3(districts[0])
            } else {
              setDistrictLevel3(null)
              setLevel3(null)
            }
          } else {
            setDistrictLevel3Array([])
            setDistrictLevel3(null)
            setLevel3(null)
          }
        }
      }
    }
  }, [
    districtLevel1,
    districtLevel2,
    addressData,
    setId,
    setLongitude,
    setLatitude,
    setLevel2,
    setLevel3,
  ])

  // districtLevel3 변경 시 위치 정보 업데이트
  useEffect(() => {
    if (districtLevel1 && districtLevel2 && districtLevel3) {
      // level3 주소를 스토어에 저장
      setLevel3(districtLevel3)

      const level1Data = addressData[districtLevel1]

      if (level1Data && 'district' in level1Data && level1Data.district) {
        const level2Data = level1Data.district[districtLevel2]

        if (level2Data && level2Data.district) {
          const level3Data = level2Data.district[districtLevel3]

          // 지역 ID, 위도, 경도 정보 스토어에 저장
          if (level3Data) {
            if (level3Data.id) setId(level3Data.id)
            if (level3Data.longitude) setLongitude(level3Data.longitude)
            if (level3Data.latitude) setLatitude(level3Data.latitude)
          }
        }
      }
    }
  }, [
    districtLevel1,
    districtLevel2,
    districtLevel3,
    addressData,
    setId,
    setLongitude,
    setLatitude,
    setLevel3,
  ])

  return (
    <>
      <Breadcrumb>
        <BreadcrumbList>
          {/* 시/도 선택 콤보박스 */}
          <BreadcrumbItem>
            <Popover open={provinceOpen} onOpenChange={setProvinceOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={provinceOpen}
                  className="flex items-center gap-1"
                >
                  {districtLevel1}
                  <ChevronDown className="opacity-50" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[200px] p-0">
                <Command>
                  <CommandInput placeholder="시/도 검색..." className="h-9" />
                  <CommandList className="max-h-[200px] overflow-auto">
                    <CommandEmpty>검색 결과가 없습니다.</CommandEmpty>
                    <CommandGroup>
                      {Object.keys(addressData).map((province) => (
                        <CommandItem
                          key={province}
                          value={province}
                          onSelect={(currentValue) => {
                            setDistrictLevel1(currentValue)
                            setProvinceOpen(false)
                          }}
                        >
                          {province}
                          <Check
                            className={cn(
                              'ml-auto',
                              districtLevel1 === province
                                ? 'opacity-100'
                                : 'opacity-0',
                            )}
                          />
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          </BreadcrumbItem>

          {districtLevel2Array.length > 0 && districtLevel2 !== '' && (
            <>
              <BreadcrumbSeparator />

              {/* 시/군/구 선택 콤보박스 */}
              <BreadcrumbItem>
                <Popover open={cityOpen} onOpenChange={setCityOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={cityOpen}
                      className="flex items-center gap-1"
                    >
                      {districtLevel2}
                      <ChevronDown className="opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[200px] p-0">
                    <Command>
                      <CommandInput
                        placeholder="시/군/구 검색..."
                        className="h-9"
                      />
                      <CommandList className="max-h-[200px] overflow-auto">
                        <CommandEmpty>검색 결과가 없습니다.</CommandEmpty>
                        <CommandGroup>
                          {districtLevel2Array.map((city) => (
                            <CommandItem
                              key={city}
                              value={city}
                              onSelect={(currentValue) => {
                                setDistrictLevel2(currentValue)
                                setCityOpen(false)
                              }}
                            >
                              {city}
                              <Check
                                className={cn(
                                  'ml-auto',
                                  districtLevel2 === city
                                    ? 'opacity-100'
                                    : 'opacity-0',
                                )}
                              />
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
              </BreadcrumbItem>
            </>
          )}

          {/* 하위 구가 있는 경우에만 표시 */}
          {districtLevel3Array.length > 0 && districtLevel3 && (
            <>
              <BreadcrumbSeparator />

              <BreadcrumbItem>
                <Popover open={districtOpen} onOpenChange={setDistrictOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={districtOpen}
                      className="flex items-center gap-1"
                    >
                      {districtLevel3}
                      <ChevronDown className="opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[200px] p-0">
                    <Command>
                      <CommandInput placeholder="구 검색..." className="h-9" />
                      <CommandList className="max-h-[200px] overflow-auto">
                        <CommandEmpty>검색 결과가 없습니다.</CommandEmpty>
                        <CommandGroup>
                          {districtLevel3Array.map((district) => (
                            <CommandItem
                              key={district}
                              value={district}
                              onSelect={(currentValue) => {
                                setDistrictLevel3(currentValue)
                                setDistrictOpen(false)
                              }}
                            >
                              {district}
                              <Check
                                className={cn(
                                  'ml-auto',
                                  districtLevel3 === district
                                    ? 'opacity-100'
                                    : 'opacity-0',
                                )}
                              />
                            </CommandItem>
                          ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
              </BreadcrumbItem>
            </>
          )}
        </BreadcrumbList>
      </Breadcrumb>
    </>
  )
}

export default AddressSelector
