'use client'

import { Breadcrumb, BreadcrumbItem, BreadcrumbList, BreadcrumbSeparator } from '@/components/ui/breadcrumb'
import { Button } from '@/components/ui/button'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import address from '@/data/address.json'
import { cn } from '@/lib/utils'
import useAddressStore from '@/store/address-store'
import { AddressData } from '@/types/address'
import { Check, ChevronDown } from 'lucide-react'
import { useEffect, useState } from 'react'

const LocationHeader: React.FC = () => {
  const addressData: AddressData = address
  const { setLevel1, setLevel2, setLevel3 } = useAddressStore()

  // 선택된 값들을 상태로 관리
  const [selectedProvince, setSelectedProvince] = useState<string>('대전광역시')
  const [selectedCity, setSelectedCity] = useState<string>('유성구')
  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null)

  // 팝오버 상태 관리
  const [provinceOpen, setProvinceOpen] = useState(false)
  const [cityOpen, setCityOpen] = useState(false)
  const [districtOpen, setDistrictOpen] = useState(false)

  // 가능한 옵션들
  const [availableCities, setAvailableCities] = useState<string[]>([])
  const [availableDistricts, setAvailableDistricts] = useState<string[]>([])

  // 시/도 변경시 도시 목록 업데이트 및 초기 도시 선택
  useEffect(() => {
    const cities = Object.keys(addressData[selectedProvince] || {}).filter((city) => city !== '')
    setAvailableCities(cities)

    if (cities.length > 0) {
      setSelectedCity(cities[0])
    } else {
      setSelectedCity('')
    }

    setSelectedDistrict(null)
  }, [selectedProvince, addressData])

  // 도시 변경시 하위 구 목록 업데이트 및 초기 하위 구 선택
  useEffect(() => {
    if (!selectedCity) {
      setAvailableDistricts([])
      setSelectedDistrict(null)
      return
    }

    const districts = addressData[selectedProvince]?.[selectedCity] || []
    setAvailableDistricts(districts)

    if (districts.length > 0) {
      setSelectedDistrict(districts[0])
    } else {
      setSelectedDistrict(null)
    }
  }, [selectedProvince, selectedCity, addressData])

  // 주소 스토어 업데이트 (level1)
  useEffect(() => {
    setLevel1(selectedProvince)
    console.log('주소 스토어 업데이트:', useAddressStore.getState())
  }, [selectedProvince, setLevel1])

  // 주소 스토어 업데이트 (level2)
  useEffect(() => {
    if (selectedCity) {
      setLevel2(selectedCity)
      console.log('주소 스토어 업데이트:', useAddressStore.getState())
    }
  }, [selectedCity, setLevel2])

  // 주소 스토어 업데이트 (level3)
  useEffect(() => {
    if (selectedDistrict) {
      setLevel3(selectedDistrict)
    } else {
      setLevel3('')
    }
    console.log('주소 스토어 업데이트:', useAddressStore.getState())
  }, [selectedDistrict, setLevel3])

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
                  {selectedProvince}
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
                            setSelectedProvince(currentValue)
                            setProvinceOpen(false)
                          }}
                        >
                          {province}
                          <Check
                            className={cn('ml-auto', selectedProvince === province ? 'opacity-100' : 'opacity-0')}
                          />
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          </BreadcrumbItem>

          {availableCities.length > 0 && selectedCity !== '' && (
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
                      {selectedCity}
                      <ChevronDown className="opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[200px] p-0">
                    <Command>
                      <CommandInput placeholder="시/군/구 검색..." className="h-9" />
                      <CommandList className="max-h-[200px] overflow-auto">
                        <CommandEmpty>검색 결과가 없습니다.</CommandEmpty>
                        <CommandGroup>
                          {availableCities.map((city) => (
                            <CommandItem
                              key={city}
                              value={city}
                              onSelect={(currentValue) => {
                                setSelectedCity(currentValue)
                                setCityOpen(false)
                              }}
                            >
                              {city}
                              <Check className={cn('ml-auto', selectedCity === city ? 'opacity-100' : 'opacity-0')} />
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
          {availableDistricts.length > 0 && selectedDistrict && (
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
                      {selectedDistrict}
                      <ChevronDown className="opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-[200px] p-0">
                    <Command>
                      <CommandInput placeholder="구 검색..." className="h-9" />
                      <CommandList className="max-h-[200px] overflow-auto">
                        <CommandEmpty>검색 결과가 없습니다.</CommandEmpty>
                        <CommandGroup>
                          {availableDistricts.map((district) => (
                            <CommandItem
                              key={district}
                              value={district}
                              onSelect={(currentValue) => {
                                setSelectedDistrict(currentValue)
                                setDistrictOpen(false)
                              }}
                            >
                              {district}
                              <Check
                                className={cn('ml-auto', selectedDistrict === district ? 'opacity-100' : 'opacity-0')}
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

export default LocationHeader
