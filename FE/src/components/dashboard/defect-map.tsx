'use client'

import DefectListOnMap from '@/components/dashboard/map/defect-list-on-map'
import DefectMapContent from '@/components/dashboard/map/defect-map-content'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { DefectDetail } from '@/types/defects'
import { ArrowRight } from 'lucide-react'
import { useEffect, useMemo, useRef } from 'react'

export default function DefectMap({
  onSelectTab,
  filteredDefectDetailList,
  selectedTab,
}: {
  onSelectTab?: (tab: string) => void
  filteredDefectDetailList: DefectDetail[]
  selectedTab?: string
}) {
  const mapCardRef = useRef<HTMLDivElement>(null)
  const detailCardRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (mapCardRef.current && detailCardRef.current) {
      detailCardRef.current.style.maxHeight = `${mapCardRef.current.clientHeight}px`
    }
  }, [mapCardRef.current?.clientHeight])

  // DefectMapContent 컴포넌트를 메모이제이션
  const memoizedMapContent = useMemo(() => <DefectMapContent />, [])

  return (
    <div
      className={`grid gap-4 md:grid-cols-3 ${selectedTab !== 'map' ? 'hidden' : ''}`}
    >
      <Card ref={mapCardRef} className="md:col-span-2">
        <CardHeader>
          <CardTitle>결함 발생 현황</CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <div className="aspect-video overflow-hidden rounded-md">
            {memoizedMapContent}
          </div>
        </CardContent>
      </Card>

      <Card ref={detailCardRef} className="flex flex-col">
        <CardHeader>
          <CardTitle>지도에 보이는 손상 필터링</CardTitle>
        </CardHeader>
        <CardContent className="grow overflow-y-hidden p-4 pt-0">
          <DefectListOnMap
            filteredDefectDetailList={filteredDefectDetailList}
          />
        </CardContent>
        <CardFooter>
          <Button
            variant="outline"
            size="sm"
            className="w-full"
            onClick={() => onSelectTab && onSelectTab('list')}
          >
            전체 이력 보기
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
