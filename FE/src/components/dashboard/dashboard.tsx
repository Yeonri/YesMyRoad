'use client'

import { coordinateAPI } from '@/api/coordinate-api'
import DefectHeatmap from '@/components/dashboard/defect-heatmap'
import DefectMap from '@/components/dashboard/defect-map'
import DefectOverall from '@/components/dashboard/defect-overall'
import DefectStats from '@/components/dashboard/defect-stats'
import Header from '@/components/dashboard/header'
import DefectList from '@/components/dashboard/list/defect-list'
import SeverityBadges from '@/components/dashboard/severity-badges'
// import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import useAddressStore from '@/store/address-store'
import { useDefectStore } from '@/store/defect-store'
import { DefectDetail } from '@/types/defects'
import { useCallback, useEffect, useMemo, useState } from 'react'

// import { statisticAPI } from '@/api/statistic-api'
import AddressSelector from './address-selector'

export default function Dashboard() {
  // Get state and actions from Zustand store
  const { updateGeoJSONData, defectDetailList, updateDefectDetailList } =
    useDefectStore()

  const [selectedTab, selectTab] = useState<string>('map') // 지도/히트맵/목록/통계 탭 선택용 상태 변수
  const [selectedFilter, selectFilter] = useState<string>('timeRange') // 목록 탭에서 시간/결함 유형/심각도 필터를 고르는 필터
  const [selectedTimeRange, selectTimeRange] = useState<string>('all') // 목록 탭에서 시간 필터
  const [selectedDefectType, selectDefectType] = useState<string>('all') // 목록 탭에서 결함 유형 필터
  const [selectedSeverity, selectSeverity] = useState<string>('all') // 목록 탭에서 심각도 필터
  const [selectedProcess, selectProcess] = useState<string>('all') // 목록 탭에서 심각도 필터
  const geoJSONData = useDefectStore((state) => state.geoJSONData)
  const mapBounds = useAddressStore((state) => state.mapBounds)

  const filteredPublicIdsByBounds = useMemo(() => {
    if (
      !geoJSONData ||
      !mapBounds ||
      !mapBounds.southWest ||
      !mapBounds.northEast
    )
      return []

    const filteredFeatures = geoJSONData.filter((feature) => {
      // coordinates[0]은 위도(latitude), coordinates[1]은 경도(longitude)
      const lat = feature.geometry.coordinates[0]
      const lng = feature.geometry.coordinates[1]

      // null 체크를 통해 안전하게 비교
      if (
        mapBounds.southWest.lat === null ||
        mapBounds.northEast.lat === null ||
        mapBounds.southWest.lng === null ||
        mapBounds.northEast.lng === null
      ) {
        return false
      }

      return (
        mapBounds.southWest.lat <= lat &&
        lat <= mapBounds.northEast.lat &&
        mapBounds.southWest.lng <= lng &&
        lng <= mapBounds.northEast.lng
      )
    })

    // 필터링된 feature들에서 publicId만 추출하여 문자열 배열로 반환
    return filteredFeatures.map(
      (feature) => feature.properties.publicId as string,
    )
  }, [geoJSONData, mapBounds])

  const filteredDefectDetailList = useMemo(() => {
    if (defectDetailList === null || defectDetailList.length === 0) return []
    return defectDetailList.filter((detail) =>
      filteredPublicIdsByBounds.includes(detail.publicId),
    )
  }, [defectDetailList, filteredPublicIdsByBounds])

  // DefectMap과 DefectHeatmap 컴포넌트를 메모이제이션
  const memoizedDefectMap = useMemo(
    () => (
      <DefectMap
        onSelectTab={selectTab}
        filteredDefectDetailList={filteredDefectDetailList}
        selectedTab={selectedTab}
      />
    ),
    [filteredDefectDetailList, selectedTab],
  )

  const memoizedDefectHeatmap = useMemo(() => <DefectHeatmap />, [])

  const loadDefectDetails = useCallback(() => {
    // 지도 데이터가 로드되지 않았거나 데이터가 없으면 종료
    if (geoJSONData === null || geoJSONData.length === 0) return

    const defectDetailList: DefectDetail[] = []

    // 지도 데이터의 각 feature에 대해 손상 상세 정보를 조회
    geoJSONData.forEach(async (feature) => {
      const publicId = feature.properties.publicId
      const response = await coordinateAPI.getDefectDetail(publicId)
      const defectDetail: DefectDetail = {
        publicId,
        address: feature.properties.address.street,
        ...response.data,
      }
      defectDetailList.push(defectDetail)
    })

    updateDefectDetailList(defectDetailList)
  }, [geoJSONData, updateDefectDetailList])

  const loadLocationData = useCallback(async () => {
    console.log(`GeoJSON 데이터 로딩 시작`)
    const response = await coordinateAPI.getDefectLocations()
    if (response.status === 200 && response.data) {
      console.log(
        `GeoJSON 데이터 로드 성공: ${response.data.features!.length || 0} 개의 데이터`,
      )
      updateGeoJSONData(response.data.features!)
    }
  }, [updateGeoJSONData])

  useEffect(() => {
    loadLocationData()
  }, [])

  useEffect(() => {
    if (geoJSONData === null || geoJSONData.length === 0) return
    loadDefectDetails()
  }, [geoJSONData])

  useEffect(() => {
    console.log('defectDetailList: ', defectDetailList)
  }, [defectDetailList])

  // // 스토어가 제대로 작동하는지 확인용 -> 정상 작동 확인완료
  // useEffect(() => {
  //   console.log("GeoJSONData updated:", geoJSONData)
  // }, [geoJSONData])

  return (
    <div className="bg-muted/40 flex min-h-screen w-full flex-col">
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <SeverityBadges />
        <DefectOverall />
        <Tabs
          defaultValue="map"
          value={selectedTab}
          onValueChange={selectTab}
          className="space-y-4"
        >
          <div className="flex justify-between">
            <div className="flex gap-5">
              <TabsList>
                <TabsTrigger value="map">지도</TabsTrigger>
                <TabsTrigger value="heatmap">히트맵</TabsTrigger>
                <TabsTrigger value="list">목록</TabsTrigger>
                <TabsTrigger value="analytics">통계</TabsTrigger>
              </TabsList>

              {(selectedTab === 'map' || selectedTab == 'heatmap') && <AddressSelector />}
            </div>
          </div>
          <TabsContent value="map" className="space-y-4" forceMount>
            {memoizedDefectMap}
          </TabsContent>
          <TabsContent value="heatmap" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>결함 밀도 히트맵</CardTitle>
                <CardDescription>결함 집중 지역을 시각화</CardDescription>
              </CardHeader>
              <CardContent className="p-4 pt-0">
                <div className="aspect-video overflow-hidden rounded-md">
                  {memoizedDefectHeatmap}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="list" className="space-y-4">
            <Card>
              <CardHeader>
                <div className="flex justify-between">
                  <div>
                    <CardTitle>결함 목록</CardTitle>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {selectedFilter === 'timeRange' && (
                      <Select
                        value={selectedTimeRange}
                        onValueChange={selectTimeRange}
                      >
                        <SelectTrigger className="h-8 w-[130px]">
                          <SelectValue placeholder="Time Range" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">전체</SelectItem>
                          <SelectItem value="daily">24시간 이내</SelectItem>
                          <SelectItem value="weekly">7일 이내</SelectItem>
                          <SelectItem value="monthly">30일 이내</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                    {selectedFilter === 'type' && (
                      <Select
                        value={selectedDefectType}
                        onValueChange={selectDefectType}
                      >
                        <SelectTrigger className="h-8 w-[130px]">
                          <SelectValue placeholder="Defect Type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">전체</SelectItem>
                          <SelectItem value="pothole">도로 홀</SelectItem>
                          <SelectItem value="crack">도로균열</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                    {selectedFilter === 'severity' && (
                      <Select
                        value={selectedSeverity}
                        onValueChange={selectSeverity}
                      >
                        <SelectTrigger className="h-8 w-[130px]">
                          <SelectValue placeholder="Severity" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">전체</SelectItem>
                          <SelectItem value="critical">심각</SelectItem>
                          <SelectItem value="danger">위험</SelectItem>
                          <SelectItem value="caution">주의</SelectItem>
                          <SelectItem value="safe">안전</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                    {selectedFilter === 'process' && (
                      <Select
                        value={selectedProcess}
                        onValueChange={selectProcess}
                      >
                        <SelectTrigger className="h-8 w-[130px]">
                          <SelectValue placeholder="Process" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">전체</SelectItem>
                          <SelectItem value="REPORTED">REPORTED</SelectItem>
                          <SelectItem value="RECEIVED">RECEIVED</SelectItem>
                          <SelectItem value="IN_PROGRESS">IN_PROGRESS</SelectItem>
                          <SelectItem value="COMPLETED">COMPLETED</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                    <Select value={selectedFilter} onValueChange={selectFilter}>
                      <SelectTrigger className="h-8 w-[130px]">
                        <SelectValue placeholder="필터 선택" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="timeRange">발생 시각</SelectItem>
                        <SelectItem value="type">결함 유형</SelectItem>
                        <SelectItem value="severity">심각도</SelectItem>
                        <SelectItem value="process">작업 현황</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <DefectList filter={selectedFilter} timeRange={selectedTimeRange} defectType={selectedDefectType} severity={selectedSeverity} process={selectedProcess} />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="analytics" className="space-y-4">
            <DefectStats />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
