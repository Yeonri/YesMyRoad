'use client'

import { coordinateAPI } from '@/api/coordinate-api'
import { statisticAPI } from '@/api/statistic-api'
import LocationHeader from '@/components/location-header'
import { Badge } from '@/components/ui/badge'
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
import {
  DefectDetail,
  DefectType,
  SeverityType,
  TimeRangeType,
  useDefectStore,
} from '@/store/defect-store'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, BarChart3, Clock, Filter, MapPin } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

import DefectHeatmap from './defect-heatmap'
import DefectList from './defect-list'
import DefectMap from './defect-map'
import DefectStats from './defect-stats'
import Header from './header'

export default function Dashboard() {
  // Get state and actions from Zustand store
  const {
    timeRange,
    setTimeRange,
    defectType,
    defectDetailList,
    setDefectType,
    severity,
    setSeverity,
    severityCounts,
    dashboardMetrics,
    // updateDefects,
    // updateDefectLocations,
    // updateRecentAlerts,
    // updateDefectStats,
    // updateDefectTrends,
    updateGeoJSONData,
    // getGeoJSONData,
    updateDefectDetailList,
  } = useDefectStore()

  const [selectedTab, selectTab] = useState<string>('map')

  // TanStack Query 사용하여 데이터 로드
  const {
    // data: geoJsonData,
    refetch: refetchGeoJson,
    // isLoading,
    // error,
  } = useQuery({
    queryKey: ['defects'],
    queryFn: async () => {
      const response = await coordinateAPI.getDefectLocations()
      if (response.status === 200 && response.data) {
        // 스토어 업데이트
        updateGeoJSONData(response.data)
        return response.data
      }
      throw new Error('데이터 로드 실패')
    },
  })

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

  const loadDefectDetails = () => {
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
  }

  const loadLocationData = async () => {
    console.log(`GeoJSON 데이터 로딩 시작`)
    const response = await coordinateAPI.getDefectLocations()
    if (response.status === 200 && response.data) {
      console.log(
        `GeoJSON 데이터 로드 성공: ${response.data.features!.length || 0} 개의 데이터`,
      )
      // console.log('데이터 목록: ', response.data.features!)

      updateGeoJSONData(response.data.features!)
    }
  }
  // 통계 데이터 로드 (timeRange 변경에 따라 자동으로 재요청)
  const { data: reportData } = useQuery({
    queryKey: ['reports', timeRange],
    queryFn: async () => {
      // API 함수 매핑 - timeRange에 따라 다른 API 호출
      const apiCalls = {
        'D': statisticAPI.getDamageDailyReport,
        'W': statisticAPI.getDamageWeeklyReport,
        'M': statisticAPI.getDamageMonthlyReport
      }

      // timeRange에 해당하는 API 함수 호출
      const apiFunction = apiCalls[timeRange] || apiCalls['D'] // 기본값은 일간 보고서
      const response = await apiFunction()

      console.log(`${timeRange} 보고서:`, response.data, '상태 코드:', response.status)
      return response.data
    },
    enabled: !!timeRange,
  })

  useEffect(() => {
    loadLocationData()
  }, [loadLocationData])

  useEffect(() => {
    if (geoJSONData === null || geoJSONData.length === 0) return
    loadDefectDetails()
  }, [geoJSONData, loadDefectDetails])

  useEffect(() => {
    console.log(defectDetailList)
  }, [defectDetailList])

  return (
    <div className="bg-muted/40 flex min-h-screen w-full flex-col">
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-wrap gap-2">
            <Badge variant="default" className="rounded-md px-3 py-1">
              <div className="mr-1 h-2 w-2 rounded-full bg-red-500" />
              심각: {severityCounts.critical}
            </Badge>
            <Badge variant="default" className="rounded-md px-3 py-1">
              <div className="mr-1 h-2 w-2 rounded-full bg-amber-500" />
              위험: {severityCounts.high}
            </Badge>
            <Badge variant="default" className="rounded-md px-3 py-1">
              <div className="mr-1 h-2 w-2 rounded-full bg-blue-500" />
              주의: {severityCounts.medium}
            </Badge>
            <Badge variant="default" className="rounded-md px-3 py-1">
              <div className="mr-1 h-2 w-2 rounded-full bg-green-500" />
              안전: {severityCounts.low}
            </Badge>
          </div>
          <div className="flex flex-wrap gap-2">
            <Select
              value={timeRange}
              onValueChange={(value) => setTimeRange(value as TimeRangeType)}
            >
              <SelectTrigger className="h-8 w-[130px]">
                <SelectValue placeholder="Time Range" />
              </SelectTrigger>
              <SelectContent>
                {/* <SelectItem value="1h">1시간 이내</SelectItem> */}
                <SelectItem value="D">24시간 이내</SelectItem>
                <SelectItem value="W">7일 이내</SelectItem>
                <SelectItem value="M">30일 이내</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={defectType}
              onValueChange={(value) => setDefectType(value as DefectType)}
            >
              <SelectTrigger className="h-8 w-[130px]">
                <SelectValue placeholder="Defect Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체</SelectItem>
                <SelectItem value="pothole">포트홀</SelectItem>
                <SelectItem value="crack">깨짐</SelectItem>
                <SelectItem value="paint">페인트 벗겨짐</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={severity}
              onValueChange={(value) => setSeverity(value as SeverityType)}
            >
              <SelectTrigger className="h-8 w-[130px]">
                <SelectValue placeholder="Severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체</SelectItem>
                <SelectItem value="critical">심각</SelectItem>
                <SelectItem value="high">높음</SelectItem>
                <SelectItem value="medium">중간</SelectItem>
                <SelectItem value="low">낮음</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={severity}
              onValueChange={(value) => setSeverity(value as SeverityType)}
            >
              <SelectTrigger className="h-8 w-[130px]">
                <SelectValue placeholder="Severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체</SelectItem>
                <SelectItem value="critical">심각</SelectItem>
                <SelectItem value="high">높음</SelectItem>
                <SelectItem value="medium">중간</SelectItem>
                <SelectItem value="low">낮음</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" className="h-8 gap-1">
              <Filter className="h-3.5 w-3.5" />
              <span>필터 더보기</span>
            </Button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">총 결함 수</CardTitle>
              <BarChart3 className="text-muted-foreground h-4 w-4" />
            </CardHeader>
            <CardContent>
              {/* <div className="text-2xl font-bold">
                {reportData?.data?.count || 0}
              </div>
              <p className="text-muted-foreground text-xs">
                {{ D: '어제', W: '지난 주', M: '지난 달' }[timeRange] || ''}{' '}
                대비{' '}
                {reportData?.data?.changeRate === null ? '-' : (reportData?.data?.changeRate || 0)}{' '}
                % 증가
              </p> */}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                심각한 결함 수
              </CardTitle>
              <AlertTriangle className="text-muted-foreground h-4 w-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {dashboardMetrics.criticalIssues}
              </div>
              <p className="text-muted-foreground text-xs">
                지난 주 대비 {dashboardMetrics.criticalIssuesChange}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                평균 작업 착수 시간
              </CardTitle>
              <Clock className="text-muted-foreground h-4 w-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {dashboardMetrics.avgResponseTime}
              </div>
              <p className="text-muted-foreground text-xs">
                지난 주 대비 {dashboardMetrics.avgResponseTimeChange}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                발생 행정구역 수
              </CardTitle>
              <MapPin className="text-muted-foreground h-4 w-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {dashboardMetrics.affectedAreas}
              </div>
              <p className="text-muted-foreground text-xs">
                지난 주 대비 {dashboardMetrics.affectedAreasChange}
              </p>
            </CardContent>
          </Card>
        </div>

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
                <TabsTrigger value="list">리스트</TabsTrigger>
                <TabsTrigger value="analytics">통계</TabsTrigger>
              </TabsList>
              <LocationHeader />
            </div>
            <Button
              onClick={() => loadLocationData()}
              className="active:bg-primary/70 active:translate-y-0.5 active:scale-95"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                <path d="M21 3v5h-5" />
                <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                <path d="M3 21v-5h5" />
              </svg>
              결함 현황 새로고침
            </Button>
          </div>
          <TabsContent value="map" className="space-y-4">
            <DefectMap
              onSelectTab={selectTab}
              filteredDefectDetailList={filteredDefectDetailList}
            />
          </TabsContent>
          <TabsContent value="heatmap" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>결함 밀도 히트맵</CardTitle>
                <CardDescription>결함 집중 지역을 시각화</CardDescription>
              </CardHeader>
              <CardContent className="p-4 pt-0">
                <div className="aspect-video overflow-hidden rounded-md">
                  <DefectHeatmap />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="list" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>결함 리스트</CardTitle>
                <CardDescription>모든 결함을 심각도별로 정렬</CardDescription>
              </CardHeader>
              <CardContent>
                <DefectList />
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
