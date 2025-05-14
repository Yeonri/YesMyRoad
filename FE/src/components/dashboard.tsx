// src/components/dashboard.tsx
'use client'

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
  DefectCard,
  DefectCardContent,
  DefectCardDescription,
  DefectCardFooter,
  DefectCardHeader,
  DefectCardTitle,
} from '@/components/ui/defect-card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  DefectType,
  SeverityType,
  TimeRangeType,
  useDefectStore,
} from '@/store/defect-store'
import {
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Calendar,
  Clock,
  Filter,
  MapPin,
} from 'lucide-react'
import { useEffect, useState } from 'react'

import DefectHeatmap from './defect-heatmap'
import DefectList from './defect-list'
import DefectMap from './defect-map'
import DefectStats from './defect-stats'
import Header from './header'
import RecentAlerts from './recent-alerts'

// src/components/dashboard.tsx

export default function Dashboard() {
  // Get state and actions from Zustand store
  const {
    timeRange,
    setTimeRange,
    defectType,
    setDefectType,
    severity,
    setSeverity,
    severityCounts,
    dashboardMetrics,
    fetchDefects,
    fetchDefectLocations,
    fetchRecentAlerts,
    fetchDefectStats,
    fetchDefectTrends,
    fetchGeoJSONData, // fetchGeoDataAndDetails 대신 fetchGeoJSONData 사용
  } = useDefectStore()

  // Fetch data on component mount
  useEffect(() => {
    console.log('Dashboard - 데이터 로드 시작')

    // 기존 API 호출
    fetchDefects()
    fetchDefectLocations()
    fetchRecentAlerts()
    fetchDefectStats()
    fetchDefectTrends()

    // GeoJSON 데이터만 먼저 로드
    fetchGeoJSONData()
    console.log('Dashboard - GeoJSON 데이터 로드 요청 완료')
  }, [
    fetchDefects,
    fetchDefectLocations,
    fetchRecentAlerts,
    fetchDefectStats,
    fetchDefectTrends,
    fetchGeoJSONData, // 의존성 배열에 fetchGeoJSONData 추가
  ])

  const [selectedTab, selectTab] = useState('map')

  return (
    <div className="bg-muted/40 flex min-h-screen w-full flex-col">
      <Header />
      <main className="flex flex-1 flex-col gap-4 p-4 md:gap-8 md:p-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">대시보드</h1>
            <p className="text-muted-foreground">
              실시간 도로 결함 모니터링 시스템
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="h-8 gap-1">
              <Calendar className="h-3.5 w-3.5" />
              <span>2025년 4월 24일</span>
            </Button>
            <Button variant="outline" size="sm" className="h-8 gap-1">
              <Clock className="h-3.5 w-3.5" />
              <span>실시간</span>
            </Button>
          </div>
        </div>

        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="rounded-md px-3 py-1">
              <div className="mr-1 h-2 w-2 rounded-full bg-red-500" />
              심각: {severityCounts.critical}
            </Badge>
            <Badge variant="outline" className="rounded-md px-3 py-1">
              <div className="mr-1 h-2 w-2 rounded-full bg-amber-500" />
              위험: {severityCounts.high}
            </Badge>
            <Badge variant="outline" className="rounded-md px-3 py-1">
              <div className="mr-1 h-2 w-2 rounded-full bg-blue-500" />
              주의: {severityCounts.medium}
            </Badge>
            <Badge variant="outline" className="rounded-md px-3 py-1">
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
                <SelectItem value="1h">1시간 이내</SelectItem>
                <SelectItem value="24h">24시간 이내</SelectItem>
                <SelectItem value="7d">7일 이내</SelectItem>
                <SelectItem value="30d">30일 이내</SelectItem>
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
              <div className="text-2xl font-bold">
                {dashboardMetrics.totalDefects}
              </div>
              <p className="text-muted-foreground text-xs">
                지난 주 대비 {dashboardMetrics.totalDefectsChange}
              </p>
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

        <Tabs defaultValue="map" className="space-y-4">
          <div className="flex gap-5">
            <TabsList>
              <TabsTrigger value="map" onClick={() => selectTab('map')}>
                지도
              </TabsTrigger>
              <TabsTrigger value="heatmap" onClick={() => selectTab('hitmap')}>
                히트맵
              </TabsTrigger>
              <TabsTrigger value="list" onClick={() => selectTab('list')}>
                리스트
              </TabsTrigger>
              <TabsTrigger
                value="analytics"
                onClick={() => selectTab('indicators')}
              >
                통계
              </TabsTrigger>
            </TabsList>
            {(selectedTab === 'map' || selectedTab === 'hitmap') && (
              <LocationHeader />
            )}
          </div>
          <TabsContent value="map" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-3">
              <Card className="md:col-span-2">
                <CardHeader>
                  <CardTitle>지도</CardTitle>
                  <CardDescription>도로 결함의 전체 분포</CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="aspect-video overflow-hidden rounded-md">
                    <DefectMap />
                  </div>
                </CardContent>
              </Card>

              <DefectCard>
                <DefectCardHeader>
                  <DefectCardTitle>최근 발생한 결함 이력</DefectCardTitle>
                  <DefectCardDescription>
                    심각한 결함 및 높은 우선순위 결함
                  </DefectCardDescription>
                </DefectCardHeader>
                <DefectCardContent>
                  <RecentAlerts />
                </DefectCardContent>
                <DefectCardFooter>
                  <Button variant="outline" size="sm" className="w-full">
                    View all alerts
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </DefectCardFooter>
              </DefectCard>
            </div>
          </TabsContent>
          <TabsContent value="heatmap" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>결함 밀도 히트맵</CardTitle>
                <CardDescription>결함 집중 지역을 시각화</CardDescription>
              </CardHeader>
              <CardContent className="p-0">
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
