import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useQuery } from '@tanstack/react-query'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'

type BarLabelOption = NonNullable<echarts.BarSeriesOption['label']>
type Region = {
  regionName: string
  totalCount: number
  holeCount: number
  crackCount: number
}

export default function TopThreeRegion(props: { cardHeight: string }) {
  /**
   * 도로파손 분포 Top 3 지역별 파손 유형 : Top 3 행정구역 가로 막대 그래프
   */
  const { cardHeight = 'h-80' } = props

  const { data: apiResponse, isLoading, error } = useQuery({
    queryKey: ['top-three-region'],
    queryFn: statisticAPI.getTop3Damagedlocations,
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1
  })

  const regionData = apiResponse?.data

  // 로딩 중이면 로딩 표시
  if (isLoading) {
    return (
      <Card className={`col-span-2 ${cardHeight} flex items-center justify-center`}>
        <div>데이터 로딩 중...</div>
      </Card>
    )
  }

  // 에러가 있으면 에러 표시
  if (error) {
    return (
      <Card className={`col-span-2 ${cardHeight} flex items-center justify-center`}>
        <div>데이터를 불러오는 중 오류가 발생했습니다.</div>
      </Card>
    )
  }

  // 데이터 추출하기
  const regions = regionData?.map((region: Region) => region.regionName) || [];
  const holeCounts = regionData?.map((region: Region) => region.holeCount) || [];
  const crackCounts = regionData?.map((region: Region) => region.crackCount) || [];

  const labelOption: BarLabelOption = { show: true, valueAnimation: true }

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: {},
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', max: 'dataMax' },
    yAxis: {
      type: 'category',
      data: regions,
      inverse: true,
    },
    series: [
      {
        name: '포트홀',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: holeCounts,
      },
      {
        name: '균열',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: crackCounts,
      },
    ],
  }

  return (
    <Card className={`col-span-3 ${cardHeight}`}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">도로파손 분포 Top 3</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}