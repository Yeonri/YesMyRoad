import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useQuery } from '@tanstack/react-query'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'

type monthItem = {
  month:string
  crackCount: number
  holeCount: number
  totalCount: number
}

export default function MonthlyChange(props: { cardHeight: string }) {
  const { cardHeight = 'h-80' } = props

  const { data: response, isLoading, error } = useQuery({
    queryKey: ['monthly-change'],
    queryFn: statisticAPI.getSummarizedDamageMonthlyReport,
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1
  })

  // const month = ["2025-04", "2025-05"]
  const month = response?.data.map((item:monthItem) => item.month)
  const crackCounts = response?.data.map((item:monthItem) => item.crackCount)
  const holeCounts = response?.data.map((item:monthItem) => item.holeCount)

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

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { show: false },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: [
      {
        type: 'category',
        data: month,
      },
    ],
    yAxis: [{ type: 'value' }],
    series: [
      // month
      {
        name: '포트홀',
        type: 'bar',
        stack: 'month',
        data: holeCounts,
      },
      {
        name: '균열',
        type: 'bar',
        stack: 'month',
        data: crackCounts,
      },
    ],
  }

  return (
    <Card className={`col-span-2 ${cardHeight}`}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">월별 도로파손 현황</CardTitle>
        <CardContent className="p-4 pt-0">
          <ReactECharts option={option} />
        </CardContent>
      </CardHeader>
    </Card>
  )
}
