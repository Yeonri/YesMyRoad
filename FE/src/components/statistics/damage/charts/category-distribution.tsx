import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { statisticAPI } from '@/api/statistic-api'
import { useQuery } from '@tanstack/react-query'

type category = {
  categoryName: string
  count: number
}

export default function CategoryDistribution(props: { cardHeight: string }) {
  const { cardHeight = 'h-80' } = props

  const { data: response, isLoading, error } = useQuery({
    queryKey: ['category-distribution'],
    queryFn: statisticAPI.getDamageReportByType,
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1
  })

  const chartData = response?.data.map((item: category) => ({
    value: item.count,
    name: item.categoryName
  })) || []


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
    tooltip: { trigger: 'item' },
    legend: { top: '5%', left: 'center', textStyle: { fontSize: 12 } },
    series: [
      {
        name: '도로파손 유형',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 10, borderColor: '#FFFFFF', borderWidth: 2 },
        label: { show: false, position: 'center' },
        emphasis: { label: { show: true, fontSize: 16, fontWeight: 'bold' } },
        labelLine: { show: false },
        data: chartData
      },
    ],
  }

  return (
    <Card className={`col-span-2 ${cardHeight}`}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">유형별 도로파손 분포</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}
