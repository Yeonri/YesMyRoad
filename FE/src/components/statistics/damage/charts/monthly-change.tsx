import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'

export default function MonthlyChange(props: { cardHeight: string }) {
  const { cardHeight = 'h-80' } = props

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: { show: false },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: [
      {
        type: 'category',
        data: [
          '1월',
          '2월',
          '3월',
          '4월',
          '5월',
          '6월',
          '7월',
          '8월',
          '9월',
          '10월',
          '11월',
          '12월',
        ],
      },
    ],
    yAxis: [{ type: 'value' }],
    series: [
      // 동구
      {
        name: '포트홀',
        type: 'bar',
        stack: '동구',
        data: [12, 3, 17, 8, 19, 2, 15, 7, 20, 1, 13, 6],
      },
      {
        name: '균열',
        type: 'bar',
        stack: '동구',
        data: [5, 18, 2, 14, 7, 11, 4, 16, 9, 12, 3, 20],
      },
      // 중구
      {
        name: '포트홀',
        type: 'bar',
        stack: '중구',
        data: [8, 15, 1, 19, 6, 13, 10, 2, 17, 5, 14, 7],
      },
      {
        name: '균열',
        type: 'bar',
        stack: '중구',
        data: [11, 4, 16, 9, 12, 3, 20, 5, 18, 2, 14, 7],
      },
      // 서구
      {
        name: '포트홀',
        type: 'bar',
        stack: '서구',
        data: [7, 20, 5, 14, 8, 15, 1, 19, 6, 13, 10, 2],
      },
      {
        name: '균열',
        type: 'bar',
        stack: '서구',
        data: [13, 10, 2, 17, 5, 14, 7, 11, 4, 16, 9, 12],
      },
      // 유성구
      {
        name: '포트홀',
        type: 'bar',
        stack: '유성구',
        data: [3, 17, 8, 19, 2, 15, 7, 20, 1, 13, 6, 12],
      },
      {
        name: '균열',
        type: 'bar',
        stack: '유성구',
        data: [16, 9, 12, 3, 20, 5, 18, 2, 14, 7, 11, 4],
      },
      // 대덕구
      {
        name: '포트홀',
        type: 'bar',
        stack: '대덕구',
        data: [19, 2, 15, 7, 20, 1, 13, 6, 12, 3, 17, 8],
      },
      {
        name: '균열',
        type: 'bar',
        stack: '대덕구',
        data: [2, 14, 7, 11, 4, 16, 9, 12, 3, 20, 5, 18],
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
