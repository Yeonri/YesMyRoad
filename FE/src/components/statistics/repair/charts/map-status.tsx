import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'

export default function MapStatus(props: { cardHeight: string }) {
  const { cardHeight = 'h-80' } = props

  // 지역별 파이 차트 데이터
  const regionData: Record<string, Array<{ value: number; name: string }>> = {
    동구: [
      { value: 15, name: '미완료' },
      { value: 10, name: '진행중' },
      { value: 5, name: '완료' },
    ],
    중구: [
      { value: 20, name: '미완료' },
      { value: 12, name: '진행중' },
      { value: 8, name: '완료' },
    ],
    서구: [
      { value: 18, name: '미완료' },
      { value: 15, name: '진행중' },
      { value: 12, name: '완료' },
    ],
    유성구: [
      { value: 25, name: '미완료' },
      { value: 15, name: '진행중' },
      { value: 10, name: '완료' },
    ],
    대덕구: [
      { value: 12, name: '미완료' },
      { value: 10, name: '진행중' },
      { value: 8, name: '완료' },
    ],
  }

  // 상태별 색상
  const statusColors: Record<string, string> = {
    미완료: '#ee6666',
    진행중: '#fac858',
    완료: '#91cc75',
  }

  const option: echarts.EChartsOption = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        // Use axis to trigger tooltip
        type: 'shadow', // 'shadow' as default; can also be 'line' or 'shadow'
      },
    },
    legend: {},
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value' },
    yAxis: {
      type: 'category',
      // regionData의 key 값을 data 배열로 사용
      data: Object.keys(regionData),
    },
    series: [
      {
        name: '미완료',
        type: 'bar',
        stack: 'total',
        label: { show: true },
        emphasis: { focus: 'series' },
        // regionData의 미완료 데이터를 data 배열로 사용
        data: Object.values(regionData).map(
          (items) => items.find((item) => item.name === '미완료')?.value || 0,
        ),
        itemStyle: { color: statusColors['미완료'] },
      },
      {
        name: '진행중',
        type: 'bar',
        stack: 'total',
        label: { show: true },
        emphasis: { focus: 'series' },
        // regionData의 진행중 데이터를 data 배열로 사용
        data: Object.values(regionData).map(
          (items) => items.find((item) => item.name === '진행중')?.value || 0,
        ),
        itemStyle: { color: statusColors['진행중'] },
      },
      {
        name: '완료',
        type: 'bar',
        stack: 'total',
        label: { show: true },
        emphasis: { focus: 'series' },
        // regionData의 완료 데이터를 data 배열로 사용
        data: Object.values(regionData).map(
          (items) => items.find((item) => item.name === '완료')?.value || 0,
        ),
        itemStyle: { color: statusColors['완료'] },
      },
    ],
  }

  return (
    <Card className={`col-span-3 grow ${cardHeight}`}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">구역별 도로 보수 현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}
