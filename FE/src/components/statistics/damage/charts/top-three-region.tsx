import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'

export default function TopThreeRegion(props: { cardHeight: string }) {
  /**
   * 도로파손 분포 Top 3 지역별 파손 유형 : Top 3 행정구역 가로 막대 그래프
   */
  const { cardHeight = 'h-80' } = props

  type BarLabelOption = NonNullable<echarts.BarSeriesOption['label']>

  const labelOption: BarLabelOption = { show: true, valueAnimation: true }

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: {},
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value', max: 'dataMax' },
    yAxis: {
      type: 'category',
      data: ['강남구', '서초구', '송파구'],
      inverse: true,
    },
    series: [
      {
        realtimeSort: true,
        name: '포트홀',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: [120, 132, 101],
      },
      {
        name: '침하',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: [220, 182, 191],
      },
      {
        name: '균열',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: [150, 132, 201],
      },
      {
        name: '패임',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: [98, 77, 101],
      },
      {
        name: '기타',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: [40, 30, 50],
      },
    ],
  }

  return (
    <Card className={cardHeight}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">도로파손 분포 Top 3</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}
