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
      data: ['동구', '대덕구', '유성구'],
      inverse: true,
    },
    series: [
      {
        name: '포트홀',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: [120, 132, 101],
      },
      {
        name: '균열',
        type: 'bar',
        stack: 'total',
        label: labelOption,
        emphasis: { focus: 'series' },
        data: [150, 132, 201],
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
