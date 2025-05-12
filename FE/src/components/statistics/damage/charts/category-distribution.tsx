import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'

export default function CategoryDistribution(props: { cardHeight: string }) {
  const { cardHeight = 'h-80' } = props

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'item' },
    legend: { top: '5%', left: 'center', textStyle: { fontSize: 12 } },
    series: [
      {
        name: '도로파손 유형',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
        label: { show: false, position: 'center' },
        emphasis: { label: { show: true, fontSize: 16, fontWeight: 'bold' } },
        labelLine: { show: false },
        data: [
          { value: 35, name: '포트홀' },
          { value: 20, name: '침하' },
          { value: 15, name: '균열' },
          { value: 10, name: '패임' },
          { value: 5, name: '기타' },
        ],
      },
    ],
  }

  return (
    <Card className={cardHeight}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">유형별 도로파손 분포</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}
