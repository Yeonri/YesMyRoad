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
        name: '도로보수 유형',
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
        label: { show: false, position: 'center' },
        emphasis: { label: { show: true, fontSize: 16, fontWeight: 'bold' } },
        data: [
          { value: 30, name: '미완료', itemStyle: { color: '#ee6666' } },
          { value: 25, name: '진행중', itemStyle: { color: '#fac858' } },
          { value: 15, name: '완료', itemStyle: { color: '#91cc75' } },
        ],
      },
    ],
  }

  return (
    <Card className={`col-span-2 ${cardHeight}`}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">대전시 도로보수 진행현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}
