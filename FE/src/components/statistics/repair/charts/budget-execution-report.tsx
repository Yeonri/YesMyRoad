import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'

export default function BudgetExecutionReport() {
  const option: echarts.EChartsOption = {
    color: ['#5470C6', '#91CC75'],
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },

    legend: {},
    xAxis: [
      {
        type: 'category',
        axisTick: { alignWithLabel: true },
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
    yAxis: [
      {
        type: 'value',
        name: '월별 집행 금액[백만원]',
        position: 'left',
        alignTicks: true,
        axisLine: { show: true, lineStyle: { color: '#5470C6' } },
        axisLabel: { formatter: '{value}' },
      },
      {
        type: 'value',
        name: '누적 집행률[%]',
        position: 'right',
        alignTicks: true,
        min: 0,
        max: 100,
        axisLine: { show: true, lineStyle: { color: '#91CC75' } },
        axisLabel: {
          formatter: function (value) {
            return value.toFixed(1) + '%'
          },
        },
      },
    ],
    series: [
      {
        name: '월별 집행 금액 (백만원)',
        type: 'bar',
        data: [120, 180, 250, 310, 420, 590, 650, 720, 830, 940, 980, 1050],
      },
      {
        name: '누적 집행률',
        type: 'line',
        yAxisIndex: 1,
        data: [8, 16, 25, 38, 48, 59, 70, 80, 88, 95, 98, 100],
      },
    ],
  }

  return (
    <Card className="col-span-3 h-auto">
      <CardHeader className="p-4">
        <CardTitle className="text-md">예산 집행 현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}
