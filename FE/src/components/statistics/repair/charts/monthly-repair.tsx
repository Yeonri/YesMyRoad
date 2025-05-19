import { maintenanceAPI } from '@/api/maintenance-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { MonthlyMaintenanceStatusResponseType } from '@/types/stats-api'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

export default function MonthlyRepair() {
  const chartRef = useRef<ReactECharts>(null)
  const [chartOption, setChartOption] = useState<echarts.EChartsOption>({})
  const [monthlyMaintenanceStatusReport, setMonthlyMaintenanceStatusReport] =
    useState<MonthlyMaintenanceStatusResponseType>({ data: [] })

  const fetchLocalMaintenanceStatusReport = useCallback(async () => {
    const response = await maintenanceAPI.getMonthlyMaintenanceStatus()
    setMonthlyMaintenanceStatusReport({ ...response.data })
  }, [])

  useEffect(() => {
    fetchLocalMaintenanceStatusReport()
  }, [])

  useEffect(() => {
    console.log(
      'monthlyMaintenanceStatusReport: ',
      monthlyMaintenanceStatusReport,
    )
  }, [monthlyMaintenanceStatusReport])

  const rawData = useMemo(
    () => [
      [320, 132, 101, 134, 90, 230, 210, 175, 280, 115, 195, 245],
      [220, 182, 191, 234, 290, 330, 310, 275, 215, 340, 185, 260],
      [150, 212, 201, 154, 190, 330, 410, 240, 180, 370, 295, 225],
    ],
    [],
  )

  const totalData = useMemo(() => {
    const result: number[] = []
    for (let i = 0; i < rawData[0].length; ++i) {
      let sum = 0
      for (let j = 0; j < rawData.length; ++j) {
        sum += rawData[j][i]
      }
      result.push(sum)
    }
    return result
  }, [rawData])

  useEffect(() => {
    if (chartRef.current) {
      const chartInstance = chartRef.current.getEchartsInstance()
      const grid = { left: 100, right: 100, top: 50, bottom: 50 }
      const gridWidth = chartInstance.getWidth() - grid.left - grid.right
      const gridHeight = chartInstance.getHeight() - grid.top - grid.bottom
      const categoryWidth = gridWidth / rawData[0].length
      const barWidth = categoryWidth * 0.6
      const barPadding = (categoryWidth - barWidth) / 2
      const color = ['#e0e0e0', '#ee6666', '#fac858', '#91cc75']

      const series: echarts.BarSeriesOption[] = [
        '미완료',
        '진행중',
        '완료',
      ].map((name, sid) => {
        return {
          name,
          type: 'bar',
          stack: 'total',
          barWidth: '60%',
          label: {
            show: true,
            formatter: (params: echarts.DefaultLabelFormatterCallbackParams) =>
              Math.round((params.value as number) * 1000) / 10 + '%',
          },
          data: rawData[sid].map((d, did) =>
            totalData[did] <= 0 ? 0 : d / totalData[did],
          ),
          itemStyle: { color: color[sid] },
        }
      })

      const option: echarts.EChartsOption = {
        legend: { selectedMode: false },
        grid,
        yAxis: { type: 'value' },
        xAxis: {
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
        series,
      }

      setChartOption(option)
    }
  }, [rawData, totalData])

  return (
    <Card className="col-span-2 h-auto">
      <CardHeader className="p-4">
        <CardTitle className="text-md">월별 도로보수 현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts ref={chartRef} option={chartOption} />
      </CardContent>
    </Card>
  )
}
