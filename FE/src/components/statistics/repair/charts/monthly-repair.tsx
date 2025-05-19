import { maintenanceAPI } from '@/api/maintenance-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { MonthlyMaintenanceStatusType } from '@/types/stats-api'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

export default function MonthlyRepair() {
  const chartRef = useRef<ReactECharts>(null)
  const [monthlyMaintenanceStatusReport, setMonthlyMaintenanceStatusReport] =
    useState<MonthlyMaintenanceStatusType[]>([])

  const fetchMonthlyMaintenanceStatusReport = useCallback(async () => {
    const response = await maintenanceAPI.getMonthlyMaintenanceStatus()
    setMonthlyMaintenanceStatusReport(response.data)
  }, [])

  useEffect(() => {
    fetchMonthlyMaintenanceStatusReport()
  }, [])

  useEffect(() => {
    console.log(
      'monthlyMaintenanceStatusReport: ',
      monthlyMaintenanceStatusReport,
    )
  }, [monthlyMaintenanceStatusReport])

  const statusLabels = [
    { key: 'reported', label: '보고됨', color: '#e0e0e0' },
    { key: 'received', label: '접수완료', color: '#ee6666' },
    { key: 'inProgress', label: '작업중', color: '#fac858' },
    { key: 'completed', label: '작업완료', color: '#91cc75' },
  ]

  const xAxisData = useMemo(
    () =>
      monthlyMaintenanceStatusReport.map((item) => {
        const date = new Date(`${item.month}-01`)
        return date.toLocaleString('ko-KR', { month: 'numeric' }) + '월'
      }),
    [monthlyMaintenanceStatusReport],
  )

  const rawData = useMemo(
    () =>
      statusLabels.map((status) =>
        monthlyMaintenanceStatusReport.map(
          (item) =>
            item[status.key as keyof MonthlyMaintenanceStatusType] as number,
        ),
      ),
    [monthlyMaintenanceStatusReport],
  )

  const totalData = useMemo(() => {
    const result: number[] = []
    for (let i = 0; i < rawData[0]?.length || 0; ++i) {
      let sum = 0
      for (let j = 0; j < rawData.length; ++j) {
        sum += rawData[j][i]
      }
      result.push(sum)
    }
    return result
  }, [rawData])

  const option = useMemo(() => {
    const grid = { left: '3%', right: '4%', bottom: '3%', containLabel: true }

    const series: echarts.BarSeriesOption[] = statusLabels.map(
      (status, sid) => {
        return {
          name: status.label,
          type: 'bar',
          stack: 'total',
          barWidth: '50%',
          label: {
            show: true,
            formatter: (params: echarts.DefaultLabelFormatterCallbackParams) =>
              Math.round((params.value as number) * 1000) / 10 + '%',
          },
          data: rawData[sid].map((d, did) =>
            totalData[did] <= 0 ? 0 : d / totalData[did],
          ),
          itemStyle: { color: status.color },
        }
      },
    )

    return {
      legend: { selectedMode: false },
      grid,
      yAxis: { type: 'value' },
      xAxis: { type: 'category', data: xAxisData },
      series,
    }
  }, [rawData, totalData, xAxisData])

  return (
    <Card className="col-span-3 h-auto">
      <CardHeader className="p-4">
        <CardTitle className="text-md">월별 도로보수 현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts ref={chartRef} option={option} />
      </CardContent>
    </Card>
  )
}
