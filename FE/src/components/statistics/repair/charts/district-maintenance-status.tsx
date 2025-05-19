import { maintenanceAPI } from '@/api/maintenance-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { DistrictMaintenanceStatusType } from '@/types/stats-api'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { useCallback, useEffect, useMemo, useState } from 'react'

export default function DistrictMaintenanceStatus() {
  const [districtMaintenanceStatus, setDistrictMaintenanceStatus] = useState<
    DistrictMaintenanceStatusType[]
  >([])

  const fetchDistrictMaintenanceStatus = useCallback(async () => {
    const response = await maintenanceAPI.getDistrictMaintenanceStatus()
    console.log('getDistrictMaintenanceStatus: ', response)
    setDistrictMaintenanceStatus(response.data)
  }, [])

  useEffect(() => {
    fetchDistrictMaintenanceStatus()
  }, [])

  useEffect(() => {
    console.log('districtMaintenanceStatus: ', districtMaintenanceStatus)
  }, [districtMaintenanceStatus])

  // 상태별 label 및 색상 정의
  const statusLabels = [
    { key: 'reported', label: '보고됨', color: '#e0e0e0' },
    { key: 'received', label: '접수완료', color: '#ee6666' },
    { key: 'inProgress', label: '작업중', color: '#fac858' },
    { key: 'completed', label: '작업완료', color: '#91cc75' },
  ]

  // 누적합 기준으로 정렬
  const sortedStatus = useMemo(() => {
    return [...districtMaintenanceStatus].sort((a, b) => {
      const sumA = statusLabels.reduce(
        (acc, cur) =>
          acc + Number(a[cur.key as keyof DistrictMaintenanceStatusType] ?? 0),
        0,
      )
      const sumB = statusLabels.reduce(
        (acc, cur) =>
          acc + Number(b[cur.key as keyof DistrictMaintenanceStatusType] ?? 0),
        0,
      )
      return sumA - sumB
    })
  }, [districtMaintenanceStatus])

  const regionNames = sortedStatus.map((item) => item.name)

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    legend: {},
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: { type: 'value' },
    yAxis: { type: 'category', data: regionNames },
    series: statusLabels.map((status) => ({
      name: status.label,
      type: 'bar',
      stack: 'total',
      label: {
        show: true,
        formatter: (params: any) => (params.value === 0 ? '' : params.value),
      },
      emphasis: { focus: 'series' },
      data: sortedStatus.map(
        (item) => item[status.key as keyof DistrictMaintenanceStatusType] ?? 0,
      ),
      itemStyle: { color: status.color },
    })),
  }

  return (
    <Card className="col-span-3 h-auto grow">
      <CardHeader className="p-4">
        <CardTitle className="text-md">구역별 도로 보수 현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}
