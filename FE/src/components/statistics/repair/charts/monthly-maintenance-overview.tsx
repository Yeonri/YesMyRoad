import { maintenanceAPI } from '@/api/maintenance-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { MonthlyMaintenanceOverviewType } from '@/types/stats-api'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { useCallback, useEffect, useState } from 'react'

export default function MonthlyMaintenanceOverview() {
  const [monthlyMaintenanceOverview, setMonthlyMaintenanceOverview] =
    useState<MonthlyMaintenanceOverviewType>({
      reported: 0,
      received: 0,
      inProgress: 0,
      completed: 0,
    })

  const fetchMonthlyMaintenanceStatus = useCallback(async () => {
    const response = await maintenanceAPI.getMaintenanceOverview()
    console.log('getMaintenanceOverview: ', response)
    setMonthlyMaintenanceOverview({ ...response.data })
  }, [])

  useEffect(() => {
    fetchMonthlyMaintenanceStatus()
  }, [])

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'item' },
    legend: { top: '5%', left: 'center', textStyle: { fontSize: 12 } },
    series: [
      {
        name: '도로보수 유형',
        type: 'pie',
        radius: ['40%', '70%'],
        color: ['#e0e0e0', '#ee6666', '#fac858', '#91cc75'],
        avoidLabelOverlap: false,
        itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
        label: { show: false, position: 'center' },
        emphasis: { label: { show: true, fontSize: 16, fontWeight: 'bold' } },
        data: [
          { value: monthlyMaintenanceOverview.reported, name: '보고됨' },
          { value: monthlyMaintenanceOverview.received, name: '접수완료' },
          { value: monthlyMaintenanceOverview.inProgress, name: '작업중' },
          { value: monthlyMaintenanceOverview.completed, name: '작업완료' },
        ],
      },
    ],
  }

  return (
    <Card className="col-span-2 h-auto">
      <CardHeader className="p-4">
        <CardTitle className="text-md">월간 도로보수 진행현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <ReactECharts option={option} />
      </CardContent>
    </Card>
  )
}
