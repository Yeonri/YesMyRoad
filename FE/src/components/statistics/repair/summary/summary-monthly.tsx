// import { statisticAPI } from '@/api/statistic-api'
import { statisticAPI } from '@/api/statistic-api'
import SummaryCard from '@/components/statistics/common/SummaryCard'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RepairMonthlyReportType } from '@/types/stats-api'
// import { useEffect, useState } from 'react'
import { useEffect, useState } from 'react'

export default function SummaryMonthly() {
  const [monthlyReport, setMonthlyReport] =
    useState<RepairMonthlyReportType | null>(null)

  // useEffect(() => {
  //   statisticAPI.getRepairMonthlyReport().then((response) => {
  //     setMonthlyReport(response.data)
  //   })
  // }, [])

  return (
    <SummaryCard
      title="월간 도로보수 건수"
      value={monthlyReport?.totalCount ?? '-'}
      className="grow"
    />
  )
}
