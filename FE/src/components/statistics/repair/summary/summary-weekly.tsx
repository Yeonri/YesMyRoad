// import { statisticAPI } from '@/api/statistic-api'
import SummaryCard from '@/components/statistics/common/SummaryCard'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RepairWeeklyReportType } from '@/types/stats-api'
// import { useEffect, useState } from 'react'
import { useState } from 'react'

export default function SummaryWeekly() {
  const [weeklyReport, setWeeklyReport] =
    useState<RepairWeeklyReportType | null>(null)

  // useEffect(() => {
  //   statisticAPI.getRepairWeeklyReport().then((response) => {
  //     setWeeklyReport(response.data.totalCount)
  //   })
  // }, [])

  return (
    <SummaryCard
      title="주간 도로보수 건수"
      value={weeklyReport?.totalCount ?? '-'}
      className="grow"
    />
  )
}
