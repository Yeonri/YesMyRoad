// import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RepairDailyReportType } from '@/types/stats-api'
// import { useEffect, useState } from 'react'
import { useState } from 'react'
import SummaryCard from '@/components/statistics/common/SummaryCard'

export default function SummaryDaily() {
  const [dailyReport, setDailyReport] = useState<RepairDailyReportType | null>(
    null,
  )

  // useEffect(() => {
  //   statisticAPI.getRepairDailyReport().then((response) => {
  //     setDailyReport(response.data)
  //   })
  // }, [])

  return (
    <SummaryCard
      title="일간 도로보수 건수"
      value={dailyReport?.totalCount ?? '-'}
      className="grow"
    />
  )
}
