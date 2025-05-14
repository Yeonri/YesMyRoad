import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { DamageWeeklyReportType } from '@/types/stats-api'
import { useEffect, useState } from 'react'

export default function SummaryWeekly() {
  const [weeklyReport, setWeeklyReport] =
    useState<DamageWeeklyReportType | null>(null)

  useEffect(() => {
    statisticAPI.getDamageWeeklyReport().then((response) => {
      setWeeklyReport(response.data)
    })
  }, [])

  return (
    <Card className="grow">
      <CardHeader className="p-4">
        <CardTitle className="text-md">주간 도로파손 건수</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        {weeklyReport && (
          <div className="flex flex-col gap-2">
            <p className="text-lg font-bold">{weeklyReport.count}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
