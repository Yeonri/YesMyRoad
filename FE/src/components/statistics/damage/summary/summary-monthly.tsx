import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { DamageMonthlyReportType } from '@/types/stats-api'
import { useEffect, useState } from 'react'

export default function SummaryMonthly() {
  const [monthlyReport, setMonthlyReport] =
    useState<DamageMonthlyReportType | null>(null)

  useEffect(() => {
    statisticAPI.getDamageMonthlyReport().then((response) => {
      setMonthlyReport(response.data)
    })
  }, [])

  return (
    <Card className="grow">
      <CardHeader className="p-4">
        <CardTitle className="text-md">월간 도로파손 건수</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        {monthlyReport && (
          <div className="flex flex-col gap-2">
            <p className="text-lg font-bold">{monthlyReport.count}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
