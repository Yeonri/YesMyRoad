import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { RepairDailyReportType } from '@/types/stats-api'
import { useEffect, useState } from 'react'

export default function SummaryDaily() {
  const [dailyReport, setDailyReport] = useState<RepairDailyReportType | null>(
    null,
  )

  useEffect(() => {
    statisticAPI.getRepairDailyReport().then((response) => {
      setDailyReport(response.data)
    })
  }, [])

  return (
    <Card className="grow">
      <CardHeader className="p-4">
        <CardTitle className="text-md">일간 도로보수 건수</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        {dailyReport && (
          <div className="flex flex-col gap-2">
            <p className="text-lg font-bold">{dailyReport.totalCount}</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
