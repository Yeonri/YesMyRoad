import { statisticAPI } from '@/api/statistic-api'
import SummaryCard from '@/components/statistics/common/SummaryCard'
import {
  DamageDailyReportType,
  DamageMonthlyReportType,
  DamageWeeklyReportType,
} from '@/types/stats-api'
import { useEffect, useState } from 'react'

export default function DamageSummary() {
  const [dailyReport, setDailyReport] = useState<DamageDailyReportType | null>(
    null,
  )
  const [weeklyReport, setWeeklyReport] =
    useState<DamageWeeklyReportType | null>(null)
  const [monthlyReport, setMonthlyReport] =
    useState<DamageMonthlyReportType | null>(null)

  useEffect(() => {
    statisticAPI.getDamageDailyReport().then((response) => {
      setDailyReport(response.data)
    })
    statisticAPI.getDamageWeeklyReport().then((response) => {
      setWeeklyReport(response.data)
    })
    statisticAPI.getDamageMonthlyReport().then((response) => {
      setMonthlyReport(response.data)
    })
  }, [])

  useEffect(() => {
    console.log('dailyReport: ', dailyReport)
    console.log('weeklyReport: ', weeklyReport)
    console.log('monthlyReport: ', monthlyReport)
  }, [dailyReport, weeklyReport, monthlyReport])

  return (
    <div className={`col-span-2 flex h-auto flex-col gap-4`}>
      <SummaryCard
        title="일간 도로파손 건수"
        value={dailyReport?.count ?? '-'}
        className="grow"
      />
      <SummaryCard
        title="주간 도로파손 건수"
        value={weeklyReport?.count ?? '-'}
        className="grow"
      />
      <SummaryCard
        title="월간 도로파손 건수"
        value={monthlyReport?.count ?? '-'}
        className="grow"
      />
    </div>
  )
}
