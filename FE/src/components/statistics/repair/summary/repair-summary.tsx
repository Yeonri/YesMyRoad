import { maintenanceAPI } from '@/api/maintenance-api'
import SummaryCard from '@/components/statistics/common/SummaryCard'
import { RepairReportSummaryType } from '@/types/stats-api'
import { useCallback, useEffect, useState } from 'react'

export default function RepairSummary() {
  const [reportCounts, setReportCounts] = useState<RepairReportSummaryType>({
    daily: 0,
    weekly: 0,
    monthly: 0,
  })

  const fetchRepairSummary = useCallback(async () => {
    const response = await maintenanceAPI.getMaintenanceCompletionStats()
    setReportCounts({
      daily: response.data.daily,
      weekly: response.data.weekly,
      monthly: response.data.monthly,
    })
  }, [])

  useEffect(() => {
    fetchRepairSummary()
  }, [])

  return (
    <div className={`col-span-2 flex h-auto flex-col gap-4`}>
      <SummaryCard
        title="일간 도로보수 건수"
        value={reportCounts.daily ?? '-'}
        className="grow"
      />
      <SummaryCard
        title="주간 도로보수 건수"
        value={reportCounts.weekly ?? '-'}
        className="grow"
      />
      <SummaryCard
        title="월간 도로보수 건수"
        value={reportCounts.monthly ?? '-'}
        className="grow"
      />
    </div>
  )
}
