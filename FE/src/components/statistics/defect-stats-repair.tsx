import BudgetExecutionReport from '@/components/statistics/repair/charts/budget-execution-report'
import MapStatus from '@/components/statistics/repair/charts/map-status'
import MonthlyMaintenanceOverview from '@/components/statistics/repair/charts/monthly-maintenance-overview'
import MonthlyRepair from '@/components/statistics/repair/charts/monthly-repair'
import RepairSummary from '@/components/statistics/repair/summary/repair-summary'

export default function DefectStatsRepair() {
  return (
    <div className="grid grid-cols-6 grid-rows-2 gap-4">
      {/* 1행 */}
      {/* 유형별 도로보수 분포 : 도넛형 */}
      <MonthlyMaintenanceOverview />
      {/* 일/월/년 도로보수 건수 */}
      <RepairSummary />
      {/* 월별 도로보수 현황 : 행정구역별 세로 막대 그래프 */}
      <MonthlyRepair />

      {/* 2행 */}
      {/* 구역별 도로 보수 현황 : 지역 heatmap */}
      <MapStatus />
      {/* 도로보수 분포 Top 3 지역별 보수 유형 : Top 3 행정구역 가로 막대 그래프 */}
      <BudgetExecutionReport />
    </div>
  )
}
