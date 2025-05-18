import BudgetExecutionReport from '@/components/statistics/repair/charts/budget-execution-report'
import CategoryDistribution from '@/components/statistics/repair/charts/category-distribution'
import MapStatus from '@/components/statistics/repair/charts/map-status'
import MonthlyRepair from '@/components/statistics/repair/charts/monthly-repair'
import SummaryDaily from '@/components/statistics/repair/summary/summary-daily'
import SummaryMonthly from '@/components/statistics/repair/summary/summary-monthly'
import SummaryWeekly from '@/components/statistics/repair/summary/summary-weekly'

const CARD_HEIGHT = 'h-auto'

export default function DefectStatsRepair() {
  return (
    <div className="grid grid-cols-6 grid-rows-2 gap-4">
      {/* 1행 */}
      {/* 유형별 도로보수 분포 : 도넛형 */}
      <CategoryDistribution cardHeight={CARD_HEIGHT} />
      {/* 일/월/년 도로보수 건수 */}
      <div className={`${CARD_HEIGHT} col-span-2 flex flex-col gap-4`}>
        <SummaryDaily />
        <SummaryWeekly />
        <SummaryMonthly />
      </div>
      {/* 월별 도로보수 현황 : 행정구역별 세로 막대 그래프 */}
      <MonthlyRepair cardHeight={CARD_HEIGHT} />

      {/* 2행 */}
      {/* 구역별 도로 보수 현황 : 지역 heatmap */}
      <MapStatus cardHeight={CARD_HEIGHT} />
      {/* 도로보수 분포 Top 3 지역별 보수 유형 : Top 3 행정구역 가로 막대 그래프 */}
      <BudgetExecutionReport cardHeight={CARD_HEIGHT} />
    </div>
  )
}
