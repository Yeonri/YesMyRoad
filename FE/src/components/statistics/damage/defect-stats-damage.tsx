import { Card, CardHeader, CardTitle } from '@/components/ui/card'

import CategoryDistribution from './charts/category-distribution'
import MapStatus from './charts/map-status'
import TopThreeRegion from './charts/top-three-region'
import SummaryDaily from './summary/summary-daily'
import SummaryMonthly from './summary/summary-monthly'
import SummaryWeekly from './summary/summary-weekly'

const CARD_HEIGHT = 'h-80'

export default function DefectStatsDamage() {
  return (
    <div className="grid grid-cols-3 grid-rows-2 gap-4">
      {/* 1행 */}
      {/* 유형별 도로파손 분포 : 도넛형 */}
      <CategoryDistribution cardHeight={CARD_HEIGHT} />
      {/* 일/월/년 도로파손 건수 */}
      <div className={`${CARD_HEIGHT} flex flex-col gap-4`}>
        <SummaryDaily />
        <SummaryWeekly />
        <SummaryMonthly />
      </div>
      {/* 월별 도로파손 현황 : 행정구역별 세로 막대 그래프 */}
      <Card className={CARD_HEIGHT}>
        <CardHeader className="p-4">
          <CardTitle className="text-md">월별 도로파손 현황</CardTitle>
        </CardHeader>
      </Card>

      {/* 2행 */}
      {/* 구역별 도로 파손 현황 : 지역 heatmap */}
      <MapStatus cardHeight={CARD_HEIGHT} />
      {/* 도로파손 분포 Top 3 지역별 파손 유형 : Top 3 행정구역 가로 막대 그래프 */}
      <TopThreeRegion cardHeight={CARD_HEIGHT} />
    </div>
  )
}
