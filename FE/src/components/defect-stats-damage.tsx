import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

const CARD_HEIGHT = 'h-80'

export default function DefectStatsDamage() {
  return (
    <div className="grid grid-cols-3 grid-rows-2 gap-4">
      {/* 1행 */}
      {/* 유형별 도로파손 분포 : 도넛형 */}
      <Card className={CARD_HEIGHT}>
        <CardHeader className="p-4">
          <CardTitle className="text-md">유형별 도로파손 분포</CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0">

        </CardContent>
      </Card>
      {/* 일/월/년 도로파손 건수 */}
      <div className={`${CARD_HEIGHT} flex flex-col gap-4`}>
        <Card className="grow">
          <CardHeader className="p-4">
            <CardTitle className="text-md">일간 도로파손 건수</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0"></CardContent>
        </Card>
        <Card className="grow">
          <CardHeader className="p-4">
            <CardTitle className="text-md">월간 도로파손 건수</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0"></CardContent>
        </Card>
        <Card className="grow">
          <CardHeader className="p-4">
            <CardTitle className="text-md">연간 도로파손 건수</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0"></CardContent>
        </Card>
      </div>
      {/* 월별 도로파손 현황 : 행정구역별 세로 막대 그래프 */}
      <Card className={CARD_HEIGHT}>
        <CardHeader className="p-4">
          <CardTitle className="text-md">월별 도로파손 현황</CardTitle>
        </CardHeader>
      </Card>

      {/* 2행 */}
      {/* 구역별 도로 파손 현황 : 지역 heatmap */}
      <Card className={`col-span-2 ${CARD_HEIGHT}`}>
        <CardHeader className="p-4">
          <CardTitle className="text-md">구역별 도로 파손 현황</CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0"></CardContent>
      </Card>
      {/* 도로파손 분포 Top 3 지역별 파손 유형 : Top 3 행정구역 가로 막대 그래프 */}
      <Card className={CARD_HEIGHT}>
        <CardHeader className="p-4">
          <CardTitle className="text-md">도로파손 분포 Top 3</CardTitle>
        </CardHeader>
      </Card>
    </div>
  )
}
