import { maintenanceAPI } from '@/api/maintenance-api'
import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, BarChart3, Clock, MapPin } from 'lucide-react'

const DefectOverall: React.FC = () => {
  const { data: totalDefectsData } = useQuery({
    queryKey: ['total-defects'],
    queryFn: async () => {
      const response = await statisticAPI.getDamageReportByType() // 함수 갈아 끼워 넣기
      if (response.error) {
        throw response.error
      }
      return response
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  // const totalDefects =
  // totalDefectsData?.data[0].count + totalDefectsData?.data[1].count

  const { data: completedRateData } = useQuery({
    queryKey: ['completed-rate'],
    queryFn: async () => {
      const response = await maintenanceAPI.getMaintenanceOverview()
      if (response.error) {
        throw response.error
      }
      return response
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  const totalCount =
    completedRateData?.data.reported +
    completedRateData?.data.received +
    completedRateData?.data.inProgress +
    completedRateData?.data.completed
  const completedRate =
    Math.round(
      ((completedRateData?.data.completed || 0) / (totalCount || 1)) *
        100 *
        100,
    ) / 100
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">총 결함 수</CardTitle>
          <BarChart3 className="text-muted-foreground h-4 w-4" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{totalCount} 건</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            심각한 결함 비율
          </CardTitle>
          <AlertTriangle className="text-muted-foreground h-4 w-4" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{0} %</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            전체 작업 대비 완료 작업 비율
          </CardTitle>
          <Clock className="text-muted-foreground h-4 w-4" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{completedRate} % 완료됨</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            발생 행정구역 수
          </CardTitle>
          <MapPin className="text-muted-foreground h-4 w-4" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{0} 곳</div>
        </CardContent>
      </Card>
    </div>
  )
}

export default DefectOverall
