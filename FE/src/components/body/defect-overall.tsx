import { maintenanceAPI } from '@/api/maintenance-api'
import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useQuery } from '@tanstack/react-query'
import {
  AlertTriangle,
  Hammer,
  MapPin,
  MessageCircleWarning,
} from 'lucide-react'

const DefectOverall: React.FC = () => {
  const { data: riskList } = useQuery({
    queryKey: ['risk-list'],
    queryFn: async () => {
      const response = await statisticAPI.getRiskList()
      if (response.error) {
        throw response.error
      }
      return response
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  const critical = riskList?.data.critical || 0
  const danger = riskList?.data.highRisk || 0
  const caution = riskList?.data.warning || 0
  const safe = riskList?.data.safe || 0

  const totalRisks = critical + danger + caution + safe || 1

  const criticalRate = Math.round((critical / totalRisks) * 100 * 100) / 100

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

  const { data: addressCount } = useQuery({
    queryKey: ['address-count'],
    queryFn: async () => {
      const response = await statisticAPI.getDefectAddressCount()
      if (response.error) {
        throw response.error
      }
      return response
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  const addrCnt = addressCount?.data.regionCount

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">총 결함 수</CardTitle>
          <MessageCircleWarning className="text-muted-foreground h-4 w-4" />
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
          <div className="text-2xl font-bold">{criticalRate} %</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            전체 작업 대비 완료 작업 비율
          </CardTitle>
          <Hammer className="text-muted-foreground h-4 w-4" />
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
          <div className="text-2xl font-bold">{addrCnt} 곳</div>
        </CardContent>
      </Card>
    </div>
  )
}

export default DefectOverall
