import { maintenanceAPI } from '@/api/maintenance-api'
import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  AlertTriangle,
  Hammer,
  MapPin,
  MessageCircleWarning,
} from 'lucide-react'
import { useEffect } from 'react'
import { useDetailedDefectStore } from '@/store/detailed-defect-list-store'

const DefectOverall: React.FC = () => {
  // QueryClient 인스턴스 가져오기
  const queryClient = useQueryClient()
  
  // 상태 변경을 감지하기 위해 defectStore에서 필요한 상태 구독
  const defectDetailList = useDetailedDefectStore((state) => state.detailedGeoJSONData)
  
  // 상태 변경 시 쿼리 무효화 (데이터 다시 가져오기) - 타입 오류 수정
  useEffect(() => {
    if (defectDetailList) {
      // 각 쿼리를 개별적으로 무효화 (객체 형태로 전달)
      queryClient.invalidateQueries({ queryKey: ['risk-list'] })
      queryClient.invalidateQueries({ queryKey: ['completed-rate'] })
      queryClient.invalidateQueries({ queryKey: ['address-count'] })
    }
  }, [defectDetailList, queryClient])

  const { data: riskList, refetch: refetchRiskList } = useQuery({
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

  const { data: completedRateData, refetch: refetchCompletedRate } = useQuery({
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

  const { data: addressCount, refetch: refetchAddressCount } = useQuery({
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

  // 필요에 따라 수동으로 새로고침할 수 있는 함수
  const refreshAllData = () => {
    refetchRiskList();
    refetchCompletedRate();
    refetchAddressCount();
  }

  // 다른 특정 이벤트(예: 커스텀 이벤트)를 수신하여 데이터 새로고침
  useEffect(() => {
    const handleDataChange = () => {
      refreshAllData();
    };

    // 커스텀 이벤트 리스너 추가
    window.addEventListener('defect-data-changed', handleDataChange);
    
    return () => {
      window.removeEventListener('defect-data-changed', handleDataChange);
    };
  }, []);

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
             완료 작업 비율
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
          <div className="text-2xl font-bold">{addrCnt} 개소</div>
        </CardContent>
      </Card>
    </div>
  )
}

export default DefectOverall