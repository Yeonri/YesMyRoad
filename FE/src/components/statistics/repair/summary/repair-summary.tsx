import { maintenanceAPI } from '@/api/maintenance-api'
import SummaryCard from '@/components/statistics/common/SummaryCard'
import { Card } from '@/components/ui/card'
import { useQuery } from '@tanstack/react-query'

export default function RepairSummary() {
  const {
    data: apiResponse,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['maintanence-completion-stats'],
    queryFn: async () => {
      const response = await maintenanceAPI.getMaintenanceCompletionStats()
      return response.data
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  // 로딩 중이면 로딩 표시
  if (isLoading) {
    return (
      <Card className={`col-span-1 flex items-center justify-center`}>
        <div>데이터 로딩 중...</div>
      </Card>
    )
  }

  // 에러가 있으면 에러 표시
  if (error) {
    return (
      <Card className={`col-span-1 flex items-center justify-center`}>
        <div>데이터를 불러오는 중 오류가 발생했습니다.</div>
      </Card>
    )
  }

  return (
    <div className={`col-span-1 flex h-auto flex-col gap-4`}>
      <SummaryCard
        title="일간 도로보수 건수"
        value={(apiResponse?.daily ?? '-') + '건'}
        className="grow"
      />
      <SummaryCard
        title="주간 도로보수 건수"
        value={(apiResponse?.weekly ?? '-') + '건'}
        className="grow"
      />
      <SummaryCard
        title="월간 도로보수 건수"
        value={(apiResponse?.monthly ?? '-') + '건'}
        className="grow"
      />
    </div>
  )
}
