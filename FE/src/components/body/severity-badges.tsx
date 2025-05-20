import { statisticAPI } from '@/api/statistic-api'
import { Badge } from '@/components/ui/badge'
import { useQuery } from '@tanstack/react-query'

const SeverityBadges: React.FC = () => {
  const { data: riskList } = useQuery({
    queryKey: ['risk-list'],
    queryFn: async () => {
      const response = await statisticAPI.getRiskList() // 함수 갈아 끼워 넣기
      if (response.error) {
        throw response.error
      }
      return response
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  return (
    <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
      <div className="flex flex-wrap gap-2">
        <Badge variant="default" className="rounded-md px-3 py-1">
          <div className="mr-1 h-2 w-2 rounded-full bg-red-500" />
          심각: {riskList?.data.critical}
        </Badge>
        <Badge variant="default" className="rounded-md px-3 py-1">
          <div className="mr-1 h-2 w-2 rounded-full bg-amber-500" />
          위험: {riskList?.data.highRisk}
        </Badge>
        <Badge variant="default" className="rounded-md px-3 py-1">
          <div className="mr-1 h-2 w-2 rounded-full bg-blue-500" />
          주의: {riskList?.data.warning}
        </Badge>
        <Badge variant="default" className="rounded-md px-3 py-1">
          <div className="mr-1 h-2 w-2 rounded-full bg-green-500" />
          안전: {riskList?.data.safe}
        </Badge>
      </div>
    </div>
  )
}

export default SeverityBadges
