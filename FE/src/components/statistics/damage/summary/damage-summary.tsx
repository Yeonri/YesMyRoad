import { statisticAPI } from '@/api/statistic-api'
import SummaryCard from '@/components/statistics/common/SummaryCard'
import { useQuery } from '@tanstack/react-query'
import { useEffect } from 'react'

export default function DamageSummary() {
  // 일간 데이터 쿼리
  const {
    data: dailyReport,
    isLoading: isDailyLoading,
    error: dailyError,
  } = useQuery({
    queryKey: ['damage-daily-report'],
    queryFn: async () => {
      const response = await statisticAPI.getDamageDailyReport()
      return response.data
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  // 주간 데이터 쿼리
  const {
    data: weeklyReport,
    isLoading: isWeeklyLoading,
    error: weeklyError,
  } = useQuery({
    queryKey: ['damage-weekly-report'],
    queryFn: async () => {
      const response = await statisticAPI.getDamageWeeklyReport()
      return response.data
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  // 월간 데이터 쿼리
  const {
    data: monthlyReport,
    isLoading: isMonthlyLoading,
    error: monthlyError,
  } = useQuery({
    queryKey: ['damage-monthly-report'],
    queryFn: async () => {
      const response = await statisticAPI.getDamageMonthlyReport()
      return response.data
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })

  useEffect(() => {
    // console.log('dailyReport: ', dailyReport)
    // console.log('weeklyReport: ', weeklyReport)
    // console.log('monthlyReport: ', monthlyReport)
  }, [dailyReport, weeklyReport, monthlyReport])

  // 로딩 상태 표시
  const isLoading = isDailyLoading || isWeeklyLoading || isMonthlyLoading

  // 에러 상태 표시
  const hasError = dailyError || weeklyError || monthlyError

  if (isLoading) {
    return (
      <div className={`col-span-1 flex h-auto flex-col gap-4`}>
        <SummaryCard title="데이터 로딩 중..." value="-" className="grow" />
      </div>
    )
  }

  if (hasError) {
    return (
      <div className={`col-span-1 flex h-auto flex-col gap-4`}>
        <SummaryCard
          title="데이터 로드 오류"
          value="다시 시도해주세요"
          className="grow"
        />
      </div>
    )
  }

  // 값 처리 함수
  const formatValue = (count: number, rate: number, period: string) => {
    if (count === undefined || count === null) return undefined
    if (count < 0) return `${period} 데이터 없음`

    const rateText =
      rate < 0 ? `${-rate}% 감소` : rate > 0 ? `${rate}% 증가` : '변동 없음'

    return `${period} 대비 ${rateText}`
  }

  return (
    <div className={`col-span-1 flex h-auto flex-col gap-4`}>
      <SummaryCard
        title="일간 도로파손 건수"
        value={Math.abs(dailyReport?.count) + '건'}
        subValue={formatValue(
          dailyReport?.count,
          dailyReport?.changeRate,
          '어제',
        )}
        className="grow"
      />
      <SummaryCard
        title="주간 도로파손 건수"
        value={Math.abs(weeklyReport?.count) + '건'}
        subValue={formatValue(
          weeklyReport?.count,
          weeklyReport?.changeRate,
          '지난 주',
        )}
        className="grow"
      />
      <SummaryCard
        title="월간 도로파손 건수"
        value={Math.abs(monthlyReport?.count) + '건'}
        subValue={formatValue(
          monthlyReport?.count,
          monthlyReport?.changeRate,
          '지난 월',
        )}
        className="grow"
      />
    </div>
  )
}
