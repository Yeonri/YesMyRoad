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
    console.log('dailyReport: ', dailyReport)
    console.log('weeklyReport: ', weeklyReport)
    console.log('monthlyReport: ', monthlyReport)
  }, [dailyReport, weeklyReport, monthlyReport])

  // 로딩 상태 표시
  const isLoading = isDailyLoading || isWeeklyLoading || isMonthlyLoading

  // 에러 상태 표시
  const hasError = dailyError || weeklyError || monthlyError

  if (isLoading) {
    return (
      <div className={`col-span-2 flex h-auto flex-col gap-4`}>
        <SummaryCard title="데이터 로딩 중..." value="-" className="grow" />
      </div>
    )
  }

  if (hasError) {
    return (
      <div className={`col-span-2 flex h-auto flex-col gap-4`}>
        <SummaryCard
          title="데이터 로드 오류"
          value="다시 시도해주세요"
          className="grow"
        />
      </div>
    )
  }

  // 값 처리 함수
  const formatValue = (
    count: number,
    rate: number = 0,
    period: string = '전날',
  ) => {
    if (!count && count !== 0) return '- 건'

    // 음수인 경우 절대값으로 변환하고 증감율은 "-"로 표시
    const absoluteCount = Math.abs(count)
    const rateText = count < 0 ? '-%' : `${rate}%`

    return `${absoluteCount}건, ${period} 대비 ${rateText} 증가`
  }

  return (
    <div className={`col-span-2 flex h-auto flex-col gap-4`}>
      <SummaryCard
        title="일간 도로파손 건수"
        value={formatValue(dailyReport?.count, dailyReport?.changeRate, '전일')}
        className="grow"
      />
      <SummaryCard
        title="주간 도로파손 건수"
        value={formatValue(
          weeklyReport?.count,
          weeklyReport?.changeRate,
          '전주',
        )}
        className="grow"
      />
      <SummaryCard
        title="월간 도로파손 건수"
        value={formatValue(
          monthlyReport?.count,
          monthlyReport?.changeRate,
          '전월',
        )}
        className="grow"
      />
    </div>
  )
}
