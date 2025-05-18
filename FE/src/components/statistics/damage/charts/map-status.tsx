import { statisticAPI } from '@/api/statistic-api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { daejeon } from '@/data/geojson/sig_daejeon'
import { useQuery } from '@tanstack/react-query'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { useEffect, useState } from 'react'

export default function MapStatus(props: { cardHeight: string }) {
  const { cardHeight = 'h-80' } = props

  const [isMapRegistered, setIsMapRegistered] = useState(false)
  const [option, setOption] = useState<echarts.EChartsOption>({})

  // 먼저 지도 등록
  useEffect(() => {
    if (!isMapRegistered) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      echarts.registerMap('daejeon', daejeon as any)
      setIsMapRegistered(true)
    }
  }, [isMapRegistered])

  // 지도 등록 후 데이터 로드
  const { data: response, isLoading, error } = useQuery({
    queryKey: ['city-map'],
    queryFn: async () => {
      const response = await statisticAPI.getLocationallyDamageMap("daejeon")
      if (response.error) {
        throw response.error 
      }
      return response
    },
    refetchOnWindowFocus: false,
    staleTime: 5 * 60 * 1000,
    retry: 1,
    enabled: isMapRegistered,
  })

  const mapData = response?.data

  // 데이터가 변경될 때마다 차트 옵션 업데이트
  useEffect(() => {
    if (isMapRegistered) {
      setOption({
        tooltip: { trigger: 'item', formatter: '{b}<br/>{c} (건)' },
        toolbox: {
          show: true,
          orient: 'vertical',
          left: 'right',
          top: 'center',
          feature: {
            dataView: { readOnly: false },
            restore: {},
            saveAsImage: {},
          },
        },
        visualMap: {
          min: 0,
          max: 50,
          text: ['High', 'Low'],
          realtime: false,
          calculable: true,
          inRange: { color: ['lightskyblue', 'yellow', 'orangered'] },
        },
        series: [
          {
            name: '대전시 행정구역별 도로파손 건수',
            type: 'map',
            map: 'daejeon',
            label: { show: true },
            // 데이터가 없으면 기본값 사용
            data: mapData || [
              { name: '동구', value: 0 },
              { name: '중구', value: 0 },
              { name: '서구', value: 0 },
              { name: '유성구', value: 0 },
              { name: '대덕구', value: 0 },
            ],
          },
        ],
      })
    }
  }, [isMapRegistered, mapData]) // mapData를 의존성 배열에 추가

  // 렌더링 부분에서 조건부로 다른 UI 표시
  let content
  if (!isMapRegistered || isLoading) {
    content = <div>데이터 로딩 중...</div>
  } else if (error) {
    content = <div>데이터를 불러오는 중 오류가 발생했습니다.</div>
  } else {
    content = (
      <ReactECharts
        option={option}
        style={{ height: '300px', width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    )
  }

  return (
    <Card className={`col-span-3 grow ${cardHeight}`}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">구역별 도로파손 현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0 flex items-center justify-center">
        {content}
      </CardContent>
    </Card>
  )
}