import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { daejeon } from '@/data/geojson/sig_daejeon'
import * as echarts from 'echarts'
import { useEffect, useRef, useState } from 'react'

export default function MapStatus(cardHeight: string) {
  const [isMapRegistered, setIsMapRegistered] = useState(false)
  const chartRef = useRef<HTMLDivElement>(null)
  const chartInstance = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!isMapRegistered) {
      echarts.registerMap('daejeon', daejeon)
      setIsMapRegistered(true)
    }
  }, [isMapRegistered])

  useEffect(() => {
    if (isMapRegistered && chartRef.current) {
      if (!chartInstance.current) {
        chartInstance.current = echarts.init(chartRef.current)
      }
      const option: echarts.EChartsOption = {
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
            data: [
              { name: '동구', value: 10 },
              { name: '중구', value: 20 },
              { name: '서구', value: 30 },
              { name: '유성구', value: 40 },
              { name: '대덕구', value: 50 },
            ],
          },
        ],
      }
      chartInstance.current.setOption(option)
      // 리사이즈 대응
      const handleResize = () => chartInstance.current?.resize()
      window.addEventListener('resize', handleResize)
      return () => {
        window.removeEventListener('resize', handleResize)
        chartInstance.current?.dispose()
        chartInstance.current = null
      }
    }
  }, [isMapRegistered])

  return (
    <Card className={`col-span-2 ${cardHeight}`}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">구역별 도로 파손 현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div ref={chartRef} className="h-[400px] w-full" />
      </CardContent>
    </Card>
  )
}
