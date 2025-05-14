import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { daejeon } from '@/data/geojson/sig_daejeon'
import * as echarts from 'echarts'
import ReactECharts from 'echarts-for-react'
import { useEffect, useState } from 'react'

export default function MapStatus(props: { cardHeight: string }) {
  const { cardHeight = 'h-80' } = props

  const [isMapRegistered, setIsMapRegistered] = useState(false)
  const [option, setOption] = useState<echarts.EChartsOption>({})

  useEffect(() => {
    if (!isMapRegistered) {
      echarts.registerMap('daejeon', daejeon)
      setIsMapRegistered(true)

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
            data: [
              { name: '동구', value: 10 },
              { name: '중구', value: 20 },
              { name: '서구', value: 30 },
              { name: '유성구', value: 40 },
              { name: '대덕구', value: 50 },
            ],
          },
        ],
      })
    }
  }, [isMapRegistered])

  return (
    <Card className={`col-span-3 grow ${cardHeight}`}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">구역별 도로 파손 현황</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        {isMapRegistered && (
          <ReactECharts
            option={option}
            style={{ height: '300px', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        )}
      </CardContent>
    </Card>
  )
}
