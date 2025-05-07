'use client'

import { useDefectStore } from '@/store/defect-store'
import { init } from 'echarts'
import type { EChartsOption } from 'echarts'
import { Loader } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

export default function DefectStats() {
  const chartRef = useRef<HTMLDivElement>(null)
  const [loading, setLoading] = useState(true)

  // Get defect statistics from Zustand store
  const { defectTypeData, severityData } = useDefectStore()

  useEffect(() => {
    // Initialize chart
    if (chartRef.current) {
      const chart = init(chartRef.current)

      // TODO: Replace with actual API call to fetch defect statistics
      // This would be implemented in the Zustand store's fetchDefectStats method

      // 차트 옵션 설정
      const option: EChartsOption = {
        tooltip: { trigger: 'item', formatter: '{a} <br/>{b}: {c} ({d}%)' },
        legend: {
          orient: 'vertical',
          left: 10,
          data: [
            'Potholes',
            'Cracks',
            'Paint Peeling',
            'Critical',
            'High',
            'Medium',
            'Low',
          ],
        },
        color: [
          '#3b82f6',
          '#22c55e',
          '#f59e0b',
          '#ef4444',
          '#f59e0b',
          '#3b82f6',
          '#22c55e',
        ],
        series: [
          {
            name: 'Defect Types',
            type: 'pie',
            radius: ['0%', '45%'],
            center: ['30%', '50%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2,
            },
            label: { show: false, position: 'center' },
            emphasis: {
              label: { show: true, fontSize: 14, fontWeight: 'bold' },
            },
            labelLine: { show: false },
            data: defectTypeData,
          },
          {
            name: 'Severity',
            type: 'pie',
            radius: ['0%', '45%'],
            center: ['75%', '50%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2,
            },
            label: { show: false, position: 'center' },
            emphasis: {
              label: { show: true, fontSize: 14, fontWeight: 'bold' },
            },
            labelLine: { show: false },
            data: severityData,
          },
        ],
      }

      // 차트 옵션 적용 및 렌더링
      chart.setOption(option)

      // 크기 변경 처리
      const handleResize = () => {
        chart.resize()
      }
      window.addEventListener('resize', handleResize)

      // 로딩 시간 시뮬레이션
      setTimeout(() => {
        setLoading(false)
      }, 1000)

      // 정리
      return () => {
        chart.dispose()
        window.removeEventListener('resize', handleResize)
      }
    }
  }, [defectTypeData, severityData])

  return (
    <div className="h-[300px] w-full">
      {loading ? (
        <div className="flex h-full w-full items-center justify-center">
          <Loader className="text-primary h-8 w-8 animate-spin" />
        </div>
      ) : (
        <div ref={chartRef} className="h-full w-full" />
      )}
    </div>
  )
}
