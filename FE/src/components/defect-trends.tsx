"use client"

import { useEffect, useRef, useState } from "react"
import { init } from "echarts"
import type { EChartsOption } from "echarts"
import { Loader } from "lucide-react"
import { useDefectStore } from "@/store/defect-store"

export default function DefectTrends() {
  const chartRef = useRef<HTMLDivElement>(null)
  const [loading, setLoading] = useState(true)

  // Get trend data from Zustand store
  const { trendData } = useDefectStore()

  useEffect(() => {
    // Initialize chart
    if (chartRef.current) {
      const chart = init(chartRef.current)

      // TODO: Replace with actual API call to fetch trend data
      // This would be implemented in the Zustand store's fetchDefectTrends method

      // Set chart options
      const option: EChartsOption = {
        tooltip: {
          trigger: "axis",
          axisPointer: {
            type: "shadow",
          },
        },
        legend: {
          data: ["Potholes", "Cracks", "Paint Peeling"],
          bottom: 0,
        },
        grid: {
          left: "3%",
          right: "4%",
          bottom: "15%",
          top: "3%",
          containLabel: true,
        },
        xAxis: {
          type: "category",
          data: trendData.dates,
        },
        yAxis: {
          type: "value",
        },
        color: ["#3b82f6", "#22c55e", "#f59e0b"],
        series: [
          {
            name: "Potholes",
            type: "bar",
            stack: "total",
            emphasis: {
              focus: "series",
            },
            data: trendData.potholesData,
          },
          {
            name: "Cracks",
            type: "bar",
            stack: "total",
            emphasis: {
              focus: "series",
            },
            data: trendData.cracksData,
          },
          {
            name: "Paint Peeling",
            type: "bar",
            stack: "total",
            emphasis: {
              focus: "series",
            },
            data: trendData.paintData,
          },
        ],
      }

      // Set chart option and render
      chart.setOption(option)

      // Handle resize
      const handleResize = () => {
        chart.resize()
      }
      window.addEventListener("resize", handleResize)

      // Simulate loading
      setTimeout(() => {
        setLoading(false)
      }, 1000)

      // Clean up
      return () => {
        chart.dispose()
        window.removeEventListener("resize", handleResize)
      }
    }
  }, [trendData])

  return (
    <div className="h-[300px] w-full">
      {loading ? (
        <div className="flex h-full w-full items-center justify-center">
          <Loader className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : (
        <div ref={chartRef} className="h-full w-full" />
      )}
    </div>
  )
}
