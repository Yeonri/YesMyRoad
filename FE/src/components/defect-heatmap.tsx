'use client'

import { useDefectStore } from '@/store/defect-store'
import MapboxLanguage from '@mapbox/mapbox-gl-language'
import { Loader } from 'lucide-react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useEffect, useRef, useState } from 'react'

mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN as string

export default function DefectHeatmap() {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [loading, setLoading] = useState(true)

  // Get heatmap locations from Zustand store
  const { heatmapLocations } = useDefectStore()

  useEffect(() => {
    if (!mapContainer.current) return

    // TODO: Replace with actual API call to fetch heatmap data
    // This would be implemented in the Zustand store

    // 지도 초기화
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [127.2984, 36.35518],
      zoom: 12,
    })

    // 지도 이동 컨트롤 추가
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right')

    const language = new MapboxLanguage()
    map.current.addControl(language)

    // 지도 로드 시 히트맵 추가
    map.current.on('load', () => {
      setLoading(false)

      // 결함 위치를 GeoJSON 형식으로 변환
      const geojsonData = {
        type: 'FeatureCollection' as const,
        features: heatmapLocations.map((defect) => ({
          type: 'Feature' as const,
          properties: {},
          geometry: { type: 'Point' as const, coordinates: [defect.lng, defect.lat] },
        })),
      }

      // 히트맵 소스 추가
      map.current!.addSource('defects', { type: 'geojson', data: geojsonData })

      // 히트맵 레이어 추가
      map.current!.addLayer({
        id: 'defects-heat',
        type: 'heatmap',
        source: 'defects',
        maxzoom: 15,
        paint: {
          // 지름이 증가할수록 weight 증가
          'heatmap-weight': 1,
          // 줌 레벨이 증가할수록 intensity 증가
          'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 15, 3],
          // 히트맵 색상 라인 파랑에서 빨강으로
          'heatmap-color': [
            'interpolate',
            ['linear'],
            ['heatmap-density'],
            0,
            'rgba(0, 0, 255, 0)',
            0.2,
            'rgb(0, 255, 255)',
            0.4,
            'rgb(0, 255, 0)',
            0.6,
            'rgb(255, 255, 0)',
            0.8,
            'rgb(255, 128, 0)',
            1,
            'rgb(255, 0, 0)',
          ],
          // 줌 레벨에 따라 히트맵 반경 조정
          'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 0, 2, 15, 20],
          // 줌 레벨에 따라 히트맵 투명도 전환
          'heatmap-opacity': ['interpolate', ['linear'], ['zoom'], 7, 1, 15, 0],
        },
      })

      // 높은 줌 레벨에서 원 레이어 추가
      map.current!.addLayer({
        id: 'defects-point',
        type: 'circle',
        source: 'defects',
        minzoom: 14,
        paint: {
          // 줌 레벨에 따라 원 반경 조정
          'circle-radius': ['interpolate', ['linear'], ['zoom'], 14, 2, 18, 10],
          'circle-color': '#ef4444',
          'circle-stroke-color': 'white',
          'circle-stroke-width': 1,
          // 줌 레벨에 따라 원 투명도 전환
          'circle-opacity': ['interpolate', ['linear'], ['zoom'], 14, 0, 15, 1],
        },
      })
    })

    // 언마운트 시 정리
    return () => {
      if (map.current) {
        map.current.remove()
      }
    }
  }, [heatmapLocations])

  return (
    <div className="relative h-full w-full min-h-[400px]">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted/20 z-10">
          <div className="flex flex-col items-center gap-2">
            <Loader className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Loading heatmap data...</p>
          </div>
        </div>
      )}
      <div ref={mapContainer} className="h-full w-full" />
    </div>
  )
}
