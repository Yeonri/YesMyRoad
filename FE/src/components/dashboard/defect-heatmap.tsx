'use client'

import useAddressStore from '@/store/address-store'
import { useDefectStore } from '@/store/defect-store'
import { FeaturePoint } from '@/types/defects'
import MapboxLanguage from '@mapbox/mapbox-gl-language'
import { Loader } from 'lucide-react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useEffect, useMemo, useRef, useState } from 'react'

mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN as string

// 컴포넌트를 분리해 내부 로직을 별도 컴포넌트로 분리
function DefectHeatmapContent() {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [loading, setLoading] = useState(true)

  const geoJsonData = useDefectStore((state) => state.geoJSONData)

  // 주소 스토어에서 위도/경도 가져오기
  const longitude = useAddressStore((state) => state.longitude)
  const latitude = useAddressStore((state) => state.latitude)

  // GeoJSON 형식으로 데이터 생성 (좌표 순서 위도-경도 -> 경도-위도로 수정)
  const heatmapData = useMemo(() => {
    if (!geoJsonData || geoJsonData.length === 0) {
      return { type: 'FeatureCollection' as const, features: [] }
    }

    return {
      type: 'FeatureCollection' as const,
      features: geoJsonData.map((feature: FeaturePoint) => ({
        type: 'Feature' as const,
        properties: feature.properties || {},
        geometry: {
          type: 'Point' as const,
          coordinates: [
            feature.geometry.coordinates[1], // 경도(longitude)가 먼저
            feature.geometry.coordinates[0], // 위도(latitude)가 나중에
          ],
        },
      })),
    } as GeoJSON.FeatureCollection
  }, [geoJsonData])

  useEffect(() => {
    if (!mapContainer.current) return

    // 지도 초기화
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [longitude!, latitude!],
      zoom: 13,
    })

    // 지도 이동 컨트롤 추가
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right')

    const language = new MapboxLanguage()
    map.current.addControl(language)

    // 지도 로드 시 히트맵 추가
    map.current.on('load', () => {
      setLoading(false)

      if (heatmapData.features.length === 0) return

      // 히트맵 소스 추가
      map.current!.addSource('defects', { type: 'geojson', data: heatmapData })

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
          'heatmap-intensity': [
            'interpolate',
            ['linear'],
            ['zoom'],
            0,
            1,
            15,
            3,
          ],
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
  }, [heatmapData])

  // longitude, latitude가 변경될 때 중심 위치만 변경하는 별도 useEffect
  useEffect(() => {
    if (map.current) {
      map.current.setCenter([longitude!, latitude!])
    }
  }, [longitude, latitude])

  return (
    <div className="relative h-full min-h-[400px] w-full">
      {loading && (
        <div className="bg-muted/20 absolute inset-0 z-10 flex items-center justify-center">
          <div className="flex flex-col items-center gap-2">
            <Loader className="text-primary h-8 w-8 animate-spin" />
            <p className="text-muted-foreground text-sm">
              히트맵 데이터 로딩중...
            </p>
          </div>
        </div>
      )}
      <div ref={mapContainer} className="h-full w-full" />
    </div>
  )
}

// 메인 컴포넌트는 메모이제이션 된 지도 컴포넌트를 반환
export default function DefectHeatmap() {
  // DefectHeatmapContent 컴포넌트를 메모이제이션하여 탭 전환 시 재생성되지 않도록 함
  const memoizedHeatmapContent = useMemo(() => <DefectHeatmapContent />, [])

  return memoizedHeatmapContent
}
