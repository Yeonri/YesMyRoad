'use client'

import { useDefectStore } from '@/store/defect-store'
import MapboxLanguage from '@mapbox/mapbox-gl-language'
import { Loader } from 'lucide-react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useEffect, useRef, useState } from 'react'

// Set your Mapbox access token here
mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN || 'YOUR_MAPBOX_ACCESS_TOKEN'

export default function DefectMap() {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [loading, setLoading] = useState(true)

  // Get defect locations from Zustand store
  const { defectLocations, defectType, severity } = useDefectStore()

  // Filter defect locations based on selected filters
  const filteredDefectLocations = defectLocations.filter((defect) => {
    const matchesType = defectType === 'all' || defect.type === defectType
    const matchesSeverity = severity === 'all' || defect.severity === severity
    return matchesType && matchesSeverity
  })

  useEffect(() => {
    if (!mapContainer.current) return

    // TODO: Replace with actual API call to fetch defect locations
    // This would be implemented in the Zustand store's fetchDefectLocations method

    // 지도 초기화
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/streets-v12',
      center: [127.2984, 36.35518],
      zoom: 13,
    })

    // 지도 이동 컨트롤 추가
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right')

    // 언어 설정
    const language = new MapboxLanguage()
    map.current.addControl(language)

    // 지도 로드 시 마커 추가
    map.current.on('load', () => {
      setLoading(false)

      // 각 결함에 대한 마커 추가
      filteredDefectLocations.forEach((defect) => {
        // 마커 요소 생성
        const markerEl = document.createElement('div')
        markerEl.className = 'defect-marker'
        markerEl.style.width = '20px'
        markerEl.style.height = '20px'
        markerEl.style.borderRadius = '50%'

        // 심각도에 따른 색상 설정
        switch (defect.severity) {
          case 'critical':
            markerEl.style.backgroundColor = '#ef4444' // red-500
            break
          case 'high':
            markerEl.style.backgroundColor = '#f59e0b' // amber-500
            break
          case 'medium':
            markerEl.style.backgroundColor = '#3b82f6' // blue-500
            break
          case 'low':
            markerEl.style.backgroundColor = '#22c55e' // green-500
            break
          default:
            markerEl.style.backgroundColor = '#6b7280' // gray-500
        }

        markerEl.style.border = '2px solid white'
        markerEl.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)'

        // 팝업 생성
        const popup = new mapboxgl.Popup({ offset: 25 }).setHTML(`
          <div>
            <h3 class="font-bold">${defect.title}</h3>
            <p>유형: ${defect.type.charAt(0).toUpperCase() + defect.type.slice(1)}</p>
            <p>심각도: ${defect.severity.charAt(0).toUpperCase() + defect.severity.slice(1)}</p>
            <p>ID: DEF-${1000 + defect.id}</p>
          </div>
        `)

        // 지도에 마커 추가
        new mapboxgl.Marker(markerEl).setLngLat([defect.lng, defect.lat]).setPopup(popup).addTo(map.current!)
      })
    })

    // 언마운트 시 정리
    return () => {
      if (map.current) {
        map.current.remove()
      }
    }
  }, [filteredDefectLocations])

  return (
    <div className="relative h-full w-full min-h-[400px]">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-muted/20 z-10">
          <div className="flex flex-col items-center gap-2">
            <Loader className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">지도 데이터 로딩중...</p>
          </div>
        </div>
      )}
      <div ref={mapContainer} className="h-full w-full" />
    </div>
  )
}
