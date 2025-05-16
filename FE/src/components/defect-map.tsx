'use client'

import useAddressStore from '@/store/address-store'
import { useDefectStore } from '@/store/defect-store'
import MapboxLanguage from '@mapbox/mapbox-gl-language'
import { Loader } from 'lucide-react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useEffect, useMemo, useRef, useState } from 'react'

mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN

export default function DefectMap() {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [loading, setLoading] = useState(true)

  const { defectLocations, defectType, severity } = useDefectStore()

  // 주소 스토어에서 위도/경도 가져오기, 없으면 서울 기본값 설정
  const longitude = useAddressStore((state) => state.longitude) ?? 127.029
  const latitude = useAddressStore((state) => state.latitude) ?? 37.4787
  // 맵 경계 좌표 설정 함수 가져오기
  const setMapBounds = useAddressStore((state) => state.setMapBounds)
  const logMapBounds = useAddressStore((state) => state.logMapBounds)

  const filteredDefectLocations = useMemo(() => {
    return defectLocations.filter((defect) => {
      const matchesType = defectType === 'all' || defect.type === defectType
      const matchesSeverity = severity === 'all' || defect.severity === severity
      return matchesType && matchesSeverity
    })
  }, [defectLocations, defectType, severity])

  useEffect(() => {
    if (!mapContainer.current) return

    // 지도 초기화
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/streets-v12',
      center: [longitude, latitude],
      zoom: 13,
    })

    // 지도 이동 컨트롤 추가
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right')

    // 언어 설정
    const language = new MapboxLanguage()
    map.current.addControl(language)

    // 지도 경계 좌표 가져오는 함수
    const getBoundaryCoordinates = () => {
      if (!map.current) return

      const bounds = map.current.getBounds()

      // 경계의 좌표 정보
      const northEast = bounds!.getNorthEast() // 북동쪽 좌표
      const southWest = bounds!.getSouthWest() // 남서쪽 좌표

      // 좌표 정보 저장
      setMapBounds({
        northEast: { lat: northEast.lat, lng: northEast.lng },
        southWest: { lat: southWest.lat, lng: southWest.lng },
      })

      // 개발 모드에서 확인용 로깅
      // logMapBounds()
    }

    // 줌이나 이동 이벤트 발생 시 좌표 정보 가져오기
    map.current.on('move', getBoundaryCoordinates)

    // 지도가 로드 완료되거나 스타일 변경 후 안정화될 때 좌표 정보 가져오기
    map.current.on('idle', getBoundaryCoordinates)

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
        new mapboxgl.Marker(markerEl)
          .setLngLat([defect.lng, defect.lat])
          .setPopup(popup)
          .addTo(map.current!)
      })
    })

    // 언마운트 시 정리
    return () => {
      if (map.current) {
        map.current.remove()
      }
    }
  }, [filteredDefectLocations, longitude, latitude, setMapBounds, logMapBounds])

  return (
    <div className="relative h-full min-h-[400px] w-full">
      {loading && (
        <div className="bg-muted/20 absolute inset-0 z-10 flex items-center justify-center">
          <div className="flex flex-col items-center gap-2">
            <Loader className="text-primary h-8 w-8 animate-spin" />
            <p className="text-muted-foreground text-sm">
              지도 데이터 로딩중...
            </p>
          </div>
        </div>
      )}
      <div ref={mapContainer} className="h-full w-full" />
    </div>
  )
}
