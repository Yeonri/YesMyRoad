'use client'

import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import useAddressStore from '@/store/address-store'
import { DefectDetail, useDefectStore } from '@/store/defect-store'
import { FeaturePoint } from '@/store/defect-store'
import MapboxLanguage from '@mapbox/mapbox-gl-language'
import { ArrowRight, Loader } from 'lucide-react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useEffect, useRef, useState } from 'react'
import { useDebounceCallback } from 'usehooks-ts'

import RecentAlerts from '../recent-alerts'

mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN

function DefectMapContent() {
  // 지도 관련 상태 관리
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const markersRef = useRef<mapboxgl.Marker[]>([])
  const [loading, setLoading] = useState(true)

  // setSelectedDefect 함수 가져오기
  const setSelectedDefect = useDefectStore((state) => state.setSelectedDefect)
  // selectedDefect 상태 가져오기
  const selectedDefect = useDefectStore((state) => state.selectedDefect)

  // 주소 스토어에서 위도/경도 가져오기, 없으면 서울 기본값 설정
  const longitude = useAddressStore((state) => state.longitude)
  const latitude = useAddressStore((state) => state.latitude)

  // 맵 경계 좌표 설정 함수 가져오기
  const setMapBounds = useAddressStore((state) => state.setMapBounds)
  const debouncedSetMapBounds = useDebounceCallback(setMapBounds, 500)

  const geoJsonData = useDefectStore((state) => state.geoJSONData)

  // 지도 초기화 및 기본 설정
  useEffect(() => {
    if (!mapContainer.current) return

    // 지도 초기화
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/streets-v12',
      center: [longitude!, latitude!],
      zoom: 13,
    })

    // 지도 이동 컨트롤 추가
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right')

    // 언어 설정
    const language = new MapboxLanguage()
    map.current.addControl(language)

    // 지도 경계 좌표 가져오는 함수
    const debouncedGetBoundaryCoordinates = () => {
      if (!map.current) return

      const bounds = map.current.getBounds()

      // 경계의 좌표 정보
      const northEast = bounds!.getNorthEast() // 북동쪽 좌표
      const southWest = bounds!.getSouthWest() // 남서쪽 좌표

      // 좌표 정보 저장
      debouncedSetMapBounds({
        northEast: { lat: northEast.lat, lng: northEast.lng },
        southWest: { lat: southWest.lat, lng: southWest.lng },
      })
    }

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
    }

    // 줌이나 이동 이벤트 발생 시 좌표 정보 가져오기
    map.current.on('move', debouncedGetBoundaryCoordinates)

    // 지도가 로드 완료되거나 스타일 변경 후 안정화될 때 좌표 정보 가져오기
    map.current.on('idle', getBoundaryCoordinates)

    map.current.on('load', () => {
      setLoading(false)
    })

    // 언마운트 시 정리
    return () => {
      if (map.current) {
        map.current.remove()
      }
    }
  }, [longitude, latitude, setMapBounds, debouncedSetMapBounds])

  // selectedDefect 변경 감지 useEffect
  useEffect(() => {
    if (selectedDefect.publicId) {
      console.log('선택된 결함 ID:', selectedDefect.publicId)
    }
  }, [selectedDefect])

  // 마커 추가 로직을 별도 useEffect로 분리
  useEffect(() => {
    if (!map.current || loading || !geoJsonData) return

    // 기존 마커 제거
    markersRef.current.forEach((marker) => marker.remove())
    markersRef.current = []

    // 각 결함에 대한 마커 추가
    geoJsonData.forEach((defect: FeaturePoint) => {
      // 마커 요소 생성
      const markerEl = document.createElement('div')
      markerEl.className = 'defect-marker'
      markerEl.style.width = '20px'
      markerEl.style.height = '20px'
      markerEl.style.border = '2px solid white'
      markerEl.style.borderRadius = '50%'
      markerEl.style.backgroundColor = '#6b7280' // gray-500
      markerEl.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)'

      // GeoJSON에서 좌표는 [lng, lat] 순서로 저장됨
      const lng = defect.geometry.coordinates[1]
      const lat = defect.geometry.coordinates[0]

      // 마커 클릭 이벤트 처리를 위한 publicId 저장
      const publicId = defect.properties.publicId

      // 팝업 생성
      const popup = new mapboxgl.Popup({ offset: 25 }).setHTML(`
        <div>
          <h3 class="font-bold">ID: ${publicId.substring(0, 8)}</h3>
          <p>주소: ${defect.properties.address.street}</p>
          <p>정확도: ${defect.properties.accuracyMeters}m</p>
        </div>
      `)

      // 지도에 마커 추가
      const marker = new mapboxgl.Marker(markerEl)
        .setLngLat([lng, lat])
        .setPopup(popup)
        .addTo(map.current!)

      // 마커 클릭 이벤트 리스너 추가
      markerEl.addEventListener('click', () => {
        setSelectedDefect(publicId)
      })

      // 마커 참조 저장
      markersRef.current.push(marker)
    })
  }, [geoJsonData, loading, setSelectedDefect])

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

export default function DefectMap({
  onSelectTab,
  filteredDefectDetailList,
}: {
  onSelectTab?: (tab: string) => void
  filteredDefectDetailList: DefectDetail[]
}) {
  const mapCardRef = useRef<HTMLDivElement>(null)
  const detailCardRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (mapCardRef.current && detailCardRef.current) {
      detailCardRef.current.style.maxHeight = `${mapCardRef.current.clientHeight}px`
    }
  }, [mapCardRef.current?.clientHeight])

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card ref={mapCardRef} className="md:col-span-2">
        <CardHeader>
          <CardTitle>결함 발생 현황</CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <div className="aspect-video overflow-hidden rounded-md">
            <DefectMapContent />
          </div>
        </CardContent>
      </Card>

      <Card ref={detailCardRef} className="flex flex-col">
        <CardHeader>
          <CardTitle>지도에 보이는 손상 필터링</CardTitle>
        </CardHeader>
        <CardContent className="grow overflow-y-hidden p-4 pt-0">
          <RecentAlerts filteredDefectDetailList={filteredDefectDetailList} />
        </CardContent>
        <CardFooter>
          <Button
            variant="outline"
            size="sm"
            className="w-full"
            onClick={() => onSelectTab && onSelectTab('list')}
          >
            전체 이력 보기
            <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
