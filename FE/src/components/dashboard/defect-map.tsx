'use client'

import DamageListOnMap from '@/components/dashboard/damage-list-on-map'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import useAddressStore from '@/store/address-store'
import { useDefectStore } from '@/store/defect-store'
import { DefectDetail, FeaturePoint } from '@/types/defects'
import MapboxLanguage from '@mapbox/mapbox-gl-language'
import { ArrowRight, Loader } from 'lucide-react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { useEffect, useRef, useState } from 'react'
import { useDebounceCallback } from 'usehooks-ts'

mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN

function DefectMapContent({
  selectedDefectPublicId,
  onSelectDefect,
}: {
  selectedDefectPublicId: string | null
  onSelectDefect: (id: string) => void
}) {
  // 지도 관련 상태 관리
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  // publicId를 키로, 마커를 값으로 가지는 객체로 변경
  const markersRef = useRef<Record<string, mapboxgl.Marker>>({})
  // 현재 열려있는 마커 ID를 추적
  const activePopupIdRef = useRef<string | null>(null)
  const [loading, setLoading] = useState(true)

  // 주소 스토어에서 위도/경도 가져오기, 없으면 서울 기본값 설정
  const longitude = useAddressStore((state) => state.longitude)
  const latitude = useAddressStore((state) => state.latitude)

  // 맵 경계 좌표 설정 함수 가져오기
  const setMapBounds = useAddressStore((state) => state.setMapBounds)
  const debouncedSetMapBounds = useDebounceCallback(setMapBounds, 500)

  const geoJsonData = useDefectStore((state) => state.geoJSONData)

  // 선택된 결함 ID가 변경될 때 해당 마커의 팝업을 표시
  useEffect(() => {
    if (!map.current) return;

    // 이전에 열려있던 팝업이 있으면 닫기
    if (activePopupIdRef.current && markersRef.current[activePopupIdRef.current]) {
      const prevMarker = markersRef.current[activePopupIdRef.current];
      const popup = prevMarker?.getPopup();
      if (popup && popup.isOpen()) {
        popup.remove();
      }
      // selectedDefectPublicId가 null이거나 다른 마커가 선택된 경우 현재 활성화된 팝업 ID 초기화
      if (!selectedDefectPublicId || selectedDefectPublicId !== activePopupIdRef.current) {
        activePopupIdRef.current = null;
      }
    }

    // 새 마커가 선택된 경우에만 팝업 열기
    if (selectedDefectPublicId && markersRef.current[selectedDefectPublicId]) {
      const marker = markersRef.current[selectedDefectPublicId];
      marker.togglePopup();
      activePopupIdRef.current = selectedDefectPublicId;
    }
  }, [selectedDefectPublicId])

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

  // 마커 추가 로직을 별도 useEffect로 분리
  useEffect(() => {
    if (!map.current || loading || !geoJsonData) return

    // 기존 마커 제거
    Object.values(markersRef.current).forEach((marker) => marker.remove())
    markersRef.current = {}
    // 활성 팝업 ID 초기화
    activePopupIdRef.current = null

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
        onSelectDefect(publicId)
      })

      // 마커를 publicId로 구분하여 저장
      markersRef.current[publicId] = marker
    })
  }, [geoJsonData, loading])

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
  selectedTab,
}: {
  onSelectTab?: (tab: string) => void
  filteredDefectDetailList: DefectDetail[]
  selectedTab?: string
}) {
  const mapCardRef = useRef<HTMLDivElement>(null)
  const detailCardRef = useRef<HTMLDivElement>(null)
  const [selectedDefectPublicId, setSelectedDefectPublicId] = useState<string | null>(null)
  
  // 주소 스토어에서 위도/경도와 맵 바운드 가져오기
  const longitude = useAddressStore((state) => state.longitude)
  const latitude = useAddressStore((state) => state.latitude)
  const mapBounds = useAddressStore((state) => state.mapBounds)

  useEffect(() => {
    if (mapCardRef.current && detailCardRef.current) {
      detailCardRef.current.style.maxHeight = `${mapCardRef.current.clientHeight}px`
    }
  }, [mapCardRef.current?.clientHeight])

  // selectedDefectPublicId 변경 시 콘솔에 로그 출력
  useEffect(() => {
    if (selectedDefectPublicId) {
      console.log('선택된 결함 ID (DefectMap):', selectedDefectPublicId)
    }
  }, [selectedDefectPublicId])
  
  // 탭 변경, 위치 변경, 맵 바운드 변경 시 결함 선택 초기화
  useEffect(() => {
    // 선택된 결함 ID 초기화
    setSelectedDefectPublicId(null)
  }, [selectedTab, longitude, latitude, mapBounds.northEast, mapBounds.southWest])

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card ref={mapCardRef} className="md:col-span-2">
        <CardHeader>
          <CardTitle>결함 발생 현황</CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0">
          <div className="aspect-video overflow-hidden rounded-md">
            <DefectMapContent
              selectedDefectPublicId={selectedDefectPublicId}
              onSelectDefect={setSelectedDefectPublicId}
            />
          </div>
        </CardContent>
      </Card>

      <Card ref={detailCardRef} className="flex flex-col">
        <CardHeader>
          <CardTitle>지도에 보이는 손상 필터링</CardTitle>
        </CardHeader>
        <CardContent className="grow overflow-y-hidden p-4 pt-0">
          <DamageListOnMap
            filteredDefectDetailList={filteredDefectDetailList}
            selectedDefectPublicId={selectedDefectPublicId}
            onSelectDefect={setSelectedDefectPublicId}
          />
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
