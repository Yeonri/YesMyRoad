// src/store/defect-store.ts
import {
  dashboardMetrics,
  defectLocations,
  defectTypeData,
  defects,
  heatmapLocations,
  recentAlerts,
  severityCounts,
  severityData,
  trendData,
} from '@/data/placeholders'
import crypto from 'crypto'
import { create } from 'zustand'

// UUID로부터 표시 ID 생성하는 함수 추가
const getDisplayId = (uuid: string, damageId: number, prefix: string = 'DEF'): string => {
  const validChars = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
  const hash = crypto.createHash('sha256').update(uuid).digest('hex')

  const shortCode = Array(5)
    .fill(0)
    .map((_, i) => {
      const index =
        parseInt(hash.slice(i * 2, i * 2 + 2), 16) % validChars.length
      return validChars[index]
    })
    .join('')

  return `${prefix}-${damageId}-${shortCode}`
}

// Define types for our store
export type DefectType = 'pothole' | 'crack' | 'paint' | 'all'
export type SeverityType = 'critical' | 'high' | 'medium' | 'low' | 'all'
export type TimeRangeType = 'D' | 'W' | 'M'
export type ProcessStatus = 'Pending' | 'Assigned' | 'In Progress' | 'Completed'

export type Defect = {
  id: string
  type: string
  severity: string
  location: string
  detectedAt: string
  status?: string
  description?: string
}

export type DefectLocation = {
  id: number
  lat: number
  lng: number
  type: string
  severity: string
  title: string
}

export type HeatmapLocation = { lat: number; lng: number }

// 손상 상세 정보를 위한 타입 정의
export type DamageItem = {
  id: number
  category: string
  status: number
  updatedAt: string
}

export type DefectDetail = {
  publicId: string
  address: string
  imageUrl: string
  risk: number
  damages: DamageItem[]
}

// GeoJSON 데이터를 위한 타입 정의 - 수정함
export type FeaturePoint = {
  type: string
  geometry: { type: string; coordinates: number[] }
  properties: {
    publicId: string
    displayId?: string // 해싱된 ID를 저장할 필드 추가
    address: { street: string }
    accuracyMeters: number
  }
}

// 저장소는 GeoJSONData 전체가 아닌 FeaturePoint 배열만 저장하도록 수정
export type FeatureCollection = FeaturePoint[]

export type ProcessedDefect = {
  id: string
  publicId: string
  type: string
  severity: string
  location: string
  detectedAt: string
  status: string
  description: string
  risk?: number
  imageUrl?: string
  damages?: DamageItem[]
}

export type DefectStoreState = {
  // Filter states
  timeRange: TimeRangeType
  defectType: DefectType
  severity: SeverityType

  // 손상 발생 위치 데이터 리스트
  geoJSONData: FeatureCollection | null // 타입 변경

  // 손상 상세 정보
  defectDetail: DefectDetail | null
  defectDetailList: DefectDetail[]

  // Data states
  defectLocations: DefectLocation[]
  heatmapLocations: HeatmapLocation[]
  defects: Defect[]
  recentAlerts: Defect[]
  defectTypeData: { value: number; name: string }[]
  severityData: { value: number; name: string }[]
  trendData: {
    dates: string[]
    potholesData: number[]
    cracksData: number[]
    paintData: number[]
  }
  dashboardMetrics: {
    totalDefects: number
    totalDefectsChange: string
    criticalIssues: number
    criticalIssuesChange: string
    avgResponseTime: string
    avgResponseTimeChange: string
    affectedAreas: number
    affectedAreasChange: string
  }
  severityCounts: {
    critical: number
    high: number
    medium: number
    low: number
  }

  // 지도 상 마커를 클릭했을 때, 선택된 결함의 publicId를 저장하는 상태
  selectedDefect: { publicId: string | null }
  processedDefects: ProcessedDefect[] // 가공된 결함 데이터
  detailsMap: Record<string, DefectDetail> // publicId를 키로 하는 상세 정보 맵
  isProcessingDefects: boolean // 데이터 처리 중 상태

  // Actions
  setTimeRange: (timeRange: TimeRangeType) => void
  setDefectType: (defectType: DefectType) => void
  setSeverity: (severity: SeverityType) => void

  updateProcessedDefects: (defects: ProcessedDefect[]) => void
  updateDetailsMap: (detailsMap: Record<string, DefectDetail>) => void
  setProcessingDefects: (isProcessing: boolean) => void

  // 상태 업데이트 함수 (API 호출 없이 상태만 업데이트)
  updateDefects: (newDefects: Defect[]) => void
  updateDefectLocations: (newLocations: DefectLocation[]) => void
  updateRecentAlerts: (newAlerts: Defect[]) => void
  updateDefectStats: (
    typeData: { value: number; name: string }[],
    sevData: { value: number; name: string }[],
  ) => void
  updateDefectTrends: (newTrends: any) => void

  // 손상 상세 정보 업데이트 함수
  updateDefectDetail: (defect: DefectDetail | null) => void
  updateDefectDetailList: (defects: DefectDetail[]) => void

  // 수정된 업데이트 함수 - 전체 GeoJSON이 아닌 features 배열만 받음
  updateGeoJSONData: (data: any) => void

  // 상태 조회 함수
  getGeoJSONData: () => FeatureCollection | null

  // selectedDefect 관련 함수
  setSelectedDefect: (publicId: string) => void
  getSelectedDefect: () => { publicId: string | null }
  getProcessedDefects: () => ProcessedDefect[]
}



export const useDefectStore = create<DefectStoreState>((set, get) => ({
  // Initial filter states
  timeRange: 'D',
  defectType: 'all',
  severity: 'all',
  geoJSONData: null,
  defectDetail: null,
  defectDetailList: [],

  // Initial data states with placeholder data
  defectLocations,
  heatmapLocations,
  defects,
  recentAlerts,
  defectTypeData,
  severityData,
  trendData,
  dashboardMetrics,
  severityCounts,

  // selectedDefect 초기값
  selectedDefect: { publicId: null },
  processedDefects: [],
  detailsMap: {},
  isProcessingDefects: false,

  // Actions for updating filters
  setTimeRange: (timeRange) => set({ timeRange }),
  setDefectType: (defectType) => set({ defectType }),
  setSeverity: (severity) => set({ severity }),

  // 상태 업데이트 함수들 (API 호출 없이 상태만 업데이트)
  updateDefects: (newDefects) => set({ defects: newDefects }),
  updateDefectLocations: (newLocations) =>
    set({ defectLocations: newLocations }),
  updateRecentAlerts: (newAlerts) => set({ recentAlerts: newAlerts }),
  updateDefectStats: (typeData, sevData) =>
    set({ defectTypeData: typeData, severityData: sevData }),
  updateDefectTrends: (newTrends) => set({ trendData: newTrends }),
  updateDefectDetail: (defect) => set({ defectDetail: defect }),
  updateDefectDetailList: (defect) => set({ defectDetailList: defect }),

  updateProcessedDefects: (defects) => set({ processedDefects: defects }),
  updateDetailsMap: (detailsMap) => set({ detailsMap }),
  setProcessingDefects: (isProcessing) => set({ isProcessingDefects: isProcessing }),

    // 가공된 defects 가져오는 함수
  getProcessedDefects: () => get().processedDefects,

  // 수정된 업데이트 함수 - features 배열을 받아 각 항목의 publicId를 기반으로 displayId 생성
  updateGeoJSONData: (data) => {
    // GeoJSON 전체 객체가 들어올 경우 features 배열만 추출
    const features = data.features ? data.features : data

    if (!features) {
      set({ geoJSONData: null })
      return
    }

    // 각 feature에 displayId 추가
    const enhancedFeatures = features.map((feature: FeaturePoint) => {
      const publicId = feature.properties.publicId
      return {
        ...feature,
        properties: {
          ...feature.properties,
          // 기본 ID는 damage ID 없이 생성 (이후 상세 데이터 로드 시 업데이트됨)
          displayId: getDisplayId(publicId, 0),
        },
      }
    })

    set({ geoJSONData: enhancedFeatures })
  },

  // 상태 조회 함수 - geoJSONData 반환
  getGeoJSONData: () => get().geoJSONData,

  // selectedDefect 설정 함수
  setSelectedDefect: (publicId) => set({ selectedDefect: { publicId } }),

  // selectedDefect 조회 함수
  getSelectedDefect: () => get().selectedDefect,
}))
