// src/store/defect-store.ts
import { coodAPI } from '@/api/coordinate-api'
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
import { create } from 'zustand'

// Define types for our store
export type DefectType = 'pothole' | 'crack' | 'paint' | 'all'
export type SeverityType = 'critical' | 'high' | 'medium' | 'low' | 'all'
export type TimeRangeType = '1h' | '24h' | '7d' | '30d'
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

// 상세 손상 정보를 위한 타입 정의
export type DamageItem = {
  id: number
  category: string
  status: number
  updatedAt: string
}

export type DetailedDefect = {
  imageUrl: string
  risk: number
  damages: DamageItem[]
}

// GeoJSON 데이터를 위한 타입 정의
export type FeaturePoint = {
  type: string
  geometry: { type: string; coordinates: number[] }
  properties: {
    publicId: string
    address: { street: string }
    accuracyMeters: number
  }
}

export type GeoJSONData = { type: string; features: FeaturePoint[] }

export type DefectStoreState = {
  // Filter states
  timeRange: TimeRangeType
  defectType: DefectType
  severity: SeverityType

  detailedDefect: DetailedDefect | null
  geoJSONData: GeoJSONData | null

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

  // Actions
  setTimeRange: (timeRange: TimeRangeType) => void
  setDefectType: (defectType: DefectType) => void
  setSeverity: (severity: SeverityType) => void

  // TODO: Add actions to fetch data from API
  fetchDefects: () => Promise<void>
  fetchDefectLocations: () => Promise<void>
  fetchRecentAlerts: () => Promise<void>
  fetchDefectStats: () => Promise<void>
  fetchDefectTrends: () => Promise<void>
  fetchDetailedDefect: (publicId: string) => Promise<void>
  fetchGeoJSONData: () => Promise<void>
  fetchGeoDataAndDetails: () => Promise<void>
}

export const useDefectStore = create<DefectStoreState>((set) => ({
  // Initial filter states
  timeRange: '24h',
  defectType: 'all',
  severity: 'all',
  detailedDefect: null,
  geoJSONData: null,

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

  // Actions for updating filters
  setTimeRange: (timeRange) => set({ timeRange }),
  setDefectType: (defectType) => set({ defectType }),
  setSeverity: (severity) => set({ severity }),

  // TODO: Implement these functions to fetch data from API endpoints
  fetchDefects: async () => {
    // TODO: Replace with actual API call
    // const response = await fetch('/api/defects')
    // const data = await response.json()
    // set({ defects: data })

    // Using placeholder data for now
    set({ defects })
  },

  fetchDefectLocations: async () => {
    // TODO: Replace with actual API call
    // const response = await fetch('/api/defect-locations')
    // const data = await response.json()
    // set({ defectLocations: data })

    // Using placeholder data for now
    set({ defectLocations })
  },

  fetchRecentAlerts: async () => {
    // TODO: Replace with actual API call
    // const response = await fetch('/api/alerts')
    // const data = await response.json()
    // set({ recentAlerts: data })

    // Using placeholder data for now
    set({ recentAlerts })
  },

  fetchDefectStats: async () => {
    // TODO: Replace with actual API call
    // const response = await fetch('/api/stats')
    // const data = await response.json()
    // set({ defectTypeData: data.typeData, severityData: data.severityData })

    // Using placeholder data for now
    set({ defectTypeData, severityData })
  },

  fetchDefectTrends: async () => {
    // TODO: Replace with actual API call
    // const response = await fetch('/api/trends')
    // const data = await response.json()
    // set({ trendData: data })

    // Using placeholder data for now
    set({ trendData })
  },

  // 상세 손상 정보 조회 액션 구현
  fetchDetailedDefect: async (publicId: string) => {
    const response = await coodAPI.getDetailedDefects(publicId)
    set({ detailedDefect: response.data })
  },

  // GeoJSON 데이터 조회 액션 구현
  fetchGeoJSONData: async () => {
    const response = await coodAPI.getDefects()
    set({ geoJSONData: response.data })
  },

  // 복합 액션: GeoJSON 데이터를 가져온 후 첫 번째 항목의 상세 정보 가져오기
  fetchGeoDataAndDetails: async () => {
    // 1. 먼저 GeoJSON 데이터 가져오기
    const response = await coodAPI.getDefects()
    set({ geoJSONData: response.data })

    // 2. 첫 번째 점의 publicId 추출
    if (response.data?.features?.length > 0) {
      const firstPointId = response.data.features[0].properties.publicId

      // 3. 추출한 publicId로 상세 정보 가져오기
      const detailResponse = await coodAPI.getDetailedDefects(firstPointId)
      set({ detailedDefect: detailResponse.data })

      console.log('GeoJSON 데이터와 상세 정보 로드 완료:', {
        geoFeatures: response.data.features.length,
        detailDamages: detailResponse.data.damages.length,
      })
    } else {
      console.warn('GeoJSON 데이터에 features가 없습니다.')
    }
  },
}))
