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
import { create } from 'zustand'

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

  // 상태 업데이트 함수 (API 호출 없이 상태만 업데이트)
  updateDefects: (newDefects: Defect[]) => void
  updateDefectLocations: (newLocations: DefectLocation[]) => void
  updateRecentAlerts: (newAlerts: Defect[]) => void
  updateDefectStats: (typeData: { value: number; name: string }[], sevData: { value: number; name: string }[]) => void
  updateDefectTrends: (newTrends: any) => void
  updateDetailedDefect: (defect: DetailedDefect | null) => void
  updateGeoJSONData: (data: GeoJSONData | null) => void

  // 상태 조회 함수
  getGeoJSONData: () => GeoJSONData | null
}

export const useDefectStore = create<DefectStoreState>((set, get) => ({
  // Initial filter states
  timeRange: 'D',
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

  // 상태 업데이트 함수들 (API 호출 없이 상태만 업데이트)
  updateDefects: (newDefects) => set({ defects: newDefects }),
  updateDefectLocations: (newLocations) => set({ defectLocations: newLocations }),
  updateRecentAlerts: (newAlerts) => set({ recentAlerts: newAlerts }),
  updateDefectStats: (typeData, sevData) => set({ defectTypeData: typeData, severityData: sevData }),
  updateDefectTrends: (newTrends) => set({ trendData: newTrends }),
  updateDetailedDefect: (defect) => set({ detailedDefect: defect }),
  updateGeoJSONData: (data) => set({ geoJSONData: data }),

  // 상태 조회 함수 - geoJSONData 반환
  getGeoJSONData: () => get().geoJSONData
}))