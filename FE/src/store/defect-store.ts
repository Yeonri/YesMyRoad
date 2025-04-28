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

export type Defect = {
  id: string
  type: string
  severity: string
  location: string
  detectedAt: string
  status?: string
  description?: string
}

export type DefectLocation = { id: number; lat: number; lng: number; type: string; severity: string; title: string }

export type HeatmapLocation = { lat: number; lng: number }

export type DefectStoreState = {
  // Filter states
  timeRange: TimeRangeType
  defectType: DefectType
  severity: SeverityType

  // Data states
  defectLocations: DefectLocation[]
  heatmapLocations: HeatmapLocation[]
  defects: Defect[]
  recentAlerts: Defect[]
  defectTypeData: { value: number; name: string }[]
  severityData: { value: number; name: string }[]
  trendData: { dates: string[]; potholesData: number[]; cracksData: number[]; paintData: number[] }
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
  severityCounts: { critical: number; high: number; medium: number; low: number }

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
}

export const useDefectStore = create<DefectStoreState>((set) => ({
  // Initial filter states
  timeRange: '24h',
  defectType: 'all',
  severity: 'all',

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
}))
