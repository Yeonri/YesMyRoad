// @/store/detailed-defect-store.ts
import { create } from 'zustand'
import { defectAPI } from '@/api/defect-api'
import { ProcessStatus } from '@/types/defects'

// 타입 정의
export type DetailedDamageType = {
  defectId: string
  damageId: number
  imageUrl: string
  type: string
  location: string
  category: string
  detectedAt: string
  publicId: string
  severity: string
  status: string
}

type SortDirection = 'asc' | 'desc'
type SortColumn = keyof DetailedDamageType
export type FilterType = 'timeRange' | 'defectType' | 'severity' | 'process' | ''

interface DetailedDefectState {
  // 상태
  detailedGeoJSONData: DetailedDamageType[]
  sortColumn: SortColumn
  sortDirection: SortDirection
  filterType: FilterType
  processFilter: string
  severityFilter: string
  defectTypeFilter: string
  timeRangeFilter: string // 추가: 발생 시각 필터
  
  // 액션
  setDetailedGeoJSONData: (data: DetailedDamageType[]) => void
  setSortColumn: (column: SortColumn) => void
  setSortDirection: (direction: SortDirection) => void
  setFilterType: (type: FilterType) => void
  setProcessFilter: (process: string) => void
  setSeverityFilter: (severity: string) => void
  setDefectTypeFilter: (defectType: string) => void
  setTimeRangeFilter: (timeRange: string) => void // 추가: 발생 시각 필터 설정
  updateDefectStatus: (damageId: number, currentStatus: string, newStatus: string, defectId: string) => Promise<void>
  
  // 선택자(selector)
  getFilteredAndSortedData: (filterType: FilterType, processFilter: string, severityFilter: string, defectTypeFilter: string, timeRangeFilter: string) => DetailedDamageType[]
  getCurrentPageData: (currentPage: number, itemsPerPage: number, filterType: FilterType, processFilter: string, severityFilter: string, defectTypeFilter: string, timeRangeFilter: string) => DetailedDamageType[]
}

export const useDetailedDefectStore = create<DetailedDefectState>((set, get) => ({
  // 초기 상태
  detailedGeoJSONData: [],
  sortColumn: 'defectId',
  sortDirection: 'asc',
  filterType: '',
  processFilter: 'all',
  severityFilter: 'all',
  defectTypeFilter: 'all',
  timeRangeFilter: 'all', // 추가: 발생 시각 필터 초기값
  
  // 액션
  setDetailedGeoJSONData: (data) => set({ detailedGeoJSONData: data }),
  setSortColumn: (column) => set({ sortColumn: column }),
  setSortDirection: (direction) => set({ sortDirection: direction }),
  setFilterType: (type) => set({ filterType: type }),
  setProcessFilter: (process) => set({ processFilter: process }),
  setSeverityFilter: (severity) => set({ severityFilter: severity }),
  setDefectTypeFilter: (defectType) => set({ defectTypeFilter: defectType }),
  setTimeRangeFilter: (timeRange) => set({ timeRangeFilter: timeRange }), // 추가: 발생 시각 필터 설정 함수
  
  updateDefectStatus: async (damageId, currentStatus, newStatus, defectId) => {
    try {
      const response = await defectAPI.updateRoadDamageStatus(damageId, newStatus)
      
      if (response.status === 200 || response.data) {
        console.log("success: ", response.data)
        
        // 상태 업데이트
        set((state) => ({
          detailedGeoJSONData: state.detailedGeoJSONData.map((defect) => {
            if (defect.defectId === defectId) {
              return { ...defect, status: newStatus }
            }
            return defect
          })
        }))
        
        alert(`작업 상태를 ${currentStatus}에서 ${newStatus}로 변경하였습니다`)
      } else {
        console.error('Status update failed:', response)
        alert('Failed to change status')
      }
    } catch (error) {
      console.error('API call error:', error)
      alert('Network error occurred')
    }
  },
  
  // 선택자(selector)
  getFilteredAndSortedData: (filterType, processFilter, severityFilter, defectTypeFilter, timeRangeFilter) => {
    const { detailedGeoJSONData, sortColumn, sortDirection } = get()
    
    // 필터링
    let filteredData = [...detailedGeoJSONData]
    
    // filterType에 따라 하나의 필터만 적용
    if (filterType === 'process' && processFilter !== 'all') {
      // 작업 상태 필터
      filteredData = filteredData.filter((defect) => defect.status === processFilter)
    } else if (filterType === 'severity' && severityFilter !== 'all') {
      // 심각도 필터
      filteredData = filteredData.filter((defect) => defect.severity === severityFilter)
    } else if (filterType === 'defectType' && defectTypeFilter !== 'all') {
      // 결함 유형 필터 (category 필드 기준)
      filteredData = filteredData.filter((defect) => defect.category === defectTypeFilter)
    } else if (filterType === 'timeRange' && timeRangeFilter !== 'all') {
      // 발생 시각 필터
      const now = new Date()
      
      filteredData = filteredData.filter((defect) => {
        const detectedDate = new Date(defect.detectedAt)
        const diffInMilliseconds = now.getTime() - detectedDate.getTime()
        const diffInDays = diffInMilliseconds / (1000 * 60 * 60 * 24)
        
        switch(timeRangeFilter) {
          case 'daily':
            // 24시간 이내 (1일)
            return diffInDays <= 1
          case 'weekly':
            // 1주일 이내 (7일)
            return diffInDays <= 7
          case 'monthly':
            // 1달 이내 (30일)
            return diffInDays <= 30
          default:
            return true
        }
      })
    }
    
    // 정렬
    return [...filteredData].sort((a, b) => {
      if (sortDirection === 'asc') {
        return a[sortColumn as keyof typeof a]! > b[sortColumn as keyof typeof b]! ? 1 : -1
      } else {
        return a[sortColumn as keyof typeof a]! < b[sortColumn as keyof typeof b]! ? 1 : -1
      }
    })
  },
  
  getCurrentPageData: (currentPage, itemsPerPage, filterType, processFilter, severityFilter, defectTypeFilter, timeRangeFilter) => {
    const filteredAndSortedData = get().getFilteredAndSortedData(filterType, processFilter, severityFilter, defectTypeFilter, timeRangeFilter)
    const startIndex = (currentPage - 1) * itemsPerPage
    return filteredAndSortedData.slice(startIndex, startIndex + itemsPerPage)
  }
}))