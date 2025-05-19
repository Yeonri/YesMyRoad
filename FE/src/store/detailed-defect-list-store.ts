// @/store/detailed-defect-store.ts
import { create } from 'zustand'
import { defectAPI } from '@/api/defect-api'

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

interface DetailedDefectState {
  // 상태
  detailedGeoJSONData: DetailedDamageType[]
  sortColumn: SortColumn
  sortDirection: SortDirection
  defectType: string
  severity: string
  
  // 액션
  setDetailedGeoJSONData: (data: DetailedDamageType[]) => void
  setSortColumn: (column: SortColumn) => void
  setSortDirection: (direction: SortDirection) => void
  setDefectType: (type: string) => void
  setSeverity: (severity: string) => void
  updateDefectStatus: (damageId: number, currentStatus: string, newStatus: string, defectId: string) => Promise<void>
  
  // 선택자(selector)
  getFilteredAndSortedData: () => DetailedDamageType[]
  getCurrentPageData: (currentPage: number, itemsPerPage: number) => DetailedDamageType[]
}

export const useDetailedDefectStore = create<DetailedDefectState>((set, get) => ({
  // 초기 상태
  detailedGeoJSONData: [],
  sortColumn: 'defectId',
  sortDirection: 'asc',
  defectType: 'all',
  severity: 'all',
  
  // 액션
  setDetailedGeoJSONData: (data) => set({ detailedGeoJSONData: data }),
  setSortColumn: (column) => set({ sortColumn: column }),
  setSortDirection: (direction) => set({ sortDirection: direction }),
  setDefectType: (type) => set({ defectType: type }),
  setSeverity: (severity) => set({ severity: severity }),
  
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
  getFilteredAndSortedData: () => {
    const { detailedGeoJSONData, defectType, severity, sortColumn, sortDirection } = get()
    
    // 필터링
    const filteredData = detailedGeoJSONData.filter((defect) => {
      const matchesType = defectType === 'all' || defect.type.toLowerCase() === defectType
      const matchesSeverity = severity === 'all' || defect.severity === severity
      return matchesType && matchesSeverity
    })
    
    // 정렬
    return [...filteredData].sort((a, b) => {
      if (sortDirection === 'asc') {
        return a[sortColumn as keyof typeof a]! > b[sortColumn as keyof typeof b]! ? 1 : -1
      } else {
        return a[sortColumn as keyof typeof a]! < b[sortColumn as keyof typeof b]! ? 1 : -1
      }
    })
  },
  
  getCurrentPageData: (currentPage, itemsPerPage) => {
    const filteredAndSortedData = get().getFilteredAndSortedData()
    const startIndex = (currentPage - 1) * itemsPerPage
    return filteredAndSortedData.slice(startIndex, startIndex + itemsPerPage)
  }
}))