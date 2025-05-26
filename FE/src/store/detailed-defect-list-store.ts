// @/store/detailed-defect-store.ts
import { defectAPI } from '@/api/defect-api'
import { create } from 'zustand'

const newStatusLabels: { [key: string]: string } = {
  'REPORTED': '보고됨으로',
  'RECEIVED': '접수됨로',
  'IN_PROGRESS': '작업중으로',
  'COMPLETED': '작업완료로',
}

const currStatusLabels: { [key: string]: string } = {
  'REPORTED': '보고됨',
  'RECEIVED': '접수됨',
  'IN_PROGRESS': '작업중',
  'COMPLETED': '작업완료',
}

// 타입 정의
export type DetailedDamageType = {
  defectId: string
  damageId: number
  imageUrl: string
  type: string
  location: string
  category: string
  createdAt: string
  publicId: string
  severity: string
  status: string
}

type SortDirection = 'asc' | 'desc'
type SortColumn = keyof DetailedDamageType
export type FilterType =
  | 'timeRange'
  | 'defectType'
  | 'severity'
  | 'process'
  | ''

interface DetailedDefectState {
  // 상태
  detailedGeoJSONData: DetailedDamageType[]
  sortColumn: SortColumn
  sortDirection: SortDirection
  filterType: FilterType
  processFilter: string
  severityFilter: string
  defectTypeFilter: string
  timeRangeFilter: string
  idSearchQuery: string
  addrSearchQuery: string

  // 액션
  setDetailedGeoJSONData: (data: DetailedDamageType[]) => void
  setSortColumn: (column: SortColumn) => void
  setSortDirection: (direction: SortDirection) => void
  setFilterType: (type: FilterType) => void
  setProcessFilter: (process: string) => void
  setSeverityFilter: (severity: string) => void
  setDefectTypeFilter: (defectType: string) => void
  setTimeRangeFilter: (timeRange: string) => void
  updateDefectStatus: (
    damageId: number,
    currentStatus: string,
    newStatus: string,
    defectId: string,
  ) => Promise<void>
  setidSearchQuery: (query: string) => void
  setaddrSearchQuery: (query: string) => void

  // 선택자(selector)
  getFilteredAndSortedData: (
    filterType: FilterType,
    processFilter: string,
    severityFilter: string,
    defectTypeFilter: string,
    timeRangeFilter: string,
    idSearchQuery: string,
    addrSearchQuery: string,
  ) => DetailedDamageType[]
  getCurrentPageData: (
    currentPage: number,
    itemsPerPage: number,
    filterType: FilterType,
    processFilter: string,
    severityFilter: string,
    defectTypeFilter: string,
    timeRangeFilter: string,
    idSearchQuery: string,
    addrSearchQuery: string,
  ) => DetailedDamageType[]
}

export const useDetailedDefectStore = create<DetailedDefectState>(
  (set, get) => ({
    // 초기 상태
    detailedGeoJSONData: [],
    sortColumn: 'defectId',
    sortDirection: 'asc',
    filterType: '',
    processFilter: 'all',
    severityFilter: 'all',
    defectTypeFilter: 'all',
    timeRangeFilter: 'all',
    idSearchQuery: '',
    addrSearchQuery: '',

    // 액션
    setDetailedGeoJSONData: (data) => set({ detailedGeoJSONData: data }),
    setSortColumn: (column) => set({ sortColumn: column }),
    setSortDirection: (direction) => set({ sortDirection: direction }),
    setFilterType: (type) => set({ filterType: type }),
    setProcessFilter: (process) => set({ processFilter: process }),
    setSeverityFilter: (severity) => set({ severityFilter: severity }),
    setDefectTypeFilter: (defectType) => set({ defectTypeFilter: defectType }),
    setTimeRangeFilter: (timeRange) => set({ timeRangeFilter: timeRange }),
    setidSearchQuery: (query) => set({ idSearchQuery: query }),
    setaddrSearchQuery: (query) => set({ addrSearchQuery: query }),

    updateDefectStatus: async (
      damageId,
      currentStatus,
      newStatus,
      defectId,
    ) => {
      try {
        const response = await defectAPI.updateRoadDamageStatus(
          damageId,
          newStatus,
        )

        if (response.status === 200 || response.data) {
          // console.log('success: ', response.data)

          // 상태 업데이트
          set((state) => ({
            detailedGeoJSONData: state.detailedGeoJSONData.map((defect) => {
              if (defect.defectId === defectId) {
                return { ...defect, status: newStatus }
              }
              return defect
            }),
          }))

          alert(
            `작업 상태를 ${currStatusLabels[currentStatus]}에서 ${newStatusLabels[newStatus]} 변경하였습니다`,
          )
        } else {
          console.error('Status update failed:', response)
          alert('Failed to change status')
        }
      } catch (error) {
        console.error('API call error:', error)
        alert('Network error occurred')
      }
    },

    // 선택자
    getFilteredAndSortedData: (
      filterType,
      processFilter,
      severityFilter,
      defectTypeFilter,
      timeRangeFilter,
      idSearchQuery,
      addrSearchQuery,
    ) => {
      const { detailedGeoJSONData, sortColumn, sortDirection } = get()

      // 필터링
      let filteredData = [...detailedGeoJSONData]

      // 기존 필터 적용 (filterType에 따라)
      if (filterType === 'process' && processFilter !== 'all') {
        filteredData = filteredData.filter(
          (defect) => defect.status === processFilter,
        )
      } else if (filterType === 'severity' && severityFilter !== 'all') {
        filteredData = filteredData.filter(
          (defect) => defect.severity === severityFilter,
        )
      } else if (filterType === 'defectType' && defectTypeFilter !== 'all') {
        filteredData = filteredData.filter(
          (defect) => defect.category === defectTypeFilter,
        )
      } else if (filterType === 'timeRange' && timeRangeFilter !== 'all') {
        // 발생 시각 필터
        const now = new Date()

        filteredData = filteredData.filter((defect) => {
          const createdDate = new Date(defect.createdAt)
          const diffInMilliseconds = now.getTime() - createdDate.getTime()
          const diffInDays = diffInMilliseconds / (1000 * 60 * 60 * 24)

          switch (timeRangeFilter) {
            case 'daily':
              return diffInDays <= 1
            case 'weekly':
              return diffInDays <= 7
            case 'monthly':
              return diffInDays <= 30
            default:
              return true
          }
        })
      }

      // 검색어 필터 적용 (다른 필터와 독립적으로)
      if (idSearchQuery && idSearchQuery.trim() !== '') {
        const query = idSearchQuery.trim().toLowerCase()
        filteredData = filteredData.filter((defect) =>
          defect.defectId.toLowerCase().includes(query),
        )
      }

      // 주소 검색어 필터 적용 (별도로 분리)
      if (addrSearchQuery && addrSearchQuery.trim() !== '') {
        filteredData = filteredData.filter((defect) =>
          defect.location.includes(addrSearchQuery.trim()),
        )
      }

      // 정렬
      return [...filteredData].sort((a, b) => {
        if (sortDirection === 'asc') {
          return a[sortColumn as keyof typeof a]! >
            b[sortColumn as keyof typeof b]!
            ? 1
            : -1
        } else {
          return a[sortColumn as keyof typeof a]! <
            b[sortColumn as keyof typeof b]!
            ? 1
            : -1
        }
      })
    },

    getCurrentPageData: (
      currentPage,
      itemsPerPage,
      filterType,
      processFilter,
      severityFilter,
      defectTypeFilter,
      timeRangeFilter,
      idSearchQuery,
      addrSearchQuery,
    ) => {
      const filteredAndSortedData = get().getFilteredAndSortedData(
        filterType,
        processFilter,
        severityFilter,
        defectTypeFilter,
        timeRangeFilter,
        idSearchQuery,
        addrSearchQuery,
      )
      const startIndex = (currentPage - 1) * itemsPerPage
      return filteredAndSortedData.slice(startIndex, startIndex + itemsPerPage)
    },
  }),
)
