'use client'

import { coordinateAPI } from '@/api/coordinate-api'
import DefectImage from '@/components/dashboard/list/defect-image'
import DefectPaginations from '@/components/dashboard/list/defect-paginations'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { StatusBadge } from '@/components/ui/status-badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  formatDate,
  getDisplayId,
  getSeverity,
  getSeverityColor,
  getStatusColor,
} from '@/lib/formatter'
import { useDefectListStore } from '@/store/defect-list-store'
import { useDefectStore } from '@/store/defect-store'
import {
  DetailedDamageType,
  useDetailedDefectStore,
} from '@/store/detailed-defect-list-store'
import { ProcessStatus } from '@/types/defects'
import { useQueries } from '@tanstack/react-query'
import { ChevronDown, ChevronUp, Clock, MapPin } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef } from 'react'

type FilterType = 'timeRange' | 'defectType' | 'severity' | 'process' | ''

type damageType = {
  id: number
  category: string
  status: number
  createdAt: string
}

type DashboardFilterProps = {
  filter: FilterType
  timeRange: string
  defectType: string
  severity: string
  process: string
  idSearchQuery: string
  addrSearchQuery: string
}

export default function DefectList({
  filter,
  timeRange,
  defectType,
  severity,
  process,
  idSearchQuery,
  addrSearchQuery,
}: DashboardFilterProps) {
  const processStatusList: ProcessStatus[] = [
    'REPORTED',
    'RECEIVED',
    'IN_PROGRESS',
    'COMPLETED',
  ]
  const processStatusListKR: Record<ProcessStatus, string> = {
    REPORTED: '보고됨',
    RECEIVED: '접수됨',
    IN_PROGRESS: '작업 중',
    COMPLETED: '작업 완료',
  }

  const {
    sortColumn,
    sortDirection,
    detailedGeoJSONData,
    setSortColumn,
    setSortDirection,
    setDetailedGeoJSONData,
    updateDefectStatus,
    getFilteredAndSortedData,
    getCurrentPageData,
    setFilterType,
    setProcessFilter,
    setSeverityFilter,
    setDefectTypeFilter,
    setTimeRangeFilter,
    setidSearchQuery,
    setaddrSearchQuery,
  } = useDetailedDefectStore()

  const { currentPage, itemsPerPage, setTotalItems, setCurrentPage } =
    useDefectListStore()
  const { geoJSONData } = useDefectStore()

  // 이전 totalItems 값을 저장하는 ref
  const prevTotalItemsRef = useRef(0)

  // React Query를 사용하여 결함 세부 정보 가져오기
  const defectQueries = useQueries({
    queries: (geoJSONData || []).map((feature) => {
      const publicId = feature.properties.publicId
      return {
        queryKey: ['defectDetail', publicId],
        queryFn: () => coordinateAPI.getDefectDetail(publicId),
        enabled: !!publicId,
      }
    }),
  })

  // 결함 세부 정보 데이터 변환 로직
  const detailedData = useMemo(() => {
    const allQueriesReady = defectQueries.every(
      (query) => query.isSuccess && query.data?.data,
    )

    if (!allQueriesReady || !geoJSONData?.length) {
      return []
    }

    return geoJSONData.flatMap((feature, index) => {
      const queryResult = defectQueries[index]
      const damages = queryResult.data?.data.damages || []
      const risk = queryResult.data?.data.risk
      const imageUrl = queryResult.data?.data.imageUrl

      if (Array.isArray(damages) && risk && imageUrl) {
        return damages.map((damage: damageType) => ({
          defectId: getDisplayId(
            damage.category,
            damage.id,
            feature.properties.publicId,
          ),
          damageId: damage.id,
          imageUrl: imageUrl,
          type: feature.geometry.type,
          location: feature.properties.address.street,
          category: damage.category,
          createdAt: damage.createdAt,
          publicId: feature.properties.publicId,
          severity: getSeverity(risk),
          status: processStatusList[damage.status],
        }))
      }
      return []
    })
  }, [defectQueries, geoJSONData, processStatusList])

  // 이전 detailedData와 현재 detailedData 비교 후 변경되었을 때만 스토어 업데이트
  const detailedDataRef = useRef<DetailedDamageType[]>([])

  useEffect(() => {
    if (detailedData.length > 0) {
      // 깊은 비교를 위해 JSON 문자열로 변환하여 비교
      const prevDataStr = JSON.stringify(detailedDataRef.current)
      const currentDataStr = JSON.stringify(detailedData)

      // 데이터가 변경된 경우에만 스토어 업데이트
      if (prevDataStr !== currentDataStr) {
        setDetailedGeoJSONData(detailedData)
        detailedDataRef.current = [...detailedData]
      }
    }
  }, [detailedData, setDetailedGeoJSONData])

  // 컴포넌트 마운트 또는 props 변경 시 모든 필터 상태 업데이트
  useEffect(() => {
    setFilterType(filter)
    setProcessFilter(process)
    setSeverityFilter(severity)
    setDefectTypeFilter(defectType)
    setTimeRangeFilter(timeRange)
    setidSearchQuery(idSearchQuery)
    setaddrSearchQuery(addrSearchQuery)
  }, [
    filter,
    process,
    severity,
    defectType,
    timeRange,
    idSearchQuery,
    addrSearchQuery,
    setFilterType,
    setProcessFilter,
    setSeverityFilter,
    setDefectTypeFilter,
    setTimeRangeFilter,
    setidSearchQuery,
    setaddrSearchQuery,
  ])

  useEffect(() => {
    // 필터 또는 검색어가 변경되면 페이지를 1로 리셋
    setCurrentPage(1)
  }, [
    filter,
    process,
    severity,
    defectType,
    timeRange,
    idSearchQuery,
    addrSearchQuery,
    setCurrentPage,
  ])

  // 필터링 및 정렬된 데이터 계산 (검색어 추가)
  const filteredAndSortedData = useMemo(() => {
    if (detailedGeoJSONData.length === 0) return []
    return getFilteredAndSortedData(
      filter,
      process,
      severity,
      defectType,
      timeRange,
      idSearchQuery,
      addrSearchQuery,
    )
  }, [
    detailedGeoJSONData,
    sortColumn,
    sortDirection,
    getFilteredAndSortedData,
    filter,
    process,
    severity,
    defectType,
    timeRange,
    idSearchQuery,
    addrSearchQuery,
  ])

  // 현재 페이지 데이터 계산 (검색어 추가)
  const currentDefects = useMemo(() => {
    if (filteredAndSortedData.length === 0) return []
    return getCurrentPageData(
      currentPage,
      itemsPerPage,
      filter,
      process,
      severity,
      defectType,
      timeRange,
      idSearchQuery,
      addrSearchQuery,
    )
  }, [
    getCurrentPageData,
    currentPage,
    itemsPerPage,
    filteredAndSortedData,
    filter,
    process,
    severity,
    defectType,
    timeRange,
    idSearchQuery,
    addrSearchQuery,
  ])

  // 데이터 길이가 실제로 변경될 때만 totalItems 업데이트
  useEffect(() => {
    const totalItems = filteredAndSortedData.length

    // 이전 값과 다를 때만 업데이트
    if (prevTotalItemsRef.current !== totalItems) {
      setTotalItems(totalItems)
      prevTotalItemsRef.current = totalItems
    }
  }, [filteredAndSortedData.length, setTotalItems])

  // 정렬 핸들러 (useCallback으로 메모이제이션)
  const handleSort = useCallback(
    (column: keyof DetailedDamageType) => {
      if (sortColumn === column) {
        setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
      } else {
        setSortColumn(column)
        setSortDirection('asc')
      }
    },
    [sortColumn, sortDirection, setSortColumn, setSortDirection],
  )

  // 상태 변경 핸들러 (useCallback으로 메모이제이션)
  const handleStatusChange = useCallback(
    (
      damageId: number,
      currentStatus: string,
      newStatus: string,
      defectId: string,
    ) => {
      updateDefectStatus(damageId, currentStatus, newStatus, defectId)
    },
    [updateDefectStatus],
  )

  return (
    <div className="w-full overflow-auto">
      <Table>
        <TableHeader>
          <TableRow className="grid grid-cols-24">
            <TableHead className="col-span-3">
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort('defectId')}
              >
                결함 ID
                {sortColumn === 'defectId' &&
                  (sortDirection === 'asc' ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  ))}
              </Button>
            </TableHead>
            <TableHead className="col-span-3">
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort('category')}
              >
                결함 유형
                {sortColumn === 'category' &&
                  (sortDirection === 'asc' ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  ))}
              </Button>
            </TableHead>
            <TableHead className="col-span-2">
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort('severity')}
              >
                심각도
                {sortColumn === 'severity' &&
                  (sortDirection === 'asc' ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  ))}
              </Button>
            </TableHead>
            <TableHead className="col-span-6">
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort('location')}
              >
                발생 위치
                {sortColumn === 'location' &&
                  (sortDirection === 'asc' ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  ))}
              </Button>
            </TableHead>
            <TableHead className="col-span-4 flex items-center pb-2">
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort('createdAt')}
              >
                발생 시각
                {sortColumn === 'createdAt' &&
                  (sortDirection === 'asc' ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  ))}
              </Button>
            </TableHead>
            <TableHead className="col-span-3">
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort('status')}
              >
                작업 현황
                {sortColumn === 'status' &&
                  (sortDirection === 'asc' ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  ))}
              </Button>
            </TableHead>
            <TableHead className="col-span-3 flex items-center pb-2">
              이미지 목록
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {currentDefects.map((defect) => (
            <TableRow key={defect.defectId} className="grid grid-cols-24">
              <TableCell className="col-span-3 font-medium">
                {defect.defectId}
              </TableCell>
              <TableCell className="col-span-3">{defect.category}</TableCell>
              <TableCell className="col-span-2">
                <StatusBadge className={getSeverityColor(defect.severity)}>
                  {defect.severity.charAt(0).toUpperCase() +
                    defect.severity.slice(1)}
                </StatusBadge>
              </TableCell>
              <TableCell className="col-span-6">
                <div className="flex items-center gap-1">
                  <MapPin className="text-muted-foreground h-3 w-3" />
                  {defect.location}
                </div>
              </TableCell>
              <TableCell className="col-span-4">
                <div className="flex items-center gap-1">
                  <Clock className="text-muted-foreground h-3 w-3" />
                  {formatDate(defect.createdAt)}
                </div>
              </TableCell>
              <TableCell className="col-span-3">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <StatusBadge className={getStatusColor(defect.status!)}>
                      {processStatusListKR[defect.status as ProcessStatus]}
                    </StatusBadge>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start">
                    <DropdownMenuLabel>
                      다음으로 작업 상태 변경
                    </DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {processStatusList.map((process) => (
                      <DropdownMenuItem
                        key={process}
                        onClick={() =>
                          handleStatusChange(
                            defect.damageId,
                            defect.status!,
                            process,
                            defect.defectId,
                          )
                        }
                      >
                        {processStatusListKR[process]}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </TableCell>
              <TableCell className="col-span-3 select-none">
                <DefectImage
                  imageUrl={defect.imageUrl}
                  alt={`결함 이미지 ${defect.defectId}`}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <DefectPaginations />
    </div>
  )
}
