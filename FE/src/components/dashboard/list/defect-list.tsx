'use client'

import { coordinateAPI } from '@/api/coordinate-api'
import { defectAPI } from '@/api/defect-api'
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
import { ProcessStatus } from '@/types/defects'
import { useQueries } from '@tanstack/react-query'
import {
  ChevronDown,
  ChevronUp,
  Clock,
  MapPin,
  MoreHorizontal,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

type damageType = {
  id: number
  category: string
  status: number
  updatedAt: string
}

type detailedDamageType = {
  defectId: string
  imageUrl: string
  type: string
  location: string
  category: string
  detectedAt: string
  publicId: string
  severity: string
  status: string
}

export default function DefectList() {
  const processStatusList: ProcessStatus[] = [
    'REPORTED',
    'RECEIVED',
    'IN_PROGRESS',
    'COMPLETED',
  ]
  const [sortColumn, setSortColumn] = useState('id')
  const [sortDirection, setSortDirection] = useState('asc')
  const defectType = 'all'
  const severity = 'all'
  const { currentPage, itemsPerPage, setTotalItems } = useDefectListStore()
  const { geoJSONData } = useDefectStore()

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

  // 최적화된 데이터 변환 로직 - useMemo 사용
  const detailedData = useMemo(() => {
    // 모든 쿼리가 완료되었는지 확인
    const allQueriesReady = defectQueries.every(
      (query) => query.isSuccess && query.data?.data,
    )

    if (!allQueriesReady || !geoJSONData?.length) {
      return [] // 준비되지 않았으면 빈 배열 반환
    }

    // 데이터 변환 로직 실행 (한 번만 계산됨)
    return geoJSONData.flatMap((feature, index) => {
      const queryResult = defectQueries[index]
      const damages = queryResult.data?.data.damages || []
      const risk = queryResult.data?.data.risk
      const imageUrl = queryResult.data?.data.imageUrl

      if (Array.isArray(damages) && risk && imageUrl) {
        return damages.map((damage: damageType) => ({
          defectId: getDisplayId(damage.id, feature.properties.publicId),
          imageUrl: imageUrl,
          type: feature.geometry.type,
          location: feature.properties.address.street,
          category: damage.category,
          detectedAt: damage.updatedAt,
          publicId: feature.properties.publicId,
          severity: getSeverity(risk),
          status: processStatusList[damage.status],
        }))
      }
      return []
    })
  }, [defectQueries, geoJSONData]) // 동일한 의존성 배열 유지

  // useState + useEffect 조합 대신 useMemo 결과를 직접 사용
  const [detailedGeoJSONData, setDetailedGeoJSONData] = useState<
    detailedDamageType[]
  >([])

  // 메모이제이션된 값이 변경될 때만 상태 업데이트
  useEffect(() => {
    if (detailedData.length > 0) {
      // 깊은 비교로 실제 내용이 변경되었는지 확인
      const isDataChanged =
        JSON.stringify(detailedData) !== JSON.stringify(detailedGeoJSONData)
      if (isDataChanged) {
        setDetailedGeoJSONData(detailedData)
      }
    }
  }, [detailedData, detailedGeoJSONData])

  // 필터링과 정렬을 useMemo로 최적화
  const filteredAndSortedData = useMemo(() => {
    // 필터링
    const filteredGeoJSONData = detailedGeoJSONData.filter((defect) => {
      const matchesType =
        defectType === 'all' || defect.type.toLowerCase() === defectType
      const matchesSeverity = severity === 'all' || defect.severity === severity
      return matchesType && matchesSeverity
    })

    // 정렬
    return [...filteredGeoJSONData].sort((a, b) => {
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
  }, [detailedGeoJSONData, defectType, severity, sortColumn, sortDirection])

  // sortedGeoJSONData.length가 바뀔 때마다 totalItems 업데이트
  useEffect(() => {
    setTotalItems(filteredAndSortedData.length)
  }, [filteredAndSortedData.length, setTotalItems])

  // 페이지네이션도 useMemo로 최적화
  const currentDefects = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage
    return filteredAndSortedData.slice(startIndex, startIndex + itemsPerPage)
  }, [filteredAndSortedData, currentPage, itemsPerPage])

  // 정렬 핸들러
  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      setSortDirection('asc')
    }
  }
  // 상태 변경 핸들러
  const handleStatusChange = async (
    damageId: number,
    currentStatus: string,
    newStatus: string,
    defectId: string,
  ) => {
    try {
      const response = await defectAPI.updateRoadDamageStatus(
        damageId,
        newStatus,
      )

      // 응답 성공 확인
      if (response.status === 200 || response.data) {
        console.log("success: ", response.data)

        // defectId를 기준으로 해당 항목의 상태 업데이트
        setDetailedGeoJSONData((prevData) =>
          prevData.map((defect) => {
            // defectId 정확히 일치하는 항목만 업데이트
            if (defect.defectId === defectId) {
              console.log(detailedGeoJSONData)
              
              return { ...defect, status: newStatus }
            }
            return defect
          }),
        )

        alert(`Changing defect ${currentStatus} to ${newStatus}`)
      } else {
        // 응답은 있지만 성공이 아닌 경우
        console.error('Status update failed:', response)
        alert('Failed to change status')
      }
    } catch (error) {
      // 요청 자체가 실패한 경우
      console.error('API call error:', error)
      alert('Network error occurred')
    }
  }

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
                onClick={() => handleSort('id')}
              >
                ID
                {sortColumn === 'id' &&
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
                onClick={() => handleSort('type')}
              >
                결함 유형
                {sortColumn === 'type' &&
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
            <TableHead className="col-span-6 flex items-center pb-2">
              발생 위치
            </TableHead>
            <TableHead className="col-span-4 flex items-center pb-2">
              발생 시각
            </TableHead>
            <TableHead className="col-span-4">
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort('detectedAt')}
              >
                작업 현황
                {sortColumn === 'detectedAt' &&
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
                onClick={() => handleSort('status')}
              >
                작업 더보기
                {sortColumn === 'status' &&
                  (sortDirection === 'asc' ? (
                    <ChevronUp className="h-4 w-4" />
                  ) : (
                    <ChevronDown className="h-4 w-4" />
                  ))}
              </Button>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {currentDefects.map((defect) => (
            <TableRow key={defect.defectId} className="grid grid-cols-24">
              <TableCell className="col-span-3 font-medium">
                {defect.defectId}
              </TableCell>
              <TableCell className="col-span-3">{defect.type}</TableCell>
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
                  {formatDate(defect.detectedAt)}
                </div>
              </TableCell>
              <TableCell className="col-span-4">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <StatusBadge className={getStatusColor(defect.status!)}>
                      {defect.status!}
                    </StatusBadge>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="start">
                    <DropdownMenuLabel>작업 상태 변경</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {processStatusList.map((process) => (
                      <DropdownMenuItem
                        key={process}
                        onClick={() =>
                          handleStatusChange(
                            1,
                            defect.status!,
                            process,
                            defect.defectId,
                          )
                        }
                      >
                        {process}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </TableCell>
              <TableCell className="col-span-2">
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8">
                      <MoreHorizontal className="h-4 w-4" />
                      <span className="sr-only">작업 더보기</span>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuLabel>작업</DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem>상세 보기</DropdownMenuItem>
                    <DropdownMenuItem>작업 할당</DropdownMenuItem>
                    <DropdownMenuItem>수리완료 처리</DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <DefectPaginations />
    </div>
  )
}
