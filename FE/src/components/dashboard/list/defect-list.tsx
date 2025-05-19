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
import { formatDate, getSeverityColor, getStatusColor } from '@/lib/formatter'
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
  id: number
  imageUrl: string
  risk: number
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
    'Pending',
    'Assigned',
    'In Progress',
    'Completed',
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
      const defectId = ""

      if (Array.isArray(damages)) {
        return damages.map((damage: damageType) => ({
          id: defectId,
          imageUrl: queryResult.data?.data.imageUrl,
          risk: queryResult.data?.data.risk,
          type: feature.geometry.type,
          location: feature.properties.address.street,
          category: damage.category,
          detectedAt: damage.updatedAt,
          publicId: feature.properties.publicId,
          severity: 'medium',
          status: 'Pending',
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
      setDetailedGeoJSONData(detailedData)
    }
  }, [detailedData])

  // 검증용 (그대로 유지)
  useEffect(() => {
    console.log('업데이트된 데이터:', detailedGeoJSONData)
  }, [detailedGeoJSONData])

  // 필터링
  const filteredGeoJSONData = detailedGeoJSONData.filter((defect) => {
    const matchesType =
      defectType === 'all' || defect.type.toLowerCase() === defectType
    const matchesSeverity = severity === 'all' || defect.severity === severity
    return matchesType && matchesSeverity
  })

  // 정렬
  const sortedGeoJSONData = [...filteredGeoJSONData].sort((a, b) => {
    if (sortDirection === 'asc') {
      return a[sortColumn as keyof typeof a]! > b[sortColumn as keyof typeof b]!
        ? 1
        : -1
    } else {
      return a[sortColumn as keyof typeof a]! < b[sortColumn as keyof typeof b]!
        ? 1
        : -1
    }
  })

  // sortedGeoJSONData.length가 바뀔 때마다 totalItems 업데이트
  useEffect(() => {
    setTotalItems(sortedGeoJSONData.length)
  }, [sortedGeoJSONData.length, setTotalItems])

  // 페이지네이션
  const startIndex = (currentPage - 1) * itemsPerPage
  const currentDefects = sortedGeoJSONData.slice(
    startIndex,
    startIndex + itemsPerPage,
  )

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
  ) => {
    const response = await defectAPI.updateRoadDamageStatus(damageId, newStatus)
    console.log(response.data)
    alert(`Changing defect ${currentStatus} to ${newStatus}`)
  }

  return (
    <div className="w-full overflow-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[100px]">
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
            <TableHead>
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
            <TableHead>
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
            <TableHead>발생 위치 및 시각</TableHead>
            <TableHead>
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
            <TableHead>
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
            <TableHead className="w-[50px]"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {currentDefects.map((defect) => {
            // 각 항목별 상세 정보 쿼리 상태 확인
            // const detailQuery = detailsQueries.find(
            //   (q) => q.data?.publicId === defect.publicId,
            // )

            return (
              <TableRow key={defect.publicId}>
                {/* <TableRow key={defect.id}> */}
                <TableCell className="font-medium">{defect.id}</TableCell>
                <TableCell>{defect.type}</TableCell>
                <TableCell>
                  <StatusBadge className={getSeverityColor(defect.severity)}>
                    {defect.severity.charAt(0).toUpperCase() +
                      defect.severity.slice(1)}
                  </StatusBadge>
                </TableCell>
                <TableCell className="flex items-center gap-1">
                  <MapPin className="text-muted-foreground h-3 w-3" />
                  {defect.location}
                </TableCell>
                <TableCell className="flex items-center gap-1">
                  <Clock className="text-muted-foreground h-3 w-3" />
                  {formatDate(defect.detectedAt)}
                </TableCell>
                <TableCell>
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
                            handleStatusChange(1, defect.status!, process)
                          }
                        >
                          {process}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
                <TableCell>
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
            )
          })}
        </TableBody>
      </Table>
      <DefectPaginations />
    </div>
  )
}
