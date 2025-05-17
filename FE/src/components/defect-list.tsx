'use client'

import { coodAPI } from '@/api/coordinate-api'
import { defectAPI } from '@/api/defect-api'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from '@/components/ui/pagination'
import { StatusBadge } from '@/components/ui/status-badge'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { ProcessStatus, useDefectStore } from '@/store/defect-store'
import { useQueries } from '@tanstack/react-query'
import {
  ChevronDown,
  ChevronUp,
  Clock,
  MapPin,
  MoreHorizontal,
} from 'lucide-react'
import { useEffect, useState } from 'react'

// DetailedDefect 타입 정의
type DetailedDefect = {
  imageUrl: string
  risk: number
  damages: { id: number; category: string; status: number; updatedAt: string }[]
}

export default function DefectList() {
  const [sortColumn, setSortColumn] = useState('id')
  const [sortDirection, setSortDirection] = useState('asc')
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 5

  // 상세 정보를 저장할 맵 상태
  const [detailsMap, setDetailsMap] = useState<Record<string, DetailedDefect>>(
    {},
  )

  // Zustand 스토어에서 데이터 가져오기
  const { defectType, severity, geoJSONData } = useDefectStore()

  // 기본 매핑 로직 - geoJSONData를 기반으로 기본 객체 생성
  const mappedDefects = geoJSONData
    ? geoJSONData.map((feature) => {
        const publicId = feature.properties.publicId
        const detail = detailsMap[publicId]

        return {
          id: feature.properties.displayId || 'Unknown',
          publicId: publicId, // API 호출용으로 보존
          type: 'Crack', // 하드코딩 값
          severity: 'medium', // 하드코딩 값
          location: feature.properties.address?.street || 'Unknown location',
          detectedAt: new Date().toISOString(),
          status: 'Pending', // 하드코딩 값
          description: 'Auto-generated defect from GeoJSON data',
          // 상세 정보가 있으면 추가
          risk: detail?.risk,
          imageUrl: detail?.imageUrl,
          damages: detail?.damages,
        }
      })
    : []

  // 필터링
  const filteredDefects = mappedDefects.filter((defect) => {
    const matchesType =
      defectType === 'all' || defect.type.toLowerCase() === defectType
    const matchesSeverity = severity === 'all' || defect.severity === severity
    return matchesType && matchesSeverity
  })

  // 정렬
  const sortedDefects = [...filteredDefects].sort((a, b) => {
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

  // 페이지네이션
  const totalPages = Math.ceil(sortedDefects.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const currentDefects = sortedDefects.slice(startIndex, endIndex)

  // 현재 페이지에 표시되는 항목들에 대한 상세 정보만 쿼리
  const detailsQueries = useQueries({
    queries: currentDefects.map((defect) => ({
      queryKey: ['defectDetail', defect.publicId],
      queryFn: async () => {
        const response = await coodAPI.getDetailedDefects(defect.publicId)
        console.log('good!: ', response.data.damages)

        return { publicId: defect.publicId, data: response.data }
      },
      enabled: !!defect.publicId, // publicId가 있는 경우에만 쿼리 실행
      staleTime: 5 * 60 * 1000, // 5분 동안 캐싱
    })),
  })

  // 상세 정보가 로드되면 detailsMap 업데이트
  useEffect(() => {
    const newDetailsMap = { ...detailsMap }
    let hasUpdates = false

    detailsQueries.forEach((query) => {
      console.log('Q: ', query.data)

      if (query.isSuccess && query.data) {
        const { publicId, data } = query.data
        if (!detailsMap[publicId]) {
          newDetailsMap[publicId] = data
          hasUpdates = true
        }
      }
    })

    if (hasUpdates) {
      setDetailsMap(newDetailsMap)
    }
  }, [detailsQueries, detailsMap])

  // 정렬 핸들러
  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      setSortDirection('asc')
    }
  }

  // 페이지 변경 핸들러
  const handlePageChange = (page: number) => {
    setCurrentPage(page)
  }

  // 페이지 번호 배열 생성
  const getPageNumbers = () => {
    const pageNumbers = []
    const maxVisiblePages = 5

    if (totalPages <= maxVisiblePages) {
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i)
      }
    } else {
      let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2))
      const endPage = Math.min(totalPages, startPage + maxVisiblePages - 1)

      if (endPage - startPage + 1 < maxVisiblePages) {
        startPage = Math.max(1, endPage - maxVisiblePages + 1)
      }

      for (let i = startPage; i <= endPage; i++) {
        pageNumbers.push(i)
      }
    }

    return pageNumbers
  }

  // 유틸리티 함수들
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500 text-white'
      case 'high':
        return 'bg-amber-500 text-white'
      case 'medium':
        return 'bg-blue-500 text-white'
      case 'low':
        return 'bg-green-500 text-white'
      default:
        return 'bg-gray-500 text-white'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Pending':
        return 'bg-gray-100 text-gray-800 hover:bg-gray-300 hover:text-gray-900 transition-colors duration-150'
      case 'Assigned':
        return 'bg-blue-100 text-blue-800 hover:bg-blue-300 hover:text-blue-900 transition-colors duration-150'
      case 'In Progress':
        return 'bg-amber-100 text-amber-800 hover:bg-amber-300 hover:text-amber-900 transition-colors duration-150'
      case 'Completed':
        return 'bg-green-100 text-green-800 hover:bg-green-300 hover:text-green-900 transition-colors duration-150'
      default:
        return 'bg-gray-100 text-gray-800 hover:bg-gray-300 hover:text-gray-900 transition-colors duration-150'
    }
  }

  // 상태 변경 핸들러
  const handleStatusChange = async (
    defectId: number,
    currentStatus: string,
    newStatus: ProcessStatus,
  ) => {
    const response = await defectAPI.updateRoadDamageStatus(defectId, newStatus)
    console.log(response.data)
    alert(`Changing defect ${currentStatus} status to ${newStatus}`)
  }

  // 로딩 상태 확인 (전체 페이지에 대한 로딩 상태가 아니라 현재 페이지에 대한 상세 정보만 확인)
  const isLoading = detailsQueries.some(
    (query) => query.isLoading && !query.isError,
  )

  return (
    <div className="w-full overflow-auto">
      {isLoading && (
        <div className="text-muted-foreground py-2 text-center text-sm">
          상세 정보를 불러오는 중...
        </div>
      )}

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
              <TableRow key={defect.id}>
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
                      <DropdownMenuItem
                        onClick={() =>
                          handleStatusChange(1, defect.status!, 'Pending')
                        }
                      >
                        Pending
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() =>
                          handleStatusChange(1, defect.status!, 'Assigned')
                        }
                      >
                        Assigned
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() =>
                          handleStatusChange(1, defect.status!, 'In Progress')
                        }
                      >
                        In Progress
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() =>
                          handleStatusChange(1, defect.status!, 'Completed')
                        }
                      >
                        Completed
                      </DropdownMenuItem>
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

      {/* 페이지네이션 */}
      <div className="mt-4">
        <Pagination>
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious
                onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
                className={
                  currentPage === 1
                    ? 'pointer-events-none opacity-50'
                    : 'cursor-pointer'
                }
              />
            </PaginationItem>

            {/* 페이지 번호 로직은 기존과 동일 */}
            {getPageNumbers()[0] > 1 && (
              <>
                <PaginationItem>
                  <PaginationLink
                    onClick={() => handlePageChange(1)}
                    className="cursor-pointer"
                  >
                    1
                  </PaginationLink>
                </PaginationItem>
                {getPageNumbers()[0] > 2 && (
                  <PaginationItem>
                    <PaginationEllipsis />
                  </PaginationItem>
                )}
              </>
            )}

            {getPageNumbers().map((page) => (
              <PaginationItem key={page}>
                <PaginationLink
                  isActive={currentPage === page}
                  onClick={() => handlePageChange(page)}
                  className="cursor-pointer"
                >
                  {page}
                </PaginationLink>
              </PaginationItem>
            ))}

            {getPageNumbers()[getPageNumbers().length - 1] < totalPages && (
              <>
                {getPageNumbers()[getPageNumbers().length - 1] <
                  totalPages - 1 && (
                  <PaginationItem>
                    <PaginationEllipsis />
                  </PaginationItem>
                )}
                <PaginationItem>
                  <PaginationLink
                    onClick={() => handlePageChange(totalPages)}
                    className="cursor-pointer"
                  >
                    {totalPages}
                  </PaginationLink>
                </PaginationItem>
              </>
            )}

            <PaginationItem>
              <PaginationNext
                onClick={() =>
                  handlePageChange(Math.min(totalPages, currentPage + 1))
                }
                className={
                  currentPage === totalPages
                    ? 'pointer-events-none opacity-50'
                    : 'cursor-pointer'
                }
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      </div>
    </div>
  )
}
