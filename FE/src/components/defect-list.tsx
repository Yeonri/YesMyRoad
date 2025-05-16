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
} from "@/components/ui/pagination"
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
import {
  ChevronDown,
  ChevronUp,
  Clock,
  MapPin,
  MoreHorizontal,
} from 'lucide-react'
import { useEffect } from 'react'
import { useState } from 'react'

export default function DefectList() {
  const [sortColumn, setSortColumn] = useState('id') // 기본 정렬을 ID로 변경
  const [sortDirection, setSortDirection] = useState('asc')
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 5 // 한 페이지에 5개의 결함만 표시

  // Get defects from Zustand store
  const { defectType, severity, geoJSONData } = useDefectStore()

  // geoJSONData를 사용하여 defects 배열 생성
  const mappedDefects = geoJSONData ? geoJSONData.map(feature => {
    // 각 feature에서 displayId를 가져와 defect 객체 생성
    return {
      id: feature.properties.displayId || 'Unknown', // displayId를 id로 사용
      type: 'Crack', // 하드코딩된 값
      severity: 'medium', // 하드코딩된 값
      location: feature.properties.address?.street || 'Unknown location', // 주소 사용
      detectedAt: new Date().toISOString(), // 현재 시간으로 하드코딩
      status: 'Pending', // 하드코딩된 값
      description: 'Auto-generated defect from GeoJSON data' // 하드코딩된 값
    }
  }) : [];

  // Filter defects based on selected filters
  const filteredDefects = mappedDefects.filter((defect) => {
    const matchesType =
      defectType === 'all' || defect.type.toLowerCase() === defectType
    const matchesSeverity = severity === 'all' || defect.severity === severity
    return matchesType && matchesSeverity
  })

  console.log('filteredDefects - 필터링 완료:', {
    total: mappedDefects.length,
    filtered: filteredDefects.length,
  })

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      setSortDirection('asc')
    }
  }

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

  // 페이지네이션을 위한 계산
  const totalPages = Math.ceil(sortedDefects.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const currentDefects = sortedDefects.slice(startIndex, endIndex)

  // 페이지 변경 핸들러
  const handlePageChange = (page: number) => {
    setCurrentPage(page)
  }

  // 페이지 번호 배열 생성 (1, 2, 3, ...)
  const getPageNumbers = () => {
    const pageNumbers = []
    const maxVisiblePages = 5 // 최대 표시할 페이지 번호 수
    
    if (totalPages <= maxVisiblePages) {
      // 총 페이지 수가 최대 표시 수보다 적은 경우, 모든 페이지 번호 표시
      for (let i = 1; i <= totalPages; i++) {
        pageNumbers.push(i)
      }
    } else {
      // 현재 페이지 주변의 페이지 번호만 표시
      let startPage = Math.max(1, currentPage - Math.floor(maxVisiblePages / 2))
      const endPage = Math.min(totalPages, startPage + maxVisiblePages - 1)
      
      // 마지막 페이지가 최대 표시 수보다 적은 경우 시작 페이지 조정
      if (endPage - startPage + 1 < maxVisiblePages) {
        startPage = Math.max(1, endPage - maxVisiblePages + 1)
      }
      
      for (let i = startPage; i <= endPage; i++) {
        pageNumbers.push(i)
      }
    }
    
    return pageNumbers
  }

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

  const handleStatusChange = async (
    defectId: number,
    currentStatus: string,
    newStatus: ProcessStatus,
  ) => {
    const response = await defectAPI.updateRoadDamageStatus(defectId, newStatus)
    console.log(response.data)
    alert(`Changing defect ${currentStatus} status to ${newStatus}`)
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
            <TableHead>발생 위치</TableHead>
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
          {currentDefects.map((defect) => (
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
          ))}
        </TableBody>
      </Table>
      
      {/* 페이지네이션 추가 */}
      <div className="mt-4">
        <Pagination>
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious 
                onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
                className={currentPage === 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
              />
            </PaginationItem>
            
            {/* 첫 페이지가 아닐 경우 첫 페이지로 가는 링크 표시 */}
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
            
            {/* 페이지 번호 표시 */}
            {getPageNumbers().map(page => (
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
            
            {/* 마지막 페이지가 아닐 경우 마지막 페이지로 가는 링크 표시 */}
            {getPageNumbers()[getPageNumbers().length - 1] < totalPages && (
              <>
                {getPageNumbers()[getPageNumbers().length - 1] < totalPages - 1 && (
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
                onClick={() => handlePageChange(Math.min(totalPages, currentPage + 1))}
                className={currentPage === totalPages ? "pointer-events-none opacity-50" : "cursor-pointer"}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      </div>
    </div>
  )
}