'use client'

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
  const [sortColumn, setSortColumn] = useState('detectedAt')
  const [sortDirection, setSortDirection] = useState('desc')

  // Get defects from Zustand store
  const { defects, defectType, severity, detailedDefect, getGeoJSONData } =
    useDefectStore()

  // 상세 데이터 로깅을 위한 useEffect 추가
  useEffect(() => {
    const geoJSONData = getGeoJSONData()
    console.log('DefectList - detailedDefect 데이터:', geoJSONData)
  }, [])

  // // useMemo를 사용하여 상세 손상 정보 캐싱
  // const cachedDetailedDefect = useMemo(() => {
  //   console.log('캐싱 - DetailedDefect 계산됨')
  //   return detailedDefect
  // }, [detailedDefect])

  // // useMemo를 사용하여 GeoJSON 데이터 캐싱
  // const cachedGeoJSONData = useMemo(() => {
  //   console.log('캐싱 - GeoJSON 계산됨')
  //   return geoJSONData
  // }, [geoJSONData])

  // 캐싱된 GeoJSON 데이터를 사용하여 UI에 표시할 정보 가공
  // const pointsInfo = useMemo(() => {
  //   if (!cachedGeoJSONData) {
  //     console.log('pointsInfo - GeoJSON 데이터 없음')
  //     return null
  //   }

  //   const result = {
  //     totalPoints: cachedGeoJSONData.features.length,
  //     points: cachedGeoJSONData.features.map((feature) => ({
  //       id: feature.properties.publicId,
  //       coordinates: feature.geometry.coordinates,
  //       address: feature.properties.address.street,
  //       accuracy: feature.properties.accuracyMeters,
  //     })),
  //   }

  //   console.log('pointsInfo - 계산 완료:', result)
  //   return result
  // }, [cachedGeoJSONData])

  // 캐싱된 상세 손상 정보를 사용하여 UI에 표시할 정보 가공
  // const damageInfo = useMemo(() => {
  //   if (!cachedDetailedDefect) {
  //     console.log('damageInfo - 상세 손상 정보 없음')
  //     return null
  //   }

  //   const result = {
  //     imageUrl: cachedDetailedDefect.imageUrl,
  //     risk: cachedDetailedDefect.risk,
  //     damageCount: cachedDetailedDefect.damages.length,
  //     categories: [
  //       ...new Set(cachedDetailedDefect.damages.map((d) => d.category)),
  //     ],
  //   }

  //   console.log('damageInfo - 계산 완료:', result)
  //   return result
  // }, [cachedDetailedDefect])

  // Filter defects based on selected filters
  const filteredDefects = defects.filter((defect) => {
    const matchesType =
      defectType === 'all' || defect.type.toLowerCase() === defectType
    const matchesSeverity = severity === 'all' || defect.severity === severity
    return matchesType && matchesSeverity
  })

  console.log('filteredDefects - 필터링 완료:', {
    total: defects.length,
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
      {/* TODO: Replace with actual API call to fetch defects */}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[100px]">ID</TableHead>
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
          {sortedDefects.map((defect) => (
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
    </div>
  )
}
