// src/hooks/useDefectList.ts

import { useState, useMemo } from 'react'
import { FeaturePoint } from '@/types/defects'

// 1) 필터 유니언 타입
export type DefectTypeFilter = 'all' | 'Potholes' | 'Cracks' | 'Paint Peeling'
export type SeverityFilter = 'all' | 'Critical' | 'High' | 'Medium' | 'Low'

// 테이블 한 줄 아이템
export interface DefectListItem {
    id: string
    publicId: string
    type: string
    severity: string
    location: string
    detectedAt: string
    status: string
}

// 훅 반환 타입
export interface UseDefectListResult {
    currentDefects: DefectListItem[]
    sortColumn: keyof DefectListItem
    sortDirection: 'asc' | 'desc'
    handleSort: (column: keyof DefectListItem) => void
    currentPage: number
    totalPages: number
    setCurrentPage: React.Dispatch<React.SetStateAction<number>>
    getPageNumbers: () => number[]
}

/**
 * geoJSONData → 매핑 → 필터 → 정렬 → 페이징
 */
export function useDefectList(
    geoJSONData: FeaturePoint[] | null,
    defectType: DefectTypeFilter,      // 유니언 타입 적용
    severityFilter: SeverityFilter,    // 유니언 타입 적용
    itemsPerPage: number,
): UseDefectListResult {
    const [sortColumn, setSortColumn] = useState<keyof DefectListItem>('id')
    const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
    const [currentPage, setCurrentPage] = useState<number>(1)

    // 1) Mapping
    const mapped = useMemo<DefectListItem[]>(() => {
        if (!geoJSONData) return []
        return geoJSONData.map(feat => ({
            id: 'Unknown',
            publicId: feat.properties.publicId,
            type: 'Crack',
            severity: 'Medium',
            location: feat.properties.address.street,
            detectedAt: new Date().toISOString(),
            status: 'Pending',
        }))
    }, [geoJSONData])

    // 2) Filtering
    const filtered = useMemo(() => {
        return mapped.filter(d =>
            (defectType === 'all' || d.type === defectType) &&
            (severityFilter === 'all' || d.severity === severityFilter)
        )
    }, [mapped, defectType, severityFilter])

    // 3) Sorting
    const sorted = useMemo(() => {
        return [...filtered].sort((a, b) => {
            const aVal = a[sortColumn]
            const bVal = b[sortColumn]
            if (aVal == null || bVal == null) return 0
            return sortDirection === 'asc'
                ? aVal > bVal ? 1 : -1
                : aVal < bVal ? 1 : -1
        })
    }, [filtered, sortColumn, sortDirection])

    // 4) Pagination
    const totalPages = Math.ceil(sorted.length / itemsPerPage)
    const currentDefects = useMemo(
        () => sorted.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage),
        [sorted, currentPage, itemsPerPage]
    )

    const getPageNumbers = () => Array.from({ length: totalPages }, (_, i) => i + 1)

    const handleSort = (column: keyof DefectListItem) => {
        if (column === sortColumn) {
            setSortDirection(dir => (dir === 'asc' ? 'desc' : 'asc'))
        } else {
            setSortColumn(column)
            setSortDirection('asc')
        }
        setCurrentPage(1)
    }

    return {
        currentDefects,
        sortColumn,
        sortDirection,
        handleSort,
        currentPage,
        totalPages,
        setCurrentPage,
        getPageNumbers,
    }
}
