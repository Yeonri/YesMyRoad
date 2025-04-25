"use client"

import { useState } from "react"
import { ChevronDown, ChevronUp, MoreHorizontal, Clock, MapPin } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useDefectStore } from "@/store/defect-store"

export default function DefectList() {
  const [sortColumn, setSortColumn] = useState("detectedAt")
  const [sortDirection, setSortDirection] = useState("desc")

  // Get defects from Zustand store
  const { defects, defectType, severity } = useDefectStore()

  // Filter defects based on selected filters
  const filteredDefects = defects.filter((defect) => {
    const matchesType = defectType === "all" || defect.type.toLowerCase() === defectType
    const matchesSeverity = severity === "all" || defect.severity === severity
    return matchesType && matchesSeverity
  })

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(column)
      setSortDirection("asc")
    }
  }

  const sortedDefects = [...filteredDefects].sort((a, b) => {
    if (sortDirection === "asc") {
      return a[sortColumn as keyof typeof a] > b[sortColumn as keyof typeof b] ? 1 : -1
    } else {
      return a[sortColumn as keyof typeof a] < b[sortColumn as keyof typeof b] ? 1 : -1
    }
  })

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "bg-red-500 text-white"
      case "high":
        return "bg-amber-500 text-white"
      case "medium":
        return "bg-blue-500 text-white"
      case "low":
        return "bg-green-500 text-white"
      default:
        return "bg-gray-500 text-white"
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "Pending":
        return "bg-gray-100 text-gray-800"
      case "Assigned":
        return "bg-blue-100 text-blue-800"
      case "In Progress":
        return "bg-amber-100 text-amber-800"
      case "Completed":
        return "bg-green-100 text-green-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
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
                onClick={() => handleSort("type")}
              >
                결함 유형
                {sortColumn === "type" &&
                  (sortDirection === "asc" ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />)}
              </Button>
            </TableHead>
            <TableHead>
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort("severity")}
              >
                심각도
                {sortColumn === "severity" &&
                  (sortDirection === "asc" ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />)}
              </Button>
            </TableHead>
            <TableHead>발생 위치</TableHead>
            <TableHead>
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort("detectedAt")}
              >
                작업 현황
                {sortColumn === "detectedAt" &&
                  (sortDirection === "asc" ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />)}
              </Button>
            </TableHead>
            <TableHead>
              <Button
                variant="ghost"
                size="sm"
                className="flex items-center gap-1 p-0 font-medium"
                onClick={() => handleSort("status")}
              >
                작업 더보기
                {sortColumn === "status" &&
                  (sortDirection === "asc" ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />)}
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
                <Badge className={getSeverityColor(defect.severity)}>
                  {defect.severity.charAt(0).toUpperCase() + defect.severity.slice(1)}
                </Badge>
              </TableCell>
              <TableCell className="flex items-center gap-1">
                <MapPin className="h-3 w-3 text-muted-foreground" />
                {defect.location}
              </TableCell>
              <TableCell className="flex items-center gap-1">
                <Clock className="h-3 w-3 text-muted-foreground" />
                {formatDate(defect.detectedAt)}
              </TableCell>
              <TableCell>
                <Badge className={getStatusColor(defect.status)}>{defect.status}</Badge>
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
