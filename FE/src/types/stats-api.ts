// 손상 통계 관련 타입
export interface DamageDailyReportType {
  day: string
  count: number
  changeRate: number
  totalCount: number
}

export interface DamageWeeklyReportType {
  weekStart: string
  count: number
  changeRate: number
  totalCount: number
}

export interface DamageMonthlyReportType {
  month: string
  count: number
  changeRate: number
  totalCount: number
}

// 보수공사 통계 관련 타입
export interface RepairReportSummaryType {
  daily: number
  weekly: number
  monthly: number
}

export type MonthlyMaintenanceOverviewType = {
  reported: number
  received: number
  inProgress: number
  completed: number
}

export type MonthlyMaintenanceStatusType = {
  month: string
  reported: number
  received: number
  inProgress: number
  completed: number
}

export interface MonthlyMaintenanceStatusResponseType {
  data: MonthlyMaintenanceStatusType[]
}
