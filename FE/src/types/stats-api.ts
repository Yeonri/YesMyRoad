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

export interface RepairDailyReportType {
  day: string
  count: number
  changeRate: number
  totalCount: number
}

export interface RepairWeeklyReportType {
  weekStart: string
  count: number
  changeRate: number
  totalCount: number
}

export interface RepairMonthlyReportType {
  month: string
  count: number
  changeRate: number
  totalCount: number
}
