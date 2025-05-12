export interface DamageDailyReportType {
  day: string
  count: number
  changeRate: number
}

export interface DamageWeeklyReportType {
  weekStart: string
  count: number
  changeRate: number
}

export interface DamageMonthlyReportType {
  month: string
  count: number
  changeRate: number
}
