// src/api/maintenance-api.ts
import apiCaller from '@/api/api-utils'

export const maintenanceAPI = {
  // 전체 보수공사(수리상태) 미완료/진행중/완료 개수
  getMaintenanceOverview: () =>
    apiCaller(
      'get',
      '/api/maintenance/overview',
      '전체 보수공사 현황 조회 실패',
    ),

  // 일간, 주간, 월간 보수공사 완료 개수
  getMaintenanceCompletionStats: () =>
    apiCaller(
      'get',
      '/api/maintenance/completion-stats',
      '일간/주간/월간 보수공사 완료 건수 조회 실패',
    ),

  // 월별 보수공사 현황
  getMonthlyMaintenanceStatus: () =>
    apiCaller(
      'get',
      '/api/maintenance/monthly-status',
      '월별 보수공사 현황 조회 실패',
    ),

  // 지역별 보수공사 현황
  getLocallyMaintenanceStatus: () =>
    apiCaller(
      'get',
      '/api/maintenance/districts',
      '지역별 보수공사 현황 조회 실패',
    ),
}
