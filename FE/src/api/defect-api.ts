// src/api/defect-api.ts
import apiCaller from '@/api/api-utils'

export const defectAPI = {
  // 도로 파손 건의 상태 변경
  updateRoadDamageStatus: (damageId: number, status: string) =>
    apiCaller(
      'patch',
      `/api/damages/${damageId}`,
      '도로 파손 건의 상태 변경 실패',
      { status },
    ),
}
