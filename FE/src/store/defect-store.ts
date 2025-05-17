// src/store/defect-store.ts
import { DefectDetail, FeaturePoint } from '@/types/defects'
import crypto from 'crypto'
import { create } from 'zustand'

// UUID로부터 표시 ID 생성하는 함수
const getDisplayId = (
  uuid: string,
  damageId: number,
  prefix: string = 'DEF',
): string => {
  const validChars = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
  const hash = crypto.createHash('sha256').update(uuid).digest('hex')

  const shortCode = Array(5)
    .fill(0)
    .map((_, i) => {
      const index =
        parseInt(hash.slice(i * 2, i * 2 + 2), 16) % validChars.length
      return validChars[index]
    })
    .join('')

  return `${prefix}-${damageId}-${shortCode}`
}

export type DefectStoreState = {
  geoJSONData: FeaturePoint[] | null // 손상 발생 위치 데이터 리스트
  defectDetailList: DefectDetail[] // 손상 상세 정보 목록
  // 상태 업데이트 함수 (API 호출 없이 상태만 업데이트)
  updateDefectDetailList: (defects: DefectDetail[]) => void
  // 수정된 업데이트 함수 - 전체 GeoJSON이 아닌 features 배열만 받음
  updateGeoJSONData: (data: any) => void
}

export const useDefectStore = create<DefectStoreState>((set) => ({
  // Initial filter states
  geoJSONData: null,
  defectDetailList: [],

  // 상태 업데이트 함수들 (API 호출 없이 상태만 업데이트)
  updateDefectDetailList: (defect) => set({ defectDetailList: defect }),

  // 수정된 업데이트 함수 - features 배열을 받아 각 항목의 publicId를 기반으로 displayId 생성
  updateGeoJSONData: (data) => {
    // GeoJSON 전체 객체가 들어올 경우 features 배열만 추출
    const features = data.features ? data.features : data

    if (!features) {
      set({ geoJSONData: null })
      return
    }

    // 각 feature에 displayId 추가
    const enhancedFeatures = features.map((feature: FeaturePoint) => {
      const publicId = feature.properties.publicId
      return {
        ...feature,
        properties: {
          ...feature.properties,
          // 기본 ID는 damage ID 없이 생성 (이후 상세 데이터 로드 시 업데이트됨)
          displayId: getDisplayId(publicId, 0),
        },
      }
    })

    set({ geoJSONData: enhancedFeatures })
  },
}))
