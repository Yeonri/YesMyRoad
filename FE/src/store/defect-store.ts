// src/store/defect-store.ts
import { DefectDetail, FeaturePoint } from '@/types/defects'
import { create } from 'zustand'

export type DefectStoreState = {
  geoJSONData: FeaturePoint[] | null // 손상 발생 위치 데이터 리스트
  defectDetailList: DefectDetail[] // 손상 상세 정보 목록

  updateGeoJSONData: (features: FeaturePoint[]) => void // geoJSONData를 담는 함수
  updateDefectDetailList: (defects: DefectDetail[]) => void
}

export const useDefectStore = create<DefectStoreState>((set) => ({
  geoJSONData: null,
  defectDetailList: [],

  updateGeoJSONData: (features: FeaturePoint[]) =>
    set({ geoJSONData: features }),
  updateDefectDetailList: (defect) => set({ defectDetailList: defect }),
}))
