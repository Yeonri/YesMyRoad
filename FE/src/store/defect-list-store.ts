// src/store/defect-list-store.ts
import { create } from 'zustand'

export interface DefectPaginationState {
    currentPage: number // 현재 페이지 번호
    itemsPerPage: number // 한 페이지당 아이템 수
    totalItems: number // 전체 아이템 개수 (부모에서 업데이트)
    maxVisiblePages: number // 최대 표시할 페이지 버튼 개수

    /*— 상태 변경 함수들 —*/
    setCurrentPage: (page: number) => void
    setItemsPerPage: (size: number) => void
    setTotalItems: (total: number) => void
}

export const useDefectListStore = create<DefectPaginationState>((set) => ({
    currentPage: 1,
    itemsPerPage: 5,
    totalItems: 0,
    maxVisiblePages: 5,

    setCurrentPage: (page: number) => set({ currentPage: page }),
    setItemsPerPage: (size: number) => set({ itemsPerPage: size }),
    setTotalItems: (total: number) => set({ totalItems: total }),
}))
