import { create } from 'zustand'

interface addressStore {
  id: number | null
  longitude: number | null
  latitude: number | null
  level1Addr: string | null
  level2Addr: string | null
  level3Addr: string | null
  mapBounds: {
    northEast: { lat: number | null; lng: number | null }
    southWest: { lat: number | null; lng: number | null }
  }
  setId: (id: number | null) => void
  setLongitude: (longitude: number | null) => void
  setLatitude: (latitude: number | null) => void
  setLevel1: (addr: string | null) => void
  setLevel2: (addr: string | null) => void
  setLevel3: (addr: string | null) => void
  setMapBounds: (bounds: {
    northEast: { lat: number | null; lng: number | null }
    southWest: { lat: number | null; lng: number | null }
  }) => void
  logMapBounds: () => void
  logState: () => void
  getAddressInfo: () => {
    id: number | null
    longitude: number | null
    latitude: number | null
    level1Addr: string | null
    level2Addr: string | null
    level3Addr: string | null
  }
  getMapBounds: () => {
    northEast: { lat: number | null; lng: number | null }
    southWest: { lat: number | null; lng: number | null }
  }
}

const useAddressStore = create<addressStore>((set, get) => ({
  id: 3020000000,
  longitude: 127.356136,
  latitude: 36.364056,
  level1Addr: '대전광역시',
  level2Addr: '유성구',
  level3Addr: null,
  mapBounds: {
    northEast: { lat: null, lng: null },
    southWest: { lat: null, lng: null },
  },
  setId: (id: number | null) => {
    set({ id })
  },
  setLongitude: (longitude: number | null) => {
    set({ longitude })
  },
  setLatitude: (latitude: number | null) => {
    set({ latitude })
  },
  setLevel1: (addr: string | null) => {
    set({ level1Addr: addr })
  },
  setLevel2: (addr: string | null) => {
    set({ level2Addr: addr })
  },
  setLevel3: (addr: string | null) => {
    set({ level3Addr: addr })
  },
  setMapBounds: (bounds: {
    northEast: { lat: number | null; lng: number | null }
    southWest: { lat: number | null; lng: number | null }
  }) => {
    set({ mapBounds: bounds })
  },
  logMapBounds: () => {
    const state = get()
    console.log('지도 경계 좌표:', state.mapBounds)
  },
  logState: () => {
    const state = get()
    console.log('주소 정보 업데이트:', {
      id: state.id,
      longitude: state.longitude,
      latitude: state.latitude,
      level1Addr: state.level1Addr,
      level2Addr: state.level2Addr,
      level3Addr: state.level3Addr,
    })
  },
  getAddressInfo: () => {
    const state = get()
    return {
      id: state.id,
      longitude: state.longitude,
      latitude: state.latitude,
      level1Addr: state.level1Addr,
      level2Addr: state.level2Addr,
      level3Addr: state.level3Addr,
    }
  },
  getMapBounds: () => {
    const state = get()
    return state.mapBounds
  },
}))

export default useAddressStore
