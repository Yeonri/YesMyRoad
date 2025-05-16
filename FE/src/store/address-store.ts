import { create } from 'zustand'

interface addressStore {
  id: number | null
  longitude: number | null
  latitude: number | null
  level1Addr: string | null
  level2Addr: string | null
  level3Addr: string | null
  setId: (id: number | null) => void
  setLongitude: (longitude: number | null) => void
  setLatitude: (latitude: number | null) => void
  setLevel1: (addr: string | null) => void
  setLevel2: (addr: string | null) => void
  setLevel3: (addr: string | null) => void
  logState: () => void
  getAddressInfo: () => {
    id: number | null
    longitude: number | null
    latitude: number | null
    level1Addr: string | null
    level2Addr: string | null
    level3Addr: string | null
  }
}

const useAddressStore = create<addressStore>((set, get) => ({
  id: null,
  longitude: null,
  latitude: null,
  level1Addr: null,
  level2Addr: null,
  level3Addr: null,
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
}))

export default useAddressStore
