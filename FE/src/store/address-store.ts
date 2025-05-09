import { create } from 'zustand'

interface addressStore {
  level1Addr: string
  level2Addr: string
  level3Addr: string
  setLevel1: (addr: string) => void
  setLevel2: (addr: string) => void
  setLevel3: (addr: string) => void
}

const useAddressStore = create<addressStore>((set) => ({
  level1Addr: '',
  level2Addr: '',
  level3Addr: '',
  setLevel1: (addr: string) => set({ level1Addr: addr }),
  setLevel2: (addr: string) => set({ level2Addr: addr }),
  setLevel3: (addr: string) => set({ level3Addr: addr }),
}))

export default useAddressStore
