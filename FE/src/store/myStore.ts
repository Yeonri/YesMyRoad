import { create } from 'zustand';

interface MyStore {
  count: number;
  increment: () => void;
}

const useMyStore = create<MyStore>((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
}));

export default useMyStore;

