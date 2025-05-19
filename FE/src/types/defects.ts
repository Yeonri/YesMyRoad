// 손상 상세 정보를 위한 타입 정의
export type DamageItem = {
  id: number
  category: string
  status: number
  updatedAt: string
}

export type DefectDetail = {
  publicId: string
  address: string
  imageUrl: string
  risk: number
  damages: DamageItem[]
}

// GeoJSON 데이터를 위한 타입 정의 - 수정함
export type FeaturePoint = {
  type: string
  geometry: { type: string; coordinates: number[] }
  properties: {
    publicId: string
    address: { street: string }
    accuracyMeters: number
  }
}

export type FeatureCollection = FeaturePoint[]

export type DetailedFeaturePoint = {
  type: string
  geometry: { type: string; coordinates: number[] }
  properties: {
    publicId: string
    address: { street: string }
    accuracyMeters: number
  }
  imageUrl: string
  risk: number
  damages: DamageItem[]
}


export type ProcessStatus = 'REPORTED' | 'RECEIVED' | 'IN_PROGRESS' | 'COMPLETED'