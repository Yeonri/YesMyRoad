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
    displayId?: string // 해싱된 ID를 저장할 필드 추가
    address: { street: string }
    accuracyMeters: number
  }
}

// 저장소는 GeoJSONData 전체가 아닌 FeaturePoint 배열만 저장하도록 수정
export type FeatureCollection = FeaturePoint[]
