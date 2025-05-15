export interface LocationInfo {
  id: number | null;
  district: Record<string, LocationInfo> | null;
  longitude: number | null;
  latitude: number | null;
}

export interface District {
  [cityName: string]: LocationInfo;
}

export interface AddressData {
  [province: string]: {
    district: District;
  } | LocationInfo;
}
