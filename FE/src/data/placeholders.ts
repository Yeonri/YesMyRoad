// TODO: Replace these placeholder values with actual data from API endpoints

// Mock data for defect locations
export const defectLocations = [
  { id: 1, lat: 36.355056, lng: 127.301378, type: 'pothole', severity: 'critical', title: '포트홀' },
  { id: 2, lat: 36.357255, lng: 127.304379, type: 'crack', severity: 'high', title: '크랙' },
  { id: 3, lat: 36.348676, lng: 127.297512, type: 'paint', severity: 'medium', title: '크랙' },
  { id: 4, lat: 36.351986, lng: 127.298512, type: 'pothole', severity: 'high', title: '포트홀' },
  { id: 5, lat: 36.363024, lng: 127.306758, type: 'crack', severity: 'low', title: '크랙' },
  { id: 6, lat: 36.360445, lng: 127.300041, type: 'pothole', severity: 'critical', title: '포트홀' },
]

// Mock data for heatmap (more points)
export const heatmapLocations = [
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.355056, lng: 127.301378 },
  { lat: 36.357255, lng: 127.304379 },
  { lat: 36.357255, lng: 127.304379 },
  { lat: 36.357255, lng: 127.304379 },
  { lat: 36.357255, lng: 127.304379 },
  { lat: 36.357255, lng: 127.304379 },
  { lat: 36.357255, lng: 127.304379 },
  { lat: 36.348676, lng: 127.297512 },
  { lat: 36.348676, lng: 127.297512 },
  { lat: 36.348676, lng: 127.297512 },
  { lat: 36.351986, lng: 127.298512 },
  { lat: 36.351986, lng: 127.298512 },
  { lat: 36.351986, lng: 127.298512 },
  { lat: 36.351986, lng: 127.298512 },
  { lat: 36.351986, lng: 127.298512 },
  { lat: 36.363024, lng: 127.306758 },
  { lat: 36.363024, lng: 127.306758 },
  { lat: 36.363024, lng: 127.306758 },
  { lat: 36.363024, lng: 127.306758 },
  { lat: 36.363024, lng: 127.306758 },
]

// Mock data for defect list
export const defects = [
  {
    id: 'DEF-1001',
    type: 'Pothole',
    severity: 'critical',
    location: '동서대로',
    detectedAt: '2025-04-24T08:30:00',
    status: 'Pending',
  },
  {
    id: 'DEF-1002',
    type: 'Crack',
    severity: 'high',
    location: '덕명로',
    detectedAt: '2025-04-24T09:15:00',
    status: 'Assigned',
  },
  {
    id: 'DEF-1003',
    type: 'Paint Peeling',
    severity: 'medium',
    location: '학하중앙로',
    detectedAt: '2025-04-24T07:45:00',
    status: 'In Progress',
  },
  {
    id: 'DEF-1004',
    type: 'Pothole',
    severity: 'high',
    location: '동서대로',
    detectedAt: '2025-04-24T10:20:00',
    status: 'Pending',
  },
  {
    id: 'DEF-1005',
    type: 'Crack',
    severity: 'low',
    location: '한밭대로',
    detectedAt: '2025-04-24T11:05:00',
    status: 'Assigned',
  },
  {
    id: 'DEF-1006',
    type: 'Pothole',
    severity: 'low',
    location: '현충원로로',
    detectedAt: '2025-04-24T08:50:00',
    status: 'Completed',
  },
]

// Mock data for recent alerts
export const recentAlerts = [
  {
    id: 'DEF-1001',
    type: 'Pothole',
    severity: 'critical',
    location: '동서대로',
    detectedAt: '2025-04-24T08:30:00',
    description: '대형 포트홀으로 인한 교통 체증',
  },
  {
    id: 'DEF-1007',
    type: 'Pothole',
    severity: 'critical',
    location: '학하로',
    detectedAt: '2025-04-24T09:30:00',
    description: '포트홀 내부로 하수관 노출',
  },
  {
    id: 'DEF-1002',
    type: 'Crack',
    severity: 'high',
    location: '학하중앙로',
    detectedAt: '2025-04-24T09:15:00',
    description: '두 차로에 걸쳐 크랙 발생',
  },
  {
    id: 'DEF-1004',
    type: 'Pothole',
    severity: 'high',
    location: '현충원로',
    detectedAt: '2025-04-24T10:20:00',
    description: '교차로에 포트홀 발생',
  },
]

// Mock data for defect statistics
export const defectTypeData = [
  { value: 42, name: 'Potholes' },
  { value: 35, name: 'Cracks' },
  { value: 23, name: 'Paint Peeling' },
]

export const severityData = [
  { value: 12, name: 'Critical' },
  { value: 28, name: 'High' },
  { value: 32, name: 'Medium' },
  { value: 28, name: 'Low' },
]

// Mock data for defect trends
export const trendData = {
  dates: ['Apr 18', 'Apr 19', 'Apr 20', 'Apr 21', 'Apr 22', 'Apr 23', 'Apr 24'],
  potholesData: [12, 19, 15, 22, 25, 28, 30],
  cracksData: [10, 15, 12, 18, 20, 22, 25],
  paintData: [8, 10, 12, 15, 13, 16, 18],
}

// Mock data for dashboard metrics
export const dashboardMetrics = {
  totalDefects: 170,
  totalDefectsChange: '+12%',
  criticalIssues: 7,
  criticalIssuesChange: '-3%',
  avgResponseTime: '2.4h',
  avgResponseTimeChange: '-18%',
  affectedAreas: 42,
  affectedAreasChange: '+5%',
}

// Mock data for severity counts
export const severityCounts = { critical: 7, high: 23, medium: 48, low: 92 }
