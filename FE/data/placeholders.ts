export const defectLocations = [
  { id: 1, lat: 40.7128, lng: -74.006, type: "pothole", severity: "critical", title: "Critical Pothole" },
  { id: 2, lat: 40.7148, lng: -74.013, type: "crack", severity: "high", title: "Major Road Crack" },
  { id: 3, lat: 40.7118, lng: -74.008, type: "paint", severity: "medium", title: "Paint Peeling" },
  { id: 4, lat: 40.7138, lng: -74.003, type: "pothole", severity: "high", title: "Large Pothole" },
  { id: 5, lat: 40.7158, lng: -74.009, type: "crack", severity: "low", title: "Minor Crack" },
  { id: 6, lat: 40.7168, lng: -74.016, type: "pothole", severity: "critical", title: "Deep Pothole" },
  { id: 7, lat: 40.7108, lng: -74.002, type: "paint", severity: "low", title: "Lane Marking Faded" },
]

// Mock data for heatmap (more points)
export const heatmapLocations = [
  // NYC area
  { lat: 40.7128, lng: -74.006 },
  { lat: 40.7148, lng: -74.013 },
  { lat: 40.7118, lng: -74.008 },
  { lat: 40.7138, lng: -74.003 },
  { lat: 40.7158, lng: -74.009 },
  { lat: 40.7168, lng: -74.016 },
  { lat: 40.7108, lng: -74.002 },
  { lat: 40.7135, lng: -74.007 },
  { lat: 40.7145, lng: -74.011 },
  { lat: 40.7125, lng: -74.005 },
  { lat: 40.7115, lng: -74.009 },
  { lat: 40.7155, lng: -74.012 },
  { lat: 40.7165, lng: -74.008 },
  { lat: 40.7105, lng: -74.004 },
  { lat: 40.7175, lng: -74.018 },
  { lat: 40.7185, lng: -74.014 },
  { lat: 40.7195, lng: -74.01 },
  { lat: 40.7205, lng: -74.006 },
  { lat: 40.7215, lng: -74.002 },
  { lat: 40.7225, lng: -74.008 },
]

// Mock data for defect list
export const defects = [
  {
    id: "DEF-1001",
    type: "Pothole",
    severity: "critical",
    location: "Main St & 5th Ave",
    detectedAt: "2025-04-24T08:30:00",
    status: "Pending",
  },
  {
    id: "DEF-1002",
    type: "Crack",
    severity: "high",
    location: "Broadway & 42nd St",
    detectedAt: "2025-04-24T09:15:00",
    status: "Assigned",
  },
  {
    id: "DEF-1003",
    type: "Paint Peeling",
    severity: "medium",
    location: "Park Ave & 23rd St",
    detectedAt: "2025-04-24T07:45:00",
    status: "In Progress",
  },
  {
    id: "DEF-1004",
    type: "Pothole",
    severity: "high",
    location: "Lexington Ave & 59th St",
    detectedAt: "2025-04-24T10:20:00",
    status: "Pending",
  },
  {
    id: "DEF-1005",
    type: "Crack",
    severity: "low",
    location: "Canal St & Bowery",
    detectedAt: "2025-04-24T11:05:00",
    status: "Assigned",
  },
  {
    id: "DEF-1006",
    type: "Paint Peeling",
    severity: "low",
    location: "Houston St & Lafayette St",
    detectedAt: "2025-04-24T08:50:00",
    status: "Completed",
  },
  {
    id: "DEF-1007",
    type: "Pothole",
    severity: "critical",
    location: "West St & Chambers St",
    detectedAt: "2025-04-24T09:30:00",
    status: "In Progress",
  },
]

// Mock data for recent alerts
export const recentAlerts = [
  {
    id: "DEF-1001",
    type: "Pothole",
    severity: "critical",
    location: "Main St & 5th Ave",
    detectedAt: "2025-04-24T08:30:00",
    description: "Large pothole causing traffic hazard",
  },
  {
    id: "DEF-1007",
    type: "Pothole",
    severity: "critical",
    location: "West St & Chambers St",
    detectedAt: "2025-04-24T09:30:00",
    description: "Deep pothole with exposed rebar",
  },
  {
    id: "DEF-1002",
    type: "Crack",
    severity: "high",
    location: "Broadway & 42nd St",
    detectedAt: "2025-04-24T09:15:00",
    description: "Extensive cracking across two lanes",
  },
  {
    id: "DEF-1004",
    type: "Pothole",
    severity: "high",
    location: "Lexington Ave & 59th St",
    detectedAt: "2025-04-24T10:20:00",
    description: "Multiple potholes in intersection",
  },
]

// Mock data for defect statistics
export const defectTypeData = [
  { value: 42, name: "Potholes" },
  { value: 35, name: "Cracks" },
  { value: 23, name: "Paint Peeling" },
]

export const severityData = [
  { value: 12, name: "Critical" },
  { value: 28, name: "High" },
  { value: 32, name: "Medium" },
  { value: 28, name: "Low" },
]

// Mock data for defect trends
export const trendData = {
  dates: ["Apr 18", "Apr 19", "Apr 20", "Apr 21", "Apr 22", "Apr 23", "Apr 24"],
  potholesData: [12, 19, 15, 22, 25, 28, 30],
  cracksData: [10, 15, 12, 18, 20, 22, 25],
  paintData: [8, 10, 12, 15, 13, 16, 18],
}

// Mock data for dashboard metrics
export const dashboardMetrics = {
  totalDefects: 170,
  totalDefectsChange: "+12%",
  criticalIssues: 7,
  criticalIssuesChange: "-3%",
  avgResponseTime: "2.4h",
  avgResponseTimeChange: "-18%",
  affectedAreas: 42,
  affectedAreasChange: "+5%",
}

// Mock data for severity counts
export const severityCounts = {
  critical: 7,
  high: 23,
  medium: 48,
  low: 92,
}
