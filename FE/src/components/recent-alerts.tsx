"use client"

import { AlertTriangle, MapPin, Clock, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { useDefectStore } from "@/store/defect-store"

export default function RecentAlerts() {
  // Get recent alerts from Zustand store
  const { recentAlerts } = useDefectStore()

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  }

  return (
    <div className="space-y-4">
      {/* TODO: Replace with actual API call to fetch recent alerts */}
      {recentAlerts.map((alert, index) => (
        <div key={alert.id}>
          <div className="flex items-start gap-2">
            <div className={`mt-0.5 rounded-full p-1 ${alert.severity === "critical" ? "bg-red-100" : "bg-amber-100"}`}>
              <AlertTriangle
                className={`h-4 w-4 ${alert.severity === "critical" ? "text-red-600" : "text-amber-600"}`}
              />
            </div>
            <div className="flex-1 space-y-1">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">
                  {alert.type}: {alert.id}
                </p>
                <Badge className={alert.severity === "critical" ? "bg-red-500" : "bg-amber-500"}>
                  {alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">{alert.description}</p>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <MapPin className="h-3 w-3" />
                  {alert.location}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {formatDate(alert.detectedAt)}
                </span>
              </div>
            </div>
          </div>
          {index < recentAlerts.length - 1 && <Separator className="my-2" />}
        </div>
      ))}
      <Button variant="outline" size="sm" className="w-full">
        View all alerts
        <ArrowRight className="ml-2 h-4 w-4" />
      </Button>
    </div>
  )
}
