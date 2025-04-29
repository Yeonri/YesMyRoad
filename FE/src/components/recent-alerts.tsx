'use client'

import {
  AlertAccordion,
  AlertAccordionContent,
  AlertAccordionItem,
  AlertAccordionTrigger,
} from '@/components/ui/alert-accordion'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { useDefectStore } from '@/store/defect-store'
import { AlertTriangle, ArrowRight, Clock, MapPin } from 'lucide-react'

export default function RecentAlerts() {
  const { recentAlerts } = useDefectStore()

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* 스크롤 리스트 영역 */}
      <div className="flex-1 overflow-y-auto max-h-full">
        {recentAlerts.map((alert, index) => (
          <div key={alert.id}>
            <div className="flex items-start gap-2">
              <div
                className={`mt-0.5 rounded-full p-1 ${alert.severity === 'critical' ? 'bg-red-100' : 'bg-amber-100'}`}
              >
                <AlertTriangle
                  className={`h-4 w-4 ${alert.severity === 'critical' ? 'text-red-600' : 'text-amber-600'}`}
                />
              </div>
              <div className="flex-1 space-y-1">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium">
                    {alert.type}: {alert.id}
                  </p>
                  <Badge className={alert.severity === 'critical' ? 'bg-red-500' : 'bg-amber-500'}>
                    {alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}
                  </Badge>
                </div>
                <AlertAccordion type="single" collapsible>
                  <AlertAccordionItem value="item-1">
                    <AlertAccordionTrigger className="p-0">
                      <div className="group">
                        <p className="text-xs text-muted-foreground group-hover:underline">{alert.description}</p>
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
                    </AlertAccordionTrigger>
                    <AlertAccordionContent className="pt-2 pb-0">
                      <img src="https://placehold.co/600x400" alt="temporary image" />
                      fill here
                    </AlertAccordionContent>
                  </AlertAccordionItem>
                </AlertAccordion>
              </div>
            </div>
            {index < recentAlerts.length - 1 && <Separator className="my-2" />}
          </div>
        ))}
      </div>
    </div>
  )
}