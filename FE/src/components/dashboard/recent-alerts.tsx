'use client'

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { DefectDetail } from '@/types/defects'
import { AlertTriangle, Clock, MapPin } from 'lucide-react'
import Image from 'next/image'

export default function RecentAlerts({
  filteredDefectDetailList,
}: {
  filteredDefectDetailList: DefectDetail[]
}) {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  // 위험도에 따라 심각도 결정
  const getSeverity = (risk: number) => {
    if (risk >= 0.7) return 'critical'
    return 'warning'
  }

  return (
    <ScrollArea className="h-full">
      {filteredDefectDetailList.map((defectDetail, index) => (
        <div key={defectDetail.publicId}>
          <div className="flex items-start gap-2">
            <div
              className={`mt-0.5 rounded-full p-1 ${
                getSeverity(defectDetail.risk) === 'critical'
                  ? 'bg-red-100'
                  : 'bg-amber-100'
              }`}
            >
              <AlertTriangle
                className={`h-4 w-4 ${
                  getSeverity(defectDetail.risk) === 'critical'
                    ? 'text-red-600'
                    : 'text-amber-600'
                }`}
              />
            </div>
            <div className="flex-1 space-y-1">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">
                  {defectDetail.damages[0]?.category ?? '알 수 없음'}:{' '}
                  {defectDetail.damages[0]?.id ?? ''}
                </p>
                <Badge
                  className={
                    getSeverity(defectDetail.risk) === 'critical'
                      ? 'bg-red-500'
                      : 'bg-amber-500'
                  }
                >
                  {getSeverity(defectDetail.risk) === 'critical'
                    ? '심각'
                    : '경고'}
                </Badge>
              </div>
              <Accordion type="single" collapsible>
                <AccordionItem value={`item-${defectDetail.publicId}`}>
                  <AccordionTrigger className="p-0">
                    <div className="group">
                      <p className="text-muted-foreground text-xs group-hover:underline">
                        {defectDetail.damages
                          .map((damage) => damage.category)
                          .join(', ')}
                      </p>
                      <div className="text-muted-foreground flex items-center gap-4 text-xs">
                        <span className="flex items-center gap-1">
                          <MapPin className="h-3 w-3" />
                          {defectDetail.address}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {defectDetail.damages.length > 0
                            ? formatDate(defectDetail.damages[0].updatedAt)
                            : ''}
                        </span>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="pt-2 pb-0">
                    <Image
                      src={defectDetail.imageUrl}
                      alt={`${defectDetail.address} 손상 이미지`}
                      width={600}
                      height={400}
                      onError={() => {
                        // 이미지 로드 실패시 콘솔에 오류 기록
                        console.error(
                          `이미지 로드 실패: ${defectDetail.imageUrl}`,
                        )
                      }}
                      unoptimized={
                        !defectDetail.imageUrl ||
                        !defectDetail.imageUrl.startsWith('https://')
                      }
                    />
                    <div className="mt-2 space-y-1">
                      <p className="text-sm font-medium">손상 정보:</p>
                      <ul className="text-muted-foreground text-xs">
                        {defectDetail.damages.map((damage) => (
                          <li
                            key={damage.id}
                            className="flex justify-between py-1"
                          >
                            <span>
                              {damage.category} (ID: {damage.id})
                            </span>
                            <span>{formatDate(damage.updatedAt)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
          </div>
          {index < filteredDefectDetailList.length - 1 && (
            <Separator className="my-2" />
          )}
        </div>
      ))}
    </ScrollArea>
  )
}
