'use client'

import imageNotFound from '@/assets/images/image-not-found.png'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { DamageItem, DefectDetail } from '@/types/defects'
import { AlertTriangle, Clock, MapPin } from 'lucide-react'
import Image, { StaticImageData } from 'next/image'
import React, { useState } from 'react'

export default function DefectListOnMap({
  filteredDefectDetailList,
}: {
  filteredDefectDetailList: DefectDetail[]
}) {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleString([], {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    })
  }

  // 손상 유형별 집계 함수 추가
  const getDamageSummary = (damages: DamageItem[]) => {
    const categoryCount = damages.reduce(
      (acc, damage) => {
        acc[damage.category] = (acc[damage.category] || 0) + 1
        return acc
      },
      {} as Record<string, number>,
    )

    return Object.entries(categoryCount)
      .map(([category, count]) => `${category} ${count}건`)
      .join(', ')
  }

  // 위험도에 따라 심각도 결정
  const getSeverity = (risk: number) => {
    if (risk >= 0.7) return 'critical'
    return 'warning'
  }

  const [imgSrcMap, setImgSrcMap] = useState<
    Record<string, string | StaticImageData>
  >({})

  return (
    <ScrollArea className="h-full">
      <Accordion type="single" collapsible>
        {filteredDefectDetailList.map((defectDetail, index) => {
          const imgSrc =
            imgSrcMap[defectDetail.publicId] || defectDetail.imageUrl
          return (
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
                      {getDamageSummary(defectDetail.damages)}
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
                  <AccordionItem value={`item-${defectDetail.publicId}`}>
                    <AccordionTrigger className="p-0">
                      <div className="group">
                        <div className="text-muted-foreground flex flex-col gap-1">
                          <p className="flex items-center gap-1">
                            <MapPin className="h-3 w-3" />
                            {defectDetail.address}
                          </p>
                          <p className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {defectDetail.damages.length > 0
                              ? formatDate(defectDetail.damages[0].updatedAt)
                              : ''}
                          </p>
                        </div>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pt-2 pb-0">
                      <Image
                        src={imgSrc}
                        alt={`${defectDetail.address} 손상 이미지`}
                        width={600}
                        height={400}
                        onError={() => {
                          setImgSrcMap((prev) => ({
                            ...prev,
                            [defectDetail.publicId]: imageNotFound,
                          }))
                        }}
                        unoptimized={
                          !imgSrc ||
                          (typeof imgSrc === 'string' &&
                            !imgSrc.startsWith('https://'))
                        }
                      />
                    </AccordionContent>
                  </AccordionItem>
                </div>
              </div>
              {index < filteredDefectDetailList.length - 1 && (
                <Separator className="my-2" />
              )}
            </div>
          )
        })}
      </Accordion>
    </ScrollArea>
  )
}
