'use client'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import Image from 'next/image'
// 추가된 import
import { useState } from 'react'

interface DefectImageProps {
  imageUrl: string
  alt?: string
}

export default function DefectImage({
  imageUrl,
  alt = '결함 이미지',
}: DefectImageProps) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <AlertDialog open={isOpen} onOpenChange={setIsOpen}>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm" className="select-none">
          이미지 보기
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent className="max-w-3xl select-none">
        <AlertDialogHeader>
          <AlertDialogTitle>결함 이미지</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="relative mt-2 h-96 w-full">
              <Image
                src={imageUrl}
                alt={alt}
                fill
                className="rounded-md object-contain select-none"
                draggable={false}
                unoptimized={imageUrl.startsWith('data:')} // data URL인 경우 최적화 비활성화
              />
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogAction className="select-none">닫기</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
