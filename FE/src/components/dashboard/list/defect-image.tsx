'use client'

import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import Image from 'next/image'
import { useState } from 'react'
import { RotateCcw, RotateCw } from 'lucide-react'

interface DefectImageProps {
  imageUrl: string
  alt?: string
}

export default function DefectImage({
  imageUrl,
  alt = '결함 이미지',
}: DefectImageProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [imageError, setImageError] = useState(false)
  const [rotation, setRotation] = useState(0)

  // 시계방향 회전 (90도씩)
  const rotateClockwise = () => {
    setRotation((prev) => (prev + 90))
  }

  // 반시계방향 회전 (90도씩)
  const rotateCounterClockwise = () => {
    setRotation((prev) => (prev - 90))
  }

  // 다이얼로그가 닫힐 때 회전 상태 초기화
  const handleOpenChange = (open: boolean) => {
    setIsOpen(open)
      setRotation(0)
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline">이미지 보기</Button>
      </DialogTrigger>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>{alt}</DialogTitle>
        </DialogHeader>
        <div className="my-4 flex h-[400px] items-center justify-center overflow-hidden">
          <div className="relative h-full w-full max-w-[600px]">
            <Image
              src={imageError ? '/assets/images/image-not-found.png' : imageUrl}
              fill
              alt={alt}
              className="rounded-md object-contain"
              style={{ transform: `rotate(${rotation}deg)`, transition: 'transform 0.3s ease' }}
              onError={() => setImageError(true)}
            />
          </div>
        </div>
        <DialogFooter className="flex justify-between sm:justify-between">
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="icon"
              onClick={rotateCounterClockwise}
              title="반시계방향 회전"
            >
              <RotateCcw className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={rotateClockwise}
              title="시계방향 회전"
            >
              <RotateCw className="h-4 w-4" />
            </Button>
          </div>
          <Button variant="secondary" onClick={() => setIsOpen(false)}>
            닫기
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}