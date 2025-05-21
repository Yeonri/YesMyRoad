'use client'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import Image from 'next/image'
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
  const [imageError, setImageError] = useState(false)
  
  return (
    <AlertDialog open={isOpen} onOpenChange={setIsOpen}>
      <AlertDialogTrigger asChild>
        <Button variant="outline">이미지 보기</Button>
      </AlertDialogTrigger>
      <AlertDialogContent className="max-w-3xl">
        <AlertDialogHeader>
          <AlertDialogTitle>{alt}</AlertDialogTitle>
        </AlertDialogHeader>
        <div className="flex justify-center items-center my-4">
          <Image
            src={imageError ? '/assets/images/image-not-found.png' : imageUrl}
            width={600}
            height={400}
            alt={alt}
            className="rounded-md object-contain"
            onError={() => setImageError(true)}
          />
        </div>
        <AlertDialogFooter>
          <AlertDialogAction>닫기</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}