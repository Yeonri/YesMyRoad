'use client'

import { Button } from "@/components/ui/button"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { useState } from "react"

interface DefectImageProps {
  imageUrl: string
  alt?: string
}

export default function DefectImage({ imageUrl, alt = "결함 이미지" }: DefectImageProps) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <AlertDialog open={isOpen} onOpenChange={setIsOpen}>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm" className="select-none">이미지 보기</Button>
      </AlertDialogTrigger>
      <AlertDialogContent className="max-w-3xl select-none">
        <AlertDialogHeader>
          <AlertDialogTitle>결함 이미지</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="relative w-full h-96 mt-2">
              <img
                src={imageUrl}
                alt={alt}
                className="object-contain w-full h-full rounded-md select-none"
                draggable="false"
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