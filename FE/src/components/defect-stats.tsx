'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import useEmblaCarousel from 'embla-carousel-react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

export default function DefectStats() {
  const [emblaRef, emblaApi] = useEmblaCarousel()
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [scrollSnaps, setScrollSnaps] = useState<number[]>([])

  const scrollPrev = useCallback(
    () => emblaApi && emblaApi.scrollPrev(),
    [emblaApi],
  )
  const scrollNext = useCallback(
    () => emblaApi && emblaApi.scrollNext(),
    [emblaApi],
  )

  const onSelect = useCallback(() => {
    if (!emblaApi) return
    setSelectedIndex(emblaApi.selectedScrollSnap())
  }, [emblaApi])

  useEffect(() => {
    if (!emblaApi) return
    onSelect()
    setScrollSnaps(emblaApi.scrollSnapList())
    emblaApi.on('select', onSelect)
  }, [emblaApi, onSelect])

  return (
    <div className="relative w-full">
      <div className="overflow-hidden" ref={emblaRef}>
        <div className="flex">
          {/* First Page */}
          <div className="min-w-0 flex-[0_0_100%]">
            <div className="grid grid-cols-3 grid-rows-2 gap-4">
              {[1, 2, 3, 4, 5, 6].map((item) => (
                <Card key={item} className="h-64">
                  <CardHeader className="p-4">
                    <CardTitle className="text-lg">Card {item}</CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 pt-0">
                    <p className="text-muted-foreground text-sm">
                      Card content
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
          {/* Second Page */}
          <div className="min-w-0 flex-[0_0_100%]">
            <div className="grid grid-cols-3 grid-rows-2 gap-4">
              {[7, 8, 9, 10, 11, 12].map((item) => (
                <Card key={item} className="h-64">
                  <CardHeader className="p-4">
                    <CardTitle className="text-lg">Card {item}</CardTitle>
                  </CardHeader>
                  <CardContent className="p-4 pt-0">
                    <p className="text-muted-foreground text-sm">
                      Card content
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Controls */}
      <div className="mt-4 flex items-center justify-center gap-4">
        <button
          onClick={scrollPrev}
          className="rounded-full p-2 transition-colors hover:bg-gray-100"
        >
          <ChevronLeft className="h-6 w-6" />
        </button>

        <div className="flex gap-2">
          {scrollSnaps.map((_, index) => (
            <button
              key={index}
              onClick={() => emblaApi?.scrollTo(index)}
              className={`h-2 w-2 rounded-full transition-colors ${
                index === selectedIndex ? 'bg-blue-600' : 'bg-gray-300'
              }`}
            />
          ))}
        </div>

        <button
          onClick={scrollNext}
          className="rounded-full p-2 transition-colors hover:bg-gray-100"
        >
          <ChevronRight className="h-6 w-6" />
        </button>
      </div>
    </div>
  )
}
