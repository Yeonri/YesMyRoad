import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import React from 'react'

interface SummaryCardProps {
  title: string
  value: React.ReactNode
  className?: string
}

export default function SummaryCard({ title, value, className }: SummaryCardProps) {
  return (
    <Card className={className}>
      <CardHeader className="p-4">
        <CardTitle className="text-md">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div className="flex flex-col gap-2">
          <p className="text-lg font-bold">{value}</p>
        </div>
      </CardContent>
    </Card>
  )
} 
