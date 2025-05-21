import { cn } from '@/lib/utils'

// components/dashboard/SummaryCard.tsx
interface SummaryCardProps {
  title: string
  value: string | React.ReactNode
  className?: string
  textSize?: 'small' | 'medium' | 'large'
  subValue?: string
}

export default function SummaryCard({
  title,
  value,
  className,
  textSize = 'medium',
  subValue,
}: SummaryCardProps) {
  const valueTextClass = {
    small: 'text-l font-semibold',
    medium: 'text-2xl font-semibold',
    large: 'text-3xl font-semibold',
  }[textSize]

  const getSubValueTextColor = (value: string) => {
    if (value.includes('감소')) return 'text-blue-600'
    if (value.includes('증가')) return 'text-red-500'
    return 'text-muted-foreground'
  }
  return (
    <div
      className={`bg-card text-card-foreground rounded-lg border p-4 shadow-sm ${className}`}
    >
      <div className="flex flex-col space-y-1.5">
        <h3 className="text-sm font-medium">{title}</h3>
        <div className={valueTextClass}>{value}</div>
        {subValue && (
          <div
            className={cn('text-sm font-bold', getSubValueTextColor(subValue))}
          >
            {subValue}
          </div>
        )}
      </div>
    </div>
  )
}
