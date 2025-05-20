// components/dashboard/SummaryCard.tsx
interface SummaryCardProps {
  title: string
  value: string | React.ReactNode
  className?: string
  textSize?: 'small' | 'medium' | 'large'
}

export default function SummaryCard({
  title,
  value,
  className,
  textSize = 'medium',
}: SummaryCardProps) {
  const valueTextClass = {
    small: 'text-l font-semibold',
    medium: 'text-2xl font-semibold',
    large: 'text-3xl font-semibold',
  }[textSize]

  return (
    <div
      className={`bg-card text-card-foreground rounded-lg border p-4 shadow-sm ${className}`}
    >
      <div className="flex flex-col space-y-1.5">
        <h3 className="text-sm font-medium">{title}</h3>
        <div className={valueTextClass}>{value}</div>
      </div>
    </div>
  )
}
