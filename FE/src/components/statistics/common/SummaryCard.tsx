// components/dashboard/SummaryCard.tsx
interface SummaryCardProps {
  title: string
  value: string | React.ReactNode
  className?: string
  textSize?: 'small' | 'medium' | 'large' // 텍스트 크기 옵션 추가
}

export default function SummaryCard({ 
  title, 
  value, 
  className,
  textSize = 'medium' // 기본값은 medium
}: SummaryCardProps) {
  // 텍스트 크기에 따른 클래스 결정
  const valueTextClass = {
    small: 'text-l font-semibold',
    medium: 'text-2xl font-semibold',
    large: 'text-3xl font-semibold'
  }[textSize]

  return (
    <div className={`rounded-lg border bg-card p-4 text-card-foreground shadow-sm ${className}`}>
      <div className="flex flex-col space-y-1.5">
        <h3 className="text-sm font-medium">{title}</h3>
        <div className={valueTextClass}>{value}</div>
      </div>
    </div>
  )
}