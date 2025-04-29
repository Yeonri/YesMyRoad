import { cn } from '@/lib/utils'
import * as React from 'react'

const DefectCard = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('rounded-xl border bg-card text-card-foreground shadow flex flex-col h-full', className)}
      {...props}
    />
  ),
)
DefectCard.displayName = 'DefectCard'

const DefectCardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('flex flex-col space-y-1.5 p-6', className)} {...props} />
  ),
)
DefectCardHeader.displayName = 'DefectCardHeader'

const DefectCardTitle = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('font-semibold leading-none tracking-tight', className)} {...props} />
  ),
)
DefectCardTitle.displayName = 'DefectCardTitle'

const DefectCardDescription = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('text-sm text-muted-foreground', className)} {...props} />
  ),
)
DefectCardDescription.displayName = 'DefectCardDescription'

const DefectCardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        // flex 컨테이너 내부에서 height 계산 위해 min-h-0 필수
        'p-6 pt-0 flex flex-col flex-1 min-h-0 overflow-hidden',
        className,
      )}
      {...props}
    />
  ),
)
DefectCardContent.displayName = 'DefectCardContent'

const DefectCardFooter = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('flex-shrink-0 flex items-center p-6 pt-0', className)} {...props} />
  ),
)
DefectCardFooter.displayName = 'DefectCardFooter'

export { DefectCard, DefectCardHeader, DefectCardTitle, DefectCardDescription, DefectCardContent, DefectCardFooter }
