import { cn } from '@/lib/utils'
import * as React from 'react'

const DefectCard = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      'bg-card text-card-foreground flex h-full flex-col rounded-xl border shadow',
      className,
    )}
    {...props}
  />
))
DefectCard.displayName = 'DefectCard'

const DefectCardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex flex-col space-y-1.5 p-6', className)}
    {...props}
  />
))
DefectCardHeader.displayName = 'DefectCardHeader'

const DefectCardTitle = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('leading-none font-semibold tracking-tight', className)}
    {...props}
  />
))
DefectCardTitle.displayName = 'DefectCardTitle'

const DefectCardDescription = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('text-muted-foreground text-sm', className)}
    {...props}
  />
))
DefectCardDescription.displayName = 'DefectCardDescription'

const DefectCardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn(
      // flex 컨테이너 내부에서 height 계산 위해 min-h-0 필수
      'flex min-h-0 flex-1 flex-col overflow-hidden p-6 pt-0',
      className,
    )}
    {...props}
  />
))
DefectCardContent.displayName = 'DefectCardContent'

const DefectCardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex flex-shrink-0 items-center p-6 pt-0', className)}
    {...props}
  />
))
DefectCardFooter.displayName = 'DefectCardFooter'

export {
  DefectCard,
  DefectCardHeader,
  DefectCardTitle,
  DefectCardDescription,
  DefectCardContent,
  DefectCardFooter,
}
