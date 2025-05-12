import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export default function DefectStatsRepair() {
  return (
    <div className="grid grid-cols-3 grid-rows-2 gap-4">
      {[7, 8, 9, 10, 11, 12].map((item) => (
        <Card key={item} className="h-64">
          <CardHeader className="p-4">
            <CardTitle className="text-lg">Card {item}</CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            <p className="text-muted-foreground text-sm">Card content</p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
