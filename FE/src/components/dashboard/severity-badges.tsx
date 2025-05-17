import { Badge } from "@/components/ui/badge"

const SeverityBadges: React.FC = () => {
    return (
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex flex-wrap gap-2">
                <Badge variant="default" className="rounded-md px-3 py-1">
                    <div className="mr-1 h-2 w-2 rounded-full bg-red-500" />
                    심각: 7 {/* TODO: replace real value*/}
                    {/* 심각: {severityCounts.critical} */}
                </Badge>
                <Badge variant="default" className="rounded-md px-3 py-1">
                    <div className="mr-1 h-2 w-2 rounded-full bg-amber-500" />
                    위험: 23 {/* TODO: replace real value*/}
                    {/* 위험: {severityCounts.high} */}
                </Badge>
                <Badge variant="default" className="rounded-md px-3 py-1">
                    <div className="mr-1 h-2 w-2 rounded-full bg-blue-500" />
                    주의: 48 {/* TODO: replace real value*/}
                    {/* 주의: {severityCounts.medium} */}
                </Badge>
                <Badge variant="default" className="rounded-md px-3 py-1">
                    <div className="mr-1 h-2 w-2 rounded-full bg-green-500" />
                    안전: 92 {/* TODO: replace real value*/}
                    {/* 안전: {severityCounts.low} */}
                </Badge>
            </div>
        </div>
    )
}

export default SeverityBadges