import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { AlertTriangle, BarChart3, Clock, MapPin } from 'lucide-react'

const DefectOverall:React.FC = () => {
    return(
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">총 결함 수</CardTitle>
              <BarChart3 className="text-muted-foreground h-4 w-4" />
            </CardHeader>
            <CardContent>
              {/* <div className="text-2xl font-bold">
                {reportData?.data?.count || 0}
              </div>
              <p className="text-muted-foreground text-xs">
                {{ D: '어제', W: '지난 주', M: '지난 달' }[timeRange] || ''}{' '}
                대비{' '}
                {reportData?.data?.changeRate === null ? '-' : (reportData?.data?.changeRate || 0)}{' '}
                % 증가
              </p> */}
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                심각한 결함 수
              </CardTitle>
              <AlertTriangle className="text-muted-foreground h-4 w-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {/* {dashboardMetrics.criticalIssues} */}
                0
              </div>
              <p className="text-muted-foreground text-xs">
                지난 주 대비 n % 증가
                {/* 지난 주 대비 {dashboardMetrics.criticalIssuesChange} */}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                평균 작업 착수 시간
              </CardTitle>
              <Clock className="text-muted-foreground h-4 w-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                0
                {/* {dashboardMetrics.avgResponseTime} */}
              </div>
              <p className="text-muted-foreground text-xs">
                지난 주 대비 n % 증가
                {/* 지난 주 대비 {dashboardMetrics.avgResponseTimeChange} */}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                발생 행정구역 수
              </CardTitle>
              <MapPin className="text-muted-foreground h-4 w-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                0
                {/* {dashboardMetrics.affectedAreas} */}
              </div>
              <p className="text-muted-foreground text-xs">
                지난 주 대비 n % 증가
                {/* 지난 주 대비 {dashboardMetrics.affectedAreasChange} */}
              </p>
            </CardContent>
          </Card>
        </div>
    )
}

export default DefectOverall