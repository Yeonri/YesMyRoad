import type { Metadata } from 'next'
import type React from 'react'

// 라우팅 보호 활성화시 아래 코드의 주석을 해제하세요
// import ClientLayout from './client-layout'
import './globals.css'

export const metadata: Metadata = {
  title: 'Road Defect Monitoring System',
  description:
    'Real-time monitoring of road defects such as potholes, cracks, and paint peeling',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        {/* Add Mapbox GL CSS */}
        <link
          href="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css"
          rel="stylesheet"
        />
      </head>
      <body>
        {/* 현재는 라우트보호가 해제되어있습니다 */}
        {children}

        {/* 아래 코드 주석 해제하면 라우트 보호가 활성화됩니다 */}
        {/* <ClientLayout>{children}</ClientLayout> */}
      </body>
    </html>
  )
}
