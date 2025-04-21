import type { Metadata } from 'next'

// import { Geist, Geist_Mono } from 'next/font/google'

import './globals.css'

// const geistSans = Geist({
//   variable: "--font-geist-sans",
//   subsets: ["latin"],
// });

// const geistMono = Geist_Mono({
//   variable: "--font-geist-mono",
//   subsets: ["latin"],
// });

export const metadata: Metadata = { title: 'Yes, my ROAD', description: '도로 파손 현황 실시간 모니터링 시스템' }

const RootLayout = ({ children }: Readonly<{ children: React.ReactNode }>) => {
  return (
    <html lang="kr">
      <head>
        <link href="https://api.mapbox.com/mapbox-gl-js/v3.11.0/mapbox-gl.css" rel="stylesheet" />
        <script src="https://api.mapbox.com/mapbox-gl-js/plugins/mapbox-gl-language/v1.0.0/mapbox-gl-language.js"></script>
      </head>
      <body>{children}</body>
    </html>
  )
}

export default RootLayout
