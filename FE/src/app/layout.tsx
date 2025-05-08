import type { Metadata } from 'next'
import type React from 'react'

import './globals.css'

import ClientLayout from './client-layout'

export const metadata: Metadata = {
  title: 'Road Defect Monitoring System',
  description: 'Real-time monitoring of road defects such as potholes, cracks, and paint peeling',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
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
        {children}
        {/* <ClientLayout>{children}</ClientLayout> */}
      </body>
    </html>
  )
}
