import ConditionHeader from '@/components/headers/ConditionHeader'
import LocationHeader from '@/components/headers/LocationHeader'
import Map from '@/components/main/Map'
import Image from '@/components/sidebar/Image'
import IssueCard from '@/components/sidebar/IssueCard'
import React from 'react'

// import { Button } from './ui/button'

const Dashboard: React.FC = () => {
  return (
    <div className="h-screen w-screen flex flex-col bg-white">
      <div className="flex-1 basis-0 mx-[10px] mt-[10px] border-2 border-[#808080] rounded-[0.5rem]">
        <ConditionHeader />
      </div>
      <div className="flex-1 basis-0 mx-[10px] mt-[10px] border-2 border-[#808080] rounded-[0.5rem]">
        <LocationHeader />
      </div>
      <div className="flex-[7] basis-0 m-[10px] border-2 border-[#808080] rounded-[0.5rem] flex flex-row gap-[10px]">
        <div className="flex-1 flex flex-col gap-[10px]">
          <div className="flex-[4] border-2 border-[#808080] rounded-[0.5rem]">
            <Image />
          </div>
          <div className="flex-[3] border-2 border-[#808080] rounded-[0.5rem]">
            <IssueCard />
          </div>
        </div>
        <div className="flex-[2] border-2 border-[#808080] rounded-[0.5rem]">
          <Map />
        </div>
      </div>
    </div>
  )
}

export default Dashboard
