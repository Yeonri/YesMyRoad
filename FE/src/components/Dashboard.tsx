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
      {/* ConditionHeader: 1/9 */}
      <div className="flex-1 basis-0 mx-[10px] mt-[10px] border-2 border-[#808080] rounded-[0.5rem]">
        <ConditionHeader />
      </div>
      {/* LocationHeader: 1/9 */}
      <div className="flex-1 basis-0 mx-[10px] mt-[10px] border-2 border-[#808080] rounded-[0.5rem]">
        <LocationHeader />
      </div>
      {/* Map, Image, IssueCard: 7/9 */}
      <div className="flex-[7] basis-0 m-[10px] border-2 border-[#808080] rounded-[0.5rem] flex flex-row gap-[10px]">
        {/* 좌측: Image + IssueCard (가로 1) */}
        <div className="flex-1 flex flex-col gap-[10px]">
          {/* Image (세로 4) */}
          <div className="flex-[4] border-2 border-[#808080] rounded-[0.5rem]">{/* <Image /> */}</div>
          {/* IssueCard (세로 3) */}
          <div className="flex-[3] border-2 border-[#808080] rounded-[0.5rem]">{/* <IssueCard /> */}</div>
        </div>
        {/* 우측: Map (가로 2) */}
        <div className="flex-[2] border-2 border-[#808080] rounded-[0.5rem]">{/* <Map /> */}</div>
      </div>
    </div>
  )
}

export default Dashboard




// import ConditionHeader from '@/components/headers/ConditionHeader'
// import LocationHeader from '@/components/headers/LocationHeader'
// import Map from '@/components/main/Map'
// import Image from '@/components/sidebar/Image'
// import IssueCard from '@/components/sidebar/IssueCard'
// import React from 'react'

// const boxStyle = "border-2 border-[#808080] rounded-[0.5rem] bg-white"

// const Dashboard: React.FC = () => {
//   return (
//     <div className="h-screen w-screen flex flex-col bg-white">
//       {/* ConditionHeader: 1/9 */}
//       <ConditionHeader className={`flex-1 basis-0 mx-[10px] mt-[10px] ${boxStyle}`} />
//       {/* LocationHeader: 1/9 */}
//       <LocationHeader className={`flex-1 basis-0 mx-[10px] mt-[10px] ${boxStyle}`} />
//       {/* Map, Image, IssueCard: 7/9 */}
//       <div className="flex-[7] basis-0 m-[10px] flex flex-row gap-[10px] border-2 border-[#808080] rounded-[0.5rem]">
//         {/* 좌측: Image + IssueCard (가로 1) */}
//         <div className="flex-1 flex flex-col gap-[10px]">
//           {/* Image (세로 4) */}
//           <Image className={`flex-[4] ${boxStyle}`} />
//           {/* IssueCard (세로 3) */}
//           <IssueCard className={`flex-[3] ${boxStyle}`} />
//         </div>
//         {/* 우측: Map (가로 2) */}
//         <Map className={`flex-[2] ${boxStyle}`} />
//       </div>
//     </div>
//   )
// }

// export default Dashboard





// type Props = {
//     className?: string
//     // ...다른 props
//   }
  
//   const ConditionHeader: React.FC<Props> = ({ className }) => (
//     <div className={className}>
//       {/* ... */}
//     </div>
//   )
  