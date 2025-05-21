import crypto from 'crypto'

// 시간 포맷팅 함수
export const formatDate = (dateString: string) => {
  const date = new Date(dateString)

  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')

  const hours = date.getHours()
  const ampm = hours >= 12 ? '오후' : '오전'
  const hour12 = hours % 12 || 12 // 12시간 형식 (0은 12로 표시)
  const minute = String(date.getMinutes()).padStart(2, '0')

  return `${year}-${month}-${day} ${ampm} ${hour12}:${minute}`
}

export const getSeverity = (risk: number) => {
  if (0 <= risk && risk < 40) return '심각'
  else if (40 <= risk && risk < 60) return '위험'
  else if (60 <= risk && risk < 80) return '주의'
  else return '안전'
}

export const getSeverityColor = (severity: string) => {
  switch (severity) {
    case '심각':
      return 'bg-red-500 text-white'
    case '위험':
      return 'bg-amber-500 text-white'
    case '주의':
      return 'bg-blue-500 text-white'
    case '안전':
      return 'bg-green-500 text-white'
    default:
      return 'bg-gray-500 text-white'
  }
}

export const getStatusColor = (status: string) => {
  switch (status) {
    case 'REPORTED':
      return 'bg-gray-100 text-gray-800 hover:bg-gray-300 hover:text-gray-900 transition-colors duration-150'
    case 'RECEIVED':
      return 'bg-blue-100 text-blue-800 hover:bg-blue-300 hover:text-blue-900 transition-colors duration-150'
    case 'IN_PROGRESS':
      return 'bg-amber-100 text-amber-800 hover:bg-amber-300 hover:text-amber-900 transition-colors duration-150'
    case 'COMPLETED':
      return 'bg-green-100 text-green-800 hover:bg-green-300 hover:text-green-900 transition-colors duration-150'
    default:
      return 'bg-gray-100 text-gray-800 hover:bg-gray-300 hover:text-gray-900 transition-colors duration-150'
  }
}

// 결함 ID를 생성하는 함수
export const getDisplayId = (
  category: string,
  damageId: number,
  publicId: string,
): string => {
  const validChars = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
  const hash = crypto.createHash('sha256').update(publicId).digest('hex')
  const prefix = category === '도로균열' ? 'RC' : 'RH' // road crack, road hole
  const shortCode = Array(5)
    .fill(0)
    .map((_, i) => {
      const index =
        parseInt(hash.slice(i * 2, i * 2 + 2), 16) % validChars.length
      return validChars[index]
    })
    .join('')

  // damageId를 문자열로 변환하고 4자리가 되도록 앞에 0을 채움
  const formattedDamageId = String(damageId).padStart(4, '0')

  return `${prefix}-${formattedDamageId}-${shortCode}`
}

// 결함 ID를 생성하는 함수: publicId만 해싱
export const getDisplayIdPublic = (publicId: string): string => {
  const validChars = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
  const hash = crypto.createHash('sha256').update(publicId).digest('hex')
  const shortCode = Array(5)
    .fill(0)
    .map((_, i) => {
      const index =
        parseInt(hash.slice(i * 2, i * 2 + 2), 16) % validChars.length
      return validChars[index]
    })
    .join('')

  return `${shortCode}`
}
