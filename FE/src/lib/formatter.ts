import crypto from 'crypto'

// 유틸리티 함수들
export const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export const getSeverityColor = (severity: string) => {
    switch (severity) {
        case 'critical':
            return 'bg-red-500 text-white'
        case 'high':
            return 'bg-amber-500 text-white'
        case 'medium':
            return 'bg-blue-500 text-white'
        case 'low':
            return 'bg-green-500 text-white'
        default:
            return 'bg-gray-500 text-white'
    }
}

export const getStatusColor = (status: string) => {
    switch (status) {
        case 'Pending':
            return 'bg-gray-100 text-gray-800 hover:bg-gray-300 hover:text-gray-900 transition-colors duration-150'
        case 'Assigned':
            return 'bg-blue-100 text-blue-800 hover:bg-blue-300 hover:text-blue-900 transition-colors duration-150'
        case 'In Progress':
            return 'bg-amber-100 text-amber-800 hover:bg-amber-300 hover:text-amber-900 transition-colors duration-150'
        case 'Completed':
            return 'bg-green-100 text-green-800 hover:bg-green-300 hover:text-green-900 transition-colors duration-150'
        default:
            return 'bg-gray-100 text-gray-800 hover:bg-gray-300 hover:text-gray-900 transition-colors duration-150'
    }
}

//   UUID로부터 표시 ID 생성하는 함수
export const getDisplayId = (
    uuid: string,
    damageId: number,
    prefix: string = 'DEF',
): string => {
    const validChars = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
    const hash = crypto.createHash('sha256').update(uuid).digest('hex')

    const shortCode = Array(5)
        .fill(0)
        .map((_, i) => {
            const index =
                parseInt(hash.slice(i * 2, i * 2 + 2), 16) % validChars.length
            return validChars[index]
        })
        .join('')

    return `${prefix}-${damageId}-${shortCode}`
}