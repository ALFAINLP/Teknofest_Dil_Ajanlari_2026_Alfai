import { APP_CONFIG } from '@/config/appConfig'

export class AnalysisApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.name = 'AnalysisApiError'
    this.status = status
  }
}

export async function analyzeVideo(file, { signal } = {}) {
  const formData = new FormData()
  formData.append('file', file)

  let response
  try {
    response = await fetch(`${APP_CONFIG.apiBaseUrl}/api/analyze-video`, {
      method: 'POST',
      body: formData,
      signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new AnalysisApiError('Backend bağlantısı kurulamadı. 8000 portundaki servisi kontrol edin.')
  }

  if (!response.ok) {
    throw new AnalysisApiError(`Sunucu hatası: ${response.status}`, response.status)
  }

  return response.json()
}

