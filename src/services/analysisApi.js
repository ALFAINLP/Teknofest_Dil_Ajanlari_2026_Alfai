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

export async function sendChatMessage({ message, analysisContext = null, history = [] }, { signal } = {}) {
  let response
  try {
    response = await fetch(`${APP_CONFIG.apiBaseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        analysis_context: analysisContext,
        history,
      }),
      signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new AnalysisApiError('Chat servisine bağlanılamadı. Backend ve Llama servislerini kontrol edin.')
  }

  if (!response.ok) {
    let detail = ''
    try {
      const payload = await response.json()
      detail = payload.detail ? `: ${payload.detail}` : ''
    } catch {
      // Sunucu JSON döndürmediyse durum kodunu göstermek yeterlidir.
    }
    throw new AnalysisApiError(`Chat sunucu hatası ${response.status}${detail}`, response.status)
  }

  const payload = await response.json()
  if (!payload.reply || typeof payload.reply !== 'string') {
    throw new AnalysisApiError('Chat servisi geçerli bir yanıt döndürmedi.')
  }
  return payload
}
