import { APP_CONFIG } from '@/config/appConfig'

export class AnalysisApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.name = 'AnalysisApiError'
    this.status = status
  }
}

async function readError(response, fallback) {
  try {
    const payload = await response.json()
    return payload.detail ? `${fallback}: ${payload.detail}` : fallback
  } catch {
    return fallback
  }
}

export async function analyzeVideo(file, { signal } = {}) {
  const formData = new FormData()
  formData.append('file', file)
  let response
  try {
    response = await fetch(`${APP_CONFIG.apiBaseUrl}/api/analyze-video`, { method: 'POST', body: formData, signal })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new AnalysisApiError('Backend bağlantısı kurulamadı. 8080 portundaki servisi kontrol edin.')
  }
  if (!response.ok) throw new AnalysisApiError(await readError(response, `Sunucu hatası ${response.status}`), response.status)
  return response.json()
}

export async function triageCameras(cameraEntries, { signal } = {}) {
  const formData = new FormData()
  cameraEntries.forEach(({ cameraId, file }) => {
    formData.append('files', file)
    formData.append('camera_ids', cameraId)
  })
  let response
  try {
    response = await fetch(`${APP_CONFIG.apiBaseUrl}/api/triage-cameras`, { method: 'POST', body: formData, signal })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new AnalysisApiError('Hızlı kamera triyaj servisine bağlanılamadı.')
  }
  if (!response.ok) throw new AnalysisApiError(await readError(response, `Triyaj sunucu hatası ${response.status}`), response.status)
  return response.json()
}

export async function sendChatMessage({ message, analysisContext = null, history = [] }, { signal } = {}) {
  let response
  try {
    response = await fetch(`${APP_CONFIG.apiBaseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, analysis_context: analysisContext, history }),
      signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') throw error
    throw new AnalysisApiError('Chat servisine bağlanılamadı. Backend ve Llama servislerini kontrol edin.')
  }
  if (!response.ok) throw new AnalysisApiError(await readError(response, `Chat sunucu hatası ${response.status}`), response.status)
  const payload = await response.json()
  if (!payload.reply || typeof payload.reply !== 'string') throw new AnalysisApiError('Chat servisi geçerli bir yanıt döndürmedi.')
  return payload
}
