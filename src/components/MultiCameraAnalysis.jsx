import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { analyzeVideo, triageCameras } from '@/services/analysisApi'
import { cn } from '@/lib/utils'

const CAMERAS = [
  { id: 'CAM_01', zone: 'Depo Girişi' },
  { id: 'CAM_02', zone: 'Üretim Hattı A' },
  { id: 'CAM_03', zone: 'Yükleme Rampası' },
  { id: 'CAM_04', zone: 'Ana Giriş' },
]

const RISK_PRIORITY = { Kritik: 4, Yüksek: 3, Orta: 2, Düşük: 1 }
const STATUS_TEXT = {
  empty: 'Video bekleniyor', ready: 'Analize hazır', analyzing: 'Analiz ediliyor…', done: 'Tamamlandı', error: 'Hata',
}
const STATUS_COLOR = {
  empty: '#64748b', ready: '#a2c9ff', analyzing: '#ffb77d', done: '#4ade80', error: '#ffb4ab',
}
const RISK_COLOR = { Kritik: '#ff6b6b', Yüksek: '#ff9f43', Orta: '#a3e635', Düşük: '#60a5fa' }

function RiskPill({ risk }) {
  if (!risk) return null
  return (
    <span className="rounded-full border px-2.5 py-1 font-sans text-[10px] font-semibold"
      style={{ color: RISK_COLOR[risk], borderColor: `${RISK_COLOR[risk]}55`, backgroundColor: `${RISK_COLOR[risk]}14` }}>
      {risk}
    </span>
  )
}

export default function MultiCameraAnalysis() {
  const [cameraItems, setCameraItems] = useState(() => CAMERAS.map(camera => ({
    ...camera, file: null, videoUrl: null, status: 'empty', analysis: null, error: null,
  })))
  const [analyzingAll, setAnalyzingAll] = useState(false)
  const [selectedCameraId, setSelectedCameraId] = useState(null)
  const [generalSummary, setGeneralSummary] = useState('')

  function updateCamera(cameraId, patch) {
    setCameraItems(current => current.map(camera => camera.id === cameraId ? { ...camera, ...patch } : camera))
  }

  function selectVideo(cameraId, file) {
    setCameraItems(current => current.map(camera => {
      if (camera.id !== cameraId) return camera
      if (camera.videoUrl) URL.revokeObjectURL(camera.videoUrl)
      return { ...camera, file, videoUrl: URL.createObjectURL(file), status: 'ready', analysis: null, error: null }
    }))
    setSelectedCameraId(cameraId)
  }

  async function analyzeAll() {
    const queued = cameraItems.filter(camera => camera.file)
    if (queued.length === 0 || analyzingAll) return
    setAnalyzingAll(true)
    queued.forEach(camera => updateCamera(camera.id, { status: 'analyzing', analysis: null, error: null }))
    try {
      const result = await triageCameras(queued.map(camera => ({ cameraId: camera.id, file: camera.file })))
      setGeneralSummary(result.general_summary ?? '')
      const resultByCamera = Object.fromEntries((result.cameras ?? []).map(camera => [camera.camera_id, camera]))
      queued.forEach(camera => {
        const triage = resultByCamera[camera.id]
        updateCamera(camera.id, triage
          ? { status: 'done', analysis: { ...triage, events: [], actions: [], analysis_mode: 'quick_triage' }, error: null }
          : { status: 'error', error: 'Model bu kamera için triyaj sonucu üretmedi.' })
      })
    } catch (error) {
      queued.forEach(camera => updateCamera(camera.id, {
        status: 'error', error: error instanceof Error ? error.message : 'Hızlı triyaj başarısız oldu.',
      }))
    } finally {
      setAnalyzingAll(false)
    }
  }

  async function runDetailedAnalysis(camera) {
    if (!camera?.file || camera.status === 'analyzing') return
    updateCamera(camera.id, { status: 'analyzing', error: null })
    try {
      const result = await analyzeVideo(camera.file)
      updateCamera(camera.id, { status: 'done', analysis: { ...result, analysis_mode: 'detailed' }, error: null })
    } catch (error) {
      updateCamera(camera.id, { status: 'error', error: error instanceof Error ? error.message : 'Detaylı analiz başarısız oldu.' })
    }
  }

  const completed = cameraItems.filter(camera => camera.analysis)
  const sortedCameras = [...cameraItems].sort((a, b) => {
    const riskDifference = (RISK_PRIORITY[b.analysis?.risk] ?? 0) - (RISK_PRIORITY[a.analysis?.risk] ?? 0)
    return riskDifference || (b.analysis?.risk_score ?? 0) - (a.analysis?.risk_score ?? 0)
  })
  const selectedCamera = cameraItems.find(camera => camera.id === selectedCameraId)
  const totalEvents = completed.reduce((total, camera) => total + (camera.analysis?.events?.length || camera.analysis?.event_count || 0), 0)
  const highestPriorityCamera = sortedCameras.find(camera => camera.analysis)

  return (
    <div className="flex-1 min-h-0 overflow-y-auto space-y-4 pr-1 pb-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-sans text-sm font-bold uppercase tracking-widest text-[#dae3ee]">Kamera Öncelik Merkezi</h2>
          <p className="mt-0.5 font-sans text-[11px] text-[#64748b]">4 video akışı · sıralı model analizi · dinamik risk önceliği</p>
        </div>
        <Button size="sm" variant="outline" disabled={analyzingAll || cameraItems.every(camera => !camera.file)} onClick={analyzeAll}
          className="border-[#a2c9ff]/40 bg-[#1d2b45]/40 font-sans text-[11px] text-[#a2c9ff] hover:bg-[#1d2b45] disabled:opacity-40">
          {analyzingAll ? 'Sırayla analiz ediliyor…' : 'Tüm Kameraları Analiz Et'}
        </Button>
      </div>

      {completed.length > 0 && (
        <div className="rounded-xl border border-[#a2c9ff]/25 bg-[#101b27] p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="font-sans text-[9px] font-semibold uppercase tracking-widest text-[#a2c9ff]">Genel Saha Öncelik Özeti</p>
              <p className="mt-1 font-sans text-sm text-[#dae3ee]">
                {generalSummary || `${completed.length} kamera hızlı triyajdan geçti, toplam ${totalEvents} ön bulgu oluştu. ${highestPriorityCamera ? `Öncelikli akış ${highestPriorityCamera.id} — ${highestPriorityCamera.analysis.risk} risk.` : ''}`}
              </p>
            </div>
            <RiskPill risk={highestPriorityCamera?.analysis?.risk} />
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        {sortedCameras.map((camera, priorityIndex) => {
          const risk = camera.analysis?.risk
          return (
            <div key={camera.id} onClick={() => setSelectedCameraId(camera.id)}
              className={cn('relative aspect-video cursor-pointer overflow-hidden rounded-xl border bg-[#060f16] transition-colors',
                selectedCameraId === camera.id ? 'border-[#a2c9ff]/70' : risk === 'Kritik' ? 'border-[#ff6b6b]/60' : 'border-[#2a3540]')}>
              {camera.videoUrl ? (
                <video src={camera.videoUrl} className="absolute inset-0 h-full w-full bg-black object-contain" loop muted playsInline controls />
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-[#334155]">
                  <span className="text-4xl">▷</span><span className="font-sans text-xs">Video yüklenmedi</span>
                </div>
              )}
              <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#0b141c]/95 via-transparent to-black/30" />
              <div className="absolute bottom-0 left-0 right-0 flex items-end justify-between gap-3 px-4 py-3">
                <div>
                  <p className="font-mono text-[13px] font-bold text-[#a2c9ff]">{camera.id}</p>
                  <p className="font-sans text-[11px] text-[#8b919d]">{camera.zone}</p>
                </div>
                <span className="rounded-full border border-white/10 bg-black/60 px-2.5 py-1 font-sans text-[10px] font-semibold"
                  style={{ color: risk ? RISK_COLOR[risk] : STATUS_COLOR[camera.status] }}>
                  {risk ? `${risk} · puan ${camera.analysis?.risk_score ?? '—'}` : STATUS_TEXT[camera.status]}
                </span>
              </div>
              <div className="absolute left-3 top-3 flex items-center gap-2">
                {camera.analysis && <span className="rounded bg-black/75 px-2 py-1 font-mono text-[10px]" style={{ color: RISK_COLOR[risk] }}>ÖNCELİK {priorityIndex + 1}</span>}
                {camera.error && <span className="max-w-[220px] truncate rounded bg-black/80 px-2 py-1 font-sans text-[10px] text-[#ffb4ab]">{camera.error}</span>}
              </div>
              <label onClick={event => event.stopPropagation()}
                className="absolute right-3 top-3 cursor-pointer rounded-md border border-[#a2c9ff]/40 bg-black/75 px-3 py-1.5 font-sans text-[10px] font-semibold text-[#a2c9ff] hover:bg-[#1d2b45]">
                {camera.file ? 'Videoyu Değiştir' : 'Video Yükle'}
                <input type="file" accept="video/*" className="hidden"
                  onChange={event => { const file = event.target.files?.[0]; if (file) selectVideo(camera.id, file) }} />
              </label>
              {camera.status === 'analyzing' && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-[1px]">
                  <p className="font-sans text-sm font-semibold text-[#ffb77d]">Model analiz ediyor…</p>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {selectedCamera?.analysis && (
        <div className="rounded-xl border border-[#2a3540] bg-[#0d1520] p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="font-mono text-xs font-bold text-[#a2c9ff]">{selectedCamera.id} · {selectedCamera.zone}</p>
              <p className="mt-1 font-sans text-xs text-[#c0c7d4]">{selectedCamera.analysis.summary}</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-sans text-[9px] uppercase tracking-widest text-[#64748b]">
                {selectedCamera.analysis.analysis_mode === 'detailed' ? 'Detaylı ajan analizi' : 'Hızlı triyaj'}
              </span>
              <RiskPill risk={selectedCamera.analysis.risk} />
              {selectedCamera.analysis.analysis_mode !== 'detailed' && (
                <Button size="sm" variant="outline" onClick={() => runDetailedAnalysis(selectedCamera)}
                  className="border-[#ffb77d]/40 bg-transparent font-sans text-[10px] text-[#ffb77d] hover:bg-[#4d2600]/30">
                  Bu Kamerayı Detaylı Analiz Et
                </Button>
              )}
            </div>
          </div>
          <div className="mt-3 grid gap-4 lg:grid-cols-2">
            <div>
              <p className="font-sans text-[9px] uppercase tracking-widest text-[#64748b]">Tespitler</p>
              <div className="mt-2 space-y-1.5">
                {(selectedCamera.analysis.events ?? []).map((event, index) => (
                  <div key={`${event.time}-${index}`} className="rounded-lg border border-[#2a3540] bg-[#141c24] px-3 py-2 font-sans text-[11px] text-[#dae3ee]">
                    <span className="mr-2 font-mono text-[#a2c9ff]">{event.time}</span>{event.event}
                  </div>
                ))}
                {(selectedCamera.analysis.events ?? []).length === 0 && (
                  <p className="font-sans text-[11px] text-[#64748b]">Olay ayrıntıları için bu kamerayı detaylı analize gönderin.</p>
                )}
              </div>
            </div>
            <div>
              <p className="font-sans text-[9px] uppercase tracking-widest text-[#64748b]">Önerilen Aksiyonlar</p>
              <div className="mt-2 space-y-1.5">
                {(selectedCamera.analysis.actions ?? []).map((action, index) => (
                  <div key={index} className="rounded-lg border border-[#2a3540] bg-[#141c24] px-3 py-2 font-sans text-[11px] text-[#c0c7d4]">
                    {index + 1}. {action}
                  </div>
                ))}
                {(selectedCamera.analysis.actions ?? []).length === 0 && (
                  <p className="font-sans text-[11px] text-[#64748b]">Aksiyonlar detaylı ajan analizinden sonra oluşturulur.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
