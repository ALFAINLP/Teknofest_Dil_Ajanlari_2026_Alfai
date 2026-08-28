import { useState, useEffect, useRef } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import { cn } from '@/lib/utils'
import {
  BarChart3, Target, ArrowRightLeft, RotateCcw, RotateCw, Gauge,
} from 'lucide-react'
import { APP_CONFIG } from '@/config/appConfig'
import { analyzeVideo, convertVideoToMp4, fetchKpiResults, fetchPerformance } from '@/services/analysisApi'
import { eventToLog, normalizeAnalysisEvent } from '@/features/analysis/eventMapper'
import { useOperatorFeedback } from '@/hooks/useOperatorFeedback'
import ChatAssistant from '@/components/ChatAssistant'

// ─── Sabitler ────────────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'heatmap',   label: 'Isı Haritası' },
  { id: 'tactical',  label: 'Taktiksel Raporlar' },
  { id: 'reports',   label: 'Raporlar' },
  { id: 'kpi',       label: 'KPI Analiz' },
  { id: 'json',      label: 'JSON Çıktısı' },
  { id: 'settings',  label: 'Ayarlar' },
]

const SETTINGS_STORAGE_KEY = 'alfai-dashboard-settings'
const DEFAULT_SETTINGS = {
  detectionThreshold: 72,
  notifyCritical: true,
  notifyWarning: true,
  showInfoLogs: false,
}

function readSavedSettings() {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY) || '{}') }
  } catch {
    return DEFAULT_SETTINGS
  }
}

function confidencePercent(value) {
  const numeric = Number.parseFloat(String(value ?? '').replace('%', '').replace(',', '.'))
  if (!Number.isFinite(numeric)) return 0
  return numeric <= 1 ? numeric * 100 : numeric
}

const ACTION_RISK_THRESHOLD = 75

function numericRiskScore(value) {
  const numeric = Number.parseFloat(String(value ?? '').replace('%', '').replace(',', '.'))
  if (!Number.isFinite(numeric)) return 0
  return numeric <= 1 ? numeric * 100 : numeric
}

function getPriorityActionEvents(events) {
  const riskWeight = { Kritik: 4, Yüksek: 3, Orta: 2, Düşük: 1 }
  return [...events]
    .filter(event => ['Kritik', 'Yüksek'].includes(event.risk))
    .filter(event => (event.ranked_actions?.length ?? 0) > 0)
    .sort((first, second) => {
      const riskDifference = (riskWeight[second.risk] ?? 0) - (riskWeight[first.risk] ?? 0)
      if (riskDifference !== 0) return riskDifference
      const confidenceDifference = confidencePercent(second.conf) - confidencePercent(first.conf)
      if (confidenceDifference !== 0) return confidenceDifference
      return (first.start_time ?? 0) - (second.start_time ?? 0)
    })
}

function formatTimelineTime(seconds) {
  const safeSeconds = Math.max(0, Math.round(seconds || 0))
  const minutes = Math.floor(safeSeconds / 60)
  const remainder = safeSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

function buildStrategicRecommendations(modelActions = []) {
  // Tool isimlerini filtrele: snake_case olan, Türkçe kelime içermeyen string'ler tool adıdır
  const isToolName = (s) => /^[a-z][a-z0-9_]{2,}$/.test(String(s).trim())
  
  // Sadece modelin ürettiği dinamik aksiyonları (human-readable olanları) alıyoruz
  const humanActions = modelActions.filter(action => String(action).trim() && !isToolName(action))

  return [...new Set(humanActions)]
}

// Backend/model zincirinden gelen risk_score değerini yalnızca arayüz açıklamasına dönüştürür.
function getRiskGuidance(riskScore) {
  if (riskScore == null) {
    return {
      level: null,
      headline: 'Analiz bekleniyor',
      detail: 'Video yüklendiğinde risk skoruna göre karar motoru burada dinamik öneriler üretecek.',
      color: '#414752',
    }
  }
  if (riskScore >= 95) {
    return {
      level: 'Kritik',
      headline: 'Devriye sıklığı artırılmalı',
      detail: `Risk skoru %${riskScore} — kritik seviyede. Devriye sıklığı ve müdahale kapasitesi artırılmalı.`,
      color: '#ffb4ab',
    }
  }
  if (riskScore >= 75) {
    return {
      level: 'Yüksek',
      headline: 'Hızlı operasyonel müdahale gerekli',
      detail: `Risk skoru %${riskScore} — yüksek seviyede. Olay yakından izlenmeli ve ilgili ekip bilgilendirilmeli.`,
      color: '#ffb77d',
    }
  }
  if (riskScore >= 50) {
    return {
      level: 'Orta',
      headline: 'İzleme sıklığı artırılmalı',
      detail: `Risk skoru %${riskScore} — orta seviyede. KKD ve ekipman denetimleri sıklaştırılmalı.`,
      color: '#a3e635',
    }
  }
  return {
    level: 'Düşük',
    headline: 'Stabil, standart protokol yeterli',
    detail: `Risk skoru %${riskScore} — düşük seviyede. Mevcut izleme protokolü stabil şekilde sürdürülebilir.`,
    color: '#4ade80',
  }
}

const CAMERAS = [
  { id: 'CAM_01', zone: 'Depo Girişi' },
  { id: 'CAM_02', zone: 'Üretim Hattı A' },
  { id: 'CAM_03', zone: 'Yükleme Rampası' },
  { id: 'CAM_04', zone: 'Ana Giriş' },
]

const CAMERA_RISK_PRIORITY = { Kritik: 4, Yüksek: 3, Orta: 2, Düşük: 1 }

// ─── JSON Syntax Renklendirme ─────────────────────────────────────────────────
function JsonSyntax({ json }) {
  if (!json) return <span style={{ color: '#2a3540' }}>{'// Henüz veri yok — video yükleyin'}</span>
  return json.split('\n').map((line, i) => {
    const parts = []
    let rest = line
    rest = rest.replace(/^(\s*)("[\w_]+")(\s*:\s*)/, (_, indent, key, colon) => {
      parts.push(
        <span key="i" style={{ color: 'inherit' }}>{indent}</span>,
        <span key="k" style={{ color: '#a2c9ff', fontWeight: 600 }}>{key}</span>,
        <span key="c" style={{ color: '#8b919d' }}>{colon}</span>,
      )
      return ''
    })
    if      (rest.match(/^".*?"[,]?$/))          parts.push(<span key="v" style={{ color: '#4ade80' }}>{rest}</span>)
    else if (rest.match(/^-?\d+\.?\d*[,]?$/))    parts.push(<span key="v" style={{ color: '#ffb77d' }}>{rest}</span>)
    else if (rest.match(/^(true|false|null)[,]?$/)) parts.push(<span key="v" style={{ color: '#c084fc' }}>{rest}</span>)
    else if (rest.trim())                         parts.push(<span key="v" style={{ color: '#8b919d' }}>{rest}</span>)

    return (
      <span key={i} style={{ display: 'block' }}>
        {parts.length ? parts : <span style={{ color: '#8b919d' }}>{line}</span>}
      </span>
    )
  })
}

// ─── Küçük Bileşenler ─────────────────────────────────────────────────────────
function LogLine({ time, msg, level, isDark = true }) {
  const darkStyles = {
    critical: 'border-[#ffb4ab] bg-[#93000a]/20 text-[#ffdad6]',
    warn:     'border-[#ffb77d] bg-[#4d2600]/20 text-[#ffb77d]',
    info:     'border-[#414752] bg-[#182028]    text-[#c0c7d4]',
    ok:       'border-[#4ade80] bg-[#052e16]/20 text-[#86efac]',
  }
  const lightStyles = {
    critical: 'border-red-400    bg-red-50     text-red-800',
    warn:     'border-amber-400  bg-amber-50   text-amber-800',
    info:     'border-slate-300  bg-slate-50   text-slate-700',
    ok:       'border-emerald-400 bg-emerald-50 text-emerald-800',
  }
  const darkTime = {
    critical: 'text-[#ffb4ab]', warn: 'text-[#ffb77d]',
    info:     'text-[#a2c9ff]', ok:   'text-[#4ade80]',
  }
  const lightTime = {
    critical: 'text-red-700', warn: 'text-amber-700',
    info:     'text-blue-700', ok:  'text-emerald-700',
  }
  const styles   = isDark ? darkStyles : lightStyles
  const timeColor = isDark ? darkTime  : lightTime
  return (
    <div className={cn('flex gap-2 border-l-2 rounded-sm px-3 py-2 text-[13px]', styles[level])}>
      <span className={cn('font-mono text-[11px] shrink-0 mt-0.5', timeColor[level])}>{time}</span>
      <span className="font-sans leading-snug">{msg}</span>
    </div>
  )
}

function RiskBadge({ level, isDark = true }) {
  const darkMap = {
    'Kritik': 'bg-[#93000a] text-[#ffb4ab] border-[#ffb4ab]/30',
    'Yüksek': 'bg-[#4d2600] text-[#ffb77d] border-[#ffb77d]/30',
    'Orta':   'bg-[#1a2a00] text-[#a3e635] border-[#a3e635]/30',
    'Düşük':  'bg-[#182028] text-[#c0c7d4] border-[#414752]',
  }
  const lightMap = {
    'Kritik': 'bg-red-50    text-red-700    border-red-300',
    'Yüksek': 'bg-amber-50  text-amber-700  border-amber-300',
    'Orta':   'bg-lime-50   text-lime-700   border-lime-300',
    'Düşük':  'bg-slate-100 text-slate-600  border-slate-300',
  }
  const map = isDark ? darkMap : lightMap
  return (
    <span className={cn('inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold tracking-wide', map[level] ?? map['Düşük'])}>
      {level}
    </span>
  )
}

// Aynı fiziksel bölgeye düşen yakın model koordinatlarını tek kartta toplar.
// 0.12, video karesinin yaklaşık %12'si; ayrı bölgeleri korurken üst üste
// binen model tahminlerini erişilebilir hâle getirir.
const HEATMAP_CLUSTER_DISTANCE = 0.12

function buildHeatmapClusters(events) {
  return events.reduce((clusters, event) => {
    const x = Number(event.center?.x)
    const y = Number(event.center?.y)
    const existing = clusters.find(cluster =>
      Math.hypot(cluster.x - x, cluster.y - y) <= HEATMAP_CLUSTER_DISTANCE)

    if (existing) {
      existing.events.push(event)
      const count = existing.events.length
      existing.x = ((existing.x * (count - 1)) + x) / count
      existing.y = ((existing.y * (count - 1)) + y) / count
      return clusters
    }

    clusters.push({ x, y, events: [event] })
    return clusters
  }, [])
}

// ─── Heatmap Canvas ───────────────────────────────────────────────────────────
function HeatmapView({ fullscreen = false, eventRows = [], videoSrc = null }) {
  const [riskFilter, setRiskFilter] = useState('all')
  const [focusedCluster, setFocusedCluster] = useState(null)
  const [zoom, setZoom] = useState(1)
  const [videoSize, setVideoSize] = useState({ width: 16, height: 9 })
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 })
  const viewportRef = useRef(null)

  useEffect(() => {
    if (!videoSrc) {
      setVideoSize({ width: 16, height: 9 })
      setZoom(1)
      return undefined
    }
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.src = videoSrc
    setZoom(1)
    video.onloadedmetadata = () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        setVideoSize({ width: video.videoWidth, height: video.videoHeight })
      }
    }
    return () => {
      video.onloadedmetadata = null
      video.removeAttribute('src')
      video.load()
    }
  }, [videoSrc])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return undefined
    const updateSize = () => setViewportSize({ width: viewport.clientWidth, height: viewport.clientHeight })
    updateSize()
    const observer = new ResizeObserver(updateSize)
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [])
  const riskColors = { Kritik: '#ff6b6b', Yüksek: '#ff9f43', Orta: '#a3e635', Düşük: '#60a5fa' }
  const visibleEvents = eventRows.filter(event => riskFilter === 'all' || event.risk === riskFilter)
  const locatedEvents = visibleEvents.filter(event =>
    Number.isFinite(Number(event.center?.x)) && Number.isFinite(Number(event.center?.y)) &&
    Number(event.center.x) >= 0 && Number(event.center.x) <= 1 &&
    Number(event.center.y) >= 0 && Number(event.center.y) <= 1)
  const missingLocationCount = visibleEvents.length - locatedEvents.length
  const clusters = buildHeatmapClusters(locatedEvents)
  const fitZoom = viewportSize.width > 0 && viewportSize.height > 0
    ? Math.min(1, viewportSize.width / videoSize.width, viewportSize.height / videoSize.height)
    : 1
  const minimumZoom = Math.min(0.5, fitZoom)
  const canvasWidth = Math.max(1, videoSize.width * zoom)
  const canvasHeight = Math.max(1, videoSize.height * zoom)
  const canvasMarginLeft = Math.max(0, (viewportSize.width - canvasWidth) / 2)
  const canvasMarginTop = Math.max(0, (viewportSize.height - canvasHeight) / 2)
  const points = clusters.map((cluster) => {
    const confidence = Math.max(...cluster.events.map(event =>
      Number.parseFloat(String(event.conf ?? event.confidence ?? '0').replace('%', '')) || 0))
    const highestRisk = ['Kritik', 'Yüksek', 'Orta', 'Düşük'].find(level =>
      cluster.events.some(event => event.risk === level)) || 'Düşük'
    const riskSummary = ['Kritik', 'Yüksek', 'Orta', 'Düşük']
      .map(level => ({ level, count: cluster.events.filter(event => event.risk === level).length }))
      .filter(item => item.count > 0)
      .map(item => `${item.count} ${item.level}`)
      .join(' · ')
    const riskSlices = ['Kritik', 'Yüksek', 'Orta', 'Düşük']
      .flatMap(level => cluster.events.filter(event => event.risk === level).map(() => riskColors[level]))
    const markerSlices = riskSlices.length > 0 ? riskSlices : [riskColors.Düşük]
    const sliceSize = 100 / markerSlices.length
    const markerBackground = `conic-gradient(${markerSlices.map((color, index) =>
      `${color} ${(index * sliceSize).toFixed(2)}% ${((index + 1) * sliceSize).toFixed(2)}%`).join(', ')})`
    return {
      cluster: { ...cluster, highestRisk, riskSummary, markerBackground },
      left: `${cluster.x * 100}%`,
      top: `${cluster.y * 100}%`,
      size: Math.min(190, 72 + confidence + (cluster.events.length - 1) * 14),
      color: riskColors[highestRisk],
    }
  })
  return (
    <div className={cn('relative w-full rounded-xl overflow-hidden border border-[#2a3540] bg-[#060f16]', fullscreen ? 'h-full' : 'min-h-[420px]')}>
      <div ref={viewportRef} className="absolute inset-0 overflow-auto">
        <div className="t-heatmap-bg relative shrink-0 bg-[#060f16]"
          style={{
            width: canvasWidth,
            height: canvasHeight,
            marginLeft: canvasMarginLeft,
            marginTop: canvasMarginTop,
            backgroundImage: 'linear-gradient(#1e2d3d 1px, transparent 1px), linear-gradient(90deg, #1e2d3d 1px, transparent 1px)',
            backgroundSize: `${40 * zoom}px ${40 * zoom}px`,
          }}>
      {points.map(({ cluster, ...point }, index) => (
        <div key={`${cluster.x}-${cluster.y}-${index}`}>
          <button type="button" onClick={() => setFocusedCluster(cluster)}
            className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full mix-blend-screen transition-transform hover:scale-110 focus:outline-none focus:ring-2 focus:ring-[#a2c9ff]"
            style={{ top: point.top, left: point.left, width: point.size, height: point.size, background: point.color, opacity: 0.42, filter: 'blur(18px)' }}
            aria-label={`${cluster.events.length} olay bulunan konum`} />
          <button type="button" onClick={() => setFocusedCluster(cluster)}
            className="absolute z-10 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-[#dae3ee]/70 p-[3px] shadow-lg transition-transform hover:scale-110"
            style={{ top: point.top, left: point.left, background: cluster.markerBackground }}
            aria-label={`Bu konumda ${cluster.events.length} olay var`}>
            <span className="flex h-full w-full items-center justify-center rounded-full bg-[#0d1520] font-mono text-[10px] font-bold text-[#dae3ee]">
              {cluster.events.length}
            </span>
          </button>
          <button type="button" onClick={() => setFocusedCluster(cluster)}
            className="absolute z-10 -translate-x-1/2 translate-y-5 whitespace-nowrap rounded-md border border-[#2a3540] bg-[#0d1520]/95 px-2 py-1 font-sans text-[9px] font-semibold text-[#c0c7d4] shadow-md"
            style={{ top: point.top, left: point.left }}>
            En yüksek: <span style={{ color: point.color }}>{cluster.highestRisk}</span>
          </button>
        </div>
      ))}
      {eventRows.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center font-sans text-sm text-[#414752]">
          Isı haritası için önce bir video analiz edin
        </div>
      )}
      {eventRows.length > 0 && points.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center px-8 text-center font-sans text-sm text-[#8b919d]">
          Model bu analizde güvenilir mekânsal koordinat üretmedi. Yanlış konum göstermemek için nokta çizilmedi.
        </div>
      )}
        </div>
      </div>
      <div className="absolute right-4 top-4 z-30 flex items-center overflow-hidden rounded-lg border border-[#a2c9ff]/30 bg-[#0d1520]/95 shadow-lg">
        <button type="button" onClick={() => setZoom(value => Math.max(minimumZoom, Number((value - 0.1).toFixed(2))))}
          disabled={zoom <= minimumZoom + 0.001} aria-label="Isı haritasını küçült"
          className="h-8 w-9 border-r border-[#2a3540] font-sans text-base text-[#a2c9ff] hover:bg-[#1d2b45] disabled:cursor-not-allowed disabled:text-[#414752]">−</button>
        <button type="button" onClick={() => setZoom(minimumZoom)} title="Isı haritasını video ekranı boyutuna getir"
          className="h-8 min-w-14 px-2 font-mono text-[10px] text-[#dae3ee] hover:bg-[#1d2b45]">%{Math.round(zoom * 100)}</button>
        <button type="button" onClick={() => setZoom(value => Math.min(4, Number((value + 0.1).toFixed(2))))}
          disabled={zoom >= 4} aria-label="Isı haritasını büyüt"
          className="h-8 w-9 border-l border-[#2a3540] font-sans text-base text-[#a2c9ff] hover:bg-[#1d2b45] disabled:cursor-not-allowed disabled:text-[#414752]">+</button>
      </div>
      {focusedCluster && (
        <div className="absolute left-5 top-14 z-20 max-h-[65%] w-[min(390px,calc(100%-40px))] overflow-y-auto rounded-lg border border-[#a2c9ff]/30 bg-[#0d1520]/95 p-3 shadow-xl">
          <div className="mb-2 flex items-center justify-between gap-4 border-b border-[#2a3540] pb-2">
            <div>
              <p className="font-sans text-xs font-semibold text-[#dae3ee]">Bu bölgede {focusedCluster.events.length} olay</p>
              <p className="mt-0.5 font-sans text-[10px] text-[#8b919d]">{focusedCluster.riskSummary} · En yüksek risk {focusedCluster.highestRisk}</p>
            </div>
            <button type="button" onClick={() => setFocusedCluster(null)} className="font-sans text-[11px] text-[#8b919d] hover:text-[#dae3ee]">Kapat ×</button>
          </div>
          <div className="space-y-2">
            {focusedCluster.events.map(event => (
              <div key={event.id} className="rounded-md border border-[#2a3540] bg-[#0b141c] p-2.5">
                <div className="flex items-center justify-between gap-4">
                  <span className="font-mono text-[11px] text-[#a2c9ff]">{event.time}</span>
                  <RiskBadge level={event.risk} />
                </div>
                <p className="mt-2 font-sans text-xs text-[#dae3ee]">{event.event}</p>
                <p className="mt-1 font-sans text-[10px] text-[#8b919d]">{event.zone} · Güven {event.conf}</p>
              </div>
            ))}
          </div>
        </div>
      )}
      <div className="absolute top-4 left-5">
        <p className="font-sans text-[10px] uppercase tracking-widest text-[#8b919d]">Video karesi olay konumları · {locatedEvents.length}/{visibleEvents.length} konumlu · {clusters.length} bölge</p>
        {missingLocationCount > 0 && <p className="mt-1 font-sans text-[9px] text-[#ffb77d]">{missingLocationCount} olay için konum üretilemedi</p>}
      </div>
      <div className="absolute bottom-4 left-5 glass rounded-lg p-3 space-y-1.5">
        {Object.entries(riskColors).map(([label, color]) => (
          <div key={label} className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
            <span className="font-sans text-[11px] text-[#c0c7d4]">{label}</span>
          </div>
        ))}
      </div>
      <div className="absolute bottom-4 right-5 flex max-w-[72%] flex-wrap justify-end gap-2">
        {[['all', 'Tümü'], ['Kritik', 'Kritik'], ['Yüksek', 'Yüksek'], ['Orta', 'Orta'], ['Düşük', 'Düşük']].map(([value, label]) => (
          <button key={value} type="button" onClick={() => setRiskFilter(value)} className={cn('font-sans text-[11px] px-3 py-1.5 rounded-lg border transition-colors',
            riskFilter === value ? 'bg-[#a2c9ff]/15 border-[#a2c9ff]/30 text-[#a2c9ff]'
                    : 'border-[#2a3540] text-[#8b919d] hover:border-[#a2c9ff]/30 hover:text-[#a2c9ff]')}>
            {label}
          </button>
        ))}
      </div>
    </div>
  )
}

// ─── Ajanlar Arası Hedef Takip Aktarımı (LangGraph State Transfer) ────────────
const HANDOFF_LOG_POOL = [
  { from: 'CAM_01', to: 'CAM_02', msg: 'Anomali tespit edildi — hedef kilitlendi, takip başlatıldı.',            state: 'TARGET_LOCKED'  },
  { from: 'CAM_02', to: 'CAM_03', msg: 'Hedef kadraj dışına çıktı — komşu kameraya state aktarımı yapılıyor.',    state: 'STATE_HANDOFF'  },
  { from: 'CAM_03', to: 'CAM_04', msg: 'LangGraph ajan zinciri: bounding box + track ID senkronize edildi.',      state: 'SYNC_OK'        },
  { from: 'CAM_04', to: 'CAM_01', msg: 'Takip tamamlandı — hedef güvenli bölgeye ulaştı, izleme kapatıldı.',      state: 'TRACK_CLOSED'   },
]

// ─── Kamera Grid ──────────────────────────────────────────────────────────────
function CamerasPage({ videoSrc, isDark = true }) {
  const statusMap = {
    active:  { dot: '#4ade80', label: 'Aktif',      ring: 'border-[#2a3540]'    },
    warn:    { dot: '#ffb77d', label: 'Uyarı',      ring: 'border-[#ffb77d]/40' },
    offline: { dot: '#ef4444', label: 'Çevrimdışı', ring: 'border-[#ef4444]/40' },
  }
  const [handoffLogs, setHandoffLogs] = useState([])
  const handoffIdx = useRef(0)

  useEffect(() => {
    const interval = setInterval(() => {
      const entry = HANDOFF_LOG_POOL[handoffIdx.current % HANDOFF_LOG_POOL.length]
      const now = new Date()
      const t = [now.getHours(), now.getMinutes(), now.getSeconds()].map(n => String(n).padStart(2, '0')).join(':')
      setHandoffLogs(prev => [{ id: Date.now(), time: t, ...entry }, ...prev].slice(0, 8))
      handoffIdx.current += 1
    }, 4000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex-1 flex flex-col overflow-y-auto min-h-0 gap-4">
      <div className="shrink-0 flex items-center justify-between">
        <div>
          <h2 className="t-text-base font-sans text-sm font-bold uppercase tracking-widest">Kamera Sistemi</h2>
          <p className="t-text-dim font-sans text-[11px] mt-0.5">4 kamera · 3 aktif · 1 uyarıda</p>
        </div>
        <Badge variant="outline" className="font-sans text-[10px] border-[#4ade80]/30 text-[#4ade80] bg-[#052e16]/40">● 3/4 AKTİF</Badge>
      </div>
      <div className="shrink-0 grid grid-cols-1 xl:grid-cols-2 gap-6">
        {CAMERAS.map(cam => {
          const s = statusMap[cam.status]
          const isMain = cam.id === 'CAM_04'
          return (
            <div key={cam.id} className={cn('t-cam-bg relative rounded-xl overflow-hidden border bg-[#060f16] w-full aspect-video', s.ring)}>
              <div className="scanline" />
              {isMain && videoSrc ? (
                <video src={videoSrc} className="absolute inset-0 w-full h-full object-contain bg-black" loop muted playsInline autoPlay />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#1e2d3d" strokeWidth="1.2">
                    <rect x="2" y="6" width="20" height="12" rx="2" /><path d="M10 9l5 3-5 3V9z" />
                  </svg>
                </div>
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-[#0b141c]/90 via-transparent to-transparent pointer-events-none" />
              <div className="absolute bottom-0 left-0 right-0 px-4 py-3 flex items-center justify-between">
                <div>
                  <p className="font-mono text-[13px] font-bold text-[#a2c9ff]">{cam.id}</p>
                  <p className="font-sans text-[11px] text-[#8b919d]">{cam.zone}</p>
                </div>
                <div className="flex items-center gap-1.5 glass rounded-full px-2.5 py-1">
                  <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: s.dot }} />
                  <span className="font-sans text-[10px] font-semibold" style={{ color: s.dot }}>{s.label}</span>
                </div>
              </div>
              <div className="absolute top-3 left-4 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-[#ef4444]" />
                <span className="font-mono text-[10px] text-[#ef4444]/70">REC</span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Ajanlar arası hedef takip aktarımı */}
      <div className={cn('shrink-0 min-h-[150px] rounded-xl border flex flex-col overflow-hidden',
        isDark ? 'border-[#2a3540] bg-[#0d1520]' : 'border-slate-200 bg-white')}>
        <div className={cn('shrink-0 flex items-center gap-2 px-4 py-2.5 border-b', isDark ? 'border-[#2a3540]' : 'border-slate-200')}>
          <ArrowRightLeft className={cn('w-3.5 h-3.5', isDark ? 'text-[#a2c9ff]' : 'text-blue-700')} />
          <p className={cn('font-sans text-[10px] uppercase tracking-widest', isDark ? 'text-[#414752]' : 'text-slate-400')}>
            Ajanlar Arası Hedef Takip Aktarımı · LangGraph State Transfer
          </p>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-2.5 space-y-1.5">
            {handoffLogs.length === 0 ? (
              <p className={cn('font-sans text-[11px] px-2 py-4 text-center', isDark ? 'text-[#414752]' : 'text-slate-400')}>
                Ajan zinciri senkronize — aktarım bekleniyor…
              </p>
            ) : handoffLogs.map(log => (
              <div key={log.id} className={cn('flex items-center gap-2.5 px-3 py-1.5 rounded-lg border text-[11px]',
                isDark ? 'border-[#2a3540] bg-[#141c24]' : 'border-slate-200 bg-slate-50')}>
                <span className={cn('font-mono text-[10px] shrink-0', isDark ? 'text-[#414752]' : 'text-slate-400')}>{log.time}</span>
                <span className={cn('font-mono font-bold shrink-0', isDark ? 'text-[#a2c9ff]' : 'text-blue-700')}>{log.from}</span>
                <ArrowRightLeft className={cn('w-3 h-3 shrink-0', isDark ? 'text-[#8b919d]' : 'text-slate-400')} />
                <span className={cn('font-mono font-bold shrink-0', isDark ? 'text-[#a2c9ff]' : 'text-blue-700')}>{log.to}</span>
                <span className={cn('flex-1 truncate', isDark ? 'text-[#dae3ee]' : 'text-slate-700')}>{log.msg}</span>
                <Badge variant="outline" className={cn('font-mono text-[9px] shrink-0',
                  isDark ? 'border-[#4ade80]/30 text-[#4ade80] bg-transparent' : 'border-emerald-300 text-emerald-700 bg-emerald-50')}>
                  {log.state}
                </Badge>
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}

// ─── JSON Çıktısı (Tam Ekran) ─────────────────────────────────────────────────
function JsonPage({ jsonOutput }) {
  const [copied, setCopied] = useState(false)
  function copy() { navigator.clipboard?.writeText(jsonOutput || '{}'); setCopied(true); setTimeout(() => setCopied(false), 2000) }
  const lines = (jsonOutput || '// Henüz veri yok — video yükleyin').split('\n')
  return (
    <div className="flex-1 min-h-0 flex flex-col overflow-hidden gap-4">
      <div className="shrink-0 flex items-center justify-between">
        <div>
          <h2 className="t-text-base font-sans text-sm font-bold uppercase tracking-widest">JSON Analiz Çıktısı</h2>
          <p className="t-text-dim font-sans text-[11px] mt-0.5">Model çıktısı · Ham veri · LangGraph ajanı</p>
        </div>
        <Button size="sm" variant="outline" onClick={copy}
          className="font-sans text-[11px] border-[#a2c9ff]/30 text-[#a2c9ff] bg-transparent hover:bg-[#1d2b45]">
          {copied ? '✓ Kopyalandı' : '⧉ Tümünü Kopyala'}
        </Button>
      </div>
      <div className="t-json-bg flex-1 min-h-0 flex flex-col overflow-hidden rounded-xl border border-[#2a3540] bg-[#060f16]">
        <div className="t-topbar shrink-0 flex items-center gap-3 px-4 py-2.5 bg-[#0d1520] border-b border-[#2a3540]">
          <div className="flex gap-1.5">
            <span className="w-3 h-3 rounded-full bg-[#ef4444]/50" />
            <span className="w-3 h-3 rounded-full bg-[#ffb77d]/50" />
            <span className="w-3 h-3 rounded-full bg-[#4ade80]/50" />
          </div>
          <span className="font-sans text-[11px] text-[#414752] ml-1">alfai_analysis_output.json</span>
          <div className="ml-auto">
            <span className={cn('font-sans text-[10px] font-semibold', jsonOutput ? 'text-[#4ade80]' : 'text-[#414752]')}>
              {jsonOutput ? '● Analiz mevcut' : '● Video bekleniyor'}
            </span>
          </div>
        </div>
        <div className="flex-1 min-h-0 overflow-auto overscroll-contain">
          <div className="flex min-h-full min-w-max">
            <div className="shrink-0 select-none py-5 px-3 border-r border-[#1e2d3d] text-right min-w-[3rem]">
              {lines.map((_, i) => (
                <div key={i} className="font-mono text-[12px] leading-6" style={{ color: '#2a3540' }}>{i + 1}</div>
              ))}
            </div>
            <pre className="flex-1 py-5 px-5 font-mono text-[13px] leading-6 whitespace-pre">
              <JsonSyntax json={jsonOutput} />
            </pre>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Raporlar ─────────────────────────────────────────────────────────────────
function ReportsPage({ eventRows, analysis, feedbackMap, operationOrders, isDark = true }) {
  const guidance = getRiskGuidance(analysis?.risk_score)
  const feedbackValues = eventRows.map(event => feedbackMap[event.id]).filter(Boolean)
  const disputedCount = feedbackValues.filter(value => value !== 'verified').length
  const operatorStatus = disputedCount > 0 ? `${disputedCount} tespite itiraz var` : feedbackValues.length > 0 ? 'Tespitler onaylandı' : 'Henüz değerlendirilmedi'
  return (
    <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain pr-1">
      <div className="flex flex-col gap-4 pb-5">
      <div className="shrink-0">
        <h2 className="t-text-base font-sans text-sm font-bold uppercase tracking-widest">Analiz Raporu</h2>
        <p className="t-text-dim font-sans text-[11px] mt-0.5">Olay özeti · AI karar motoru çıktısı</p>
      </div>

      <Card className={cn('shrink-0 border', isDark ? 'bg-[#141c24] border-[#2a3540]' : 'bg-white border-slate-200')}>
        <CardContent className="px-4 py-4 space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: guidance.color }} />
              <p className={cn('font-sans text-[13px] font-semibold', isDark ? 'text-[#dae3ee]' : 'text-slate-800')}>{guidance.headline}</p>
            </div>
            {analysis && (
              <span className="font-mono text-[12px] font-bold shrink-0" style={{ color: guidance.color }}>
                %{analysis.risk_score} · {analysis.risk}
              </span>
            )}
          </div>
          <p className={cn('font-sans text-[12px] leading-relaxed', isDark ? 'text-[#8b919d]' : 'text-slate-500')}>
            {analysis?.summary ?? guidance.detail}
          </p>
          {analysis?.actions?.length > 0 && (
            <ul className="space-y-1.5 pt-1">
              {analysis.actions.map((action, i) => (
                <li key={i} className={cn('flex items-start gap-2 font-sans text-[12px] leading-relaxed', isDark ? 'text-[#dae3ee]' : 'text-slate-700')}>
                  <Target className={cn('w-3 h-3 shrink-0 mt-0.5', isDark ? 'text-[#a3e635]' : 'text-emerald-600')} />
                  {action}
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      {analysis?.kpi_metrics && (
        <Card className={cn('shrink-0 border', isDark ? 'bg-[#141c24] border-[#2a3540]' : 'bg-white border-slate-200')}>
          <CardHeader className="px-4 py-3 t-border border-b">
            <CardTitle className="t-text-muted font-sans text-[11px] uppercase tracking-widest">AI KPI Metrikleri</CardTitle>
          </CardHeader>
          <CardContent className="px-4 py-4 grid grid-cols-4 gap-3">
            <div>
              <p className={cn('font-sans text-[10px] uppercase tracking-widest', isDark ? 'text-[#414752]' : 'text-slate-400')}>İşlem Durumu</p>
              <p className={cn('font-sans text-[13px] font-semibold mt-1', isDark ? 'text-[#dae3ee]' : 'text-slate-800')}>{analysis.kpi_metrics.ai_processing_status}</p>
            </div>
            <div>
              <p className={cn('font-sans text-[10px] uppercase tracking-widest', isDark ? 'text-[#414752]' : 'text-slate-400')}>Operatör Değerlendirmesi</p>
              <p className={cn('font-sans text-[12px] font-bold mt-1', disputedCount > 0 ? 'text-[#ffb77d]' : feedbackValues.length > 0 ? 'text-[#4ade80]' : 'text-[#8b919d]')}>
                {operatorStatus}
              </p>
            </div>
            <div>
              <p className={cn('font-sans text-[10px] uppercase tracking-widest', isDark ? 'text-[#414752]' : 'text-slate-400')}>Güven Skoru</p>
              <p className="font-mono text-[13px] font-bold mt-1 text-[#a2c9ff]">{analysis.kpi_metrics.confidence_score}</p>
            </div>
            <div>
              <p className={cn('font-sans text-[10px] uppercase tracking-widest', isDark ? 'text-[#414752]' : 'text-slate-400')}>Şema Doğrulaması</p>
              <p className={cn('font-mono text-[13px] font-bold mt-1', analysis.kpi_metrics.is_schema_validated ? 'text-[#4ade80]' : 'text-[#ffb4ab]')}>
                {analysis.kpi_metrics.is_schema_validated ? '✓ Geçerli' : '✗ Geçersiz'}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {Object.keys(operationOrders).length > 0 && (
        <Card className="t-surface shrink-0 border-[#4ade80]/30">
          <CardHeader className="px-4 py-3 t-border border-b">
            <CardTitle className="font-sans text-[11px] uppercase tracking-widest text-[#4ade80]">
              Kaydedilen Operasyon Emirleri · {Object.keys(operationOrders).length}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 p-3">
            {Object.values(operationOrders).map(order => (
              <div key={order.event_id} className="flex items-start justify-between gap-4 rounded-lg border border-[#2a3540] bg-[#0b141c] px-3 py-2.5">
                <div className="min-w-0">
                  <p className="font-sans text-[10px] text-[#8b919d]">{order.event_time} · {order.zone}</p>
                  <p className="mt-1 font-sans text-[12px] font-semibold text-[#dae3ee]">{order.action.label}</p>
                  <p className="mt-1 font-sans text-[10px] text-[#607086]">{order.event_description}</p>
                </div>
                <span className="shrink-0 rounded-full bg-[#052e16] px-2 py-1 font-sans text-[9px] font-bold text-[#4ade80]">✓ Emir verildi</span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {eventRows.length > 0 && (
        <Card className="t-surface shrink-0">
          <CardHeader className="px-4 py-3 t-border border-b">
            <CardTitle className="t-text-muted font-sans text-[11px] uppercase tracking-widest">Tespit Özeti · {eventRows.length} olay</CardTitle>
          </CardHeader>
          <CardContent className="p-0 max-h-48 overflow-auto">
            <Table>
              <TableHeader>
                <TableRow className="t-border border-b hover:bg-transparent">
                  {['Zaman','Bölge','Olay','Risk','Güven'].map(h => (
                    <TableHead key={h} className="t-text-dim font-sans text-[10px] uppercase tracking-widest h-8 px-3">{h}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {[...eventRows].reverse().map((row, i) => (
                  <TableRow key={i} className="t-border border-b hover:bg-[#1a2330]/50">
                    <TableCell className={cn('font-mono text-[11px] px-3 py-2', isDark ? 'text-[#a2c9ff]' : 'text-blue-700')}>{row.time}</TableCell>
                    <TableCell className="t-text-muted font-sans text-[12px] px-3 py-2">{row.zone}</TableCell>
                    <TableCell className="t-text-base font-sans text-[12px] px-3 py-2">{row.event}</TableCell>
                    <TableCell className="px-3 py-2"><RiskBadge level={row.risk} isDark={isDark} /></TableCell>
                    <TableCell className="t-text-muted font-mono text-[11px] px-3 py-2">{row.conf}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
      </div>
    </div>
  )
}

// ─── Taktiksel Raporlar ─────────────────────────────────────────────────────
function TacticalReportsPage({ eventRows, analysis, isDark }) {
  const zoneStats = {}
  eventRows.forEach(r => {
    if (!zoneStats[r.zone]) zoneStats[r.zone] = { total: 0, kritik: 0, yuksek: 0 }
    zoneStats[r.zone].total += 1
    if (r.risk === 'Kritik') zoneStats[r.zone].kritik += 1
    if (r.risk === 'Yüksek') zoneStats[r.zone].yuksek += 1
  })
  const zoneList = Object.entries(zoneStats).sort((a, b) => b[1].total - a[1].total)

  const guidance = getRiskGuidance(analysis?.risk_score)
  const recommendations = buildStrategicRecommendations(analysis?.actions ?? [])

  const now = new Date()
  const dateStr = now.toLocaleDateString('tr-TR')
  const timeStr = [now.getHours(), now.getMinutes()].map(n => String(n).padStart(2, '0')).join(':')

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl space-y-4 pb-6">
        <div className="shrink-0">
          <h2 className="t-text-base font-sans text-sm font-bold uppercase tracking-widest">Taktiksel Raporlar</h2>
          <p className="t-text-dim font-sans text-[11px] mt-0.5">Karar destek özeti · Amir brifingi · LangGraph çıkarım motoru</p>
        </div>

        <Card className={cn('border', isDark ? 'bg-[#141c24] border-[#2a3540]' : 'bg-white border-slate-200')}>
          <CardHeader className={cn('px-5 py-4 border-b flex flex-row items-center justify-between gap-3', isDark ? 'border-[#2a3540]' : 'border-slate-200')}>
            <div>
              <CardTitle className={cn('font-mono text-[13px] font-bold uppercase tracking-widest', isDark ? 'text-[#a2c9ff]' : 'text-blue-700')}>
                Taktiksel Vardiya Raporu
              </CardTitle>
              <p className={cn('font-sans text-[10px] mt-1', isDark ? 'text-[#414752]' : 'text-slate-400')}>
                Rapor No: TR-{dateStr.replace(/\./g, '')}-{timeStr.replace(':', '')} · Sadece Yetkili Personel
              </p>
            </div>
            <Badge variant="outline" className={cn('font-mono text-[10px] shrink-0', isDark ? 'border-[#ffb77d]/40 text-[#ffb77d] bg-[#4d2600]/20' : 'border-amber-300 text-amber-700 bg-amber-50')}>
              ● BRİFİNGE HAZIR
            </Badge>
          </CardHeader>

          <CardContent className="px-5 py-5 space-y-5">
            <div>
              <p className={cn('font-sans text-[10px] uppercase tracking-widest mb-2', isDark ? 'text-[#8b919d]' : 'text-slate-400')}>1. Yönetici Özeti</p>
              <p className={cn('font-sans text-[13px] leading-relaxed', isDark ? 'text-[#dae3ee]' : 'text-slate-700')}>
                {analysis?.summary ?? guidance.detail}
              </p>
            </div>

            <div>
              <p className={cn('font-sans text-[10px] uppercase tracking-widest mb-2', isDark ? 'text-[#8b919d]' : 'text-slate-400')}>2. Bölgesel Tehdit Değerlendirmesi</p>
              {zoneList.length > 0 ? (
                <div className="space-y-1.5">
                  {zoneList.map(([zone, s]) => (
                    <div key={zone} className={cn('flex items-center gap-3 px-3 py-2 rounded-lg border text-[12px]', isDark ? 'border-[#2a3540] bg-[#0b141c]' : 'border-slate-200 bg-slate-50')}>
                      <span className={cn('font-mono font-semibold flex-1', isDark ? 'text-[#a2c9ff]' : 'text-blue-700')}>{zone}</span>
                      <span className={isDark ? 'text-[#8b919d]' : 'text-slate-500'}>{s.total} olay</span>
                      <RiskBadge level={s.kritik > 0 ? 'Kritik' : s.yuksek > 0 ? 'Yüksek' : 'Düşük'} isDark={isDark} />
                    </div>
                  ))}
                </div>
              ) : (
                <p className={cn('font-sans text-[12px]', isDark ? 'text-[#414752]' : 'text-slate-400')}>Vardiya boyunca kayıtlı bölgesel olay bulunmuyor.</p>
              )}
            </div>

            <div>
              <p className={cn('font-sans text-[10px] uppercase tracking-widest mb-2', isDark ? 'text-[#8b919d]' : 'text-slate-400')}>3. Stratejik Öneriler</p>
              {recommendations.length > 0 ? (
                <ul className="space-y-2">
                  {recommendations.map((rec, i) => (
                    <li key={i} className={cn('flex items-start gap-2.5 text-[13px] leading-relaxed', isDark ? 'text-[#dae3ee]' : 'text-slate-700')}>
                      <Target className={cn('w-3.5 h-3.5 shrink-0 mt-0.5', isDark ? 'text-[#a3e635]' : 'text-emerald-600')} />
                      {rec}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className={cn('font-sans text-[12px]', isDark ? 'text-[#414752]' : 'text-slate-400')}>
                  Video yüklenip analiz tamamlandığında öneriler risk skoruna göre burada otomatik oluşturulacaktır.
                </p>
              )}
            </div>

            <div className={cn('flex items-center justify-between pt-4 border-t font-sans text-[11px]', isDark ? 'border-[#2a3540] text-[#414752]' : 'border-slate-200 text-slate-400')}>
              <span>
                Hazırlayan: LangGraph Karar Motoru v2
                {analysis && ` · Risk Skoru: %${analysis.risk_score} (${analysis.risk})`}
              </span>
              <span>{dateStr} {timeStr}</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// ─── KPI Analiz ───────────────────────────────────────────────────────────────
function KpiAnalysisPage({ benchmark, loading, error }) {
  const aggregate = benchmark?.aggregate ?? {}
  const run = benchmark?.run_summary ?? {}
  const results = benchmark?.results ?? []
  const percent = value => value == null ? '—' : `%${(Number(value) * 100).toFixed(1)}`
  const terminalRows = [
    ['Genel ağırlıklı başarı', percent(aggregate.overall_score), 'Ağırlıklı katman skoru'],
    ['Olay tespit doğruluğu (F1)', percent(aggregate.event_f1_score), `Precision ${percent(aggregate.event_precision)} · Recall ${percent(aggregate.event_recall)}`],
    ['Kritik olay yakalama oranı', percent(aggregate.critical_recall), 'Kritik GT olaylarının bulunma oranı'],
    ['Özet kalitesi', percent(aggregate.summary_score), 'ROUGE ve özlülük birleşik skoru'],
    ['Aksiyon önerisi doğruluğu', percent(aggregate.tool_jaccard_score), 'Beklenen ve seçilen aksiyon Jaccard skoru'],
    ['Risk sınıfı doğruluğu', percent(aggregate.risk_accuracy), 'Tahmin edilen risk sınıfı doğruluğu'],
    ['Ortalama işlem süresi', aggregate.processing_seconds == null ? '—' : `${Number(aggregate.processing_seconds).toFixed(1)} sn`, `RTF ${aggregate.real_time_factor == null ? '—' : Number(aggregate.real_time_factor).toFixed(2)}`],
    ['Tepe GPU / RSS belleği', aggregate.peak_gpu_gb == null ? '—' : `${Number(aggregate.peak_gpu_gb).toFixed(2)} / ${aggregate.peak_rss_gb == null ? '—' : Number(aggregate.peak_rss_gb).toFixed(2)} GB`, 'Benchmark boyunca ölçülen tepe kullanım'],
  ]

  return (
    <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain pr-1">
      <div className="mx-auto w-full max-w-6xl space-y-4 pb-8">
        <div className="sticky top-0 z-10 -mx-1 border-b border-[#2a3540] bg-[#0b141c]/95 px-1 pb-3 backdrop-blur">
          <h2 className="font-sans text-sm font-bold uppercase tracking-widest text-[#dae3ee]">KPI Analiz</h2>
          <p className="mt-0.5 font-sans text-[11px] text-[#607086]">kpi_enhanced.py · benchmark değerlendirme çıktısı</p>
        </div>

        {loading ? (
          <Card className="border-[#2a3540] bg-[#141c24]">
            <CardContent className="flex min-h-56 flex-col items-center justify-center gap-2 p-8 text-center">
              <Gauge className="h-8 w-8 text-[#2a3540]" />
              <p className="font-sans text-sm font-semibold text-[#8b919d]">KPI sonuçları yükleniyor</p>
            </CardContent>
          </Card>
        ) : error || !benchmark ? (
          <Card className="border-[#ffb4ab]/30 bg-[#141c24]">
            <CardContent className="flex min-h-56 flex-col items-center justify-center gap-2 p-8 text-center">
              <p className="font-sans text-sm font-semibold text-[#ffb4ab]">KPI sonuçları alınamadı</p>
              <p className="font-sans text-[11px] text-[#8b919d]">{error ?? 'Önce kpi_enhanced.py benchmark çalıştırılmalı.'}</p>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card className="overflow-hidden border-[#2a3540] bg-[#090f16] shadow-xl">
              <CardHeader className="border-b border-[#2a3540] bg-[#101820] px-4 py-3">
                <CardTitle className="font-mono text-[11px] font-bold uppercase tracking-wider text-[#a2c9ff]">KPI Değerlendirme Raporu</CardTitle>
              </CardHeader>
              <CardContent className="p-0 font-mono">
                <div className="border-b border-[#1e2d3d] px-4 py-2 text-[9px] text-[#4ade80]">
                  $ python kpi_enhanced.py --output benchmark_results_enhanced.json
                </div>
                <div className="divide-y divide-[#17222e] px-4 py-1">
                  {terminalRows.map(([label, value, detail]) => (
                    <div key={label} className="grid grid-cols-[minmax(210px,1fr)_110px_minmax(220px,1.4fr)] items-center gap-4 py-2.5">
                      <span className="text-[11px] text-[#c0c7d4]">• {label}</span>
                      <span className="text-right text-[12px] font-bold text-[#a2c9ff]">: {value}</span>
                      <span className="truncate text-[9px] text-[#607086]">{detail}</span>
                    </div>
                  ))}
                </div>
                <div className="border-t border-[#1e2d3d] px-4 py-2 font-mono text-[9px] text-[#4ade80]">
                  ✓ {run.videos_evaluated ?? results.length}/{run.videos_total ?? results.length} video değerlendirildi · {run.videos_failed ?? 0} başarısız
                </div>
              </CardContent>
            </Card>

            <Card className="overflow-hidden border-[#2a3540] bg-[#090f16]">
              <CardHeader className="border-b border-[#2a3540] bg-[#101820] px-4 py-3">
                <CardTitle className="font-mono text-[11px] font-bold uppercase tracking-wider text-[#a2c9ff]">Video Bazlı KPI Sonuçları · {results.length}</CardTitle>
              </CardHeader>
              <CardContent className="overflow-x-auto p-0">
                <table className="w-full min-w-[760px] table-fixed border-collapse font-mono">
                  <thead className="bg-[#0d151e]">
                    <tr className="h-10 border-b border-[#2a3540]">
                      <th className="w-44 px-4 text-left text-[9px] font-semibold uppercase tracking-wider text-[#a2c9ff]">Video</th>
                      {['Genel','F1','Kritik','Özet','Aksiyon','Risk'].map(label => (
                        <th key={label} className="w-20 px-2 text-center text-[9px] font-semibold uppercase tracking-wider text-[#a2c9ff]">{label}</th>
                      ))}
                      <th className="w-24 px-3 text-right text-[9px] font-semibold uppercase tracking-wider text-[#a2c9ff]">Süre</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((row, rowIndex) => (
                      <tr key={row.video_id} className={cn('h-10 border-b border-[#1e2d3d] transition-colors hover:bg-[#14202c]', rowIndex % 2 === 1 && 'bg-[#0c141c]')}>
                        <td className="w-44 overflow-hidden text-ellipsis whitespace-nowrap px-4 text-[10px] font-semibold text-[#a2c9ff]">{row.video_id}</td>
                        {[row.overall_score,row.event_f1_score,row.critical_recall,row.summary_score,row.tool_jaccard_score,row.risk_accuracy].map((value, index) => (
                          <td key={index} className="w-20 whitespace-nowrap px-2 text-center text-[10px] font-semibold text-[#dae3ee]">{percent(value)}</td>
                        ))}
                        <td className="w-24 whitespace-nowrap px-3 text-right text-[10px] font-semibold text-[#4ade80]">{row.processing_seconds == null ? '—' : `${Number(row.processing_seconds).toFixed(1)} sn`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  )
}

// ─── Ayarlar ──────────────────────────────────────────────────────────────────
function SettingsPage({ settings, onChange }) {
  const { detectionThreshold, notifyCritical, notifyWarning, showInfoLogs } = settings

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl space-y-6 pb-6">
        <div className="shrink-0">
          <h2 className="t-text-base font-sans text-sm font-bold uppercase tracking-widest">Sistem Ayarları</h2>
          <p className="t-text-dim font-sans text-[11px] mt-0.5">Model konfigürasyonu · Bildirimler · Analiz parametreleri</p>
        </div>

        {/* Ajan Analiz Hassasiyeti */}
        <Card className="t-surface bg-[#141c24] border-[#2a3540]">
          <CardHeader className="px-5 py-4 t-border border-b border-[#2a3540]">
            <CardTitle className="t-text-muted font-sans text-[11px] uppercase tracking-widest text-[#8b919d]">Ajan Analiz Hassasiyeti</CardTitle>
          </CardHeader>
          <CardContent className="px-5 py-5 space-y-4">
            <div className="flex items-center justify-between mb-1">
              <span className="t-text-base font-sans text-[13px] text-[#dae3ee]">Tespit Eşiği</span>
              <span className="font-mono text-[13px] font-bold text-[#a2c9ff]">%{detectionThreshold}</span>
            </div>
            <Slider
              value={[detectionThreshold]}
              onValueChange={([value]) => onChange({ detectionThreshold: value })}
              min={10} max={99} step={1}
              className="w-full"
            />
            <div className="flex justify-between font-sans text-[10px] t-text-dim text-[#414752]">
              <span>Daha fazla tespit</span>
              <span>Dengeli</span>
              <span>Daha seçici</span>
            </div>
            <p className="font-sans text-[11px] t-text-dim text-[#414752] mt-2">
              Güven puanı bu değerin altında kalan olaylar sonuçlara alınmaz. Düşük eşik daha fazla olay, yüksek eşik daha seçici sonuç üretir.
            </p>
          </CardContent>
        </Card>

        {/* Bildirim Tercihleri */}
        <Card className="t-surface bg-[#141c24] border-[#2a3540]">
          <CardHeader className="px-5 py-4 t-border border-b border-[#2a3540]">
            <CardTitle className="t-text-muted font-sans text-[11px] uppercase tracking-widest text-[#8b919d]">Bildirim Tercihleri</CardTitle>
          </CardHeader>
          <CardContent className="px-5 py-4 space-y-0 divide-y divide-[#2a3540]">
            {[
              { key: 'notifyCritical', label: 'Kritik Alarm Bildirimleri', desc: 'Kritik olayları ekranda anlık bildir', checked: notifyCritical, dot: '#ffb4ab' },
              { key: 'notifyWarning',  label: 'Uyarı Bildirimleri', desc: 'Yüksek ve orta riskli olayları ekranda bildir', checked: notifyWarning, dot: '#ffb77d' },
              { key: 'showInfoLogs',   label: 'Bilgi Günlüğü', desc: 'Düşük riskli bilgi loglarını operasyonel logda göster', checked: showInfoLogs, dot: '#a2c9ff' },
            ].map(({ key, label, desc, checked, dot }) => (
              <div key={label} className="flex items-center justify-between py-3.5">
                <div className="flex items-center gap-3">
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: dot }} />
                  <div>
                    <p className="t-text-base font-sans text-[13px] text-[#dae3ee]">{label}</p>
                    <p className="t-text-dim font-sans text-[11px] text-[#8b919d]">{desc}</p>
                  </div>
                </div>
                <Switch checked={checked} onCheckedChange={(value) => onChange({ [key]: value })} />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// ─── Video Analiz Görünümü ────────────────────────────────────────────────────
function RankedActionButtons({ event, selectedRank, savedOrder, onSelect, onSave, compact = false }) {
  const actions = event?.ranked_actions ?? []
  if (actions.length === 0) return <span className="font-sans text-[10px] text-[#414752]">Öneri bulunamadı</span>
  const isSaved = Boolean(selectedRank && savedOrder?.action?.rank === selectedRank)

  return (
    <div className={cn('flex gap-2', compact ? 'min-w-[260px] flex-col' : 'mt-3 flex-col lg:grid lg:grid-cols-3')}>
      {actions.map(action => (
        <button key={`${event.id}-${action.rank}-${action.label}`} type="button" title={action.reason}
          onClick={() => onSelect(event, action)}
          className={cn('rounded-lg border px-3 py-2 text-left transition-colors',
            selectedRank === action.rank
              ? 'border-[#4ade80] bg-[#052e16]/70 text-[#86efac] ring-1 ring-[#4ade80]/30'
              : action.rank === 1
              ? 'border-[#ffb77d]/50 bg-[#4d2600]/20 text-[#ffca9f] hover:bg-[#4d2600]/40'
              : 'border-[#2a3540] bg-[#0b141c] text-[#c0c7d4] hover:border-[#a2c9ff]/40')}>
          <span className="block font-sans text-[9px] font-bold uppercase tracking-widest opacity-70">
            {action.rank}. {action.rank === 1 ? 'En acil' : action.rank === 2 ? 'Alternatif' : 'Uzun vadeli'}
          </span>
          <span className="mt-1 block font-sans text-[11px] font-semibold">{action.label}</span>
          {!compact && <span className="mt-1 block font-sans text-[10px] leading-relaxed opacity-70">{action.reason}</span>}
        </button>
      ))}
      <button type="button" disabled={!selectedRank || isSaved}
        onClick={() => onSave(event)}
        className={cn(
          'rounded-lg border px-3 py-2 text-center font-sans text-[10px] font-bold transition-colors',
          selectedRank && !isSaved
            ? 'border-[#4ade80]/60 bg-[#052e16]/50 text-[#86efac] hover:bg-[#14532d]/60'
            : isSaved
              ? 'cursor-default border-[#4ade80]/30 bg-[#052e16]/30 text-[#4ade80]'
              : 'cursor-not-allowed border-[#2a3540] text-[#414752]',
        )}>
        {isSaved ? '✓ Operasyon emri kaydedildi' : 'Seçili Emri Kaydet'}
      </button>
    </div>
  )
}

function ActionCenter({ open, events, actionDrafts, operationOrders, onSelect, onSave, onClose }) {
  const pendingCount = events.filter(event => !operationOrders[event.id]).length

  useEffect(() => {
    if (!open) return undefined
    function closeOnEscape(event) {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [open, onClose])

  if (!open) return null
  return (
    <aside role="dialog" aria-label="Aksiyon Merkezi"
      className="fixed inset-y-0 right-0 z-50 flex w-[420px] max-w-[92vw] flex-col border-l border-[#2a3540] bg-[#0d1520] shadow-2xl">
      <div className="flex shrink-0 items-start justify-between border-b border-[#2a3540] p-4">
        <div>
          <p className="font-sans text-[11px] font-bold uppercase tracking-widest text-[#a2c9ff]">Aksiyon Merkezi</p>
          <p className="mt-1 font-sans text-[10px] text-[#8b919d]">{events.length} olay · {pendingCount} bekleyen emir</p>
        </div>
        <button type="button" onClick={onClose} aria-label="Aksiyon merkezini kapat"
          className="rounded-lg border border-[#2a3540] px-3 py-1.5 font-sans text-[11px] text-[#8b919d] hover:border-[#a2c9ff]/40 hover:text-white">
          Kapat ×
        </button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto overscroll-contain p-4">
        {events.map(event => (
          <section key={event.id} className={cn('rounded-xl border p-3', operationOrders[event.id] ? 'border-[#4ade80]/30 bg-[#052e16]/10' : 'border-[#2a3540] bg-[#101b27]')}>
            <div className="mb-3 flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="font-mono text-[9px] text-[#8b919d]">{event.time} · {event.zone}</p>
                <p className="mt-1 font-sans text-[12px] font-semibold leading-relaxed text-[#dae3ee]">{event.event}</p>
              </div>
              <RiskBadge level={event.risk} />
            </div>
            <RankedActionButtons event={event}
              selectedRank={actionDrafts[event.id]}
              savedOrder={operationOrders[event.id]}
              onSelect={onSelect}
              onSave={onSave}
              compact />
          </section>
        ))}
      </div>
    </aside>
  )
}

function PerformancePanel({ analyzing, elapsedSeconds, analysis, telemetry }) {
  const finalPerformance = analysis?.performance
  const gpu = telemetry ?? finalPerformance?.gpu
  if (!analyzing && !analysis) return null
  const totalSeconds = analyzing ? elapsedSeconds : finalPerformance?.total_processing_seconds
  const inferenceMs = finalPerformance?.vision_inference_ms
  const usedGb = gpu?.available ? (gpu.vram_used_mb / 1024).toFixed(1) : null
  const totalGb = gpu?.available ? (gpu.vram_total_mb / 1024).toFixed(1) : null

  return (
    <div className="mx-3 mb-3 shrink-0 rounded-xl border border-[#2a3540] bg-[#0b141c] p-3 shadow-lg">
      <div className="mb-3 flex items-center justify-between">
        <p className="font-sans text-[9px] font-bold uppercase tracking-widest text-[#a2c9ff]">Performans Paneli</p>
        <span className={cn('rounded-full px-2 py-0.5 font-sans text-[9px] font-bold', analyzing ? 'bg-[#052e16] text-[#4ade80]' : 'bg-[#1d2b45] text-[#a2c9ff]')}>
          {analyzing ? 'CANLI' : 'TAMAMLANDI'}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="min-w-0 rounded-lg bg-[#101b27] p-2"><p className="font-sans text-[7px] uppercase tracking-wider text-[#607086]">Toplam süre</p><p className="mt-1 whitespace-nowrap font-mono text-[12px] font-bold text-[#dae3ee]">{totalSeconds != null ? `${Number(totalSeconds).toFixed(1)} sn` : '—'}</p></div>
        <div className="min-w-0 rounded-lg bg-[#101b27] p-2"><p className="font-sans text-[7px] uppercase tracking-wider text-[#607086]">Model inference</p><p className="mt-1 whitespace-nowrap font-mono text-[11px] font-bold text-[#dae3ee]">{inferenceMs != null ? `${Number(inferenceMs).toFixed(0)} ms` : analyzing ? 'Ölçülüyor…' : '—'}</p></div>
        <div className="min-w-0 rounded-lg bg-[#101b27] p-2"><p className="font-sans text-[7px] uppercase tracking-wider text-[#607086]">GPU kullanımı</p><p className="mt-1 whitespace-nowrap font-mono text-[12px] font-bold text-[#4ade80]">{gpu?.available ? `%${gpu.gpu_utilization_percent}` : 'Yok'}</p></div>
        <div className="min-w-0 rounded-lg bg-[#101b27] p-2"><p className="font-sans text-[7px] uppercase tracking-wider text-[#607086]">VRAM</p><p className="mt-1 whitespace-nowrap font-mono text-[10px] font-bold text-[#ffb77d]">{gpu?.available ? `${usedGb}/${totalGb}` : '—'}</p>{gpu?.available && <p className="mt-0.5 whitespace-nowrap font-mono text-[8px] text-[#607086]">GB · %{gpu.vram_utilization_percent}</p>}</div>
      </div>
    </div>
  )
}

function VideoView({ videoSrc, fileName, onFileSelect, onDeleteVideo, analyzing, eventRows, selectedEvent, onFeedback, feedbackMap, actionDrafts, operationOrders, onActionSelect, onOperationSave }) {
  const fileInputRef = useRef(null)
  const videoRef     = useRef(null)
  const [videoDuration, setVideoDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [feedbackDraft, setFeedbackDraft] = useState(null)
  const [feedbackSaved, setFeedbackSaved] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1)

  const lastEventTime = Math.max(...eventRows.map(event => event.end_time ?? event.start_time ?? 0), 0)
  const timelineDuration = Math.max(videoDuration, lastEventTime, 1)
  const timelineTicks = Array.from({ length: 5 }, (_, index) => timelineDuration * index / 4)

  useEffect(() => {
    if (videoSrc && videoRef.current) {
      setVideoDuration(0)
      setCurrentTime(0)
      videoRef.current.load()
      videoRef.current.playbackRate = playbackRate
      videoRef.current.play().catch(() => {})
    }
  }, [videoSrc])

  useEffect(() => {
    if (!selectedEvent || !videoRef.current) return
    videoRef.current.currentTime = selectedEvent.start_time ?? 0
    videoRef.current.play().catch(() => {})
  }, [selectedEvent])

  useEffect(() => {
    setFeedbackDraft(selectedEvent ? (feedbackMap[selectedEvent.id] ?? null) : null)
    setFeedbackSaved(false)
  }, [selectedEvent?.id])

  return (
    <div className="space-y-3 w-full">
      {/* Video */}
      <div className="t-video-bg relative w-full rounded-xl overflow-hidden border border-[#2a3540] bg-[#060f16]" style={{ height: 560 }}>
        <div className="scanline" />

        {videoSrc ? (
          <video ref={videoRef} src={videoSrc}
            className="w-full h-full object-contain bg-black"
            controls loop playsInline
            onLoadedMetadata={event => setVideoDuration(event.currentTarget.duration || 0)}
            onTimeUpdate={event => setCurrentTime(event.currentTarget.currentTime || 0)} />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center gap-3">
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#1e2d3d" strokeWidth="1.2">
              <rect x="2" y="6" width="20" height="12" rx="2" /><path d="M10 9l5 3-5 3V9z" />
            </svg>
            <p className="font-sans text-[12px] text-[#414752]">Video Bekleniyor</p>
            <p className="font-sans text-[10px] text-[#2a3540]">↑ Video Yükle butonuna basın</p>
          </div>
        )}

        {/* Video Yükle — sağ üst, controls ile çakışmasın */}
        <div className="absolute top-3 right-3 flex items-center gap-2 pointer-events-auto z-10">
          {videoSrc && (
            <span className="font-sans text-[10px] font-semibold text-[#4ade80] bg-black/70 px-2 py-1 rounded border border-[#4ade80]/20 max-w-[160px] truncate">
              ▶ {fileName}
            </span>
          )}
          <Input ref={fileInputRef} type="file" accept=".mp4,.avi,video/mp4,video/x-msvideo" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) onFileSelect(f) }} />
          <Button size="sm" variant="outline"
            onClick={() => fileInputRef.current?.click()}
            className="h-7 px-3 font-sans text-[11px] font-semibold
              bg-black/70 backdrop-blur border-[#a2c9ff]/50 text-[#a2c9ff]
              hover:bg-[#1d2b45] hover:text-white transition-colors">
            ↑ Video Yükle
          </Button>
          {videoSrc && (
            <Button size="sm" variant="outline"
              title="Videoyu ve mevcut analiz sonucunu temizle"
              onClick={() => {
                onDeleteVideo()
                if (fileInputRef.current) fileInputRef.current.value = ''
              }}
              className="h-7 border-[#ffb4ab]/50 bg-black/70 px-3 font-sans text-[11px] font-semibold text-[#ffb4ab] hover:bg-[#35191d]">
              Videoyu Sil
            </Button>
          )}
        </div>

        {/* Alt HUD — sadece video yokken */}
        {!videoSrc && (
          <div className="absolute bottom-3 left-3 flex gap-3">
            <div className="glass px-3 py-2 rounded-lg">
              <p className="font-sans text-[9px] uppercase tracking-widest text-[#8b919d] mb-0.5">Vardiya</p>
              <p className="font-sans text-sm font-semibold">08:00 – 16:00</p>
            </div>
            <div className="glass border-l-2 border-l-[#a2c9ff] px-3 py-2 rounded-lg">
              <p className="font-sans text-[9px] uppercase tracking-widest text-[#8b919d] mb-0.5">Bölge</p>
              <p className="font-sans text-sm font-semibold">Ana Giriş</p>
            </div>
          </div>
        )}

        {/* Kritik olay sayacı — video oynarken */}
        {videoSrc && eventRows.some(r => r.risk === 'Kritik') && (
          <div className="absolute bottom-16 left-3 glass border-l-2 border-l-[#ffb4ab] px-3 py-2 rounded-lg">
            <p className="font-sans text-[9px] uppercase tracking-widest text-[#8b919d] mb-0.5">Kritik Olay</p>
            <p className="font-sans text-sm font-bold text-[#ffb4ab]">{eventRows.filter(r => r.risk === 'Kritik').length} Tespit</p>
          </div>
        )}
      </div>

      {videoSrc && (
        <div className="flex min-h-12 items-center justify-between gap-3 rounded-xl border border-[#2a3540] bg-gradient-to-r from-[#0e1823] to-[#101b27] px-3 py-2 shadow-sm">
          <div className="flex min-w-0 items-center gap-1.5">
            <button type="button" title="5 saniye geri" aria-label="Videoyu 5 saniye geri sar"
              onClick={() => { if (videoRef.current) videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime - 5) }}
              className="relative flex h-8 w-9 shrink-0 items-center justify-center rounded-lg border border-[#2a3540] bg-[#0b141c] text-[#c0c7d4] transition-colors hover:border-[#a2c9ff]/50 hover:text-[#a2c9ff]">
              <RotateCcw className="h-4 w-4" /><span className="absolute text-[7px] font-bold">5</span>
            </button>
            <div className="min-w-[76px] whitespace-nowrap text-center font-mono text-[9px] text-[#a2c9ff]">
              {formatTimelineTime(currentTime)} <span className="text-[#607086]">/</span> {formatTimelineTime(videoDuration)}
            </div>
            <button type="button" title="5 saniye ileri" aria-label="Videoyu 5 saniye ileri sar"
              onClick={() => { if (videoRef.current) videoRef.current.currentTime = Math.min(videoDuration || Infinity, videoRef.current.currentTime + 5) }}
              className="relative flex h-8 w-9 shrink-0 items-center justify-center rounded-lg border border-[#2a3540] bg-[#0b141c] text-[#c0c7d4] transition-colors hover:border-[#a2c9ff]/50 hover:text-[#a2c9ff]">
              <RotateCw className="h-4 w-4" /><span className="absolute text-[7px] font-bold">5</span>
            </button>
          </div>
          <label className="flex h-8 shrink-0 items-center gap-1.5 rounded-lg border border-[#2a3540] bg-[#0b141c] px-2 text-[#607086]">
            <Gauge className="h-3.5 w-3.5" />
            <span className="font-sans text-[8px] font-semibold uppercase tracking-wider">Hız</span>
            <select value={playbackRate} onChange={event => {
                  const rate = Number(event.target.value)
                  setPlaybackRate(rate)
                  if (videoRef.current) videoRef.current.playbackRate = rate
                }}
              aria-label="Video oynatma hızı"
              className="cursor-pointer bg-transparent font-mono text-[10px] font-bold text-[#a2c9ff] outline-none">
              {[0.5, 1, 1.5, 2].map(rate => <option key={rate} value={rate} className="bg-[#0b141c] text-white">{rate}x</option>)}
            </select>
          </label>
        </div>
      )}

      {selectedEvent && (
        <Card className="t-surface border-[#a2c9ff]/30 bg-[#101b27]">
          <CardContent className="p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="font-sans text-[9px] font-semibold uppercase tracking-widest text-[#a2c9ff]">Kanıt Zinciri</p>
                <h3 className="mt-1 font-sans text-sm font-semibold text-[#dae3ee]">{selectedEvent.event}</h3>
                <p className="mt-1 font-mono text-[10px] text-[#8b919d]">
                  {selectedEvent.time} · {selectedEvent.end_time_label ?? 'kritik an'} · Güven {selectedEvent.conf}
                </p>
              </div>
              <RiskBadge level={selectedEvent.risk} />
            </div>

            <div className="mt-3 grid gap-3 lg:grid-cols-2">
              <div className="space-y-1.5">
                {(selectedEvent.evidence ?? []).map((item, index) => (
                  <button key={`${item.time}-${index}`} type="button"
                    onClick={() => {
                      if (!videoRef.current) return
                      videoRef.current.currentTime = item.seconds ?? selectedEvent.start_time ?? 0
                      videoRef.current.play().catch(() => {})
                    }}
                    className="flex w-full items-start gap-2 rounded-lg border border-[#2a3540] bg-[#0b141c] px-3 py-2 text-left transition-colors hover:border-[#a2c9ff]/40">
                    <span className="font-mono text-[10px] text-[#a2c9ff]">{item.time}</span>
                    <span className="font-sans text-[11px] leading-relaxed text-[#c0c7d4]">{item.description}</span>
                  </button>
                ))}
              </div>
              <div className="rounded-lg border border-[#ffb77d]/20 bg-[#4d2600]/10 p-3">
                <p className="font-sans text-[9px] font-semibold uppercase tracking-widest text-[#ffb77d]">Belirsizlik Notu</p>
                <p className="mt-2 font-sans text-[11px] leading-relaxed text-[#c0c7d4]">
                  {(selectedEvent.uncertainties ?? ['Ek bir belirsizlik bildirilmedi.']).join(' ')}
                </p>
              </div>
            </div>

            <div className="mt-3 border-t border-[#2a3540] pt-3">
              <p className="font-sans text-[9px] font-semibold uppercase tracking-widest text-[#a2c9ff]">Sıralı Aksiyon Önerileri</p>
              <RankedActionButtons event={selectedEvent}
                selectedRank={actionDrafts[selectedEvent.id]}
                savedOrder={operationOrders[selectedEvent.id]}
                onSelect={onActionSelect}
                onSave={onOperationSave} />
            </div>

            <div className="mt-3 border-t border-[#2a3540] pt-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="mr-1 font-sans text-[9px] uppercase tracking-widest text-[#414752]">Operatör doğrulaması</span>
                {[
                  ['verified', 'Doğru tespit'],
                  ['false_alarm', 'Yanlış alarm'],
                  ['wrong_risk', 'Risk yanlış'],
                  ['wrong_time', 'Zaman yanlış'],
                ].map(([value, label]) => (
                  <button key={value} type="button" onClick={() => { setFeedbackDraft(value); setFeedbackSaved(false) }}
                    className={cn(
                      'rounded-md border px-2.5 py-1 font-sans text-[10px] transition-colors',
                      feedbackDraft === value
                        ? 'border-[#a2c9ff]/60 bg-[#a2c9ff]/10 text-[#a2c9ff]'
                        : 'border-[#2a3540] text-[#8b919d] hover:border-[#a2c9ff]/40 hover:text-[#dae3ee]'
                    )}>
                    {label}
                  </button>
                ))}
              </div>
              <div className="mt-4 flex items-center justify-end gap-3">
                {feedbackSaved && <span className="font-sans text-[10px] text-[#4ade80]">✓ Değerlendirme kaydedildi</span>}
                <button type="button" disabled={!feedbackDraft}
                  onClick={() => {
                    if (!feedbackDraft) return
                    onFeedback(selectedEvent, feedbackDraft)
                    setFeedbackSaved(true)
                  }}
                  className={cn(
                    'min-w-[190px] rounded-lg border px-5 py-2 font-sans text-[11px] font-semibold transition-colors',
                    feedbackDraft
                      ? 'border-[#4ade80]/50 bg-[#052e16]/50 text-[#86efac] hover:bg-[#14532d]/50'
                      : 'cursor-not-allowed border-[#2a3540] text-[#414752]'
                  )}>
                  Değerlendirmeyi Kaydet
                </button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Zaman çizelgesi */}
      <Card className="t-surface bg-[#141c24] border-[#2a3540]">
        <CardContent className="p-4">
          <div className="flex justify-between items-center mb-3">
            <span className="font-sans text-[11px] uppercase tracking-widest text-[#8b919d]">Olay Zaman Çizelgesi · Model Tespitleri</span>
            <span className="font-sans text-[11px] t-text-dim text-[#414752]">{eventRows.length} olay</span>
          </div>
          <div className="relative h-10 bg-[#0b141c] rounded-lg overflow-hidden border border-[#2a3540] flex items-center">
            <div className="absolute inset-0 flex justify-between px-3 pointer-events-none opacity-10">
              {Array.from({ length: 8 }).map((_, i) => <span key={i} className="h-full w-px bg-white" />)}
            </div>
            {eventRows.length > 0 ? (
              <>
                {eventRows.map((r) => ({
                  id: r.id,
                  seconds: r.start_time ?? 0,
                  left: `${Math.min(100, Math.max(0, ((r.start_time ?? 0) / timelineDuration) * 100))}%`,
                  color: r.risk === 'Kritik' ? '#ffb4ab' : r.risk === 'Yüksek' ? '#ffb77d' : '#a2c9ff',
                  label: r.event, pulse: r.risk === 'Kritik',
                })).map(({ id, seconds, left, color, label, pulse }) => (
                  <button key={id} type="button" title={`${formatTimelineTime(seconds)} · ${label}`}
                    onClick={() => {
                      if (!videoRef.current) return
                      videoRef.current.currentTime = seconds
                      videoRef.current.play().catch(() => {})
                    }}
                    className={cn('absolute -translate-x-1/2 w-3 h-3 rounded-full border-2 border-[#0b141c]', pulse && 'animate-pulse')}
                    style={{ left, backgroundColor: color }} />
                ))}
                <div
                  className="absolute h-full w-px bg-[#a2c9ff] transition-[left] duration-100"
                  style={{ left: `${Math.min(100, (currentTime / timelineDuration) * 100)}%`, boxShadow: '0 0 6px #a2c9ff' }}
                />
              </>
            ) : (
              <p className="w-full text-center font-sans text-[10px] text-[#414752]">Henüz analiz verisi yok</p>
            )}
          </div>
          <div className="flex justify-between font-sans text-[10px] t-text-dim text-[#414752] mt-2">
            {timelineTicks.map((seconds, index) => <span key={index}>{formatTimelineTime(seconds)}</span>)}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// ─── İkon Bileşenleri ─────────────────────────────────────────────────────────
// ─── Ana Bileşen ──────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [activeNav, setActiveNav]         = useState('dashboard')
  const [dashboardTab, setDashboardTab]   = useState('video')
  const [logs, setLogs]                   = useState([])
  const [eventRows, setEventRows]         = useState([])
  const [selectedEvent, setSelectedEvent] = useState(null)
  const { feedbackMap, saveFeedback, clearFeedback } = useOperatorFeedback()
  const [tick, setTick]                   = useState(0)
  const [videoSrc, setVideoSrc]           = useState(null)
  const [videoFileName, setVideoFileName] = useState(null)
  const isDark                            = true
  const [analyzing, setAnalyzing]         = useState(false)
  const [analysis, setAnalysis]           = useState(null)
  const [analysisError, setAnalysisError] = useState(null)
  const [settings, setSettings]           = useState(readSavedSettings)
  const [alerts, setAlerts]               = useState([])
  const [serviceStatus, setServiceStatus] = useState('checking')
  const [livePerformance, setLivePerformance] = useState(null)
  const [analysisElapsed, setAnalysisElapsed] = useState(0)
  const [operationNotice, setOperationNotice] = useState(null)
  const [actionDrafts, setActionDrafts] = useState({})
  const [operationOrders, setOperationOrders] = useState({})
  const [actionCenterOpen, setActionCenterOpen] = useState(false)
  const [kpiBenchmark, setKpiBenchmark] = useState(null)
  const [kpiLoading, setKpiLoading] = useState(false)
  const [kpiError, setKpiError] = useState(null)
  const eventIdCounter = useRef(1)
  const analysisStartedAt = useRef(null)
  const analysisAbortRef = useRef(null)
  const actionCenterTimerRef = useRef(null)

  useEffect(() => {
    document.documentElement.classList.remove('light')
  }, [])

  useEffect(() => {
    if (activeNav !== 'kpi') return undefined
    const controller = new AbortController()
    setKpiLoading(true)
    setKpiError(null)
    fetchKpiResults({ signal: controller.signal })
      .then(setKpiBenchmark)
      .catch(error => {
        if (error?.name !== 'AbortError') setKpiError(error instanceof Error ? error.message : 'KPI sonuçları alınamadı.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setKpiLoading(false)
      })
    return () => controller.abort()
  }, [activeNav])

  useEffect(() => {
    let cancelled = false

    async function checkServices() {
      try {
        const response = await fetch(`${APP_CONFIG.apiBaseUrl}/api/health`)
        const data = await response.json()
        if (!cancelled) setServiceStatus(response.ok && data.status === 'healthy' ? 'online' : 'offline')
      } catch {
        if (!cancelled) setServiceStatus('offline')
      }
    }

    checkServices()
    const timer = window.setInterval(checkServices, 15000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings))
  }, [settings])

  useEffect(() => {
    if (!analyzing) return undefined
    let cancelled = false

    async function updatePerformance() {
      if (analysisStartedAt.current != null) {
        setAnalysisElapsed((performance.now() - analysisStartedAt.current) / 1000)
      }
      try {
        const telemetry = await fetchPerformance()
        if (!cancelled) setLivePerformance(telemetry)
      } catch {
        // Analiz sonucu yine dönebilir; telemetri hatası analizi durdurmaz.
      }
    }

    updatePerformance()
    const timer = window.setInterval(updatePerformance, 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [analyzing])

  function updateSettings(patch) {
    setSettings(current => ({ ...current, ...patch }))
  }

  function showEventAlert(event) {
    const isCritical = event.risk === 'Kritik'
    const isWarning = event.risk === 'Yüksek' || event.risk === 'Orta'
    if ((!isCritical || !settings.notifyCritical) && (!isWarning || !settings.notifyWarning)) return

    const id = `${event.id}-${Date.now()}`
    setAlerts(current => [...current.slice(-2), { id, event }])
    window.setTimeout(() => {
      setAlerts(current => current.filter(alert => alert.id !== id))
    }, 6000)
  }

  const reportAnalysis = analysis
    ? { ...analysis, operation_orders: Object.values(operationOrders) }
    : null
  const jsonOutput = reportAnalysis ? JSON.stringify(reportAnalysis, null, 2) : ''

  function showEvidence(event) {
    setSelectedEvent(event)
    setDashboardTab('video')
  }

  function selectOperationAction(event, action) {
    setActionDrafts(current => ({ ...current, [event.id]: action.rank }))
  }

  function saveOperationOrder(event) {
    const selectedRank = actionDrafts[event.id]
    const action = event.ranked_actions?.find(item => item.rank === selectedRank)
    if (!action) return
    const order = {
      event_id: event.id,
      event_time: event.time,
      zone: event.zone,
      event_description: event.event,
      action,
      status: 'Operasyon emri verildi',
      saved_at: new Date().toISOString(),
    }
    setOperationOrders(current => ({ ...current, [event.id]: order }))
    const notice = { id: Date.now(), event, action }
    setOperationNotice(notice)
    window.setTimeout(() => {
      setOperationNotice(current => current?.id === notice.id ? null : current)
    }, 5000)
  }

  function handleDeleteVideo() {
    if (actionCenterTimerRef.current) window.clearTimeout(actionCenterTimerRef.current)
    actionCenterTimerRef.current = null
    analysisAbortRef.current?.abort()
    analysisAbortRef.current = null
    if (videoSrc) URL.revokeObjectURL(videoSrc)
    setVideoSrc(null)
    setVideoFileName(null)
    setLogs([])
    setEventRows([])
    setSelectedEvent(null)
    setTick(0)
    setAnalysis(null)
    setAnalysisError(null)
    setAlerts([])
    setAnalyzing(false)
    setLivePerformance(null)
    setAnalysisElapsed(0)
    setOperationNotice(null)
    setActionDrafts({})
    setOperationOrders({})
    setActionCenterOpen(false)
    clearFeedback()
  }

  async function handleFileSelect(file) {
    const extension = file.name.split('.').pop()?.toLowerCase()
    if (!['mp4', 'avi'].includes(extension)) {
      setAnalysisError('Yalnızca MP4 ve AVI video dosyaları desteklenmektedir.')
      return
    }

    if (actionCenterTimerRef.current) window.clearTimeout(actionCenterTimerRef.current)
    actionCenterTimerRef.current = null
    analysisAbortRef.current?.abort()
    const controller = new AbortController()
    analysisAbortRef.current = controller
    if (videoSrc) URL.revokeObjectURL(videoSrc)
    setVideoSrc(null)
    setVideoFileName(file.name)
    setLogs([])
    setEventRows([])
    eventIdCounter.current = 1
    setSelectedEvent(null)
    setTick(0)
    setAnalysis(null)
    setAnalysisError(null)
    setAlerts([])
    setActionDrafts({})
    setOperationOrders({})
    setActionCenterOpen(false)
    setLivePerformance(null)
    setAnalysisElapsed(0)
    analysisStartedAt.current = performance.now()
    clearFeedback()
    setAnalyzing(true)

    try {
      const analysisFile = extension === 'avi'
        ? await convertVideoToMp4(file, { signal: controller.signal })
        : file
      setVideoSrc(URL.createObjectURL(analysisFile))
      setVideoFileName(analysisFile.name)

      const data = await analyzeVideo(analysisFile, { signal: controller.signal })
      const acceptedEvents = (data.events ?? []).filter(
        event => confidencePercent(event.confidence) >= settings.detectionThreshold,
      )
      setAnalysis({ ...data, events: acceptedEvents })

      // Model toplu yanıt verdiği için olayları yapay gecikmeyle değil, analiz biter bitmez göster.
      const normalizedEvents = acceptedEvents.map(entry => {
        const fallbackId = `EVT-${String(eventIdCounter.current++).padStart(3, '0')}`
        return normalizeAnalysisEvent(entry, fallbackId)
      })
      setLogs(acceptedEvents.map(eventToLog).reverse())
      setEventRows(normalizedEvents)
      setTick(normalizedEvents.length)
      normalizedEvents.forEach(showEventAlert)
      setAnalyzing(false)
      if (numericRiskScore(data.risk_score) >= ACTION_RISK_THRESHOLD && getPriorityActionEvents(normalizedEvents).length > 0) {
        actionCenterTimerRef.current = window.setTimeout(() => {
          setActionCenterOpen(true)
          actionCenterTimerRef.current = null
        }, 3000)
      }
    } catch (err) {
      if (err?.name === 'AbortError') return
      setAnalysisError(err instanceof Error ? err.message : 'Analiz sırasında beklenmeyen bir hata oluştu.')
      setAnalyzing(false)
    } finally {
      if (analysisAbortRef.current === controller) analysisAbortRef.current = null
    }
  }

  useEffect(() => () => {
    analysisAbortRef.current?.abort()
    if (actionCenterTimerRef.current) window.clearTimeout(actionCenterTimerRef.current)
    if (videoSrc) URL.revokeObjectURL(videoSrc)
  }, [])

  const visibleLogs = settings.showInfoLogs
    ? logs
    : logs.filter(log => log.level !== 'info')

  const actionRiskScore = numericRiskScore(analysis?.risk_score)
  const priorityActionEvents = getPriorityActionEvents(eventRows)
  const canShowActionCenter = actionRiskScore >= ACTION_RISK_THRESHOLD && priorityActionEvents.length > 0

  return (
    <div className="t-surface-2 flex h-screen overflow-hidden" style={{ backgroundColor: 'var(--background)', color: 'var(--foreground)' }}>

      <div className="fixed right-5 top-5 z-50 w-80 space-y-2">
        {alerts.map(({ id, event }) => (
          <button
            key={id}
            type="button"
            onClick={() => setAlerts(current => current.filter(alert => alert.id !== id))}
            className={cn(
              'w-full rounded-lg border p-3 text-left shadow-xl backdrop-blur',
              event.risk === 'Kritik'
                ? 'border-[#ffb4ab]/50 bg-[#35191d]/95 text-[#ffb4ab]'
                : 'border-[#ffb77d]/50 bg-[#33251b]/95 text-[#ffb77d]',
            )}
          >
            <p className="font-sans text-[10px] font-bold uppercase tracking-widest">
              {event.risk === 'Kritik' ? 'Kritik alarm' : 'Risk uyarısı'}
            </p>
            <p className="mt-1 font-sans text-[12px] text-[#f1f5f9]">{event.event}</p>
            <p className="mt-1 font-mono text-[10px] opacity-75">{event.time} · {event.zone}</p>
          </button>
        ))}
      </div>

      {operationNotice && (
        <button type="button" onClick={() => setOperationNotice(null)}
          className="fixed right-5 top-24 z-[60] w-80 rounded-xl border border-[#4ade80]/50 bg-[#052e16]/95 p-4 text-left shadow-2xl backdrop-blur">
          <p className="font-sans text-[10px] font-bold uppercase tracking-widest text-[#4ade80]">Operasyon emri verildi</p>
          <p className="mt-1 font-sans text-sm font-semibold text-[#f1f5f9]">{operationNotice.action.label}</p>
          <p className="mt-1 font-mono text-[10px] text-[#86efac]">{operationNotice.event.time} · {operationNotice.event.zone} · {operationNotice.action.rank}. sıra</p>
          <p className="mt-2 font-sans text-[10px] text-[#8b919d]">Komut yerel operasyon kaydına işlendi.</p>
        </button>
      )}

      <ActionCenter open={actionCenterOpen}
        events={priorityActionEvents}
        actionDrafts={actionDrafts}
        operationOrders={operationOrders}
        onSelect={selectOperationAction}
        onSave={saveOperationOrder}
        onClose={() => setActionCenterOpen(false)} />


      {/* ── Sidebar ────────────────────────────────────────────────── */}
      <aside className="t-sidebar min-h-0 w-60 shrink-0 flex flex-col overflow-y-auto overscroll-contain bg-[#0d1520] border-r border-[#2a3540]">
        <div className="px-5 py-5 t-border border-b border-[#2a3540]">
          <div className="flex items-center gap-2.5">
            <BarChart3 className="w-8 h-8 shrink-0 text-[#a2c9ff] drop-shadow-[0_0_6px_rgba(162,201,255,0.55)]" strokeWidth={2.25} />
            <div>
              <p className="font-mono text-base font-bold tracking-widest text-[#a2c9ff]">ALFAI</p>
              <p className={cn(
                'font-sans text-[10px] mt-0.5 tracking-wide',
                serviceStatus === 'online' ? 'text-[#4ade80]' : serviceStatus === 'offline' ? 'text-[#ffb4ab]' : 'text-[#8b919d]',
              )}>
                {serviceStatus === 'online' ? 'Model ve API hazır' : serviceStatus === 'offline' ? 'Servis bağlantısı yok' : 'Servisler kontrol ediliyor'}
              </p>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-3 space-y-0.5">
          {NAV_ITEMS.map(({ id, label }) => (
            <button key={id} onClick={() => setActiveNav(id)}
              className={cn(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-150 text-left',
                'font-sans text-[11px] font-bold tracking-widest uppercase',
                activeNav === id
                  ? (isDark ? 'bg-[#1d2b45] text-white' : 'bg-blue-100 text-blue-900 border border-blue-200')
                  : (isDark ? 't-text-muted text-[#8b919d] hover:bg-[#1a2330] hover:text-[#dae3ee]' : 'text-slate-500 hover:bg-slate-200 hover:text-slate-900')
              )}>
              <span className="w-1.5 h-1.5 rounded-full shrink-0"
                style={{ backgroundColor: activeNav === id ? '#a2c9ff' : '#2a3540' }} />
              {label}
            </button>
          ))}
        </nav>

        <PerformancePanel analyzing={analyzing} elapsedSeconds={analysisElapsed} analysis={analysis} telemetry={livePerformance} />

        {/* Global sayaçlar: her bölümde görünür, ana içerik yüksekliğini daraltmaz. */}
        <div className="t-border shrink-0 grid grid-cols-2 gap-2 p-3 border-t border-[#2a3540]">
          {[
            {
              label: 'Model Durumu',
              value: serviceStatus === 'online' ? 'Hazır' : serviceStatus === 'offline' ? 'Kapalı' : '…',
              sub: serviceStatus === 'online' ? 'Qwen bağlantısı aktif' : 'Servisleri kontrol edin',
              colorDark: serviceStatus === 'online' ? 'text-[#4ade80]' : 'text-[#ffb4ab]',
              colorLight: serviceStatus === 'online' ? 'text-emerald-700' : 'text-red-700',
            },
            {
              label: 'Analiz Olayı',
              value: String(eventRows.length),
              sub: `${eventRows.filter(r => r.risk === 'Kritik').length} kritik`,
              colorDark: 'text-[#ffb4ab]', colorLight: 'text-red-700',
            },
          ].map(({ label, value, sub, colorDark, colorLight }) => (
            <div key={label} className="t-surface min-w-0 rounded-lg border border-[#2a3540] bg-[#141c24] px-3 py-2.5">
              <p className="t-text-dim truncate font-sans text-[8px] uppercase tracking-[0.12em] text-[#414752]">{label}</p>
              <p className={cn('mt-1 font-mono text-lg font-bold leading-none', isDark ? colorDark : colorLight)}>{value}</p>
              <p className="t-text-dim mt-1 truncate font-sans text-[9px] text-[#414752]">{sub}</p>
            </div>
          ))}
        </div>
      </aside>

      {/* ── İçerik Alanı ───────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Topbar */}
        {/* Ana içerik — tam genişlik, sağ panel yok */}
        <div className="flex-1 overflow-hidden">
          <div className="h-full flex flex-col overflow-hidden px-5 py-4">

            {activeNav === 'heatmap'  && (
              <div className="flex-1 flex flex-col overflow-hidden gap-3">
                <div className="shrink-0">
                  <h2 className="t-text-base font-sans text-sm font-bold uppercase tracking-widest">Isı Haritası</h2>
                  <p className="t-text-dim font-sans text-[11px] mt-0.5">Hareket yoğunluğu · Anomali dağılımı</p>
                </div>
                <div className="flex-1 overflow-hidden"><HeatmapView fullscreen eventRows={eventRows} videoSrc={videoSrc} /></div>
              </div>
            )}
            {activeNav === 'tactical' && <TacticalReportsPage eventRows={eventRows} analysis={analysis} isDark={isDark} />}
            {activeNav === 'reports'  && <ReportsPage eventRows={eventRows} analysis={reportAnalysis} feedbackMap={feedbackMap} operationOrders={operationOrders} isDark={isDark} />}
            {activeNav === 'kpi'      && <KpiAnalysisPage benchmark={kpiBenchmark} loading={kpiLoading} error={kpiError} />}
            {activeNav === 'json'     && <JsonPage jsonOutput={jsonOutput} />}
            {activeNav === 'settings' && <SettingsPage settings={settings} onChange={updateSettings} />}

            {activeNav === 'dashboard' && (
              <Tabs value={dashboardTab} onValueChange={setDashboardTab} className="flex-1 flex flex-col overflow-hidden">
                <div className="mb-4 flex shrink-0 items-center justify-between gap-3">
                  <TabsList className="t-surface w-fit rounded-lg border border-[#2a3540] bg-[#141c24] p-1">
                    {[['video','Video Analiz'],['events','Olay Tablosu']].map(([v, lbl]) => (
                      <TabsTrigger key={v} value={v}
                        className="font-sans text-[11px] uppercase tracking-widest data-[state=active]:bg-[#a2c9ff] data-[state=active]:text-[#00315c]">
                        {lbl}
                      </TabsTrigger>
                    ))}
                  </TabsList>
                  {canShowActionCenter && (
                    <button type="button" onClick={() => setActionCenterOpen(true)}
                      className="rounded-lg border border-[#a2c9ff]/40 bg-[#101b27] px-3 py-2 font-sans text-[10px] font-bold uppercase tracking-widest text-[#a2c9ff] hover:bg-[#1d2b45]">
                      Aksiyon Merkezi · {priorityActionEvents.filter(event => !operationOrders[event.id]).length} bekleyen
                    </button>
                  )}
                </div>

                {/* Video Analiz — 2/3 video + 1/3 log paneli */}
                <TabsContent value="video" className="flex-1 overflow-hidden mt-0">
                  <div className="h-full grid grid-cols-3 gap-4">

                    {/* Video + zaman çizelgesi (2 pay) */}
                    <div className="col-span-2 overflow-y-auto">
                      <VideoView
                        videoSrc={videoSrc}
                        fileName={videoFileName}
                        onFileSelect={handleFileSelect}
                        onDeleteVideo={handleDeleteVideo}
                        analyzing={analyzing}
                        eventRows={eventRows}
                        selectedEvent={selectedEvent}
                        onFeedback={saveFeedback}
                        feedbackMap={feedbackMap}
                        actionDrafts={actionDrafts}
                        operationOrders={operationOrders}
                        onActionSelect={selectOperationAction}
                        onOperationSave={saveOperationOrder}
                      />
                    </div>

                    {/* Canlı Log Paneli (1 pay) */}
                    <div className={cn(
                      'col-span-1 min-h-0 flex flex-col overflow-hidden rounded-xl border',
                      isDark ? 'border-[#2a3540] bg-[#0d1520]' : 'border-slate-200 bg-white'
                    )}>

                      {/* Karar motoru */}
                      <div className={cn('shrink-0 p-4 border-b', isDark ? 'border-[#2a3540]' : 'border-slate-200')}>
                        <div className="flex items-center justify-between mb-3">
                          <p className={cn('font-sans text-[10px] uppercase tracking-widest', isDark ? 'text-[#414752]' : 'text-slate-400')}>Karar Motoru</p>
                          <span className={cn('font-sans text-[9px] font-semibold', analyzing ? 'text-[#ffb77d]' : videoSrc ? 'text-[#4ade80]' : 'text-[#414752]')}>
                            {analyzing ? 'ANALİZ EDİLİYOR' : videoSrc ? 'TAMAMLANDI' : 'BEKLEMEDE'}
                          </span>
                        </div>
                        <div className="space-y-1.5">
                          {[
                            { color: analyzing ? '#ffb77d' : (isDark ? '#a2c9ff' : '#1d4ed8'),
                              text: analyzing ? 'Video LangGraph ajanına gönderildi…' : analysis ? 'Analiz tamamlandı' : 'Video bekleniyor' },
                            { color: analysis ? getRiskGuidance(analysis.risk_score).color : (isDark ? '#414752' : '#94a3b8'),
                              text: analysis
                                ? <>Risk skoru <b style={{ color: getRiskGuidance(analysis.risk_score).color }}>%{analysis.risk_score}</b> — {analysis.risk}</>
                                : 'Risk değerlendiriliyor…' },
                            { color: analysis ? getRiskGuidance(analysis.risk_score).color : (isDark ? '#98cbff' : '#2563eb'),
                              text: analysisError
                                ? <span className="text-[#ffb4ab]">{analysisError}</span>
                                : analysis ? getRiskGuidance(analysis.risk_score).headline : 'Aksiyon bekleniyor' },
                          ].map(({ color, text }, i) => (
                            <div key={i} className={cn('flex items-center gap-2.5 text-[13px]', isDark ? 'text-[#dae3ee]' : 'text-slate-700')}>
                              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                              <span className="font-sans">{text}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Log başlık */}
                      <div className={cn('shrink-0 flex items-center justify-between px-4 py-3 border-b', isDark ? 'border-[#2a3540]' : 'border-slate-200')}>
                        <p className={cn('font-sans text-[10px] uppercase tracking-widest', isDark ? 'text-[#414752]' : 'text-slate-400')}>Operasyonel Log</p>
                        <span className={cn('font-sans text-[9px] font-semibold', videoSrc ? 'text-[#4ade80]' : 'text-[#414752]')}>
                          {analyzing && tick === 0 ? 'başlıyor...' : videoSrc ? `${tick} olay` : 'video bekleniyor'}
                        </span>
                      </div>

                      {/* Log listesi */}
                      <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain">
                        <div className="p-3 space-y-1.5">
                          {visibleLogs.length === 0 ? (
                            <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
                              <span className="text-2xl opacity-20">🎥</span>
                              <p className={cn('font-sans text-[10px] leading-relaxed', isDark ? 'text-[#414752]' : 'text-slate-400')}>
                                Video analizi tamamlandığında<br />tespit sonuçları burada gösterilir
                              </p>
                            </div>
                          ) : (
                            visibleLogs.map(log => <LogLine key={log.id} {...log} isDark={isDark} />)
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* Olay Tablosu */}
                <TabsContent value="events" className="flex-1 overflow-hidden mt-0">
                  <Card className="t-surface bg-[#141c24] border-[#2a3540] h-full flex flex-col">
                    <CardHeader className="px-5 py-4 t-border border-b border-[#2a3540] shrink-0">
                      <CardTitle className="t-text-muted font-sans text-[11px] uppercase tracking-widest text-[#8b919d] flex items-center justify-between">
                        <span>Zaman Damgalı Olay Günlüğü</span>
                        <span className="t-text-dim text-[#414752]">{eventRows.length} olay</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0 flex-1 overflow-auto">
                      {eventRows.length === 0 ? (
                        <div className="flex items-center justify-center h-full">
                          <p className="t-text-dim font-sans text-[11px] text-[#414752]">Video yüklendiğinde olaylar burada listelenir</p>
                        </div>
                      ) : (
                        <Table>
                          <TableHeader>
                            <TableRow className="t-border border-b border-[#2a3540] hover:bg-transparent">
                              {['Zaman','Bölge','Olay','Risk','Güven','Aksiyonlar','Kanıt'].map(h => (
                                <TableHead key={h} className="t-text-dim font-sans text-[10px] uppercase tracking-widest text-[#414752] h-9">{h}</TableHead>
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {[...eventRows].reverse().map((row, i) => (
                              <TableRow key={i} className="t-border border-b border-[#2a3540] hover:bg-[#1a2330]/50">
                                <TableCell className={cn('font-mono text-[12px]', isDark ? 'text-[#a2c9ff]' : 'text-blue-700')}>{row.time}</TableCell>
                                <TableCell className={cn('font-sans text-[13px]', isDark ? 'text-[#8b919d]' : 'text-slate-500')}>{row.zone}</TableCell>
                                <TableCell className={cn('font-sans text-[13px]', isDark ? 'text-[#dae3ee]' : 'text-slate-800')}>{row.event}</TableCell>
                                <TableCell><RiskBadge level={row.risk} isDark={isDark} /></TableCell>
                                <TableCell className={cn('font-mono text-[12px]', isDark ? 'text-[#8b919d]' : 'text-slate-500')}>{row.conf}</TableCell>
                                <TableCell><RankedActionButtons event={row}
                                  selectedRank={actionDrafts[row.id]}
                                  savedOrder={operationOrders[row.id]}
                                  onSelect={selectOperationAction}
                                  onSave={saveOperationOrder}
                                  compact /></TableCell>
                                <TableCell>
                                  <button type="button" onClick={() => showEvidence(row)}
                                    className="rounded-md border border-[#a2c9ff]/30 px-2.5 py-1 font-sans text-[10px] text-[#a2c9ff] transition-colors hover:bg-[#1d2b45]">
                                    Kanıtı Göster
                                  </button>
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            )}
          </div>
        </div>
      </div>
      <ChatAssistant analysis={analysis} analyzing={analyzing} />
    </div>
  )
}

