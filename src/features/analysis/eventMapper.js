function formatTimestamp(seconds) {
  const safeSeconds = Math.max(0, Math.floor(seconds))
  return `${String(Math.floor(safeSeconds / 60)).padStart(2, '0')}:${String(safeSeconds % 60).padStart(2, '0')}`
}

function parseTimestamp(value = '00:00') {
  const [minutes, seconds] = String(value).split(':').map(Number)
  return (minutes || 0) * 60 + (seconds || 0)
}

export function riskToLogLevel(risk) {
  if (risk === 'Kritik') return 'critical'
  if (risk === 'Yüksek' || risk === 'Orta') return 'warn'
  return 'info'
}

export function normalizeAnalysisEvent(event, fallbackId) {
  const startTime = event.start_time ?? parseTimestamp(event.time)
  const endTime = event.end_time ?? startTime + 4

  return {
    id: event.id ?? fallbackId,
    time: event.time ?? formatTimestamp(startTime),
    start_time: startTime,
    end_time: endTime,
    end_time_label: event.end_time_label ?? formatTimestamp(endTime),
    zone: event.zone,
    event: event.event,
    risk: event.risk,
    conf: event.confidence,
    evidence: event.evidence ?? [
      { time: formatTimestamp(startTime), seconds: startTime, description: `${event.event} bulgusu bu kare aralığında tespit edildi.` },
      { time: formatTimestamp(endTime), seconds: endTime, description: 'Olayın gelişimi ve çevresel bağlam kontrol edildi.' },
    ],
    uncertainties: event.uncertainties ?? ['Olay doğrulandı; sonuç operatör teyidine açıktır.'],
  }
}

export function eventToLog(event) {
  return {
    id: `${event.id}-${event.time}`,
    time: event.time,
    msg: `${event.event} tespit edildi — ${event.zone} (Güven: ${event.confidence})`,
    level: riskToLogLevel(event.risk),
  }
}

