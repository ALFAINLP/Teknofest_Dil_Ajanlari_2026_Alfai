import { useEffect, useRef } from 'react'
import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import Footer from '../components/Footer'

const SURVEILLANCE_IMG = 'https://lh3.googleusercontent.com/aida-public/AB6AXuBnEy8490R7CUSElVbctrpJv1WF-rIQrclFcE-cOoTLbap7Pd0QaGXyGL77MeuHeZpw283wAEkWg9gJgWMNsMVTh3G7MAQO8YKcqCNee5NnPrM78loE4rjZTCnhj7FpfD2AX3cbyvdvqpyCSJiEJbtQ6CAOlO2AVMCnUy7fiKAEyhk_qkDg1M9UQA-yU-ZrwYG6EfcaK79snGe5X6KePjWjYA7nSWrUPmypEa2e3Py4cG9lq9KHkLcBzLHAFI0g1DzrJc2oCFNVALjv'

const initialLogs = [
  { time: '00:15', msg: "Forklift hız sınırını aştı. Plaka: TR-34-FKL-01", level: 'primary' },
  { time: '00:42', msg: "Personel 'Zon A' bölgesinden çıkış yaptı.", level: 'normal' },
  { time: '04:22', msg: 'Kaza Öncesi Uyarı: Forklift hızlanıyor, zemin ıslak. Çarpışma riski %85.', level: 'error' },
  { time: '04:25', msg: 'Güvenlik kapısı otomatik kilitlendi.', level: 'normal' },
]

function LogEntry({ time, msg, level }) {
  const borderColor = level === 'error' ? 'border-[#ef4444]' : level === 'primary' ? 'border-[#3b82f6]' : 'border-[#1c2b45]'
  const bg          = level === 'error' ? 'bg-[#1c0a0a]/60 risk-glow' : 'bg-[#0c1120]'
  const timeColor   = level === 'error' ? 'text-[#ef4444]' : 'text-[#3b82f6]'
  const textColor   = level === 'error' ? 'text-[#fca5a5]' : 'text-[#7d93b8]'
  return (
    <div className={`p-3 rounded border-l-2 text-[12px] ${borderColor} ${bg}`}>
      <span className={`font-mono mr-2 font-bold ${timeColor}`}>{time}</span>
      <span className={textColor}>{msg}</span>
    </div>
  )
}

export default function VardiyaGozuDashboard() {
  const logsRef = useRef(null)

  useEffect(() => {
    const id = setInterval(() => {
      if (!logsRef.current) return
      const now = new Date()
      const t = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}`
      const div = document.createElement('div')
      div.className = 'p-3 rounded border-l-2 border-[#3b82f6] bg-[#0c1120] text-[12px]'
      div.innerHTML = `<span class="font-mono font-bold text-[#3b82f6] mr-2">${t}</span><span class="text-[#7d93b8]">Sistem log girişi güncellendi.</span>`
      logsRef.current.prepend(div)
      if (logsRef.current.children.length > 8) logsRef.current.lastChild.remove()
    }, 5000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="bg-[#070b14] text-[#e8f0ff]">
      <Header title="Operasyonel Kontrol Merkezi" />
      <Sidebar />

      {/* Main: starts below header (mt-16), fills remaining viewport height minus footer (h-[calc(100vh-4rem-2rem)]) */}
      <main className="ml-60 mt-16 h-[calc(100vh-4rem-2rem)] flex overflow-hidden">

        {/* Middle Column */}
        <section className="flex-1 p-4 border-r border-[#1c2b45] bg-[#070b14] flex flex-col gap-4 overflow-y-auto min-h-0">

          {/* Upload Dropzone */}
          <div className="border-2 border-dashed border-[#1c2b45] rounded-xl p-4 flex flex-row items-center justify-center gap-4 hover:border-[#2563eb]/50 transition-colors group cursor-pointer bg-[#050a12] flex-shrink-0">
            <span className="material-symbols-outlined text-[#7d93b8] group-hover:text-[#3b82f6] transition-colors" style={{ fontSize: '2rem' }}>cloud_upload</span>
            <div className="flex flex-col">
              <p className="text-sm text-[#e8f0ff] font-medium">Yeni Video Yükle</p>
              <p className="text-[10px] text-[#3a4f6e]">MP4, MKV (Max 2GB)</p>
            </div>
          </div>

          {/* Surveillance Feed */}
          <div className="flex-1 relative glass-panel rounded-xl overflow-hidden min-h-64">
            <div className="absolute top-4 left-4 z-20 flex flex-col gap-1">
              <div className="flex items-center gap-2 px-3 py-1 bg-[#1c0a0a] text-[#ef4444] rounded-full text-xs font-bold border border-[#ef4444]/30 animate-pulse">
                <span className="w-2 h-2 rounded-full bg-[#ef4444]" />
                CANLI ANALİZ: AKTİF
              </div>
              <div className="bg-black/60 backdrop-blur-md px-3 py-1 rounded text-[11px] font-mono text-white/70 border border-white/10">
                CAM_04 // Fabrika Ana Giriş
              </div>
            </div>

            <div className="scanline" />

            <div className="w-full h-full bg-cover bg-center min-h-64" style={{ backgroundImage: `url('${SURVEILLANCE_IMG}')` }} />

            <div className="absolute inset-0 pointer-events-none bg-gradient-to-t from-[#070b14]/50 to-transparent" />

            <div className="absolute top-4 right-4 z-20 font-mono text-[10px] text-[#3b82f6] space-y-1 text-right">
              <div>LAT: 41.0082° N</div>
              <div>LON: 28.9784° E</div>
              <div>RES: 3840×2160</div>
              <div>FPS: 60.0</div>
            </div>

            <div className="absolute bottom-4 left-4 z-20 flex gap-3">
              <div className="glass-panel px-3 py-2 rounded-lg">
                <span className="text-[9px] text-[#3a4f6e] font-mono uppercase tracking-widest block">Vardiya</span>
                <span className="text-sm font-bold text-[#e8f0ff]">08:00 - 16:00</span>
              </div>
              <div className="glass-panel px-3 py-2 rounded-lg border-l-4 border-[#2563eb]">
                <span className="text-[9px] text-[#3a4f6e] font-mono uppercase tracking-widest block">Bölge</span>
                <span className="text-sm font-bold text-[#e8f0ff]">Fabrika Ana Giriş</span>
              </div>
            </div>
          </div>

          {/* Timeline */}
          <div className="glass-panel rounded-xl p-4 flex flex-col gap-3 flex-shrink-0">
            <div className="flex justify-between items-center">
              <h3 className="font-mono text-[10px] tracking-widest uppercase text-[#3a4f6e]">Temporal Analysis (PySceneDetect)</h3>
              <span className="text-[10px] font-mono text-[#3a4f6e]">Detected Events: 04</span>
            </div>
            <div className="relative h-10 bg-[#0c1120] rounded-lg overflow-hidden flex items-center border border-[#1c2b45]">
              <div className="absolute inset-0 flex justify-between px-2 opacity-20 pointer-events-none">
                {[...Array(6)].map((_, i) => <span key={i} className="h-full w-px bg-[#3a4f6e]" />)}
              </div>
              {[
                { left: '15%', color: '#f59e0b', title: '00:15 - Forklift Girişi', pulse: false },
                { left: '42%', color: '#ef4444', title: '04:22 - Hız İhlali',      pulse: true },
                { left: '65%', color: '#3b82f6', title: '08:12 - Personel Geçişi', pulse: false },
                { left: '88%', color: '#3b82f6', title: '12:45 - Kapı Kapandı',   pulse: false },
              ].map(({ left, color, title, pulse }, i) => (
                <div
                  key={i}
                  className={`absolute w-3 h-3 rounded-full border-2 border-[#070b14] cursor-help group/tip ${pulse ? 'animate-pulse' : ''}`}
                  style={{ left, backgroundColor: color }}
                  title={title}
                >
                  <div className="hidden group-hover/tip:block absolute bottom-6 left-1/2 -translate-x-1/2 bg-[#0c1120] p-2 rounded text-[10px] whitespace-nowrap z-30 border border-[#1c2b45] text-[#e8f0ff]">
                    {title}
                  </div>
                </div>
              ))}
              <div className="absolute left-[55%] h-full w-[2px] bg-[#2563eb] z-10" style={{ boxShadow: '0 0 8px #2563eb' }} />
            </div>
            <div className="flex justify-between text-[10px] font-mono text-[#3a4f6e]">
              {['00:00','05:00','10:00','15:00'].map(t => <span key={t}>{t}</span>)}
            </div>
          </div>
        </section>

        {/* Right Panel */}
        <aside className="w-96 bg-[#050a12] flex flex-col min-h-0 overflow-y-auto">

          {/* Görsel Anlama */}
          <div className="p-5 border-b border-[#1c2b45] flex flex-col gap-3" style={{ minHeight: 0 }}>
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[#3b82f6]">visibility</span>
              <h2 className="text-base font-semibold text-[#e8f0ff]">Görsel Anlama</h2>
              <span className="ml-auto text-[9px] font-mono text-[#3a4f6e] uppercase tracking-widest">Qwen2.5-VL-72B</span>
            </div>
            <div className="space-y-2 pr-1 max-h-52 overflow-y-auto" ref={logsRef}>
              {initialLogs.map((log, i) => <LogEntry key={i} {...log} />)}
            </div>
          </div>

          {/* Akıllı Karar Motoru */}
          <div className="p-5 border-b border-[#1c2b45] flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[#8b5cf6]">psychology</span>
              <h2 className="text-base font-semibold text-[#e8f0ff]">Akıllı Karar Motoru</h2>
              <span className="ml-auto text-[9px] font-mono text-[#3a4f6e] uppercase tracking-widest">LangGraph</span>
            </div>
            <div className="flex flex-col gap-2">
              {[
                { color: '#3b82f6',  label: 'Analiz Ediliyor',                                                    bold: false },
                { color: '#ef4444',  label: <>Risk Hesaplandı: <span className="text-[#ef4444] font-bold">YÜKSEK</span></>, bold: true },
                { color: '#60a5fa',  label: <>Aksiyon: <span className="bg-[#1d4ed8]/20 px-2 py-0.5 rounded text-[#60a5fa]">Ambulans Çağır</span></>, bold: false },
              ].map(({ color, label, bold }, i) => (
                <div key={i}>
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-md flex items-center justify-center" style={{ backgroundColor: `${color}18`, border: `1px solid ${color}33` }}>
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                    </div>
                    <div className={`text-[12px] text-[#e8f0ff] ${bold ? 'font-bold' : ''}`}>{label}</div>
                  </div>
                  {i < 2 && <div className="w-px h-3 bg-[#1c2b45] ml-[11px]" />}
                </div>
              ))}
              <div className="mt-2 p-3 rounded-lg bg-[#0c1120] border border-[#1c2b45] flex items-center justify-between">
                <span className="text-[10px] font-mono uppercase tracking-widest text-[#3a4f6e]">Agent State</span>
                <span className="text-[10px] font-mono text-[#10b981] font-bold">COMPLETED</span>
              </div>
            </div>
          </div>

          {/* JSON Output */}
          <div className="p-5 flex flex-col gap-3 flex-1 min-h-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[#3a4f6e]">terminal</span>
                <h2 className="text-base font-semibold text-[#e8f0ff]">JSON Çıktısı</h2>
              </div>
              <button
                className="text-[10px] font-mono text-[#3b82f6] hover:underline uppercase tracking-widest"
                onClick={() => navigator.clipboard?.writeText(`{"event_id":"EVT_49221","timestamp":"2023-10-27T04:22:15Z","risk_score":0.85}`)}
              >
                COPY
              </button>
            </div>
            <div className="flex-1 bg-[#0c1120] border border-[#1c2b45] rounded-lg p-3 font-mono text-[11px] min-h-0 overflow-hidden">
              <pre className="text-[#7d93b8] overflow-y-auto h-full">{`{
  "event_id": "EVT_49221",
  "timestamp": "2023-10-27T04:22:15Z",
  "risk_score": 0.85,
  "objects": [
    { "type": "forklift", "speed": 18.4 },
    { "type": "personnel", "status": "alert" }
  ],
  "agent_reasoning": "High speed on wet floor...",
  "recommended_action": "TRIGGER_ALARM"
}`}</pre>
            </div>
          </div>
        </aside>
      </main>

      <Footer />
    </div>
  )
}
