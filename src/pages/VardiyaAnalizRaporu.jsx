import Header from '../components/Header'
import Sidebar from '../components/Sidebar'
import Footer from '../components/Footer'

const metrics = [
  {
    label: 'TOPLAM ANALİZ SÜRESİ',
    icon: 'timer',
    iconColor: '#3b82f6',
    value: '08:00:00',
    badge: '+0.2%',
    badgeColor: '#10b981',
    bar: <div className="mt-2 w-full bg-[#111827] h-1 rounded-full overflow-hidden"><div className="bg-[#2563eb] h-full w-full" /></div>,
  },
  {
    label: 'TESPİT EDİLEN OLAYLAR',
    icon: 'visibility',
    iconColor: '#60a5fa',
    value: '14 olay',
    badge: 'Normal Seviye',
    badgeColor: '#f59e0b',
    bar: (
      <div className="mt-2 flex gap-1">
        {[1,1,0,0].map((fill,i) => (
          <div key={i} className={`h-1 rounded-full w-full ${fill ? 'bg-[#3b82f6]' : 'bg-[#111827]'}`} />
        ))}
      </div>
    ),
  },
  {
    label: 'KRİTİK RİSK DURUMU',
    icon: 'warning',
    iconColor: '#ef4444',
    value: '3 Yüksek Risk',
    valueColor: '#ef4444',
    badge: 'DİKKAT',
    badgeColor: '#ef4444',
    extra: <div className="mt-2 text-[10px] font-mono text-[#7d93b8] uppercase">Son 1 saatte 1 yeni tespit</div>,
    glow: true,
  },
  {
    label: 'SİSTEM SAĞLIĞI',
    icon: 'bolt',
    iconColor: '#10b981',
    value: '%99.8 Uptime',
    valueColor: '#10b981',
    bar: (
      <div className="mt-2 flex items-center gap-2">
        <div className="flex-1 flex gap-0.5 items-end h-4">
          {[2,3,2,4,2].map((h,i) => (
            <div key={i} className="w-1 bg-[#10b981] rounded-sm" style={{ height: `${h*4}px`, opacity: 0.4 + i*0.15 }} />
          ))}
        </div>
        <span className="text-[10px] font-mono text-[#7d93b8]">LATENCY: 12ms</span>
      </div>
    ),
  },
]

export default function VardiyaAnalizRaporu() {
  return (
    <div className="bg-[#070b14] text-[#e8f0ff]">
      <Header title="Vardiya Analiz Raporu" />
      <Sidebar />

      <main className="ml-60 pt-16 pb-8 min-h-screen">
        <div className="p-6 pb-16 space-y-5">

          {/* Mimari şerit — alfaii.py'dan */}
          <div className="grid grid-cols-4 gap-2">
            {[
              { color: '#2563eb', num: '01', title: 'VİDEO İŞLEME',  sub: 'FFmpeg · PySceneDetect' },
              { color: '#8b5cf6', num: '02', title: 'GÖRSEL ANALİZ', sub: 'Qwen2.5-VL-72B' },
              { color: '#f59e0b', num: '03', title: 'KARAR MOTORU',  sub: 'LangGraph · Llama-3.3-70B' },
              { color: '#10b981', num: '04', title: 'ÇIKTI & RAPOR', sub: 'Türkçe özet · JSON · KPI' },
            ].map(({ color, num, title, sub }) => (
              <div key={num} className="rounded-lg p-3 text-center bg-[#0c1120] border flex flex-col justify-center" style={{ borderColor: `${color}25` }}>
                <div className="font-mono text-[8px] font-bold tracking-widest mb-1" style={{ color }}>{num}</div>
                <div className="text-[11px] font-bold text-[#e8f0ff] mb-0.5">{title}</div>
                <div className="text-[9px] text-[#3a4f6e]">{sub}</div>
              </div>
            ))}
          </div>

          {/* Row 1: Metrics */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {metrics.map(({ label, icon, iconColor, value, valueColor, badge, badgeColor, bar, extra, glow }, i) => (
              <div
                key={i}
                className={`p-4 rounded-xl flex flex-col justify-between transition-all duration-300 hover:border-[#2563eb]/40 bg-[#0c1120] border ${glow ? 'critical-glow' : 'border-[#1c2b45]'}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-mono text-[9px] tracking-widest uppercase" style={{ color: glow ? '#ef4444' : '#3a4f6e' }}>{label}</span>
                  <span className={`material-symbols-outlined text-lg ${glow ? 'pulse-warning' : ''}`} style={{ color: iconColor }}>{icon}</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span className="font-mono text-xl font-bold" style={{ color: valueColor || '#e8f0ff' }}>{value}</span>
                  {badge && <span className="text-[10px] font-bold" style={{ color: badgeColor }}>{badge}</span>}
                </div>
                {bar}
                {extra}
              </div>
            ))}
          </div>

          {/* Row 2: Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Risk Timeline */}
            <div className="bg-[#0c1120] border border-[#1c2b45] p-5 rounded-xl">
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-base font-semibold text-[#e8f0ff] flex items-center gap-2">
                  <span className="material-symbols-outlined text-[#3b82f6]">analytics</span>
                  Risk Yoğunluğu
                </h3>
                <div className="flex gap-2 items-center">
                  <span className="w-2.5 h-2.5 bg-[#ef4444] rounded-full" />
                  <span className="text-[9px] font-mono uppercase tracking-widest text-[#3a4f6e]">KRİTİK EŞİK</span>
                </div>
              </div>
              <div className="relative h-56 w-full">
                <svg className="w-full h-full" viewBox="0 0 400 150">
                  {[0,50,100,150].map(y => <line key={y} stroke="#1c2b45" strokeWidth="0.5" x1="0" x2="400" y1={y} y2={y} />)}
                  <line opacity="0.5" stroke="#ef4444" strokeDasharray="4" strokeWidth="1" x1="0" x2="400" y1="40" y2="40" />
                  <defs>
                    <linearGradient id="grad1" x1="0%" x2="0%" y1="0%" y2="100%">
                      <stop offset="0%"   stopColor="#2563eb" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#2563eb" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path d="M0,150 L0,120 L50,130 L100,80 L150,110 L200,30 L250,90 L300,100 L350,50 L400,70 L400,150 Z" fill="url(#grad1)" />
                  <path className="chart-line" d="M0,120 L50,130 L100,80 L150,110 L200,30 L250,90 L300,100 L350,50 L400,70" fill="none" stroke="#3b82f6" strokeWidth="2" />
                </svg>
                <div className="absolute bottom-0 w-full flex justify-between px-2 pt-2 border-t border-[#1c2b45] font-mono text-[10px] text-[#3a4f6e]">
                  {['08:00','10:00','12:00','14:00','16:00'].map(t => <span key={t}>{t}</span>)}
                </div>
              </div>
            </div>

            {/* Olay Dağılımı */}
            <div className="bg-[#0c1120] border border-[#1c2b45] p-5 rounded-xl">
              <h3 className="text-base font-semibold text-[#e8f0ff] flex items-center gap-2 mb-5">
                <span className="material-symbols-outlined text-[#60a5fa]">pie_chart</span>
                Olay Dağılımı
              </h3>
              <div className="flex items-center h-52 gap-6">
                <div
                  className="relative w-40 h-40 rounded-full border-[18px] border-[#111827] flex-shrink-0"
                  style={{ borderTopColor: '#3b82f6', borderRightColor: '#f59e0b', borderBottomColor: '#ef4444' }}
                >
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="font-mono text-2xl font-bold text-[#e8f0ff]">14</span>
                    <span className="text-[9px] font-mono uppercase tracking-widest text-[#3a4f6e]">TOPLAM</span>
                  </div>
                </div>
                <div className="flex-1 space-y-4">
                  {[
                    { color: '#3b82f6', label: 'Hız İhlali',      count: 6 },
                    { color: '#f59e0b', label: 'Yetkisiz Giriş',  count: 5 },
                    { color: '#ef4444', label: 'İş Kazası Riski', count: 3 },
                  ].map(({ color, label, count }) => (
                    <div key={label} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color }} />
                        <span className="text-sm text-[#7d93b8]">{label}</span>
                      </div>
                      <span className="font-mono font-bold text-[#e8f0ff]">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Row 3: AI Summary */}
          <div className="bg-[#0c1120] border border-[#1c2b45] rounded-xl overflow-hidden">
            <div className="bg-[#2563eb]/10 border-b border-[#1c2b45] px-5 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[#3b82f6]" style={{ fontVariationSettings: "'FILL' 1" }}>neurology</span>
                <h3 className="font-mono text-[10px] tracking-widest uppercase text-[#3b82f6]">YAPAY ZEKA VARDİYA ÖZETİ — Qwen2.5-VL-72B</h3>
              </div>
              <div className="flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#3b82f6] opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#3b82f6]" />
                </span>
                <span className="text-[9px] font-mono text-[#3b82f6]">ANALİZ TAMAMLANDI</span>
              </div>
            </div>
            <div className="p-5">
              <p className="text-sm leading-relaxed text-[#7d93b8] max-w-4xl">
                Bugünkü vardiyada ana giriş bölgesinde forklift trafiği{' '}
                <span className="text-[#3b82f6] font-bold">%20 artış</span> gösterdi.
                Saat 10:30 sularında zemin ıslaklığı nedeniyle{' '}
                <span className="text-[#f59e0b] font-bold">2 adet ramak kala olay</span> tespit edildi.
                Personelin kask kullanımı{' '}
                <span className="text-[#10b981] font-bold">%95 seviyesinde</span> seyretmekte olup,
                sevkiyat alanında baretsiz 1 kişi tespit edilerek anlık uyarı verilmiştir.
                Genel risk skoru: <span className="text-[#3b82f6]">Düşük-Orta</span>.
              </p>
              <div className="mt-5 flex gap-3">
                {[
                  { icon: 'download', label: 'PDF OLARAK İNDİR' },
                  { icon: 'share',    label: 'YÖNETİME GÖNDER' },
                ].map(({ icon, label }) => (
                  <button
                    key={label}
                    className="px-4 py-2 bg-[#111827] hover:bg-[#1a2540] rounded-lg border border-[#1c2b45] transition-colors flex items-center gap-2 cursor-pointer"
                  >
                    <span className="material-symbols-outlined text-sm text-[#3b82f6]">{icon}</span>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-[#7d93b8]">{label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

        </div>
      </main>

      <Footer />
    </div>
  )
}
