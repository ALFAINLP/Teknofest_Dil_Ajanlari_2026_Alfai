import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import Header from '../components/Header'
import Footer from '../components/Footer'

const BLUEPRINT_IMG = 'https://lh3.googleusercontent.com/aida-public/AB6AXuC8IMaGJv9dtr_h-b9_ZXVGJjhih6RMsJ9JYeBc6Z278_DCI15tTifIH4Lo4lFVUZh2NePSRN8SZ7CFDWR7v3BpjF3nqgIECxdkdSCkgW2r3U3_jnZm3c6a1fWSJOy99LJ6I1kBq7Bid9x6FOncJfwPUwwev6PvdDIUDiMEgqMR9ogsBX9n6DdoPVAAVsX9uxaYFD7X98idigA8L_FFRhttC0xLo5dRYHiKzq2RfZIUuO2h77xX2xhVwOiCTfmv-yOUl-udY5NNSWjk'

const navItems = [
  { to: '/',        icon: 'dashboard',   label: 'Dashboard' },
  { to: '/rapor',   icon: 'description', label: 'Shift Reports' },
  { to: '/heatmap', icon: 'grain',       label: 'Heatmaps' },
]

const toggleButtons = [
  { icon: 'stream',     label: 'Canlı Yoğunluk' },
  { icon: 'history',    label: 'Geçmiş Risk Verisi' },
  { icon: 'filter_alt', label: 'Anomali Filtresi' },
]

export default function AnomaliIsiHaritasi() {
  const [activeToggle, setActiveToggle] = useState(0)

  return (
    <div className="bg-[#070b14] text-[#e8f0ff]">
      <Header title="Anomali Isı Haritası" />

      {/* fixed-height layout below header, above footer */}
      <div className="flex" style={{ height: 'calc(100vh - 4rem - 2rem)', marginTop: '4rem' }}>

        {/* Sidebar */}
        <aside className="w-60 flex-shrink-0 bg-[#070b14] border-r border-[#1c2b45] flex flex-col py-5 px-4 gap-y-3 overflow-y-auto">
          <div className="flex items-center gap-3 px-1 mb-3">
            <div className="w-9 h-9 rounded-lg bg-[#2563eb]/20 flex items-center justify-center border border-[#2563eb]/30">
              <span className="material-symbols-outlined text-[#3b82f6]">shield</span>
            </div>
            <div>
              <p className="font-mono text-[10px] font-bold tracking-widest text-[#3b82f6] uppercase">Operator Alpha-1</p>
              <p className="font-mono text-[9px] text-[#3a4f6e]">Shift Active: 04:22</p>
            </div>
          </div>
          <nav className="flex flex-col gap-1">
            {navItems.map(({ to, icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 font-mono text-[11px] font-bold tracking-widest uppercase ${
                    isActive
                      ? 'bg-[#2563eb] text-white shadow-lg shadow-[#2563eb]/25'
                      : 'text-[#7d93b8] hover:bg-[#0c1120]'
                  }`
                }
              >
                <span className="material-symbols-outlined">{icon}</span>
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
          <div className="mt-auto">
            <div className="p-3 bg-[#0c1120] rounded-lg border border-[#1c2b45]">
              <div className="flex justify-between items-center mb-2">
                <span className="font-mono text-[9px] uppercase tracking-widest text-[#3a4f6e]">System Status</span>
                <span className="flex h-2 w-2 rounded-full bg-[#3b82f6] animate-pulse" />
              </div>
              <p className="font-mono text-[10px] font-bold tracking-widest text-[#3b82f6] uppercase">SYSTEM READY</p>
            </div>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 flex overflow-hidden bg-[#070b14]">

          {/* Map area: flex-col so toggles sit below the scroll container */}
          <div className="flex-1 flex flex-col overflow-hidden">

            {/* Scrollable map canvas */}
            <div className="flex-1 blueprint-grid flex items-center justify-center p-6 overflow-auto min-h-0">
              <div className="relative w-[1200px] h-[780px] border border-[#1c2b45] bg-[#050a12]/60 shadow-2xl rounded-xl overflow-hidden flex-shrink-0">

                <div
                  className="absolute inset-0 w-full h-full opacity-30 mix-blend-screen bg-cover bg-center"
                  style={{ backgroundImage: `url('${BLUEPRINT_IMG}')` }}
                />

                {/* Heatmap overlays */}
                <div className="absolute inset-0 pointer-events-none">
                  <div className="absolute top-1/4 left-1/3 w-64 h-64 bg-[#ef4444]/35 blur-[60px] rounded-full heatmap-pulse mix-blend-screen" />
                  <div className="absolute top-[28%] left-[36%] w-16 h-16 bg-[#ef4444] blur-[20px] rounded-full" />
                  <div className="absolute bottom-1/4 right-1/4 w-48 h-48 bg-[#ef4444]/25 blur-[50px] rounded-full heatmap-pulse mix-blend-screen" />
                  <div className="absolute top-[15%] right-[20%] w-40 h-40 bg-[#f59e0b]/25 blur-[40px] rounded-full mix-blend-screen" />
                  <div className="absolute bottom-[20%] left-[15%] w-56 h-56 bg-[#f59e0b]/15 blur-[45px] rounded-full mix-blend-screen" />

                  {/* HUD marker */}
                  <div className="absolute top-[28%] left-[36%] border-t-2 border-l-2 border-[#ef4444] w-8 h-8 -translate-x-1/2 -translate-y-1/2" />
                  <div className="absolute top-[28%] left-[36%] mt-4 ml-4">
                    <span className="bg-[#ef4444] text-white font-mono text-[10px] px-2 py-0.5 rounded shadow-lg">ANOMALİ TESPİT EDİLDİ</span>
                  </div>
                </div>

                <div className="absolute top-4 left-5 pointer-events-none">
                  <h2 className="font-mono text-[10px] uppercase tracking-widest text-[#3a4f6e]">Kat Planı: Seviye 02 — Lojistik Merkezi</h2>
                </div>

                {/* Legend */}
                <div className="absolute bottom-4 left-5 bg-[#0c1120]/90 backdrop-blur-sm border border-[#1c2b45] p-3 rounded-lg flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#ef4444]" />
                    <span className="font-mono text-[9px] text-[#7d93b8]">Kritik Risk / Yoğun Anomali</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#f59e0b]" />
                    <span className="font-mono text-[9px] text-[#7d93b8]">Orta Seviye Hareketlilik</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Toggle Controls — always visible, outside the scroll container */}
            <div className="flex-shrink-0 flex justify-center py-3 bg-[#070b14] border-t border-[#1c2b45]">
              <div className="flex gap-3 bg-[#0c1120] border border-[#1c2b45] p-1.5 rounded-xl">
                {toggleButtons.map(({ icon, label }, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveToggle(i)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-mono text-[11px] uppercase tracking-widest transition-all active:scale-95 ${
                      activeToggle === i
                        ? 'bg-[#2563eb] text-white shadow-lg shadow-[#2563eb]/25'
                        : 'text-[#7d93b8] hover:bg-[#111827]'
                    }`}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: '0.9rem' }}>{icon}</span>
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Sidebar */}
          <aside className="w-80 flex-shrink-0 bg-[#050a12] border-l border-[#1c2b45] flex flex-col p-5 overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-sm font-bold text-[#e8f0ff] tracking-tight uppercase">Bölge Bazlı Risk Analizi</h3>
              <span className="material-symbols-outlined text-[#3b82f6] cursor-pointer">info</span>
            </div>

            {/* Section A */}
            <div className="space-y-3 mb-6">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[9px] uppercase tracking-widest text-[#3a4f6e]">SENSOR GROUP: ENTRY_A</span>
                <span className="bg-[#1c0a0a] text-[#ef4444] border border-[#ef4444]/30 font-mono text-[9px] px-2 py-0.5 rounded font-bold">KRİTİK</span>
              </div>
              <div className="p-4 rounded-lg bg-[#1c0a0a]/40 border border-[#ef4444]/20">
                <div className="flex items-start gap-3 mb-3">
                  <span className="material-symbols-outlined text-[#ef4444] text-sm">warning</span>
                  <div>
                    <p className="text-sm font-semibold text-[#e8f0ff]">A Bölgesi (Ana Giriş)</p>
                    <p className="text-xs text-[#7d93b8] mt-1 leading-relaxed">
                      Güvenlik protokolü dışında kümelenme tespit edildi. 4+ personel girişi 03:45'ten itibaren sabit.
                    </p>
                  </div>
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="w-full bg-[#111827] h-1.5 rounded-full overflow-hidden">
                    <div className="bg-[#ef4444] h-full" style={{ width: '88%' }} />
                  </div>
                  <span className="font-mono text-[9px] text-[#ef4444] self-end">Risk Puanı: 88/100</span>
                </div>
              </div>
            </div>

            {/* Section B */}
            <div className="space-y-3 mb-6">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[9px] uppercase tracking-widest text-[#3a4f6e]">SENSOR GROUP: LOAD_D</span>
                <span className="bg-[#1c1408] text-[#f59e0b] border border-[#f59e0b]/30 font-mono text-[9px] px-2 py-0.5 rounded font-bold">ORTA</span>
              </div>
              <div className="p-4 rounded-lg bg-[#1c1408]/40 border border-[#f59e0b]/20">
                <div className="flex items-start gap-3 mb-3">
                  <span className="material-symbols-outlined text-[#f59e0b] text-sm">inventory_2</span>
                  <div>
                    <p className="text-sm font-semibold text-[#e8f0ff]">Yükleme Alanı</p>
                    <p className="text-xs text-[#7d93b8] mt-1 leading-relaxed">
                      Ekipman depolama sınırında beklenmeyen hareketlilik. Forklift trafiği normalin %12 üzerinde.
                    </p>
                  </div>
                </div>
                <div className="flex flex-col gap-1.5">
                  <div className="w-full bg-[#111827] h-1.5 rounded-full overflow-hidden">
                    <div className="bg-[#f59e0b] h-full" style={{ width: '42%' }} />
                  </div>
                  <span className="font-mono text-[9px] text-[#f59e0b] self-end">Risk Puanı: 42/100</span>
                </div>
              </div>
            </div>

            {/* AI Insights */}
            <div className="mt-auto pt-5 border-t border-[#1c2b45]">
              <h4 className="font-mono text-[9px] uppercase tracking-widest text-[#3b82f6] mb-3">AI GÖZLEMLERİ</h4>
              <ul className="space-y-2">
                {[
                  'Geçen haftaya göre anomali artışı: %8.4',
                  'En güvenli bölge: Ofis Katı (Risk 0)',
                  'Tahmini bakım uyarısı: 14 saat içinde.',
                ].map((item, i) => (
                  <li key={i} className="font-mono text-[11px] text-[#7d93b8] flex gap-2">
                    <span className="text-[#3b82f6]">•</span>
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </aside>
        </main>
      </div>

      <Footer />
    </div>
  )
}
