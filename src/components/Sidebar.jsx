import { NavLink } from 'react-router-dom'

const navItems = [
  { to: '/',        icon: 'dashboard',   label: 'Dashboard' },
  { to: '/rapor',   icon: 'description', label: 'Shift Reports' },
  { to: '/heatmap', icon: 'grain',       label: 'Heatmaps' },
]

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-full w-60 bg-[#070b14] border-r border-[#1c2b45] flex flex-col py-6 px-4 gap-y-4 z-40">
      <div className="pt-16 pb-4">
        <div className="p-3 rounded-lg bg-[#0c1120] border border-[#1c2b45] flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[#2563eb]/20 flex items-center justify-center text-[#3b82f6] border border-[#2563eb]/30">
            <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>shield</span>
          </div>
          <div>
            <div className="font-mono text-[10px] font-bold tracking-widest text-[#3b82f6] uppercase">Operator Alpha-1</div>
            <div className="text-[10px] font-mono text-[#7d93b8]">Shift Active: 04:22</div>
          </div>
        </div>
      </div>

      <nav className="flex-1 flex flex-col gap-1">
        {navItems.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 font-mono text-[11px] font-bold tracking-widest uppercase ${
                isActive
                  ? 'bg-[#2563eb] text-white shadow-lg shadow-[#2563eb]/25'
                  : 'text-[#7d93b8] hover:text-[#e8f0ff] hover:bg-[#0c1120]'
              }`
            }
          >
            <span className="material-symbols-outlined">{icon}</span>
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto flex flex-col gap-3">
        {/* Sistem durumu */}
        {[
          { color: '#10b981', label: 'Qwen2.5-VL-72B', sub: 'Görsel analiz — AKTİF' },
          { color: '#10b981', label: 'Llama-3.3-70B',  sub: 'Karar motoru — AKTİF' },
          { color: '#8b5cf6', label: 'vLLM',            sub: '8× A40 GPU' },
        ].map(({ color, label, sub }) => (
          <div key={label} className="flex items-center gap-2.5 px-3 py-2 rounded-lg bg-[#0c1120] border border-[#1c2b45]">
            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: color, boxShadow: `0 0 5px ${color}` }} />
            <div>
              <div className="font-mono text-[10px] font-bold text-[#e8f0ff]">{label}</div>
              <div className="font-mono text-[9px] text-[#3a4f6e]">{sub}</div>
            </div>
          </div>
        ))}
        <div className="p-3 bg-[#2563eb] text-white rounded-lg text-center font-bold tracking-widest text-[10px] font-mono shadow-lg shadow-[#2563eb]/25">
          SYSTEM READY
        </div>
      </div>
    </aside>
  )
}
