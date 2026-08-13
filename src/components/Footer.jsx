export default function Footer() {
  return (
    <footer className="fixed bottom-0 right-0 left-60 bg-[#050a12] border-t border-[#1c2b45] flex justify-between items-center px-8 h-8 z-40">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 text-[#f59e0b] font-mono text-[12px]">
          <span className="w-2 h-2 rounded-full bg-[#f59e0b]" />
          GPU Usage: 42%
        </div>
        <div className="flex items-center gap-2 text-[#60a5fa] font-mono text-[12px]">
          <span className="w-2 h-2 rounded-full bg-[#60a5fa]" />
          vLLM: ACTIVE
        </div>
        <div className="flex items-center gap-2 text-[#10b981] font-mono text-[12px]">
          <span className="w-2 h-2 rounded-full bg-[#10b981]" />
          Local Server: Online
        </div>
      </div>
      <div className="font-mono text-[11px] text-[#3a4f6e]">
        Vardiya Gözü v2.5 | GPU Cluster: 8× A40 | Latency: 45ms
      </div>
    </footer>
  )
}
