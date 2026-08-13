export default function Header({ title }) {
  return (
    <header className="fixed top-0 w-full z-50 flex items-center gap-4 px-6 h-16 bg-[#070b14]/95 backdrop-blur-md border-b border-[#1c2b45]">
      <span className="text-lg font-bold text-[#e8f0ff] tracking-tight">Vardiya Gözü</span>
      <div className="h-4 w-px bg-[#1c2b45]" />
      <span className="text-sm text-[#7d93b8]">{title}</span>
    </header>
  )
}
