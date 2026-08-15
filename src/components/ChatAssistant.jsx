import { useEffect, useRef, useState } from 'react'
import { Bot, LoaderCircle, MessageSquare, Send, Trash2, User, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { sendChatMessage } from '@/services/analysisApi'

const SUGGESTIONS = [
  'En yüksek riskli olay neydi?',
  'Bu risk neden verildi?',
  'Operatör şimdi ne yapmalı?',
]

export default function ChatAssistant({ analysis, analyzing }) {
  const [open, setOpen] = useState(false)
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [messages, setMessages] = useState([])
  const messagesEndRef = useRef(null)

  useEffect(() => {
    if (open) messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, sending, open])

  async function submitMessage(rawMessage) {
    const message = rawMessage.trim()
    if (!message || sending) return

    const history = messages.map(({ role, content }) => ({ role, content }))
    setMessages(previous => [...previous, { id: `user-${Date.now()}`, role: 'user', content: message }])
    setInput('')
    setError('')
    setSending(true)

    try {
      const result = await sendChatMessage({ message, analysisContext: analysis, history })
      setMessages(previous => [...previous, {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: result.reply,
      }])
    } catch (chatError) {
      setError(chatError instanceof Error ? chatError.message : 'Yanıt üretilirken beklenmeyen bir hata oluştu.')
    } finally {
      setSending(false)
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    submitMessage(input)
  }

  return (
    <div className="fixed bottom-5 right-5 z-50 font-sans">
      {open && (
        <section aria-label="ALFAI Chat Asistanı"
          className="mb-3 flex h-[min(560px,calc(100vh-110px))] w-[min(390px,calc(100vw-32px))] flex-col overflow-hidden rounded-2xl border border-[#334252] bg-[#0d1520] shadow-2xl shadow-black/50">
          <header className="flex shrink-0 items-center justify-between border-b border-[#2a3540] bg-[#141c24] px-4 py-3.5">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-[#a2c9ff]/25 bg-[#1d2b45]">
                <Bot className="h-5 w-5 text-[#a2c9ff]" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-[12px] font-bold tracking-[0.14em] text-[#dae3ee]">ALFAI ASİSTAN</p>
                  <span className="h-1.5 w-1.5 rounded-full bg-[#4ade80] shadow-[0_0_6px_#4ade80]" />
                </div>
                <p className="mt-0.5 truncate text-[10px] text-[#708090]">
                  {analysis ? 'Aktif video analizi bağlamı hazır' : 'Analiz bağlamı bekleniyor'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              {messages.length > 0 && (
                <button type="button" onClick={() => { setMessages([]); setError('') }} aria-label="Sohbeti temizle"
                  className="rounded-lg p-2 text-[#708090] transition-colors hover:bg-[#1a2330] hover:text-[#dae3ee]">
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
              <button type="button" onClick={() => setOpen(false)} aria-label="Chat penceresini kapat"
                className="rounded-lg p-2 text-[#708090] transition-colors hover:bg-[#1a2330] hover:text-[#dae3ee]">
                <X className="h-4 w-4" />
              </button>
            </div>
          </header>

          <div className="flex-1 overflow-y-auto overscroll-contain px-4 py-4">
            {messages.length === 0 && (
              <div className="space-y-4">
                <div className="rounded-xl border border-[#2a3540] bg-[#141c24] p-3.5 text-[12px] leading-relaxed text-[#aeb9c6]">
                  {analysis
                    ? 'Video analizi hazır. Olayların nedeni, risk seviyesi veya önerilen aksiyonlar hakkında soru sorabilirsiniz.'
                    : 'Henüz aktif bir video analizi yok. Önce video yükleyerek bağlama dayalı sorular sorabilirsiniz.'}
                </div>
                {analysis && (
                  <div>
                    <p className="mb-2 text-[9px] font-semibold uppercase tracking-[0.16em] text-[#586675]">Önerilen sorular</p>
                    <div className="space-y-2">
                      {SUGGESTIONS.map(suggestion => (
                        <button key={suggestion} type="button" onClick={() => submitMessage(suggestion)}
                          className="block w-full rounded-lg border border-[#2a3540] bg-[#101923] px-3 py-2.5 text-left text-[11px] text-[#a2c9ff] transition-colors hover:border-[#a2c9ff]/45 hover:bg-[#1d2b45]">
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="space-y-3">
              {messages.map(message => (
                <div key={message.id} className={cn('flex gap-2.5', message.role === 'user' && 'flex-row-reverse')}>
                  <div className={cn('mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border',
                    message.role === 'user' ? 'border-[#a2c9ff]/25 bg-[#1d2b45] text-[#a2c9ff]' : 'border-[#2a3540] bg-[#141c24] text-[#8b919d]')}>
                    {message.role === 'user' ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
                  </div>
                  <div className={cn('max-w-[82%] whitespace-pre-wrap rounded-xl px-3.5 py-2.5 text-[12px] leading-relaxed',
                    message.role === 'user' ? 'rounded-tr-sm bg-[#1d3a5d] text-[#e5efff]' : 'rounded-tl-sm border border-[#2a3540] bg-[#141c24] text-[#c2ccd7]')}>
                    {message.content}
                  </div>
                </div>
              ))}
              {sending && (
                <div className="flex items-center gap-2.5 text-[11px] text-[#708090]">
                  <div className="flex h-7 w-7 items-center justify-center rounded-lg border border-[#2a3540] bg-[#141c24]"><Bot className="h-3.5 w-3.5" /></div>
                  <div className="flex items-center gap-2 rounded-xl border border-[#2a3540] bg-[#141c24] px-3 py-2.5">
                    <LoaderCircle className="h-3.5 w-3.5 animate-spin text-[#a2c9ff]" />Yanıt hazırlanıyor
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {error && <div role="alert" className="mx-4 mb-2 rounded-lg border border-[#ffb4ab]/25 bg-[#3a1719] px-3 py-2 text-[10px] leading-relaxed text-[#ffb4ab]">{error}</div>}

          <form onSubmit={handleSubmit} className="shrink-0 border-t border-[#2a3540] bg-[#101923] p-3">
            <div className="flex items-center gap-2">
              <Input value={input} onChange={event => setInput(event.target.value)}
                placeholder={analyzing ? 'Analiz tamamlanırken soru yazabilirsiniz…' : 'Analiz hakkında bir soru sorun…'}
                maxLength={1000} aria-label="Chat mesajı"
                className="h-10 border-[#334252] bg-[#0b141c] text-[12px] text-[#dae3ee] placeholder:text-[#586675] focus-visible:ring-[#a2c9ff]/60" />
              <Button type="submit" disabled={!input.trim() || sending} aria-label="Mesajı gönder"
                className="h-10 w-10 shrink-0 bg-[#a2c9ff] p-0 text-[#00315c] hover:bg-[#b9d7ff] disabled:bg-[#263442] disabled:text-[#586675]">
                {sending ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
            <p className="mt-2 text-center text-[9px] text-[#4e5b68]">Yanıtlar aktif analiz ve kayıtlı hafıza ile sınırlandırılır</p>
          </form>
        </section>
      )}

      <button type="button" onClick={() => setOpen(previous => !previous)} aria-expanded={open}
        aria-label={open ? 'Chat asistanını kapat' : 'Chat asistanını aç'}
        className="ml-auto flex h-12 items-center gap-2.5 rounded-2xl border border-[#a2c9ff]/30 bg-[#1d2b45] px-4 text-[#dceaff] shadow-xl shadow-black/40 transition-all hover:-translate-y-0.5 hover:border-[#a2c9ff]/55 hover:bg-[#243754]">
        {open ? <X className="h-5 w-5" /> : <MessageSquare className="h-5 w-5 text-[#a2c9ff]" />}
        {!open && <span className="text-[11px] font-bold tracking-[0.1em]">AI ASİSTAN</span>}
      </button>
    </div>
  )
}
